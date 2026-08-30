"""
跌倒风险监控服务 — 整合完整链路:
视频拉流 → 人体检测 → 关键点提取 → 帧过滤 → 特征计算 → 基线对比 → 偏离检测 → 分级预警
音频采集 → 声音事件识别 → 音频预警升级 (并行线程)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.alerts.engine import AlertEngine, AlertEvent, RiskLevel
from src.api.database import Database
from src.api.websocket import manager, raw_video_manager, video_manager
from src.data.audio_capture import AudioCapture
from src.data.environment_detector import EnvironmentBox, EnvironmentDetector
from src.data.frame_filter import FrameFilter
from src.data.human_detector import DetectionBox, HumanDetector, PrimaryPersonTracker, box_iou
from src.data.keypoint_extractor import create_keypoint_extractor
from src.data.video_capture import VideoCapture
from src.data.yolo_pose_extractor import YoloPoseExtractor
from src.inference.audio_analyzer import AudioAnalysisResult, AudioAnalyzer, AudioEvent
from src.inference.baseline import BaselineManager
from src.inference.deviation import DeviationDetector, DeviationResult
from src.inference.features import FeatureCalculator, FeatureVector
from src.inference.post_impact import PostImpactDetector, PostImpactResult
from src.inference.visual_risk import (
    CausalTrajectoryProvider,
    EngineeringRiskFusion,
    FusionDecision,
    MotionRiskTracker,
    compute_clutter_risk,
    compute_environment_risk,
    compute_interaction_risk,
    compute_lighting_risk,
    risk_state,
    unavailable_wet_floor,
)
from src.notifications.service import NotificationService
from src.utils.config import get_config
from src.utils.draw import draw_overlay, encode_jpeg
from src.utils.keypoints import KeypointFrame
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MonitorStatus:
    """监控状态"""

    is_running: bool = False
    person_id: str = "default"
    device_id: str = "default"
    source: str = ""
    source_type: str = "unknown"
    source_name: str = ""
    temporary_source_path: str | None = None
    frames_processed: int = 0
    frames_valid: int = 0
    last_feature: FeatureVector | None = None
    last_deviation: DeviationResult | None = None
    last_alert: AlertEvent | None = None
    last_notification: dict | None = None
    current_risk_level: RiskLevel = RiskLevel.LOW
    baseline_ready: bool = False
    baseline_samples: int = 0
    recent_keypoints: list[KeypointFrame] = field(default_factory=list)
    audio_enabled: bool = False
    audio_source: str = ""
    last_audio_result: AudioAnalysisResult | None = None
    audio_chunks_processed: int = 0
    audio_error: str | None = None
    human_detection: DetectionBox | None = None
    human_detections: list[DetectionBox] = field(default_factory=list)
    human_keypoints_present: bool = False
    human_keypoint_state: str = "MISSING"
    human_keypoint_reason: str = "person_missing"
    human_keypoint_score: float | None = None
    human_error: str | None = None
    person_match: bool | None = None
    environment_boxes: list[EnvironmentBox] = field(default_factory=list)
    environment_persons: list[DetectionBox] = field(default_factory=list)
    environment_error: str | None = None
    environment_stale: bool = True
    illumination: float | None = None
    environment_last_updated: float | None = None
    motion_score: float | None = None
    deviation_risk_score: float | None = None
    human_risk_score: float | None = None
    environment_risk_score: float | None = None
    interaction_risk_score: float | None = None
    environment_state: str = "UNKNOWN"
    top_hazards: list = field(default_factory=list)
    risk_extensions: dict = field(default_factory=dict)
    post_impact: PostImpactResult | None = None
    fusion: FusionDecision | None = None
    current_risk_score: float = 0.0
    current_risk_message: str = "等待视觉风险数据"
    base_risk_level: RiskLevel = RiskLevel.LOW
    base_risk_message: str = "人体基线风险正常"
    _pending_audio_events: list[AudioEvent] = field(default_factory=list)
    _audio_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class FallRiskMonitor:
    """跌倒风险监控服务(单例)"""

    _instance: FallRiskMonitor | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.config = get_config()
        self.person_id = "default"
        self.device_id = "default"
        self.status = MonitorStatus()

        # 核心组件
        self.video_capture: VideoCapture | None = None
        self.human_detector = HumanDetector()
        self.environment_detector = EnvironmentDetector()
        self.primary_person_tracker = PrimaryPersonTracker()
        self.environment_person_tracker = PrimaryPersonTracker()
        self.pose_backend = self.config.pose_estimation.get("backend", "mediapipe")
        self.keypoint_extractor = create_keypoint_extractor()
        self.frame_filter = FrameFilter()
        self.feature_calculator = FeatureCalculator()
        self.baseline_manager = BaselineManager()
        self.deviation_detector = DeviationDetector()
        self.alert_engine = AlertEngine()
        self.notification_service = NotificationService()
        engineering_config = self.config.engineering_risk
        self.motion_risk_tracker = MotionRiskTracker(
            scale=float(engineering_config.motion_velocity_scale)
        )
        self.trajectory_provider = CausalTrajectoryProvider(
            dict(self.config.risk_extensions.trajectory)
        )
        self.engineering_fusion = EngineeringRiskFusion(
            upgrade_confirmations=int(engineering_config.upgrade_confirmations),
            downgrade_confirmations=int(engineering_config.downgrade_confirmations),
        )
        self.post_impact_detector = PostImpactDetector(dict(self.config.get("post_impact", {})))
        self._last_risk_record_time = 0.0
        self._last_emitted_alert_time = 0.0
        self._last_emitted_alert_level = RiskLevel.LOW
        self._last_emitted_reason_signature = ""

        # 音频组件 (start() 时按 audio_source 决定是否启用)
        self.audio_analyzer: AudioAnalyzer | None = None
        self.audio_capture: AudioCapture | None = None
        self._audio_thread: threading.Thread | None = None

        # 运行控制
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._temporary_source_lock = threading.Lock()
        self._keypoint_buffer: list[KeypointFrame] = []
        self._buffer_window = 30  # 特征计算窗口帧数

    def start(
        self,
        source: str,
        person_id: str = "default",
        device_id: str = "default",
        audio_source: str | None = None,
        temporary_source_path: str | None = None,
    ) -> bool:
        """启动监控"""
        if self.status.is_running:
            return False

        self._keypoint_buffer.clear()
        self.deviation_detector.reset()
        self.alert_engine.reset()
        for component_name in (
            "primary_person_tracker",
            "environment_person_tracker",
            "motion_risk_tracker",
            "trajectory_provider",
            "engineering_fusion",
            "post_impact_detector",
        ):
            component = getattr(self, component_name, None)
            if component is not None:
                component.reset()
        if isinstance(self.keypoint_extractor, YoloPoseExtractor):
            self.keypoint_extractor.reset()
        self._last_risk_record_time = 0.0
        self._last_emitted_alert_time = 0.0
        self._last_emitted_alert_level = RiskLevel.LOW
        self._last_emitted_reason_signature = ""
        self.person_id = person_id
        self.device_id = device_id

        audio_cfg_enabled = self.config.get("audio", {}).get("enabled", False)
        cfg_audio_source = self.config.get("audio", {}).get("source", "off")
        effective_audio_source = audio_source if audio_source is not None else cfg_audio_source

        if effective_audio_source in ("auto", "video_source"):
            effective_audio_source = (
                source if source.lower().startswith(("rtsp://", "rtmp://")) else "off"
            )

        audio_enabled = audio_cfg_enabled and effective_audio_source not in ("off", "")

        self.status = MonitorStatus(
            is_running=True,
            person_id=person_id,
            device_id=device_id,
            source=source,
            source_type="uploaded" if temporary_source_path else "stream",
            source_name=Path(source).name if temporary_source_path else "",
            temporary_source_path=temporary_source_path,
            audio_enabled=audio_enabled,
            audio_source=effective_audio_source if audio_enabled else "",
        )
        self._stop_flag.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        if audio_enabled:
            self.audio_analyzer = AudioAnalyzer()
            self.audio_capture = AudioCapture(
                source=effective_audio_source,
                sample_rate=int(self.config.audio.sample_rate),
                chunk_seconds=int(self.config.audio.chunk_seconds),
                input_device=self.config.audio.get("input_device"),
                ffmpeg_path=str(self.config.audio.get("ffmpeg_path", "")),
                stop_event=self._stop_flag,
            )
            self._audio_thread = threading.Thread(target=self._run_audio, daemon=True)
            self._audio_thread.start()

        return True

    def stop(self):
        """停止监控"""
        self._stop_flag.set()
        # 先关闭采集器(解除ffmpeg阻塞读), 再 join 线程, 避免 stop 卡满超时
        if self.audio_capture:
            self.audio_capture.close()
            self.audio_capture = None
        if self._thread:
            self._thread.join(timeout=5)
        if self._audio_thread:
            self._audio_thread.join(timeout=5)
            self._audio_thread = None
        self.status.is_running = False
        if self.video_capture:
            self.video_capture.close()
            self.video_capture = None
        self._cleanup_temporary_source()

    def _cleanup_temporary_source(self) -> None:
        """删除当前会话创建的上传文件，不触碰普通视频源。"""
        lock = getattr(self, "_temporary_source_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._temporary_source_lock = lock
        with lock:
            path_value = self.status.temporary_source_path
            self.status.temporary_source_path = None
        if not path_value:
            return
        try:
            Path(path_value).unlink(missing_ok=True)
        except OSError as exc:
            log.warning(f"清理上传视频失败: {exc}")

    def _run(self):
        """监控主循环(在子线程运行)"""
        inference_interval = self.config.inference.inference_interval_ms / 1000
        person_id = self.person_id
        device_id = self.device_id
        environment_config = self.config.environment_detection
        environment_interval = max(int(environment_config.interval_frames), 1)
        environment_ttl = float(environment_config.cache_ttl_seconds)
        persistence_interval = float(self.config.engineering_risk.persistence_interval_seconds)
        _last_broadcast = 0.0
        _last_raw_broadcast = 0.0
        _last_base_evaluation = 0.0

        try:
            baseline = self.baseline_manager.load_baseline(person_id)
            if baseline is None:
                baseline = self.baseline_manager.compute_baseline(person_id)
            self.status.baseline_ready = baseline.is_ready
            self.status.baseline_samples = baseline.sample_count

            with VideoCapture(source=self.status.source) as cap:
                self.video_capture = cap

                for video_frame in cap.frames():
                    if self._stop_flag.is_set():
                        break
                    self.status.frames_processed += 1
                    now = time.time()

                    # YOLO Pose 后端一次推理同时取得人体框与关键点。
                    kp_frame: KeypointFrame | None = None
                    keypoint_score: float | None = None
                    human_boxes: list[DetectionBox] = []
                    primary_person: DetectionBox | None = None
                    try:
                        if isinstance(self.keypoint_extractor, YoloPoseExtractor):
                            pose_result = self.keypoint_extractor.extract_result(video_frame)
                            human_boxes = [person.box for person in pose_result.people]
                            if pose_result.primary is not None:
                                primary_person = pose_result.primary.box
                                kp_frame = pose_result.primary.keypoint_frame
                                keypoint_score = pose_result.primary.keypoint_score
                        else:
                            human_boxes = self.human_detector.detect(video_frame.frame)
                            primary_person = self.primary_person_tracker.select(human_boxes)
                            if primary_person is not None:
                                kp_frame = self.keypoint_extractor.extract(video_frame)
                        self.status.human_error = None
                    except Exception as exc:
                        self.status.human_error = str(exc)
                        log.error(f"人体识别失败: {exc}")

                    self.status.human_detections = human_boxes
                    self.status.human_detection = primary_person
                    self.status.human_keypoints_present = kp_frame is not None
                    self.status.human_keypoint_state = (
                        "MISSING"
                        if kp_frame is None
                        else "OK"
                        if kp_frame.is_valid
                        else "LOW_QUALITY"
                    )
                    self.status.human_keypoint_reason = (
                        "person_missing"
                        if primary_person is None
                        else "keypoints_missing"
                        if kp_frame is None
                        else kp_frame.invalid_reason or ""
                    )
                    self.status.human_keypoint_score = keypoint_score
                    self.status.motion_score = self.motion_risk_tracker.update(
                        video_frame.timestamp, primary_person
                    )
                    self.status.post_impact = self.post_impact_detector.update(primary_person)

                    lighting = compute_lighting_risk(
                        video_frame.frame, dict(self.config.risk_extensions.lighting)
                    )
                    self.status.illumination = lighting.get("evidence", {}).get("mean_luminance")

                    should_update_environment = (
                        self.status.environment_last_updated is None
                        or (self.status.frames_processed - 1) % environment_interval == 0
                    )
                    if should_update_environment:
                        try:
                            detection_result = self.environment_detector.detect_result(
                                video_frame.frame
                            )
                            self.status.environment_persons = detection_result.persons
                            self.status.environment_boxes = detection_result.objects
                            self.status.environment_last_updated = now
                            self.status.environment_error = None
                        except Exception as exc:
                            self.status.environment_error = str(exc)
                            log.error(f"环境识别失败: {exc}")

                    self.status.environment_stale = (
                        self.status.environment_last_updated is None
                        or now - self.status.environment_last_updated > environment_ttl
                    )
                    environment_boxes = (
                        self.status.environment_boxes if not self.status.environment_stale else []
                    )
                    self.status.person_match = self._match_environment_person(
                        primary_person, self.status.environment_persons
                    )

                    if self.status.environment_stale:
                        environment = {
                            "source": "env_risk_v0",
                            "available": False,
                            "score": None,
                            "state": "UNKNOWN",
                            "top_hazards": [],
                            "reason_codes": ["environment_cache_stale"],
                        }
                        clutter = {
                            "source": "clutter_v0",
                            "available": False,
                            "risk_index": None,
                            "state": "UNKNOWN",
                            "evidence": {},
                            "reason_codes": ["environment_cache_stale"],
                            "obstacles": [],
                        }
                    else:
                        environment = compute_environment_risk(
                            primary_person,
                            environment_boxes,
                            dict(self.config.environment_risk),
                        )
                        clutter = compute_clutter_risk(
                            primary_person,
                            environment_boxes,
                            dict(self.config.risk_extensions.clutter),
                            video_frame.frame.shape,
                        )

                    trajectory = self.trajectory_provider.update(
                        video_frame.timestamp, primary_person
                    )
                    wet_floor = unavailable_wet_floor()
                    interaction = (
                        compute_interaction_risk(
                            trajectory,
                            environment_boxes,
                            wet_floor,
                            dict(self.config.risk_extensions.interaction),
                        )
                        if not self.status.environment_stale
                        else {
                            "source": "interaction_v0",
                            "available": False,
                            "risk_index": None,
                            "state": "UNKNOWN",
                            "evidence": {},
                            "reason_codes": ["environment_cache_stale"],
                            "intersections": [],
                        }
                    )

                    self.status.top_hazards = environment["top_hazards"]
                    self.status.environment_risk_score = environment.get("score")
                    self.status.environment_state = environment.get("state", "UNKNOWN")
                    self.status.interaction_risk_score = interaction.get("risk_index")
                    self.status.risk_extensions = {
                        "source": "risk_extensions_v0_3_2",
                        "human_risk_index": self.status.human_risk_score,
                        "human_risk_state": risk_state(self.status.human_risk_score),
                        "environment_risk_index": None,
                        "environment_risk_state": "UNKNOWN",
                        "interaction_risk_index": self.status.interaction_risk_score,
                        "interaction_risk_state": risk_state(self.status.interaction_risk_score),
                        "lighting": lighting,
                        "clutter": clutter,
                        "trajectory": trajectory,
                        "wet_floor": wet_floor,
                        "interaction": interaction,
                        "post_impact": (
                            self.status.post_impact.to_dict()
                            if self.status.post_impact
                            else None
                        ),
                    }

                    new_deviation: DeviationResult | None = None
                    feature: FeatureVector | None = None
                    if kp_frame is not None:
                        self.frame_filter.filter(kp_frame)
                        if kp_frame.is_valid:
                            self.status.frames_valid += 1
                            self._keypoint_buffer.append(kp_frame)
                            if len(self._keypoint_buffer) > self._buffer_window:
                                self._keypoint_buffer = self._keypoint_buffer[-self._buffer_window:]
                            self.status.recent_keypoints = list(self._keypoint_buffer)
                            if len(self._keypoint_buffer) >= 10:
                                feature = self.feature_calculator.calculate(self._keypoint_buffer)
                                self.status.last_feature = feature
                                if not baseline.is_ready:
                                    self.baseline_manager.add_sample(self.person_id, feature)
                                    baseline = self.baseline_manager.compute_baseline(self.person_id)
                                    self.status.baseline_ready = baseline.is_ready
                                    self.status.baseline_samples = baseline.sample_count
                                if baseline.is_ready:
                                    new_deviation = self.deviation_detector.check(feature, baseline)
                                    self.status.last_deviation = new_deviation
                                    self.status.deviation_risk_score = min(
                                        100.0,
                                        max(0.0, new_deviation.mahalanobis_distance / 6.0 * 100),
                                    )

                    with self.status._audio_lock:
                        pending_events = list(self.status._pending_audio_events)
                        self.status._pending_audio_events.clear()

                    if new_deviation is not None or pending_events or now - _last_base_evaluation >= 1:
                        base_alert = self.alert_engine.evaluate(
                            new_deviation or self.status.last_deviation or DeviationResult(),
                            video_frame.timestamp,
                            has_activity=primary_person is not None,
                            audio_events=pending_events or None,
                            emit=False,
                        )
                        _last_base_evaluation = now
                        self.status.base_risk_level = base_alert.level
                        self.status.base_risk_message = base_alert.message

                    motion_index = (
                        None
                        if self.status.motion_score is None
                        else self.status.motion_score * 100.0
                    )
                    human_values = [
                        value
                        for value in (motion_index, self.status.deviation_risk_score)
                        if value is not None
                    ]
                    human_index = max(human_values) if human_values else None
                    environment_values = [
                        value
                        for value in (
                            None
                            if environment.get("score") is None
                            else environment["score"] * 100.0,
                            clutter.get("risk_index"),
                        )
                        if value is not None
                    ]
                    environment_index = max(environment_values) if environment_values else None
                    self.status.human_risk_score = human_index
                    self.status.risk_extensions["human_risk_index"] = human_index
                    self.status.risk_extensions["human_risk_state"] = risk_state(human_index)
                    self.status.risk_extensions["environment_risk_index"] = environment_index
                    self.status.risk_extensions["environment_risk_state"] = risk_state(environment_index)
                    self.status.risk_extensions["interaction_risk_index"] = interaction.get("risk_index")
                    self.status.risk_extensions["interaction_risk_state"] = risk_state(
                        interaction.get("risk_index")
                    )

                    previous_level = self.status.current_risk_level
                    decision = self.engineering_fusion.evaluate(
                        human_index=human_index,
                        environment_index=environment_index if primary_person is not None else None,
                        interaction_index=(
                            interaction.get("risk_index") if primary_person is not None else None
                        ),
                        base_level=self.status.base_risk_level.value,
                        base_reason_codes=[f"base_{self.status.base_risk_level.value}"],
                        acute_critical=(
                            self.status.base_risk_level == RiskLevel.CRITICAL
                            or (motion_index is not None and motion_index >= 70)
                        ),
                    )
                    self._apply_post_impact_override(decision)
                    self.status.fusion = decision
                    self.status.current_risk_score = decision.overall_score
                    self.status.current_risk_level = RiskLevel(decision.overall_level)
                    self.status.current_risk_message = self._risk_message(decision)
                    if self.status.post_impact is not None and self.status.post_impact.confirmed:
                        self.status.current_risk_message = (
                            "已跌倒（FALL）：检测到人体倒地且持续不活动，请立即确认现场情况"
                        )
                    self.status.risk_extensions["overall_engineering_state_v0_3"] = decision.overall_level.upper()
                    if self.status.current_risk_level == RiskLevel.LOW:
                        self._last_emitted_alert_level = RiskLevel.LOW

                    if self.status.current_risk_level != previous_level:
                        manager.broadcast_threadsafe(
                            {
                                "type": "risk_update",
                                "level": self.status.current_risk_level.value,
                                "score": self.status.current_risk_score,
                                "reason_codes": decision.reason_codes,
                            }
                        )

                    if self._should_emit_alert(decision, now):
                        alert = AlertEvent(
                            level=self.status.current_risk_level,
                            timestamp=video_frame.timestamp,
                            message=self.status.current_risk_message,
                            deviation=self.status.last_deviation,
                        )
                        self.alert_engine.emit_event(alert)
                        self.status.last_alert = alert
                        alert_id = self._persist_alert(alert, decision, person_id, device_id)
                        try:
                            self.status.last_notification = self.notification_service.dispatch(
                                alert,
                                alert_id=alert_id,
                                risk_score=decision.overall_score,
                                person_id=person_id,
                                device_id=device_id,
                                reason_codes=decision.reason_codes,
                            )
                        except Exception as exc:
                            self.status.last_notification = None
                            log.error(f"通知编排失败，保留原有告警链路: {exc}")
                        manager.broadcast_threadsafe(
                            {
                                "type": "alert",
                                "level": alert.level.value,
                                "score": decision.overall_score,
                                "message": alert.message,
                                "reason_codes": decision.reason_codes,
                                "notification_id": (
                                    self.status.last_notification.get("notification_id")
                                    if self.status.last_notification
                                    else None
                                ),
                            }
                        )

                    if now - self._last_risk_record_time >= persistence_interval:
                        self._persist_risk_record(
                            decision,
                            feature,
                            environment,
                            person_id,
                            device_id,
                        )
                        self._last_risk_record_time = now

                    if video_manager.has_clients and now - _last_broadcast >= 0.1:
                        _last_broadcast = now
                        try:
                            overlay = draw_overlay(
                                video_frame.frame,
                                kp_frame,
                                risk_level=self.status.current_risk_level.value,
                                baseline_ready=self.status.baseline_ready,
                                frames_processed=self.status.frames_processed,
                                human_box=primary_person,
                                human_boxes=human_boxes,
                                environment_boxes=environment_boxes,
                                illumination=self.status.illumination,
                                top_hazards=self.status.top_hazards,
                                trajectory=trajectory,
                                risk_score=self.status.current_risk_score,
                                human_risk_score=self.status.human_risk_score,
                                environment_risk_score=(
                                    self.status.fusion.environment_risk_index
                                    if self.status.fusion
                                    else None
                                ),
                                interaction_risk_score=(
                                    self.status.fusion.interaction_risk_index
                                    if self.status.fusion
                                    else None
                                ),
                            )
                            video_manager.broadcast_frame(encode_jpeg(overlay))
                        except Exception as exc:
                            log.debug(f"分析画面叠加失败: {exc}")

                    if raw_video_manager.has_clients and now - _last_raw_broadcast >= 0.1:
                        _last_raw_broadcast = now
                        try:
                            raw_video_manager.broadcast_frame(encode_jpeg(video_frame.frame))
                        except Exception:
                            pass

                    time.sleep(inference_interval)
        except Exception as e:
            log.error(f"监控线程异常: {e}")
        finally:
            self.status.is_running = False
            self.video_capture = None
            self._cleanup_temporary_source()

    @staticmethod
    def _match_environment_person(
        primary: DetectionBox | None,
        environment_persons: list[DetectionBox],
    ) -> bool | None:
        if primary is None:
            return None
        if not environment_persons:
            return None
        matched = max(environment_persons, key=lambda box: box_iou(primary, box))
        overlap = box_iou(primary, matched)
        center_x, center_y = primary.center
        other_x, other_y = matched.center
        center_distance = ((center_x - other_x) ** 2 + (center_y - other_y) ** 2) ** 0.5
        return overlap >= 0.15 or center_distance <= 0.7 * max(primary.height, 1.0)

    def _apply_post_impact_override(self, decision: FusionDecision) -> None:
        """FALL 已确认时强制整体风险为 CRITICAL。

        注入原因码使告警签名变化，复用现有 _should_emit_alert 在上升沿
        发出一次告警；持续期间按 alert_reminder_seconds 周期提醒。
        """
        result = self.status.post_impact
        if result is None or not result.confirmed:
            return
        decision.overall_level = "critical"
        decision.overall_score = max(decision.overall_score, 92.0)
        decision.pending_direction = None
        decision.pending_count = 0
        decision.pending_required = 0
        if "post_impact_fall_confirmed" not in decision.reason_codes:
            decision.reason_codes.append("post_impact_fall_confirmed")

    def _risk_message(self, decision: FusionDecision) -> str:
        if decision.pending_direction == "upgrade":
            return (
                f"检测到风险变化，正在确认 "
                f"({decision.pending_count}/{decision.pending_required})"
            )
        if decision.overall_level == "critical":
            return "人体异常与环境风险叠加，请立即确认现场情况"
        if decision.overall_level == "warning":
            return "检测到高环境或路径交互风险，建议尽快排查"
        if decision.overall_level == "attention":
            return "检测到人体或环境风险变化，建议持续关注"
        if self.status.human_detection is None:
            return "暂未检测到监测对象，环境诊断保持独立显示"
        return "人体与周边环境状态正常，持续监测中"

    def _should_emit_alert(self, decision: FusionDecision, now: float) -> bool:
        level = RiskLevel(decision.overall_level)
        if level == RiskLevel.LOW or decision.pending_direction is not None:
            return False
        signature = "|".join(sorted(decision.reason_codes))
        reminder = float(self.config.engineering_risk.alert_reminder_seconds)
        should_emit = (
            level.priority > self._last_emitted_alert_level.priority
            or signature != self._last_emitted_reason_signature
            or now - self._last_emitted_alert_time >= reminder
        )
        if should_emit:
            self._last_emitted_alert_time = now
            self._last_emitted_alert_level = level
            self._last_emitted_reason_signature = signature
        return should_emit

    def _persist_alert(
        self,
        alert: AlertEvent,
        decision: FusionDecision,
        person_id: str,
        device_id: str,
    ) -> int | None:
        try:
            return Database().insert_alert_event(
                alert_level=alert.level.value,
                message=alert.message,
                risk_score=decision.overall_score,
                person_id=person_id,
                device_id=device_id,
                risk_score_source="overall_engineering_v1",
                reason_codes=decision.reason_codes,
            )
        except Exception as exc:
            log.error(f"告警持久化失败: {exc}")
            return None

    def _persist_risk_record(
        self,
        decision: FusionDecision,
        feature: FeatureVector | None,
        environment: dict,
        person_id: str,
        device_id: str,
    ) -> None:
        gait_features = None
        if feature is not None:
            gait_features = {
                "walking_rhythm": feature.walking_rhythm,
                "step_amplitude": feature.step_amplitude,
                "trunk_stability": feature.trunk_stability,
                "activity_density": feature.activity_density,
            }
        try:
            Database().insert_risk_record(
                risk_score=decision.overall_score,
                risk_level=decision.overall_level,
                person_id=person_id,
                device_id=device_id,
                gait_features=gait_features,
                env_features={
                    "environment": environment,
                    "risk_extensions": self.status.risk_extensions,
                },
                risk_score_source="overall_engineering_v1",
                raw_deviation_score=(
                    self.status.last_deviation.mahalanobis_distance
                    if self.status.last_deviation
                    else None
                ),
                human_risk_score=decision.human_risk_index,
                environment_risk_score=decision.environment_risk_index,
                interaction_risk_score=decision.interaction_risk_index,
                reason_codes=decision.reason_codes,
            )
        except Exception as exc:
            log.error(f"风险记录持久化失败: {exc}")

    def _run_audio(self):
        """音频采集+分析循环(独立线程, 与视频并行)"""
        if self.audio_capture is None or self.audio_analyzer is None:
            return

        person_id = getattr(self, "person_id", self.status.person_id)
        device_id = getattr(self, "device_id", self.status.device_id)

        try:
            with self.audio_capture:
                for chunk in self.audio_capture.chunks():
                    if self._stop_flag.is_set():
                        break

                    try:
                        result = self.audio_analyzer.analyze_waveform(
                            chunk.waveform, chunk.sample_rate, chunk.timestamp
                        )
                        self.status.last_audio_result = result
                        self.status.audio_chunks_processed += 1

                        if result.events:
                            with self.status._audio_lock:
                                self.status._pending_audio_events.extend(result.events)
                                if len(self.status._pending_audio_events) > 100:
                                    self.status._pending_audio_events = self.status._pending_audio_events[-100:]

                            try:
                                db = Database()
                                db.insert_audio_events(
                                    result.events,
                                    person_id=person_id,
                                    device_id=device_id,
                                )
                            except Exception as e:
                                log.error(f"音频事件持久化失败: {e}")

                    except Exception as e:
                        self.status.audio_error = str(e)
                        log.error(f"音频分析失败: {e}")
        except Exception as e:
            self.status.audio_error = str(e)
            log.error(f"音频采集线程异常: {e}")

    def get_status(self) -> dict:
        """获取监控状态"""
        risk_label = self.status.current_risk_level.label
        primary = self.status.human_detection
        fusion = self.status.fusion
        extensions = self.status.risk_extensions
        lighting = extensions.get("lighting", {})
        clutter = extensions.get("clutter", {})
        trajectory = extensions.get("trajectory", {})
        wet_floor = extensions.get("wet_floor", {})
        interaction = extensions.get("interaction", {})
        environment_boxes = (
            [] if self.status.environment_stale else self.status.environment_boxes
        )
        environment_cache_age = (
            None
            if self.status.environment_last_updated is None
            else max(0.0, time.time() - self.status.environment_last_updated)
        )
        return {
            "is_running": self.status.is_running,
            "person_id": self.status.person_id,
            "device_id": self.status.device_id,
            "source": (
                self.status.source_type
                if self.status.source_type == "uploaded"
                else self.status.source
            ),
            "source_type": self.status.source_type,
            "source_name": self.status.source_name,
            "frames_processed": self.status.frames_processed,
            "frames_valid": self.status.frames_valid,
            "current_risk_level": self.status.current_risk_level.value,
            "current_risk_label": risk_label,
            "current_risk_score": self.status.current_risk_score,
            "current_risk_message": self.status.current_risk_message,
            "risk_evaluable": fusion is not None,
            "gait_risk_evaluable": self.status.baseline_ready,
            "baseline_ready": self.status.baseline_ready,
            "baseline_samples": self.status.baseline_samples,
            "last_feature": (
                self.status.last_feature.to_array().tolist()
                if self.status.last_feature
                else None
            ),
            "last_deviation": (
                {
                    "level": self.status.last_deviation.level.value,
                    "mahalanobis_distance": self.status.last_deviation.mahalanobis_distance,
                    "detail": self.status.last_deviation.detail,
                }
                if self.status.last_deviation
                else None
            ),
            "last_alert": (
                {
                    "level": self.status.last_alert.level.value,
                    "message": self.status.last_alert.message,
                    "timestamp": self.status.last_alert.timestamp,
                }
                if self.status.last_alert
                else None
            ),
            "last_notification": self.status.last_notification,
            "audio_enabled": self.status.audio_enabled,
            "audio_source": self.status.audio_source,
            "audio_chunks_processed": self.status.audio_chunks_processed,
            "audio_error": self.status.audio_error,
            "person": {
                "present": primary is not None,
                "bbox": (
                    [primary.x1, primary.y1, primary.x2, primary.y2]
                    if primary is not None
                    else None
                ),
                "confidence": primary.confidence if primary is not None else None,
                "candidate_count": len(self.status.human_detections),
                "keypoints_present": self.status.human_keypoints_present,
                "keypoint_state": self.status.human_keypoint_state,
                "keypoint_reason": self.status.human_keypoint_reason,
                "keypoint_score": self.status.human_keypoint_score,
                "match": self.status.person_match,
            },
            "human_detected": primary is not None,
            "human_error": self.status.human_error,
            "environment_count": len(environment_boxes),
            "environment_boxes": [box.to_dict() for box in environment_boxes],
            "environment_error": self.status.environment_error,
            "illumination": self.status.illumination,
            "environment_model_loaded": (
                getattr(getattr(self, "environment_detector", None), "_model", None) is not None
            ),
            "pose_model_loaded": (
                getattr(getattr(self, "keypoint_extractor", None), "_model", None) is not None
            ),
            "environment_last_updated": self.status.environment_last_updated,
            "motion": {
                "source": "motion_heuristic_v0",
                "score": self.status.motion_score,
                "state": risk_state(
                    None if self.status.motion_score is None else self.status.motion_score * 100
                ),
                "deviation_score": self.status.deviation_risk_score,
                "risk_index": self.status.human_risk_score,
            },
            "environment": {
                "source": "env_risk_v0",
                "available": (
                    (not self.status.environment_stale and primary is not None)
                    or bool(lighting.get("available"))
                ),
                "score": self.status.environment_risk_score,
                "risk_index": fusion.environment_risk_index if fusion else None,
                "state": risk_state(fusion.environment_risk_index if fusion else None),
                "proximity_state": self.status.environment_state,
                "objects": [box.to_dict() for box in environment_boxes],
                "top_hazards": self.status.top_hazards,
                "updated_at": self.status.environment_last_updated,
                "cache_age_seconds": environment_cache_age,
                "stale": self.status.environment_stale,
            },
            "fusion": {
                "source": "overall_engineering_v1",
                "overall_score": fusion.overall_score if fusion else 0.0,
                "overall_state": fusion.overall_level if fusion else "low",
                "candidate_state": fusion.candidate_level if fusion else "low",
                "reason_codes": fusion.reason_codes if fusion else [],
                "context_elevated": fusion.context_elevated if fusion else False,
                "confirmation": {
                    "direction": fusion.pending_direction if fusion else None,
                    "count": fusion.pending_count if fusion else 0,
                    "required": fusion.pending_required if fusion else 0,
                },
            },
            "risk_extensions": extensions,
            "post_impact": (
                self.status.post_impact.to_dict() if self.status.post_impact else None
            ),
            "reason_codes": fusion.reason_codes if fusion else [],
            "top_hazards": self.status.top_hazards,
            "low_light": {
                **lighting,
                "brightness": lighting.get("evidence", {}).get("mean_luminance"),
            },
            "obstacle": clutter,
            "trajectory": trajectory,
            "wet_floor": wet_floor,
            "interaction": interaction,
            "last_audio_result": (
                {
                    "events": [
                        {
                            "category": e.category.value,
                            "label": e.label,
                            "score": e.score,
                            "timestamp": e.timestamp,
                        }
                        for e in self.status.last_audio_result.events
                    ],
                    "top_labels": [
                        [label, score]
                        for label, score in self.status.last_audio_result.top_labels
                    ],
                    "duration_sec": self.status.last_audio_result.duration_sec,
                    "elapsed_ms": self.status.last_audio_result.elapsed_ms,
                }
                if self.status.last_audio_result
                else None
            ),
        }

    def get_alert_history(self, level: str | None = None, limit: int = 100) -> list[dict]:
        """获取预警历史"""
        risk_level = RiskLevel(level) if level else None
        events = self.alert_engine.get_events(risk_level, limit)
        return [
            {
                "level": e.level.value,
                "label": e.level.label,
                "message": e.message,
                "timestamp": e.timestamp,
                "created_at": e.created_at,
                "notified": e.notified,
            }
            for e in events
        ]

    def reset_baseline(self, person_id: str | None = None) -> bool:
        """重置基线"""
        target_person_id = person_id or self.person_id
        self.baseline_manager.reset_baseline(target_person_id)
        if target_person_id == self.person_id:
            self.status.baseline_ready = False
            self.status.baseline_samples = 0
        return True
