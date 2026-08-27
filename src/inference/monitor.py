"""
跌倒风险监控服务 — 整合完整链路:
视频拉流 → 人体检测 → 关键点提取 → 帧过滤 → 特征计算 → 基线对比 → 偏离检测 → 分级预警
音频采集 → 声音事件识别 → 音频预警升级 (并行线程)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from src.alerts.engine import AlertEngine, AlertEvent, RiskLevel
from src.api.database import Database
from src.api.websocket import raw_video_manager, video_manager
from src.data.audio_capture import AudioCapture
from src.data.frame_filter import FrameFilter
from src.data.human_detector import HumanDetector
from src.data.keypoint_extractor import create_keypoint_extractor
from src.data.video_capture import VideoCapture
from src.inference.audio_analyzer import AudioAnalysisResult, AudioAnalyzer, AudioEvent
from src.inference.baseline import BaselineManager
from src.inference.deviation import DeviationDetector, DeviationResult
from src.inference.features import FeatureCalculator, FeatureVector
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
    frames_processed: int = 0
    frames_valid: int = 0
    last_feature: FeatureVector | None = None
    last_deviation: DeviationResult | None = None
    last_alert: AlertEvent | None = None
    current_risk_level: RiskLevel = RiskLevel.LOW
    baseline_ready: bool = False
    baseline_samples: int = 0
    recent_keypoints: list[KeypointFrame] = field(default_factory=list)
    audio_enabled: bool = False
    audio_status: str = "DISABLED"
    audio_source: str = ""
    last_audio_result: AudioAnalysisResult | None = None
    audio_chunks_processed: int = 0
    audio_error: str | None = None
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
        self.pose_backend = self.config.pose_estimation.get("backend", "mediapipe")
        self.keypoint_extractor = create_keypoint_extractor()
        self.frame_filter = FrameFilter()
        self.feature_calculator = FeatureCalculator()
        self.baseline_manager = BaselineManager()
        self.deviation_detector = DeviationDetector()
        self.alert_engine = AlertEngine()

        # 音频组件 (start() 时按 audio_source 决定是否启用)
        self.audio_analyzer: AudioAnalyzer | None = None
        self.audio_capture: AudioCapture | None = None
        self._audio_thread: threading.Thread | None = None

        # 运行控制
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._audio_stop_flag = threading.Event()
        self._keypoint_buffer: list[KeypointFrame] = []
        self._buffer_window = 30  # 特征计算窗口帧数

    @staticmethod
    def _resolve_audio_source(video_source: str, requested_source: str | None) -> str:
        requested = (requested_source or "off").strip()
        lowered = requested.lower()
        video_lower = video_source.lower()

        if lowered in ("", "off"):
            return "off"
        if lowered == "mic":
            return "mic"
        if lowered in ("video_source", "camera"):
            if "://" not in video_lower:
                raise ValueError("视频源收音仅支持 RTSP、RTMP 或 HTTP 网络流")
            return video_source
        if lowered == "auto":
            return video_source if "://" in video_lower else "off"
        if "://" in lowered:
            return requested
        if requested:
            return requested
        return "off"

    def _build_audio_capture(self, source: str) -> AudioCapture:
        return AudioCapture(
            source=source,
            sample_rate=int(self.config.audio.sample_rate),
            chunk_seconds=int(self.config.audio.chunk_seconds),
            input_device=self.config.audio.get("input_device"),
            ffmpeg_path=str(self.config.audio.get("ffmpeg_path", "")),
            stop_event=self._audio_stop_flag,
        )

    def start(
        self,
        source: str,
        person_id: str = "default",
        device_id: str = "default",
        audio_source: str | None = None,
    ) -> bool:
        """启动监控"""
        if self.status.is_running:
            return False

        self._keypoint_buffer.clear()
        self.deviation_detector.reset()
        self.alert_engine.reset()
        self.person_id = person_id
        self.device_id = device_id

        audio_cfg_enabled = self.config.get("audio", {}).get("enabled", False)
        cfg_audio_source = self.config.get("audio", {}).get("source", "off")
        requested_audio_source = audio_source if audio_source is not None else cfg_audio_source
        try:
            effective_audio_source = self._resolve_audio_source(source, requested_audio_source)
        except ValueError as exc:
            log.error(str(exc))
            return False

        audio_enabled = audio_cfg_enabled and effective_audio_source not in ("off", "")

        self.status = MonitorStatus(
            is_running=True,
            person_id=person_id,
            device_id=device_id,
            source=source,
            audio_enabled=audio_enabled,
            audio_status="STARTING" if audio_enabled else "DISABLED",
            audio_source=effective_audio_source if audio_enabled else "",
        )
        self._stop_flag.clear()
        if not hasattr(self, "_audio_stop_flag"):
            self._audio_stop_flag = threading.Event()
        self._audio_stop_flag.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        if audio_enabled:
            self._start_audio_thread(effective_audio_source)

        return True

    def stop(self):
        """停止监控"""
        self._stop_flag.set()
        self.stop_audio()
        if self._thread:
            self._thread.join(timeout=5)
        self.status.is_running = False
        if self.video_capture:
            self.video_capture.close()
            self.video_capture = None

    def _start_audio_thread(self, audio_source: str) -> bool:
        self.audio_analyzer = AudioAnalyzer()
        resources = self.audio_analyzer.resource_status()
        if not resources["resources_ready"]:
            missing = []
            if not resources["checkpoint_exists"]:
                missing.append("PANNs checkpoint")
            if not resources["labels_exist"]:
                missing.append("AudioSet 标签文件")
            self.status.audio_enabled = False
            self.status.audio_status = "UNAVAILABLE"
            self.status.audio_error = f"缺少音频资源: {', '.join(missing)}"
            return False

        self.status.audio_enabled = True
        self.status.audio_status = "STARTING"
        self.status.audio_source = audio_source
        self.status.audio_error = None
        self._audio_stop_flag.clear()
        self.audio_capture = self._build_audio_capture(audio_source)
        self._audio_thread = threading.Thread(target=self._run_audio, daemon=True)
        self._audio_thread.start()
        return True

    def start_audio(self, audio_source: str = "video_source") -> bool:
        """在视频监控运行期间单独启动音频分支。"""
        if not self.status.is_running:
            self.status.audio_status = "UNAVAILABLE"
            self.status.audio_error = "视频监控尚未启动"
            return False
        if self._audio_thread and self._audio_thread.is_alive():
            return False
        if not bool(self.config.get("audio", {}).get("enabled", False)):
            self.status.audio_status = "UNAVAILABLE"
            self.status.audio_error = "音频分析已在配置中禁用"
            return False
        try:
            resolved = self._resolve_audio_source(self.status.source, audio_source)
        except ValueError as exc:
            self.status.audio_status = "UNAVAILABLE"
            self.status.audio_error = str(exc)
            return False
        if resolved == "off":
            self.status.audio_status = "DISABLED"
            self.status.audio_error = None
            return False
        return self._start_audio_thread(resolved)

    def stop_audio(self):
        """只停止音频分支，不影响视频监控。"""
        if not hasattr(self, "_audio_stop_flag"):
            self._audio_stop_flag = threading.Event()
        self._audio_stop_flag.set()
        capture = self.audio_capture
        if capture is not None:
            capture.close()
        if self._audio_thread and self._audio_thread is not threading.current_thread():
            self._audio_thread.join(timeout=3)
        self._audio_thread = None
        self.audio_capture = None
        self.audio_analyzer = None
        self.status.audio_enabled = False
        if self.status.audio_status != "UNAVAILABLE":
            self.status.audio_status = "DISABLED"
            self.status.audio_error = None
        self.status.audio_source = ""
        with self.status._audio_lock:
            self.status._pending_audio_events.clear()

    def _run(self):
        """监控主循环(在子线程运行)"""
        inference_interval = self.config.inference.inference_interval_ms / 1000
        person_id = getattr(self, "person_id", "default")
        device_id = getattr(self, "device_id", "default")
        _last_broadcast = 0.0
        _last_raw_broadcast = 0.0

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

                    if self.pose_backend != "yolo_pose":
                        detection = self.human_detector.detect_best(video_frame.frame)
                        if detection is None:
                            self.status.frames_processed += 1
                            continue

                    self.status.frames_processed += 1
                    kp_frame = self.keypoint_extractor.extract(video_frame)

                    now = time.time()
                    if video_manager.has_clients and now - _last_broadcast >= 0.1:
                        _last_broadcast = now
                        try:
                            risk_level = self.status.current_risk_level.value
                            overlay = draw_overlay(
                                video_frame.frame, kp_frame,
                                risk_level=risk_level,
                                baseline_ready=self.status.baseline_ready,
                                frames_processed=self.status.frames_processed,
                            )
                            jpeg = encode_jpeg(overlay)
                            video_manager.broadcast_frame(jpeg)
                        except Exception:
                            pass
                    if raw_video_manager.has_clients and now - _last_raw_broadcast >= 0.1:
                        _last_raw_broadcast = now
                        try:
                            raw_jpeg = encode_jpeg(video_frame.frame)
                            raw_video_manager.broadcast_frame(raw_jpeg)
                        except Exception:
                            pass

                    if kp_frame is None:
                        continue

                    self.frame_filter.filter(kp_frame)

                    if not kp_frame.is_valid:
                        continue

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

                        # 从音频线程排空待处理事件, 用于视频偏差联合评估
                        now_timestamp = time.time()
                        merge_window = float(
                            self.config.alert.audio.get("merge_window_seconds", 15)
                        )
                        with self.status._audio_lock:
                            pending_events = [
                                event
                                for event in self.status._pending_audio_events
                                if 0 <= now_timestamp - event.timestamp <= merge_window
                            ]
                            self.status._pending_audio_events.clear()

                        if baseline.is_ready:
                            deviation = self.deviation_detector.check(feature, baseline)
                            self.status.last_deviation = deviation

                            alert = self.alert_engine.evaluate(
                                deviation, now_timestamp,
                                has_activity=True,
                                audio_events=pending_events or None,
                            )
                            self.status.last_alert = alert
                            self.status.current_risk_level = alert.level

                            try:
                                db = Database()
                                db.insert_risk_record(
                                    risk_score=deviation.mahalanobis_distance,
                                    risk_level=alert.level.value,
                                    person_id=person_id,
                                    device_id=device_id,
                                    gait_features={
                                        "walking_rhythm": feature.walking_rhythm,
                                        "step_amplitude": feature.step_amplitude,
                                        "trunk_stability": feature.trunk_stability,
                                        "activity_density": feature.activity_density,
                                    },
                                )
                                if alert.level != RiskLevel.LOW:
                                    db.insert_alert_event(
                                        alert_level=alert.level.value,
                                        message=alert.message,
                                        risk_score=deviation.mahalanobis_distance,
                                        person_id=person_id,
                                        device_id=device_id,
                                    )
                            except Exception as e:
                                log.error(f"持久化失败: {e}")

                    time.sleep(inference_interval)
        except Exception as e:
            log.error(f"监控线程异常: {e}")
        finally:
            self.status.is_running = False
            self.video_capture = None

    def _run_audio(self):
        """音频采集+分析循环(独立线程, 与视频并行)"""
        if not hasattr(self, "_audio_stop_flag"):
            self._audio_stop_flag = threading.Event()
        if self.audio_capture is None or self.audio_analyzer is None:
            return

        person_id = getattr(self, "person_id", "default")
        device_id = getattr(self, "device_id", "default")

        config_audio = getattr(getattr(self, "config", None), "audio", None)
        try:
            reconnect_attempts = int(config_audio.get("reconnect_attempts", 3)) if config_audio else 3
        except (TypeError, ValueError):
            reconnect_attempts = 3
        try:
            reconnect_delay = float(config_audio.get("reconnect_delay_seconds", 2)) if config_audio else 2.0
        except (TypeError, ValueError):
            reconnect_delay = 2.0
        source = self.status.audio_source
        is_network = "://" in source.lower()
        attempt = 0

        try:
            baseline_manager = getattr(self, "baseline_manager", None)
            baseline = baseline_manager.load_baseline(person_id) if baseline_manager else None
            chunks_since_reload = 0

            while not self._audio_stop_flag.is_set():
                capture = self.audio_capture or self._build_audio_capture(source)
                self.audio_capture = capture
                if not capture.open():
                    raise RuntimeError(f"无法打开音频源: {source}")
                self.status.audio_status = "RUNNING"
                self.status.audio_error = None

                for chunk in capture.chunks():
                    if self._audio_stop_flag.is_set():
                        break

                    try:
                        result = self.audio_analyzer.analyze_waveform(
                            chunk.waveform, chunk.sample_rate, chunk.timestamp
                        )
                        self.status.last_audio_result = result
                        self.status.audio_chunks_processed += 1
                        chunks_since_reload += 1

                        if chunks_since_reload >= 6:
                            baseline = baseline_manager.load_baseline(person_id) if baseline_manager else None
                            chunks_since_reload = 0

                        if result.events:
                            if not baseline or not baseline.is_ready:
                                audio_alert = self.alert_engine.evaluate_audio_only(
                                    result.events, chunk.timestamp,
                                )
                                if audio_alert.level != RiskLevel.LOW:
                                    self.status.last_alert = audio_alert
                                    self.status.current_risk_level = audio_alert.level
                                    try:
                                        db = Database()
                                        db.insert_alert_event(
                                            alert_level=audio_alert.level.value,
                                            message=audio_alert.message,
                                            risk_score=0,
                                            person_id=person_id,
                                            device_id=device_id,
                                        )
                                    except Exception as e:
                                        log.error(f"音频告警持久化失败: {e}")
                            else:
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
                capture.close(signal_stop=False)
                self.audio_capture = None
                if self._audio_stop_flag.is_set() or not is_network:
                    break
                attempt += 1
                if attempt > reconnect_attempts:
                    raise RuntimeError(f"网络音频重连失败，已尝试 {reconnect_attempts} 次")
                self.status.audio_status = "STARTING"
                self.status.audio_error = f"网络音频中断，正在进行第 {attempt} 次重连"
                if self._audio_stop_flag.wait(reconnect_delay):
                    break
        except Exception as e:
            self.status.audio_error = str(e)
            self.status.audio_status = "UNAVAILABLE"
            self.status.audio_enabled = False
            log.error(f"音频采集线程异常: {e}")
        finally:
            if self.audio_capture:
                self.audio_capture.close(signal_stop=False)
                self.audio_capture = None
            if self.status.audio_status == "RUNNING":
                self.status.audio_status = "DISABLED" if not is_network or self._audio_stop_flag.is_set() else "UNAVAILABLE"
                self.status.audio_enabled = False

    def get_status(self) -> dict:
        """获取监控状态"""
        risk_label = (
            self.status.current_risk_level.label
            if self.status.baseline_ready
            else "基线采集中"
        )
        return {
            "is_running": self.status.is_running,
            "person_id": self.status.person_id,
            "device_id": self.status.device_id,
            "source": self.status.source,
            "frames_processed": self.status.frames_processed,
            "frames_valid": self.status.frames_valid,
            "current_risk_level": self.status.current_risk_level.value,
            "current_risk_label": risk_label,
            "risk_evaluable": self.status.baseline_ready,
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
            "audio_enabled": self.status.audio_enabled,
            "audio_status": self.status.audio_status,
            "audio_source": self.status.audio_source,
            "audio_chunks_processed": self.status.audio_chunks_processed,
            "audio_error": self.status.audio_error,
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
