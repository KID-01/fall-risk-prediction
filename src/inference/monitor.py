"""
跌倒风险监控服务 — 整合完整链路:
视频拉流 → 人体检测 → 关键点提取 → 帧过滤 → 特征计算 → 基线对比 → 偏离检测 → 分级预警
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from src.alerts.engine import AlertEngine, AlertEvent, RiskLevel
from src.data.frame_filter import FrameFilter
from src.data.human_detector import HumanDetector
from src.data.keypoint_extractor import KeypointExtractor
from src.data.video_capture import VideoCapture
from src.inference.baseline import BaselineManager
from src.inference.deviation import DeviationDetector, DeviationResult
from src.inference.features import FeatureCalculator, FeatureVector
from src.utils.config import get_config
from src.utils.keypoints import KeypointFrame


@dataclass
class MonitorStatus:
    """监控状态"""

    is_running: bool = False
    person_id: str = "default"
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
        self.status = MonitorStatus()

        # 核心组件
        self.video_capture: VideoCapture | None = None
        self.human_detector = HumanDetector()
        self.keypoint_extractor = KeypointExtractor()
        self.frame_filter = FrameFilter()
        self.feature_calculator = FeatureCalculator()
        self.baseline_manager = BaselineManager()
        self.deviation_detector = DeviationDetector()
        self.alert_engine = AlertEngine()

        # 运行控制
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._keypoint_buffer: list[KeypointFrame] = []
        self._buffer_window = 30  # 特征计算窗口帧数

    def start(self, source: str, person_id: str = "default") -> bool:
        """启动监控"""
        if self.status.is_running:
            return False

        self.person_id = person_id
        self.status.source = source
        self.status.person_id = person_id
        self._stop_flag.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.status.is_running = True
        return True

    def stop(self):
        """停止监控"""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.status.is_running = False
        if self.video_capture:
            self.video_capture.close()
            self.video_capture = None

    def _run(self):
        """监控主循环(在子线程运行)"""
        inference_interval = self.config.inference.inference_interval_ms / 1000

        with VideoCapture(source=self.status.source) as cap:
            self.video_capture = cap

            for video_frame in cap.frames():
                if self._stop_flag.is_set():
                    break

                # 阶段1: 人体检测
                detection = self.human_detector.detect_best(video_frame.frame)
                if detection is None:
                    continue

                # 阶段2: 关键点提取
                kp_frame = self.keypoint_extractor.extract(video_frame)
                if kp_frame is None:
                    continue

                # 阶段3: 帧质量过滤
                self.frame_filter.filter(kp_frame)
                self.status.frames_processed += 1

                if not kp_frame.is_valid:
                    continue

                self.status.frames_valid += 1
                self._keypoint_buffer.append(kp_frame)

                # 保留最近N帧
                if len(self._keypoint_buffer) > self._buffer_window:
                    self._keypoint_buffer = self._keypoint_buffer[-self._buffer_window:]

                self.status.recent_keypoints = list(self._keypoint_buffer)

                # 阶段4: 特征计算 (积累足够帧后)
                if len(self._keypoint_buffer) >= 10:
                    feature = self.feature_calculator.calculate(self._keypoint_buffer)
                    self.status.last_feature = feature

                    # 阶段5: 基线采集
                    self.baseline_manager.add_sample(self.person_id, feature)
                    baseline = self.baseline_manager.compute_baseline(self.person_id)
                    self.status.baseline_ready = baseline.is_ready
                    self.status.baseline_samples = baseline.sample_count

                    # 阶段6: 偏离检测
                    if baseline.is_ready:
                        deviation = self.deviation_detector.check(feature, baseline)
                        self.status.last_deviation = deviation

                        # 阶段7: 预警评估
                        alert = self.alert_engine.evaluate(
                            deviation, feature.timestamp, has_activity=True
                        )
                        self.status.last_alert = alert
                        self.status.current_risk_level = alert.level

                time.sleep(inference_interval)

    def get_status(self) -> dict:
        """获取监控状态"""
        return {
            "is_running": self.status.is_running,
            "person_id": self.status.person_id,
            "source": self.status.source,
            "frames_processed": self.status.frames_processed,
            "frames_valid": self.status.frames_valid,
            "current_risk_level": self.status.current_risk_level.value,
            "current_risk_label": self.status.current_risk_level.label,
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

    def reset_baseline(self) -> bool:
        """重置基线"""
        self.baseline_manager.reset_baseline(self.person_id)
        self.status.baseline_ready = False
        self.status.baseline_samples = 0
        return True
