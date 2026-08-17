"""
音频分析模块 — 基于 PANNs Cnn14 (panns-inference) 的跌倒相关声音事件识别
覆盖: 声音事件分类(人声呼救/撞击声) / 音频重采样 / 多标签打分 / 阈值筛选
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ==== 声音类别标签映射 (AudioSet 527 类索引) ====
VOCAL_DISTRESS_LABELS: dict[int, str] = {
    8: "Shout",
    10: "Whoop",
    11: "Yell",
    14: "Screaming",
    22: "Crying, sobbing",
    23: "Baby cry, infant cry",
    25: "Wail, moan",
    38: "Groan",
    44: "Gasp",
}

IMPACT_LABELS: dict[int, str] = {
    358: "Slam",
    359: "Knock",
    441: "Glass",
    443: "Shatter",
    460: "Thump, thud",
    466: "Bang",
    469: "Smash, crash",
    470: "Breaking",
}


class SoundCategory(Enum):
    """声音事件类别"""

    VOCAL_DISTRESS = "vocal_distress"
    IMPACT = "impact"

    @property
    def label(self) -> str:
        """中文类别名"""
        return {
            SoundCategory.VOCAL_DISTRESS: "人声呼救",
            SoundCategory.IMPACT: "撞击声",
        }[self]


@dataclass(frozen=True)
class AudioEvent:
    """单一声音事件"""

    category: SoundCategory
    label: str
    class_index: int
    score: float
    timestamp: float


@dataclass
class AudioAnalysisResult:
    """一次音频分析的完整结果"""

    events: list[AudioEvent]
    top_labels: list[tuple[str, float]]
    duration_sec: float
    elapsed_ms: float


class AudioAnalyzer:
    """PANNs Cnn14 音频分析器 — 懒加载模型, 线程安全"""

    def __init__(self, config: OmegaConf | None = None):
        cfg = config if config is not None else get_config()
        if "audio" not in cfg:
            raise RuntimeError("配置缺少 audio 段")
        audio = cfg.audio
        self._enabled = bool(audio.enabled)
        self.model_type = str(audio.model_type)
        self.checkpoint_path = str(audio.checkpoint_path)
        self.device = str(audio.device)
        self.sample_rate = int(audio.sample_rate)
        self.chunk_seconds = int(audio.chunk_seconds)
        self.top_k = int(audio.top_k)
        self.min_event_score = float(audio.min_event_score)
        self.vocal_threshold = float(audio.vocal_distress.threshold)
        self.vocal_indices = [int(i) for i in audio.vocal_distress.indices]
        self.impact_threshold = float(audio.impact.threshold)
        self.impact_indices = [int(i) for i in audio.impact.indices]

        self._model = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """音频分析是否启用"""
        return self._enabled

    def analyze_waveform(self, waveform: np.ndarray, sample_rate: int) -> AudioAnalysisResult:
        """分析一段波形, 返回声音事件与 top-k 标签"""
        if not self._enabled:
            raise RuntimeError("音频分析未启用 (audio.enabled=false)")

        start = time.perf_counter()
        wave = np.asarray(waveform)
        if wave.dtype == np.int16:
            wave = wave.astype(np.float32) / 32768.0
        else:
            wave = wave.astype(np.float32)

        if wave.ndim == 2:
            # 兼容 (channels, samples) 与 (samples, channels) 两种布局, 按较小轴(声道)平均
            wave = wave.mean(axis=0 if wave.shape[0] <= wave.shape[1] else -1)
        duration_sec = len(wave) / sample_rate

        if sample_rate != self.sample_rate:
            wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=self.sample_rate)

        self._ensure_model()
        clipwise, _embedding = self._model.inference(
            np.ascontiguousarray(wave[None, :], dtype=np.float32)
        )
        scores = clipwise[0]

        events = self._build_events(scores)
        top_labels = self._build_top_labels(scores)
        elapsed_ms = (time.perf_counter() - start) * 1000

        log.info(f"音频分析完成: {elapsed_ms:.0f}ms, 事件 {len(events)} 个, 时长 {duration_sec:.1f}s")
        return AudioAnalysisResult(
            events=events,
            top_labels=top_labels,
            duration_sec=duration_sec,
            elapsed_ms=elapsed_ms,
        )

    def analyze_file(self, path: str | Path) -> AudioAnalysisResult:
        """分析音频文件 (wav/flac/ogg 等 soundfile 支持的格式)"""
        waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        return self.analyze_waveform(waveform, int(sample_rate))

    def _build_events(self, scores: np.ndarray) -> list[AudioEvent]:
        """按类别阈值筛选声音事件"""
        events: list[AudioEvent] = []
        for idx in range(len(scores)):
            score = float(scores[idx])
            if score < self.min_event_score:
                continue
            if idx in VOCAL_DISTRESS_LABELS:
                if score >= self.vocal_threshold:
                    events.append(
                        AudioEvent(
                            category=SoundCategory.VOCAL_DISTRESS,
                            label=VOCAL_DISTRESS_LABELS[idx],
                            class_index=idx,
                            score=score,
                            timestamp=0.0,
                        )
                    )
            elif idx in IMPACT_LABELS:
                if score >= self.impact_threshold:
                    events.append(
                        AudioEvent(
                            category=SoundCategory.IMPACT,
                            label=IMPACT_LABELS[idx],
                            class_index=idx,
                            score=score,
                            timestamp=0.0,
                        )
                    )
        return events

    def _build_top_labels(self, scores: np.ndarray) -> list[tuple[str, float]]:
        """全局 top_k 标签, (label, score) 降序, 仅保留超过最低分数的"""
        candidates: list[tuple[str, float]] = []
        for idx, score in enumerate(scores):
            if float(score) < self.min_event_score:
                continue
            label = (
                VOCAL_DISTRESS_LABELS.get(idx)
                or IMPACT_LABELS.get(idx)
                or str(idx)
            )
            candidates.append((label, float(score)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[: self.top_k]

    def _ensure_model(self) -> None:
        """懒加载 PANNs 模型 — 双检锁, 构造期间临时修补 torch.load"""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            checkpoint = Path(self.checkpoint_path).expanduser()
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"checkpoint 不存在: {checkpoint} — 请下载 Cnn14_mAP=0.431.pth 并放置到该路径"
                )
            labels_csv = Path.home() / "panns_data" / "class_labels_indices.csv"
            if not labels_csv.is_file():
                raise RuntimeError(
                    f"标签文件不存在: {labels_csv} — 请放置 class_labels_indices.csv (527 类)"
                )

            import torch

            original_load = torch.load
            try:
                # PyTorch 2.6+ 默认 weights_only=True, 无法加载含非张量对象的 PANNs checkpoint
                torch.load = lambda *a, **k: original_load(*a, **{**k, "weights_only": False})
                audio_tagging_cls = _import_audiotagging()
                self._model = audio_tagging_cls(
                    checkpoint_path=str(checkpoint), device=self.device
                )
            finally:
                torch.load = original_load


def _import_audiotagging():
    """延迟导入 panns_inference.AudioTagging (便于测试注入)"""
    from panns_inference import AudioTagging

    return AudioTagging
