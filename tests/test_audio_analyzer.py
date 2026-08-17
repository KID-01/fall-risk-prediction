"""
音频分析模块单元测试 — AudioAnalyzer / SoundCategory / AudioEvent
纯逻辑测试注入 FakeModel, 无需 318MB checkpoint; 集成测试按 checkpoint 存在性跳过
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.inference.audio_analyzer import (
    AudioAnalysisResult,
    AudioAnalyzer,
    AudioEvent,
    SoundCategory,
)
from src.utils.config import get_config


# ============================================================
# 测试辅助: 最小 audio 配置 (与 configs/base.yaml 保持一致)
# ============================================================
def _make_cfg(**overrides) -> OmegaConf:
    """构造最小 audio 配置, 支持字段覆盖"""
    cfg = OmegaConf.create(
        {
            "audio": {
                "enabled": True,
                "model_type": "Cnn14",
                "checkpoint_path": "~/panns_data/Cnn14_mAP=0.431.pth",
                "device": "cpu",
                "sample_rate": 32000,
                "chunk_seconds": 10,
                "top_k": 5,
                "min_event_score": 0.05,
                "vocal_distress": {
                    "threshold": 0.25,
                    "indices": [8, 10, 11, 14, 22, 23, 25, 38, 44],
                },
                "impact": {
                    "threshold": 0.30,
                    "indices": [358, 359, 441, 443, 460, 466, 469, 470],
                },
            }
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create({"audio": overrides}))


class FakeModel:
    """假 PANNs 模型 — 记录输入并返回受控 clipwise/embedding"""

    def __init__(self) -> None:
        self.last_input: np.ndarray | None = None
        self.scores = np.zeros((1, 527), dtype=np.float32)

    def inference(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.last_input = audio
        clipwise = self.scores.copy()
        embedding = np.zeros((1, 2048), dtype=np.float32)
        return clipwise, embedding


# ============================================================
# TestAudioConfig — 真实 base.yaml 配置
# ============================================================
class TestAudioConfig:
    def test_audio_section_exists(self):
        cfg = get_config()
        assert "audio" in cfg

    def test_required_keys_exist(self):
        audio = get_config().audio
        for key in (
            "enabled",
            "model_type",
            "checkpoint_path",
            "device",
            "sample_rate",
            "chunk_seconds",
            "top_k",
            "min_event_score",
        ):
            assert key in audio

    def test_thresholds_are_float(self):
        audio = get_config().audio
        assert isinstance(audio.vocal_distress.threshold, float)
        assert isinstance(audio.impact.threshold, float)

    def test_indices_are_int_lists(self):
        audio = get_config().audio
        # OmegaConf 2.3 ListConfig 继承 MutableSequence 而非 list, 用 Sequence 兼容两者
        assert isinstance(audio.vocal_distress.indices, Sequence)
        assert all(isinstance(i, int) for i in audio.vocal_distress.indices)
        assert isinstance(audio.impact.indices, Sequence)
        assert all(isinstance(i, int) for i in audio.impact.indices)

    def test_vocal_distress_indices(self):
        assert get_config().audio.vocal_distress.indices == [8, 10, 11, 14, 22, 23, 25, 38, 44]

    def test_impact_indices(self):
        assert get_config().audio.impact.indices == [358, 359, 441, 443, 460, 466, 469, 470]


# ============================================================
# TestSoundCategory
# ============================================================
class TestSoundCategory:
    def test_values(self):
        assert SoundCategory.VOCAL_DISTRESS.value == "vocal_distress"
        assert SoundCategory.IMPACT.value == "impact"

    def test_chinese_labels(self):
        for category in SoundCategory:
            assert isinstance(category.label, str)
            assert len(category.label) > 0
            assert re.search(r"[\u4e00-\u9fff]", category.label)


# ============================================================
# TestAudioEvent
# ============================================================
class TestAudioEvent:
    def test_fields(self):
        event = AudioEvent(
            category=SoundCategory.VOCAL_DISTRESS,
            label="Screaming",
            class_index=14,
            score=0.9,
            timestamp=1.5,
        )
        assert event.category == SoundCategory.VOCAL_DISTRESS
        assert event.label == "Screaming"
        assert event.class_index == 14
        assert event.score == 0.9
        assert event.timestamp == 1.5

    def test_frozen(self):
        event = AudioEvent(
            category=SoundCategory.IMPACT,
            label="Thump, thud",
            class_index=460,
            score=0.8,
            timestamp=2.0,
        )
        with pytest.raises(FrozenInstanceError):
            event.score = 0.1


# ============================================================
# TestAudioAnalyzerMapping — FakeModel 注入, 无真实 checkpoint
# ============================================================
class TestAudioAnalyzerMapping:
    def _make_analyzer(self, **overrides) -> AudioAnalyzer:
        """构造注入 FakeModel 的 analyzer"""
        analyzer = AudioAnalyzer(config=_make_cfg(**overrides))
        analyzer._model = FakeModel()
        return analyzer

    def test_lazy_load_model_none_after_init(self):
        analyzer = AudioAnalyzer(config=_make_cfg())
        assert analyzer._model is None

    def test_waveform_resampled_and_mono(self):
        """2s@16kHz 立体声 (2,32000) → 模型收到 (1,64000) float32 单声道"""
        analyzer = self._make_analyzer()
        stereo = np.zeros((2, 32000), dtype=np.float32)
        result = analyzer.analyze_waveform(stereo, 16000)
        assert isinstance(result, AudioAnalysisResult)
        assert analyzer._model.last_input is not None
        assert analyzer._model.last_input.shape == (1, 64000)
        assert analyzer._model.last_input.dtype == np.float32

    def test_event_emitted_above_threshold(self):
        """scores[14]=0.9 → VOCAL_DISTRESS 事件, label=Screaming"""
        analyzer = self._make_analyzer()
        analyzer._model.scores[0, 14] = 0.9
        result = analyzer.analyze_waveform(np.zeros(32000, dtype=np.float32), 32000)
        assert len(result.events) == 1
        event = result.events[0]
        assert event.category == SoundCategory.VOCAL_DISTRESS
        assert event.class_index == 14
        assert event.label == "Screaming"
        assert event.score == pytest.approx(0.9)

    def test_no_event_below_threshold(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze_waveform(np.zeros(32000, dtype=np.float32), 32000)
        assert result.events == []

    def test_impact_category_mapping(self):
        """scores[460]=0.9 → IMPACT 事件, label=Thump, thud"""
        analyzer = self._make_analyzer()
        analyzer._model.scores[0, 460] = 0.9
        result = analyzer.analyze_waveform(np.zeros(32000, dtype=np.float32), 32000)
        assert len(result.events) == 1
        event = result.events[0]
        assert event.category == SoundCategory.IMPACT
        assert event.class_index == 460
        assert event.label == "Thump, thud"

    def test_min_event_score_floor(self):
        """分数在 floor(0.05) 与类别阈值(0.25) 之间 → 无事件"""
        analyzer = self._make_analyzer()
        analyzer._model.scores[0, 14] = 0.1
        result = analyzer.analyze_waveform(np.zeros(32000, dtype=np.float32), 32000)
        assert result.events == []

    def test_top_k_selection(self):
        """10 个类别超过 floor → top_labels ≤ top_k(5), 降序, 分数 ≥ floor"""
        analyzer = self._make_analyzer()
        for i in range(10):
            analyzer._model.scores[0, 100 + i] = 0.9 - i * 0.08
        result = analyzer.analyze_waveform(np.zeros(32000, dtype=np.float32), 32000)
        assert len(result.top_labels) <= 5
        scores = [score for _, score in result.top_labels]
        assert scores == sorted(scores, reverse=True)
        assert all(score >= 0.05 for _, score in result.top_labels)

    def test_enabled_false_raises(self):
        analyzer = AudioAnalyzer(config=_make_cfg(enabled=False))
        with pytest.raises(RuntimeError):
            analyzer.analyze_waveform(np.zeros(32000, dtype=np.float32), 32000)

    def test_missing_checkpoint_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.pth"
        analyzer = AudioAnalyzer(config=_make_cfg(checkpoint_path=str(missing)))
        with pytest.raises(RuntimeError, match="checkpoint"):
            analyzer._ensure_model()

    def test_torch_load_patch_restored(self, tmp_path):
        """_ensure_model 临时替换 torch.load 后必须恢复原样"""
        checkpoint = tmp_path / "dummy.pth"
        checkpoint.write_bytes(b"dummy checkpoint content")
        received = {}

        class FakeAudioTagging:
            def __init__(self, checkpoint_path: str, device: str = "cpu"):
                received["checkpoint_path"] = checkpoint_path

        original_load = torch.load
        with patch("src.inference.audio_analyzer._import_audiotagging", return_value=FakeAudioTagging):
            analyzer = AudioAnalyzer(config=_make_cfg(checkpoint_path=str(checkpoint)))
            analyzer._ensure_model()
        assert torch.load is original_load
        assert received["checkpoint_path"] == str(checkpoint)
        assert analyzer._model is not None

    def test_unknown_high_score_no_event(self):
        """未知 AudioSet 类别高分 → 不产出事件, 仅保留于 top_labels"""
        analyzer = self._make_analyzer()
        analyzer._model.scores[0, 200] = 0.9
        result = analyzer.analyze_waveform(np.zeros(32000, dtype=np.float32), 32000)
        assert result.events == []
        assert any(label == "200" for label, _ in result.top_labels)


# ============================================================
# TestAudioAnalyzerIntegration — 需要真实 Cnn14 checkpoint
# ============================================================
class TestAudioAnalyzerIntegration:
    pytestmark = pytest.mark.skipif(
        not Path("~/panns_data/Cnn14_mAP=0.431.pth").expanduser().exists(),
        reason="Cnn14 checkpoint not downloaded",
    )

    def test_real_inference_synthetic_tone(self):
        """真实模型推理: 合成 3s 440Hz 正弦波 + 噪声"""
        analyzer = AudioAnalyzer()
        sr = 32000
        t = np.arange(sr * 3) / sr
        wave = (0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(sr * 3)).astype(
            np.float32
        )
        result = analyzer.analyze_waveform(wave, sr)
        assert result.duration_sec == pytest.approx(3.0, abs=0.05)
        assert len(result.top_labels) <= 5
        assert result.elapsed_ms > 0

    def test_analyze_file_roundtrip(self, tmp_path):
        """写 wav 文件 → analyze_file 返回结果"""
        import soundfile as sf

        sr = 32000
        t = np.arange(sr * 2) / sr
        wave = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = tmp_path / "tone.wav"
        sf.write(str(path), wave, sr)
        analyzer = AudioAnalyzer()
        result = analyzer.analyze_file(str(path))
        assert result is not None
        assert result.duration_sec > 0
