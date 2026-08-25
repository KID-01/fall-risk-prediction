"""音频分析接口测试 — status 契约与 analyze 错误映射 (503/400/413/415)。

注入 FakeModel 的真实 AudioAnalyzer 并 monkeypatch get_analyzer,
无需 318MB checkpoint 即可锁定路由行为。
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

import src.api.audio_routes as routes
from src.inference.audio_analyzer import AudioAnalyzer


# ============================================================
# 测试辅助: 最小 audio 配置 + FakeModel 注入
# ============================================================
def _make_cfg(**overrides) -> OmegaConf:
    """构造最小 audio 配置 (与 configs/base.yaml 数值一致), 支持字段覆盖"""
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
    """假 PANNs 模型 — 返回受控 clipwise/embedding"""

    def __init__(self) -> None:
        self.scores = np.zeros((1, 527), dtype=np.float32)

    def inference(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipwise = self.scores.copy()
        embedding = np.zeros((1, 2048), dtype=np.float32)
        return clipwise, embedding


def _make_client(monkeypatch: pytest.MonkeyPatch, analyzer: AudioAnalyzer) -> TestClient:
    """monkeypatch 模块级 get_analyzer 后挂载 audio_router"""
    monkeypatch.setattr(routes, "get_analyzer", lambda: analyzer)
    app = FastAPI()
    app.include_router(routes.audio_router)
    return TestClient(app)


def _wav_bytes(seconds: float = 2.0, sr: int = 16000) -> bytes:
    """生成内存中的合法 wav 字节流"""
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def _upload(name: str, data: bytes, content_type: str):
    return {"files": {"file": (name, data, content_type)}}


# ============================================================
# GET /api/v1/audio/status — 配置与状态契约
# ============================================================
class TestAudioStatus:
    def test_status_contract(self, monkeypatch, tmp_path):
        """返回完整契约字段; checkpoint 存在且模型未加载 → checkpoint_exists=True, model_loaded=False"""
        checkpoint = tmp_path / "cnn14.pth"
        checkpoint.write_bytes(b"fake")
        analyzer = AudioAnalyzer(config=_make_cfg(checkpoint_path=str(checkpoint)))
        client = _make_client(monkeypatch, analyzer)

        response = client.get("/api/v1/audio/status")

        assert response.status_code == 200
        assert response.json() == {
            "enabled": True,
            "model_type": "Cnn14",
            "device": "cpu",
            "sample_rate": 32000,
            "chunk_seconds": 10,
            "top_k": 5,
            "min_event_score": 0.05,
            "vocal_distress_threshold": 0.25,
            "impact_threshold": 0.30,
            "checkpoint_exists": True,
            "model_loaded": False,
        }

    def test_status_model_loaded_after_injection(self, monkeypatch, tmp_path):
        """注入 FakeModel 后 status 反映 model_loaded=True"""
        checkpoint = tmp_path / "cnn14.pth"
        checkpoint.write_bytes(b"fake")
        analyzer = AudioAnalyzer(config=_make_cfg(checkpoint_path=str(checkpoint)))
        analyzer._model = FakeModel()
        client = _make_client(monkeypatch, analyzer)

        response = client.get("/api/v1/audio/status")

        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_status_checkpoint_missing(self, monkeypatch, tmp_path):
        """checkpoint 路径不存在 → checkpoint_exists=False"""
        analyzer = AudioAnalyzer(
            config=_make_cfg(checkpoint_path=str(tmp_path / "missing.pth"))
        )
        client = _make_client(monkeypatch, analyzer)

        response = client.get("/api/v1/audio/status")

        assert response.status_code == 200
        assert response.json()["checkpoint_exists"] is False


# ============================================================
# POST /api/v1/audio/analyze — 正常路径与错误映射
# ============================================================
class TestAudioAnalyzeHappyPath:
    def test_impact_event_returned(self, monkeypatch):
        """合法 wav + scores[460]=0.9 → 200, impact 事件与 top_labels 契约"""
        analyzer = AudioAnalyzer(config=_make_cfg())
        analyzer._model = FakeModel()
        analyzer._model.scores[0, 460] = 0.9
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze", **_upload("tone.wav", _wav_bytes(), "audio/wav")
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["duration_sec"] == pytest.approx(2.0, abs=0.1)
        assert payload["elapsed_ms"] > 0
        assert len(payload["events"]) == 1
        event = payload["events"][0]
        assert event["category"] == "impact"
        assert event["label"] == "Thump, thud"
        assert event["class_index"] == 460
        assert event["score"] == pytest.approx(0.9)
        assert event["timestamp"] == 0.0
        assert payload["top_labels"][0] == ["Thump, thud", pytest.approx(0.9)]

    def test_timestamp_propagation(self, monkeypatch):
        """?timestamp=12.5 → 事件 timestamp 为 12.5"""
        analyzer = AudioAnalyzer(config=_make_cfg())
        analyzer._model = FakeModel()
        analyzer._model.scores[0, 14] = 0.9
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze?timestamp=12.5",
            **_upload("tone.wav", _wav_bytes(), "audio/wav"),
        )

        assert response.status_code == 200
        assert len(response.json()["events"]) == 1
        assert response.json()["events"][0]["timestamp"] == 12.5

    def test_no_events_when_silent(self, monkeypatch):
        """全零分数 → events=[], top_labels=[]"""
        analyzer = AudioAnalyzer(config=_make_cfg())
        analyzer._model = FakeModel()
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze", **_upload("tone.wav", _wav_bytes(), "audio/wav")
        )

        assert response.status_code == 200
        assert response.json()["events"] == []
        assert response.json()["top_labels"] == []


class TestAudioAnalyzeErrors:
    def test_disabled_returns_503(self, monkeypatch):
        """audio.enabled=false → 503"""
        analyzer = AudioAnalyzer(config=_make_cfg(enabled=False))
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze", **_upload("tone.wav", _wav_bytes(), "audio/wav")
        )

        assert response.status_code == 503

    def test_model_load_failure_returns_503(self, monkeypatch, tmp_path):
        """推理阶段 RuntimeError (checkpoint 缺失) → 503"""
        analyzer = AudioAnalyzer(
            config=_make_cfg(checkpoint_path=str(tmp_path / "missing.pth"))
        )
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze", **_upload("tone.wav", _wav_bytes(), "audio/wav")
        )

        assert response.status_code == 503
        assert "checkpoint" in response.json()["detail"]

    def test_wrong_extension_returns_415(self, monkeypatch):
        """.txt 扩展名 → 415 (即使 content-type 合法)"""
        analyzer = AudioAnalyzer(config=_make_cfg())
        analyzer._model = FakeModel()
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze", **_upload("note.txt", b"hello", "audio/wav")
        )

        assert response.status_code == 415

    def test_wrong_content_type_returns_415(self, monkeypatch):
        """text/plain 内容类型 → 415 (即使扩展名合法)"""
        analyzer = AudioAnalyzer(config=_make_cfg())
        analyzer._model = FakeModel()
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze", **_upload("tone.wav", _wav_bytes(), "text/plain")
        )

        assert response.status_code == 415

    def test_oversize_returns_413(self, monkeypatch):
        """超过 20MB 上限 → 413"""
        analyzer = AudioAnalyzer(config=_make_cfg())
        analyzer._model = FakeModel()
        client = _make_client(monkeypatch, analyzer)
        oversize = b"\x00" * (routes.MAX_UPLOAD_BYTES + 1)

        response = client.post(
            "/api/v1/audio/analyze", **_upload("big.wav", oversize, "audio/wav")
        )

        assert response.status_code == 413

    def test_corrupt_bytes_return_400(self, monkeypatch):
        """扩展名/类型合法但内容无法解码 → 400"""
        analyzer = AudioAnalyzer(config=_make_cfg())
        analyzer._model = FakeModel()
        client = _make_client(monkeypatch, analyzer)

        response = client.post(
            "/api/v1/audio/analyze",
            **_upload("broken.wav", b"this is not audio" * 16, "audio/wav"),
        )

        assert response.status_code == 400
