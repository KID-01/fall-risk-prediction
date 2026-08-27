"""独立音频启停接口测试。"""
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routes as routes


def test_audio_start_and_stop_do_not_require_video_restart(monkeypatch):
    fake_monitor = SimpleNamespace(
        status=SimpleNamespace(
            is_running=True,
            audio_enabled=False,
            audio_status="DISABLED",
            audio_source="",
            audio_error=None,
        ),
        start_audio=Mock(return_value=True),
        stop_audio=Mock(),
    )
    monkeypatch.setattr(routes, "monitor", fake_monitor)
    app = FastAPI()
    app.include_router(routes.monitor_router)

    with TestClient(app) as client:
        started = client.post("/api/v1/stream/audio/start", json={"audio_source": "video_source"})
        stopped = client.post("/api/v1/stream/audio/stop")

    assert started.status_code == 200
    fake_monitor.start_audio.assert_called_once_with("video_source")
    assert stopped.status_code == 200
    fake_monitor.stop_audio.assert_called_once_with()


def test_audio_start_rejects_when_video_is_not_running(monkeypatch):
    fake_monitor = SimpleNamespace(
        status=SimpleNamespace(is_running=False, audio_enabled=False),
        start_audio=Mock(),
    )
    monkeypatch.setattr(routes, "monitor", fake_monitor)
    app = FastAPI()
    app.include_router(routes.monitor_router)

    with TestClient(app) as client:
        response = client.post("/api/v1/stream/audio/start")

    assert response.status_code == 409
    fake_monitor.start_audio.assert_not_called()
