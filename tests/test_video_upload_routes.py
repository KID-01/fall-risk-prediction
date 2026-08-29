from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import src.api.routes as routes
from src.inference.monitor import FallRiskMonitor, MonitorStatus


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(routes.monitor_router)
    fake_monitor = SimpleNamespace(
        status=SimpleNamespace(is_running=False, audio_source=""),
        person_id="default",
        device_id="default",
        start=Mock(return_value=True),
        stop=Mock(),
    )
    monkeypatch.setattr(routes, "monitor", fake_monitor)
    uploaded = tmp_path / "upload.mp4"
    uploaded.write_bytes(b"video")

    async def fake_save(_file):
        return uploaded, "sample.mp4"

    monkeypatch.setattr(routes, "_save_uploaded_video", fake_save)
    with TestClient(app) as client:
        yield client, fake_monitor, uploaded
    routes.cleanup_staged_videos(force=True)


def test_video_upload_stages_without_starting_monitor(api_client):
    client, monitor, uploaded = api_client

    response = client.post(
        "/api/v1/stream/upload",
        files={"file": ("sample.mp4", b"ignored", "video/mp4")},
        data={"person_id": "person-1", "device_id": "device-1", "audio_source": "off"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["upload_id"]
    assert payload["source_name"] == "sample.mp4"
    assert str(uploaded) not in response.text
    monitor.start.assert_not_called()

    start_response = client.post(
        "/api/v1/stream/upload/start",
        json={
            "upload_id": payload["upload_id"],
            "person_id": "person-1",
            "device_id": "device-1",
            "audio_source": "off",
        },
    )
    assert start_response.status_code == 200
    assert str(uploaded) not in start_response.text
    monitor.start.assert_called_once_with(
        source=str(uploaded),
        person_id="person-1",
        device_id="device-1",
        audio_source="off",
        temporary_source_path=str(uploaded),
    )


def test_video_upload_allowed_while_running(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(routes.monitor_router)
    fake_monitor = SimpleNamespace(status=SimpleNamespace(is_running=True))
    monkeypatch.setattr(routes, "monitor", fake_monitor)
    uploaded = tmp_path / "upload.mp4"
    uploaded.write_bytes(b"video")

    async def fake_save(_file):
        return uploaded, "sample.mp4"

    monkeypatch.setattr(routes, "_save_uploaded_video", fake_save)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/stream/upload",
            files={"file": ("sample.mp4", b"ignored", "video/mp4")},
        )

    assert response.status_code == 200
    assert response.json()["monitor_running"] is True


def test_start_replaces_running_monitor(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(routes.monitor_router)
    uploaded = tmp_path / "upload.mp4"
    uploaded.write_bytes(b"video")
    fake_monitor = SimpleNamespace(
        status=SimpleNamespace(is_running=True, audio_source=""),
        person_id="person-1",
        device_id="device-1",
        start=Mock(return_value=True),
        stop=Mock(),
    )
    monkeypatch.setattr(routes, "monitor", fake_monitor)
    upload_id = "replace-me"
    with routes._staged_videos_lock:
        routes._staged_videos[upload_id] = routes.StagedVideo(
            path=uploaded, original_name="sample.mp4", expires_at=time.time() + 60
        )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/stream/upload/start",
            json={"upload_id": upload_id, "person_id": "person-1", "device_id": "device-1"},
        )

    assert response.status_code == 200
    fake_monitor.stop.assert_called_once_with()
    fake_monitor.start.assert_called_once()


def test_cancel_upload_deletes_staged_file(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(routes.monitor_router)
    monkeypatch.setattr(routes, "monitor", SimpleNamespace(status=SimpleNamespace(is_running=False)))
    uploaded = tmp_path / "upload.mp4"
    uploaded.write_bytes(b"video")
    upload_id = "cancel-me"
    with routes._staged_videos_lock:
        routes._staged_videos[upload_id] = routes.StagedVideo(
            path=uploaded, original_name="sample.mp4", expires_at=time.time() + 60
        )

    with TestClient(app) as client:
        response = client.delete(f"/api/v1/stream/upload/{upload_id}")

    assert response.status_code == 200
    assert not uploaded.exists()


def test_video_upload_rejects_unsupported_extension():
    from starlette.datastructures import UploadFile
    from io import BytesIO

    upload = UploadFile(filename="notes.txt", file=BytesIO(b"not video"))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes._save_uploaded_video(upload))
    assert getattr(exc_info.value, "status_code", None) == 415


def test_monitor_cleans_only_registered_temporary_source(tmp_path):
    monitor = FallRiskMonitor()
    uploaded = tmp_path / "session.mp4"
    uploaded.write_bytes(b"video")
    monitor.status = MonitorStatus(
        source=str(uploaded),
        source_type="uploaded",
        source_name="session.mp4",
        temporary_source_path=str(uploaded),
    )

    monitor._cleanup_temporary_source()

    assert not uploaded.exists()
    assert monitor.status.temporary_source_path is None


def test_monitor_status_redacts_uploaded_absolute_path(tmp_path):
    monitor = FallRiskMonitor()
    uploaded = tmp_path / "session.mp4"
    monitor.status = MonitorStatus(
        source=str(uploaded),
        source_type="uploaded",
        source_name="session.mp4",
        temporary_source_path=str(uploaded),
    )

    payload = monitor.get_status()

    assert payload["source"] == "uploaded"
    assert payload["source_name"] == "session.mp4"
    assert str(uploaded) not in str(payload)
