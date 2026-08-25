"""萤石设备、播放器和一键监控接口测试。"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.ezviz_routes as routes

SERIAL = "TEST123456"
DEVICE = {
    "deviceSerial": SERIAL,
    "deviceName": "测试摄像头",
    "status": 1,
    "channelNo": 1,
    "isEncrypt": 0,
}


@pytest.fixture
def api_client():
    app = FastAPI()
    app.include_router(routes.ezviz_router)
    app.state.ezviz_client = AsyncMock()
    with TestClient(app) as client:
        yield client, app.state.ezviz_client


def test_devices_requires_local_config():
    app = FastAPI()
    app.include_router(routes.ezviz_router)
    app.state.ezviz_client = None

    with TestClient(app) as client:
        response = client.get("/api/v1/ezviz/devices")

    assert response.status_code == 503
    assert "configs/ezviz.yaml" in response.json()["detail"]


def test_devices_are_masked(api_client):
    client, ezviz = api_client
    ezviz.list_devices.return_value = [DEVICE]

    response = client.get("/api/v1/ezviz/devices")

    assert response.status_code == 200
    payload = response.json()["devices"][0]
    assert payload["online"] is True
    assert payload["encrypted"] is False
    assert payload["channels"] == [1]
    assert SERIAL not in json.dumps(payload)


def test_offline_device_cannot_play(api_client):
    client, ezviz = api_client
    ezviz.list_devices.return_value = [{**DEVICE, "status": 0}]

    response = client.post(
        "/api/v1/ezviz/player",
        json={"device_id": routes._device_id(SERIAL), "channel_no": 1},
    )

    assert response.status_code == 409
    assert "离线" in response.json()["detail"]


def test_player_returns_no_store_ezopen_config(api_client):
    client, ezviz = api_client
    ezviz.list_devices.return_value = [DEVICE]
    ezviz.ensure_token.return_value = "temporary-test-token"

    response = client.post(
        "/api/v1/ezviz/player",
        json={"device_id": routes._device_id(SERIAL), "channel_no": 1},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["accessToken"] == "temporary-test-token"
    assert response.json()["url"] == f"ezopen://open.ys7.com/{SERIAL}/1.live"
    assert SERIAL not in json.dumps(response.json()["device"])


def test_monitor_start_uses_private_analysis_stream(api_client, monkeypatch):
    client, ezviz = api_client
    ezviz.list_devices.return_value = [DEVICE]
    ezviz.ensure_token.return_value = "temporary-test-token"
    ezviz.get_analysis_stream.return_value = "rtmp://example.invalid/private-analysis"
    fake_monitor = SimpleNamespace(
        status=SimpleNamespace(is_running=False),
        start=Mock(return_value=True),
    )
    monkeypatch.setattr(routes, "monitor", fake_monitor)
    device_id = routes._device_id(SERIAL)

    response = client.post(
        "/api/v1/ezviz/monitor/start",
        json={"device_id": device_id, "channel_no": 1, "person_id": "test-person"},
    )

    assert response.status_code == 200
    ezviz.get_analysis_stream.assert_awaited_once_with(SERIAL, 1)
    fake_monitor.start.assert_called_once_with(
        source="rtmp://example.invalid/private-analysis",
        person_id="test-person",
        device_id=device_id,
    )
    assert "private-analysis" not in response.text
