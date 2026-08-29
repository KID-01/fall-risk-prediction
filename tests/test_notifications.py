from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.alerts.engine import AlertEvent, RiskLevel
from src.api.database import Database
from src.notifications.service import NotificationService, WechatCloudFunctionAdapter


def _cleanup(db: Database, notification_ids: list[str]) -> None:
    with db._get_conn() as conn:
        for notification_id in notification_ids:
            conn.execute(
                "DELETE FROM notification_deliveries WHERE notification_id=?",
                (notification_id,),
            )
            conn.execute(
                "DELETE FROM notifications WHERE notification_id=?",
                (notification_id,),
            )


def test_policy_has_three_external_levels():
    policy = NotificationService.policy()
    assert set(policy["levels"]) == {"low", "attention", "critical"}
    assert policy["levels"]["low"]["channels"] == []
    assert policy["levels"]["attention"]["channels"] == ["cloud_function"]
    assert policy["levels"]["critical"]["channels"] == ["cloud_function", "app", "sms"]


def test_attention_pushes_to_http_cloud_function():
    db = Database()
    adapter = WechatCloudFunctionAdapter(url="https://example.test/fallAlarmPush", enabled=True)
    service = NotificationService(database=db, fallback_seconds=30, wechat_adapter=adapter)
    ids: list[str] = []
    try:
        class Response:
            status_code = 200
            is_success = True
            def json(self):
                return {"code": 0, "msg": "告警已入库"}
        with patch("src.notifications.service.httpx.post", return_value=Response()) as post:
            payload = service.dispatch(
                AlertEvent(RiskLevel.ATTENTION, time.time(), "关注测试"),
                alert_id=None,
                risk_score=42.5,
                person_id="notify_attention",
                device_id="test",
                reason_codes=["human_medium"],
            )
        ids.append(payload["notification_id"])
        assert payload["risk_level"] == "attention"
        assert payload["channels"] == ["cloud_function"]
        assert payload["ack_required"] is False
        post.assert_called_once_with(
            "https://example.test/fallAlarmPush",
            json={"action": "push", "data": {
                "risk_label": "风险关注", "title": "风险关注提醒", "message": "关注测试",
                "risk_level": "attention", "risk_score": 42.5, "isRead": False,
            }},
            timeout=5.0,
        )
        assert payload["cloud_push"]["status"] == "sent"
        assert db.get_notification(payload["notification_id"])["deliveries"][0]["status"] == "sent"
    finally:
        service.close()
        _cleanup(db, ids)


def test_internal_warning_maps_to_external_attention():
    db = Database()
    service = NotificationService(database=db)
    ids: list[str] = []
    try:
        with patch("src.notifications.service.httpx.post"):
            payload = service.dispatch(
                AlertEvent(RiskLevel.WARNING, time.time(), "趋势测试"),
                alert_id=None,
                risk_score=58.0,
                person_id="notify_warning",
                device_id="test",
                reason_codes=["base_warning"],
            )
        ids.append(payload["notification_id"])
        assert payload["source_risk_level"] == "warning"
        assert payload["risk_level"] == "attention"
        assert payload["channels"] == ["cloud_function"]
    finally:
        service.close()
        _cleanup(db, ids)


def test_critical_fallback_is_cancelled_by_ack():
    db = Database()
    service = NotificationService(database=db, fallback_seconds=30)
    ids: list[str] = []
    alert_id = int(time.time() * 1000) % 2_000_000_000
    try:
        with patch("src.notifications.service.httpx.post"):
            payload = service.dispatch(
                AlertEvent(RiskLevel.CRITICAL, time.time(), "高危测试"),
                alert_id=alert_id,
                risk_score=88.0,
                person_id="notify_critical",
                device_id="test",
                reason_codes=["human_high"],
            )
        ids.append(payload["notification_id"])
        assert payload["channels"] == ["cloud_function", "app", "sms"]
        assert {item["channel"] for item in payload["deliveries"]} == {"cloud_function", "app", "sms"}
        assert service.acknowledge_alert(alert_id) is True
        updated = db.get_notification(payload["notification_id"])
        assert updated["acknowledged"] is True
        assert updated["fallback"]["state"] == "cancelled"
        time.sleep(0.05)
        assert not any(item["channel"] == "phone" for item in db.get_notification(payload["notification_id"])["deliveries"])
    finally:
        service.close()
        _cleanup(db, ids)


def test_critical_fallback_creates_phone_stub_after_deadline():
    db = Database()
    service = NotificationService(database=db, fallback_seconds=0)
    ids: list[str] = []
    try:
        with patch("src.notifications.service.httpx.post"):
            payload = service.dispatch(
                AlertEvent(RiskLevel.CRITICAL, time.time(), "兜底测试"),
                alert_id=None,
                risk_score=91.0,
                person_id="notify_fallback",
                device_id="test",
                reason_codes=["interaction_high"],
            )
        ids.append(payload["notification_id"])
        deadline = time.time() + 1
        while time.time() < deadline:
            current = db.get_notification(payload["notification_id"])
            if any(item["channel"] == "phone" for item in current["deliveries"]):
                break
            time.sleep(0.01)
        current = db.get_notification(payload["notification_id"])
        assert current["fallback"]["state"] == "triggered"
        phone = [item for item in current["deliveries"] if item["channel"] == "phone"]
        assert phone and phone[0]["status"] == "not_configured"
    finally:
        service.close()
        _cleanup(db, ids)


def test_notification_policy_and_detail_routes():
    from src.api.main import app

    with TestClient(app) as client:
        response = client.get("/api/v1/notifications/policy")
        assert response.status_code == 200
        assert set(response.json()["levels"]) == {"low", "attention", "critical"}
        missing = client.get("/api/v1/notifications/does-not-exist")
        assert missing.status_code == 404


def test_wechat_cloud_function_payload_and_success_delivery():
    db = Database()
    adapter = WechatCloudFunctionAdapter(url="https://example.test/fallAlarmPush", enabled=True)
    service = NotificationService(database=db, wechat_adapter=adapter)
    ids: list[str] = []
    class Response:
        status_code = 200
        is_success = True
        def json(self):
            return {"code": 0, "msg": "ok"}
    try:
        with patch("src.notifications.service.httpx.post", return_value=Response()) as post:
            payload = service.dispatch(
                AlertEvent(RiskLevel.CRITICAL, time.time(), "高危云函数测试"),
                alert_id=None,
                risk_score=86.4,
                person_id="E001",
                device_id="test",
                reason_codes=["human_high"],
            )
        ids.append(payload["notification_id"])
        post.assert_called_once_with(
            "https://example.test/fallAlarmPush",
            json={"action": "push", "data": {
                "risk_label": "跌倒高危", "title": "跌倒高危告警", "message": "高危云函数测试",
                "risk_level": "critical", "risk_score": 86.4, "isRead": False,
            }},
            timeout=5.0,
        )
        assert payload["cloud_push"]["status"] == "sent"
        assert payload["cloud_push"]["response"]["code"] == 0
    finally:
        service.close()
        _cleanup(db, ids)
