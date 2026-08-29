"""多渠道风险通知服务。

外部 APP、短信和电话供应商暂不接入，默认使用可替换的投递桩并记录状态。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from uuid import uuid4

from src.alerts.engine import AlertEvent, RiskLevel
from src.api.database import Database
from src.api.websocket import manager
from src.utils.logger import get_logger

log = get_logger(__name__)

NOTIFICATION_POLICY = {
    "low": {
        "label": "低风险",
        "channels": [],
        "ack_required": False,
        "fallback": None,
        "description": "仅本地日志留存，不主动推送",
    },
    "attention": {
        "label": "关注级",
        "channels": ["websocket"],
        "ack_required": False,
        "fallback": None,
        "description": "通过 WebSocket 推送看板提醒",
    },
    "critical": {
        "label": "高危级",
        "channels": ["websocket", "app", "sms"],
        "ack_required": True,
        "fallback": {"enabled": True, "channel": "phone", "delay_seconds": 30},
        "description": "Web、APP、短信同步通知，30 秒未确认触发电话兜底",
    },
}
EXTERNAL_LEVEL_MAP = {
    "low": "low",
    "attention": "attention",
    # 兼容现有内部四级引擎：warning 对外归入黄色关注级。
    "warning": "attention",
    "critical": "critical",
}


class NotificationService:
    """创建风险通知、记录渠道投递并管理高危电话兜底。"""

    def __init__(self, database: Database | None = None, fallback_seconds: int = 30):
        self.db = database or Database()
        self.fallback_seconds = int(fallback_seconds)
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self.recover_pending_fallbacks()

    @staticmethod
    def policy() -> dict:
        return {
            "version": "risk-notification-v1",
            "levels": NOTIFICATION_POLICY,
            "transport": {
                "websocket": "/ws/alerts",
                "rest": "/api/v1/notifications",
                "acknowledge": "/api/v1/alerts/{alert_id}/acknowledge",
            },
        }

    def dispatch(
        self,
        alert: AlertEvent,
        *,
        alert_id: int | None,
        risk_score: float,
        person_id: str,
        device_id: str,
        reason_codes: list[str],
    ) -> dict:
        source_level = alert.level.value
        level = EXTERNAL_LEVEL_MAP.get(source_level, "attention")
        policy = NOTIFICATION_POLICY.get(level, NOTIFICATION_POLICY["low"])
        notification_id = uuid4().hex
        # 对外时间统一使用服务端 Unix 时间；视频源时间轴可能是相对秒数。
        occurred_at = time.time()
        # 告警时间可能来自视频流的相对时间轴，兜底截止必须使用服务端墙钟。
        ack_deadline_at = (
            time.time() + self.fallback_seconds if policy["ack_required"] else None
        )
        fallback_state = "scheduled" if policy["ack_required"] else None
        payload = {
            "notification_id": notification_id,
            "alert_id": alert_id,
            "risk_level": level,
            "risk_label": policy["label"],
            "source_risk_level": source_level,
            "risk_score": round(float(risk_score), 2),
            "person_id": person_id,
            "device_id": device_id,
            "occurred_at": occurred_at,
            "title": "跌倒风险告警",
            "message": alert.message,
            "reason_codes": list(reason_codes),
            "channels": list(policy["channels"]),
            "ack_required": bool(policy["ack_required"]),
            "acknowledged": False,
            "acknowledged_at": None,
            "ack_deadline_at": ack_deadline_at,
            "fallback": {
                "enabled": bool(policy["fallback"]),
                "channel": "phone" if policy["fallback"] else None,
                "state": fallback_state,
            },
            "created_at": datetime.now().isoformat(),
            "deliveries": [],
        }
        self.db.create_notification(payload)

        for channel in policy["channels"]:
            status = "queued" if channel == "websocket" else "not_configured"
            self.db.insert_notification_delivery(notification_id, channel, status)

        if level == RiskLevel.ATTENTION.value:
            self._broadcast(payload)
        elif level == RiskLevel.CRITICAL.value:
            self._broadcast(payload)
            self._schedule_fallback(notification_id, max(0.0, (ack_deadline_at or 0) - time.time()))
        else:
            log.info(f"低风险事件仅本地留存: person_id={person_id} device_id={device_id}")
        return self._refresh_payload(notification_id)

    def acknowledge_alert(self, alert_id: int) -> bool:
        """确认告警并取消关联高危通知的电话兜底。"""
        notification = self.db.get_notification_by_alert_id(alert_id)
        if not notification:
            return False
        self.db.acknowledge_notification(notification["notification_id"])
        with self._lock:
            timer = self._timers.pop(notification["notification_id"], None)
        if timer:
            timer.cancel()
        if notification.get("fallback", {}).get("state") == "scheduled":
            self.db.update_notification_fallback(notification["notification_id"], "cancelled")
        self._broadcast(self._refresh_payload(notification["notification_id"]))
        return True

    def recover_pending_fallbacks(self) -> None:
        for notification in self.db.query_notifications(
            risk_level=RiskLevel.CRITICAL.value,
            acknowledged=0,
            fallback_state="scheduled",
            limit=1000,
        ):
            deadline = notification.get("ack_deadline_at") or time.time()
            self._schedule_fallback(
                notification["notification_id"],
                max(0.0, float(deadline) - time.time()),
            )

    def close(self) -> None:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()

    def _schedule_fallback(self, notification_id: str, delay: float) -> None:
        with self._lock:
            if notification_id in self._timers:
                return
            timer = threading.Timer(delay, self._trigger_fallback, args=(notification_id,))
            timer.daemon = True
            self._timers[notification_id] = timer
            timer.start()

    def _trigger_fallback(self, notification_id: str) -> None:
        with self._lock:
            self._timers.pop(notification_id, None)
        notification = self.db.get_notification(notification_id)
        if not notification or notification.get("acknowledged"):
            return
        if notification.get("fallback", {}).get("state") != "scheduled":
            return
        self.db.update_notification_fallback(notification_id, "triggered")
        self.db.insert_notification_delivery(
            notification_id,
            "phone",
            "not_configured",
            fallback_due_at=notification.get("ack_deadline_at"),
        )
        payload = self._refresh_payload(notification_id)
        log.warning(f"高危通知未在30秒内确认，电话兜底待接入: {notification_id}")
        self._broadcast(payload)

    @staticmethod
    def _broadcast(payload: dict) -> None:
        manager.broadcast_threadsafe({"type": "risk_notification", "data": payload})

    def _refresh_payload(self, notification_id: str) -> dict:
        return self.db.get_notification(notification_id) or {"notification_id": notification_id}
