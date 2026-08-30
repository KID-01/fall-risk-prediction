"""多渠道风险通知服务。

微信云函数上报适配器可通过环境变量启用；APP、短信和电话供应商仍使用投递桩。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
import os
from uuid import uuid4

import httpx

from src.alerts.engine import AlertEvent, RiskLevel
from src.api.database import Database
from src.utils.logger import get_logger

log = get_logger(__name__)

WECHAT_FALL_ALARM_PUSH_URL = os.getenv(
    "WECHAT_FALL_ALARM_PUSH_URL",
    "https://cloud1-d2gl1lav2eb6e440e-1477389215.ap-shanghai.app.tcloudbase.com/fallAlarmPush",
)


class WechatCloudFunctionAdapter:
    """将风险事件上报微信云函数；未启用时不发起外部请求。"""

    def __init__(
        self,
        url: str | None = None,
        enabled: bool | None = None,
        timeout: float = 5.0,
        payload_mode: str | None = None,
    ):
        self.url = url or WECHAT_FALL_ALARM_PUSH_URL
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("WECHAT_FALL_ALARM_PUSH_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        )
        self.timeout = timeout
        # legacy 保持现有 action/data 契约；hybrid 同时携带顶层字段，
        # 可兼容只读取 elderId/riskLevel/riskScore 的公网网关版本。
        self.payload_mode = (payload_mode or os.getenv("WECHAT_FALL_ALARM_PUSH_PAYLOAD_MODE", "legacy")).lower()

    def send(
        self,
        *,
        risk_label: str,
        title: str,
        message: str,
        risk_level: str,
        risk_score: float,
        person_id: str = "default",
    ) -> dict:
        if not self.enabled:
            return {"enabled": False, "status": "not_configured"}
        data = {
            "risk_label": risk_label,
            "title": title,
            "message": message,
            "risk_level": risk_level,
            "risk_score": round(float(risk_score), 2),
            "isRead": False,
        }
        body = {
            "action": "push",
            "data": data,
        }
        if self.payload_mode in {"hybrid", "flat"}:
            # 顶层字段是方案 A 公网网关的最小契约；保留 data/action 供旧云函数使用。
            body.update({
                "elderId": person_id,
                "riskLevel": risk_level,
                "riskScore": round(float(risk_score), 2),
            })
        if self.payload_mode == "flat":
            body = {
                "elderId": person_id,
                "riskLevel": risk_level,
                "riskScore": round(float(risk_score), 2),
            }
        try:
            response = httpx.post(self.url, json=body, timeout=self.timeout)
            # CloudBase 网关可能以 text/plain 返回空正文；HTTP 2xx 已表示请求被网关接收，
            # 不能因 response.json() 解析失败而把已送达的告警标记为失败。
            has_text = hasattr(response, "text")
            raw_text = (getattr(response, "text", "") or "").strip()
            try:
                # 测试替身可能只实现 json()；真实 httpx 响应则用正文是否为空判断。
                result = response.json() if (raw_text or not has_text) else None
            except ValueError:
                result = None
            if response.is_success and (result is None or result.get("code") == 0):
                return {"enabled": True, "status": "sent", "http_status": response.status_code, "response": result}
            return {
                "enabled": True,
                "status": "failed",
                "http_status": response.status_code,
                "response": result,
                "error": f"cloud_function_code={result.get('code') if result else 'unknown'}",
            }
        except Exception as exc:
            log.warning(f"微信云函数上报失败: {exc}")
            return {"enabled": True, "status": "failed", "error": str(exc)}

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
        "channels": ["cloud_function"],
        "ack_required": False,
        "fallback": None,
        "description": "通过云函数 HTTP 网关写入小程序告警列表",
    },
    "critical": {
        "label": "高危级",
        "channels": ["cloud_function", "app", "sms"],
        "ack_required": True,
        "fallback": {"enabled": True, "channel": "phone", "delay_seconds": 30},
        "description": "云函数 HTTP 网关写入小程序告警列表，30 秒未确认触发电话兜底",
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

    def __init__(
        self,
        database: Database | None = None,
        fallback_seconds: int = 30,
        wechat_adapter: WechatCloudFunctionAdapter | None = None,
    ):
        self.db = database or Database()
        self.fallback_seconds = int(fallback_seconds)
        self.wechat_adapter = wechat_adapter or WechatCloudFunctionAdapter()
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self.recover_pending_fallbacks()

    @staticmethod
    def policy() -> dict:
        return {
            "version": "risk-notification-v1",
            "levels": NOTIFICATION_POLICY,
            "transport": {
                "rest": "/api/v1/notifications",
                "acknowledge": "/api/v1/alerts/{alert_id}/acknowledge",
                "wechat_cloud_function": {
                    "enabled_env": "WECHAT_FALL_ALARM_PUSH_ENABLED",
                    "url_env": "WECHAT_FALL_ALARM_PUSH_URL",
                    "payload_mode_env": "WECHAT_FALL_ALARM_PUSH_PAYLOAD_MODE",
                    "payload_modes": ["legacy", "hybrid", "flat"],
                    "mode": "http_gateway",
                    "payload": ["action", "data.risk_label", "data.title", "data.message", "data.risk_level", "data.risk_score", "data.isRead"],
                },
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
        title = "跌倒高危告警" if level == RiskLevel.CRITICAL.value else "风险关注提醒"
        cloud_risk_label = "跌倒高危" if level == RiskLevel.CRITICAL.value else "风险关注"
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
            "title": title,
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

        # 小程序方案 A：关注级和高危级均写入云函数 fall_alerts；前端通过 action=pull 轮询。
        cloud_push = (
            self.wechat_adapter.send(
                risk_label=cloud_risk_label,
                title=title,
                message=alert.message,
                risk_level=level,
                risk_score=risk_score,
                person_id=person_id,
            )
            if level != RiskLevel.LOW.value
            else {"enabled": False, "status": "not_applicable", "reason": "low_local_only"}
        )
        payload["cloud_push"] = cloud_push
        self.db.update_notification_cloud_push(notification_id, cloud_push)

        for channel in policy["channels"]:
            if channel == "cloud_function":
                self.db.insert_notification_delivery(
                    notification_id,
                    channel,
                    cloud_push["status"],
                    error_message=cloud_push.get("error"),
                )
            else:
                self.db.insert_notification_delivery(notification_id, channel, "not_configured")

        if level == RiskLevel.CRITICAL.value:
            self._schedule_fallback(notification_id, max(0.0, (ack_deadline_at or 0) - time.time()))
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
        log.warning(f"高危通知未在30秒内确认，电话兜底待接入: {notification_id}")

    def _refresh_payload(self, notification_id: str) -> dict:
        return self.db.get_notification(notification_id) or {"notification_id": notification_id}
