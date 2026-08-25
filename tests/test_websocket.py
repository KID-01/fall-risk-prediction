"""
WebSocket 模块单元测试 — ConnectionManager / VideoConnectionManager / 端点
覆盖告警推送与视频帧推送两条链路; 端点级测试使用 FastAPI TestClient
"""
from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from src.api.websocket import ConnectionManager, VideoConnectionManager


class FakeWebSocket:
    """假 WebSocket — 记录 accept 状态与已发送内容"""

    def __init__(self):
        self.accepted = False
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def accept(self):
        self.accepted = True

    async def send_text(self, data: str):
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)


# ============================================================
# 单元测试: VideoConnectionManager (视频帧推送, 依赖 asyncio)
# ============================================================
class TestVideoConnectionManager:
    """VideoConnectionManager — 跨线程视频帧广播管理器"""

    def test_connect_sets_running_loop(self):
        """connect 应记录当前事件循环供跨线程广播使用"""
        manager = VideoConnectionManager()
        ws = FakeWebSocket()
        asyncio.run(manager.connect(ws))
        assert ws.accepted is True
        assert manager._loop is not None
        assert len(manager.active_connections) == 1

    def test_broadcast_frame_without_clients_is_silent(self):
        """无客户端且无事件循环时 broadcast_frame 应静默返回"""
        manager = VideoConnectionManager()
        manager.broadcast_frame(b"\xff\xd8")  # 不应抛出任何异常


# ============================================================
# 单元测试: ConnectionManager (告警推送, 回归锁定)
# ============================================================
class TestConnectionManager:
    """ConnectionManager — 告警消息广播管理器"""

    def test_connect_and_broadcast(self):
        """connect 后 broadcast 应推送到全部在线客户端"""
        manager = ConnectionManager()
        ws = FakeWebSocket()
        asyncio.run(manager.connect(ws))

        async def _broadcast() -> None:
            await manager.broadcast({"type": "alert", "level": "warning"})

        asyncio.run(_broadcast())
        assert len(ws.sent_text) == 1
        msg = json.loads(ws.sent_text[0])
        assert msg["type"] == "alert"
        assert msg["level"] == "warning"


# ============================================================
# 端点级测试: /ws/alerts 与 /ws/video (ping/pong 心跳)
# ============================================================
class TestWebSocketEndpoints:
    """FastAPI TestClient 端点测试 — 覆盖真实路由注册"""

    def test_alerts_endpoint_ping_pong(self):
        """/ws/alerts 应响应 ping→pong (告警端点回归锁定)"""
        from src.api.main import app

        with TestClient(app) as client:
            with client.websocket_connect("/ws/alerts") as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                data = ws.receive_json()
                assert data == {"type": "pong"}

    def test_video_endpoint_ping_pong(self):
        """/ws/video 应响应 ping→pong (修复前: asyncio NameError 崩溃)"""
        from src.api.main import app

        with TestClient(app) as client:
            with client.websocket_connect("/ws/video") as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                data = ws.receive_json()
                assert data == {"type": "pong"}
