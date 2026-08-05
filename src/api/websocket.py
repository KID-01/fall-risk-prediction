"""
WebSocket 实时推送服务 — 告警实时推送到前端
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from src.utils.logger import get_logger

log = get_logger(__name__)


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"WebSocket 客户端连接, 当前在线: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info(f"WebSocket 客户端断开, 当前在线: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """广播消息到所有连接的客户端"""
        text = json.dumps(message, ensure_ascii=False, default=str)
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_text(text)
            except Exception as e:
                log.warning(f"推送失败: {e}")
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, message: dict):
        """发送消息给单个客户端"""
        await websocket.send_text(json.dumps(message, ensure_ascii=False, default=str))


# 全局连接管理器单例
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点处理函数"""
    await manager.connect(websocket)
    try:
        while True:
            # 接收客户端心跳/消息
            data = await websocket.receive_text()
            msg = json.loads(data) if data else {}
            if msg.get("type") == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WebSocket异常: {e}")
        manager.disconnect(websocket)
