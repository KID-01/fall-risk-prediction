"""
WebSocket 实时推送服务 — 告警实时推送 + 视频帧推送
"""
from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect

from src.utils.logger import get_logger

log = get_logger(__name__)


class ConnectionManager:
    """WebSocket 连接管理器 (告警用)"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
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

    def broadcast_threadsafe(self, message: dict) -> None:
        """允许监控线程向告警 WebSocket 广播结构化消息。"""
        if self._loop is None or not self.active_connections:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)

    async def send_personal(self, websocket: WebSocket, message: dict):
        """发送消息给单个客户端"""
        await websocket.send_text(json.dumps(message, ensure_ascii=False, default=str))


class VideoConnectionManager:
    """视频帧 WebSocket 连接管理器 — 推送 JPEG 二进制帧，支持跨线程调用"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        log.info(f"视频 WebSocket 连接, 当前在线: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info(f"视频 WebSocket 断开, 当前在线: {len(self.active_connections)}")

    def broadcast_frame(self, jpeg_bytes: bytes):
        """线程安全 — 可从任意线程调用，广播 JPEG 帧到所有客户端"""
        if self._loop is None or not self.active_connections:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast(jpeg_bytes), self._loop
        )

    async def _broadcast(self, jpeg_bytes: bytes):
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_bytes(jpeg_bytes)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    @property
    def has_clients(self) -> bool:
        return len(self.active_connections) > 0


# 全局连接管理器单例
manager = ConnectionManager()
video_manager = VideoConnectionManager()
raw_video_manager = VideoConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点 — 告警推送"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data) if data else {}
            if msg.get("type") == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WebSocket异常: {e}")
        manager.disconnect(websocket)


async def video_websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点 — 视频帧推送 (骨骼叠加)"""
    await video_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data) if data else {}
            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        video_manager.disconnect(websocket)
    except Exception as e:
        log.error(f"视频WebSocket异常: {e}")
        video_manager.disconnect(websocket)


async def raw_video_websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点 — 原始视频帧推送 (无骨骼叠加)"""
    await raw_video_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data) if data else {}
            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        raw_video_manager.disconnect(websocket)
    except Exception as e:
        log.error(f"原始视频WebSocket异常: {e}")
        raw_video_manager.disconnect(websocket)
