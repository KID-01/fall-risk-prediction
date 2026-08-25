#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时后台监控脚本
用法: python scripts/monitor_backend.py [--interval 2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import websockets

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_config


class BackendMonitor:
    def __init__(self, base_url: str = "http://localhost:8000", ws_url: str = "ws://localhost:8000/ws/video"):
        self.base_url = base_url
        self.ws_url = ws_url
        self.client = httpx.AsyncClient(timeout=5.0)
        self.running = True

    async def check_health(self) -> dict:
        """健康检查"""
        try:
            resp = await self.client.get(f"{self.base_url}/health")
            return {"status": "healthy" if resp.status_code == 200 else "unhealthy", "code": resp.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def check_config(self) -> dict:
        """获取配置摘要"""
        try:
            resp = await self.client.get(f"{self.base_url}/config")
            data = resp.json()
            return {
                "project": data.get("project", {}).get("name", "unknown"),
                "version": data.get("project", {}).get("version", "unknown"),
                "audio_enabled": data.get("audio", {}).get("enabled", False),
                "ezviz_configured": bool(data.get("ezviz", {}).get("app_key")),
            }
        except Exception as e:
            return {"error": str(e)}

    async def check_ezviz_devices(self) -> dict:
        """萤石设备列表"""
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/ezviz/devices")
            if resp.status_code == 200:
                devices = resp.json().get("devices", [])
                return {"count": len(devices), "devices": devices}
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
        except Exception as e:
            return {"error": str(e)}

    async def check_alerts(self) -> dict:
        """告警历史"""
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/alerts?limit=5")
            if resp.status_code == 200:
                data = resp.json()
                return {"total": data.get("total", 0), "recent": data.get("items", [])}
            return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    async def check_websocket(self) -> dict:
        """WebSocket 连接测试"""
        try:
            async with websockets.connect(self.ws_url, open_timeout=3) as ws:
                await ws.send(json.dumps({"type": "ping"}))
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                return {"status": "connected", "response": resp[:100]}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def check_stats(self) -> dict:
        """统计面板"""
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/stats")
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    async def run_once(self) -> dict:
        """单次全量检查"""
        results = await asyncio.gather(
            self.check_health(),
            self.check_config(),
            self.check_ezviz_devices(),
            self.check_alerts(),
            self.check_stats(),
            self.check_websocket(),
            return_exceptions=True,
        )
        keys = ["health", "config", "ezviz", "alerts", "stats", "websocket"]
        return {k: (v if not isinstance(v, Exception) else {"error": str(v)}) for k, v in zip(keys, results)}

    def format_output(self, data: dict) -> str:
        """格式化输出"""
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [f"\n{'='*60}", f"[{ts}] 后台监控", f"{'='*60}"]

        # Health
        h = data.get("health", {})
        status = h.get("status", "unknown")
        icon = "[OK]" if status == "healthy" else "[FAIL]"
        lines.append(f"  Health: {icon} {status}")

        # Config
        c = data.get("config", {})
        if "error" not in c:
            lines.append(f"  Project: {c.get('project')} v{c.get('version')}")
            lines.append(f"  Audio: {'ON' if c.get('audio_enabled') else 'OFF'}")
            lines.append(f"  Ezviz: {'配置済' if c.get('ezviz_configured') else '未配置'}")
        else:
            lines.append(f"  Config: ❌ {c['error']}")

        # Ezviz
        e = data.get("ezviz", {})
        if "error" not in e:
            lines.append(f"  Devices: {e.get('count', 0)} 台")
        else:
            lines.append(f"  Ezviz: ❌ {e.get('error')}")

        # Alerts
        a = data.get("alerts", {})
        if "error" not in a:
            lines.append(f"  Alerts: {a.get('total', 0)} 条 (最近 {len(a.get('recent', []))} 条)")
        else:
            lines.append(f"  Alerts: ❌ {a['error']}")

        # Stats
        s = data.get("stats", {})
        if "error" not in s:
            lines.append(f"  Stats: sessions={s.get('active_sessions', 0)} risks={s.get('risk_count', 0)}")
        else:
            lines.append(f"  Stats: ❌ {s['error']}")

        # WebSocket
        w = data.get("websocket", {})
        ws_icon = "[OK]" if w.get("status") == "connected" else "[FAIL]"
        lines.append(f"  WebSocket: {ws_icon} {w.get('status', 'unknown')}")

        return "\n".join(lines)

    async def run_loop(self, interval: float):
        """循环监控"""
        print(f"[MONITOR] Starting real-time monitoring (interval {interval}s) - Ctrl+C to exit")
        try:
            while self.running:
                os.system("cls" if os.name == "nt" else "clear")
                data = await self.run_once()
                print(self.format_output(data))
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
        finally:
            await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser(description="实时后台监控")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="检查间隔(秒)")
    parser.add_argument("--host", default="localhost", help="后端主机")
    parser.add_argument("--port", type=int, default=8000, help="后端端口")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    ws = f"ws://{args.host}:{args.port}/ws/video"

    monitor = BackendMonitor(base, ws)
    await monitor.run_loop(args.interval)


if __name__ == "__main__":
    asyncio.run(main())