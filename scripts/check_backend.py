#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单次后台状态检查
用法: python scripts/check_backend.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import websockets

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_config


class BackendChecker:
    def __init__(self, base_url: str = "http://localhost:8000", ws_url: str = "ws://localhost:8000/ws/video"):
        self.base_url = base_url
        self.ws_url = ws_url
        self.client = httpx.AsyncClient(timeout=5.0)

    async def check_health(self) -> dict:
        try:
            resp = await self.client.get(f"{self.base_url}/health")
            return {"status": "healthy" if resp.status_code == 200 else "unhealthy", "code": resp.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def check_config(self) -> dict:
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
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/ezviz/devices")
            if resp.status_code == 200:
                devices = resp.json().get("devices", [])
                return {"count": len(devices), "devices": devices}
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
        except Exception as e:
            return {"error": str(e)}

    async def check_alerts(self) -> dict:
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/alerts?limit=5")
            if resp.status_code == 200:
                data = resp.json()
                return {"total": data.get("total", 0), "recent": data.get("items", [])}
            return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    async def check_stats(self) -> dict:
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/stats")
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    async def check_websocket(self) -> dict:
        try:
            async with websockets.connect(self.ws_url, open_timeout=3) as ws:
                await ws.send(json.dumps({"type": "ping"}))
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                return {"status": "connected", "response": resp[:100]}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def run_all(self) -> dict:
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

    def print_report(self, data: dict):
        print("\n" + "=" * 50)
        print("BACKEND STATUS REPORT")
        print("=" * 50)

        h = data.get("health", {})
        status = h.get("status", "unknown")
        ok = status == "healthy"
        print(f"Health:      {'[OK]' if ok else '[FAIL]'} {status}")

        c = data.get("config", {})
        if "error" not in c:
            print(f"Project:     {c.get('project')} v{c.get('version')}")
            print(f"Audio:       {'ON' if c.get('audio_enabled') else 'OFF'}")
            print(f"Ezviz:       {'Configured' if c.get('ezviz_configured') else 'Not configured'}")
        else:
            print(f"Config:      [FAIL] {c['error']}")

        e = data.get("ezviz", {})
        if "error" not in e:
            print(f"Devices:     {e.get('count', 0)} online")
        else:
            print(f"Ezviz:       [FAIL] {e.get('error')}")

        a = data.get("alerts", {})
        if "error" not in a:
            print(f"Alerts:      {a.get('total', 0)} total, {len(a.get('recent', []))} recent")
        else:
            print(f"Alerts:      [FAIL] {a['error']}")

        s = data.get("stats", {})
        if "error" not in s:
            print(f"Stats:       sessions={s.get('active_sessions', 0)} risks={s.get('risk_count', 0)}")
        else:
            print(f"Stats:       [FAIL] {s['error']}")

        w = data.get("websocket", {})
        ws_ok = w.get("status") == "connected"
        print(f"WebSocket:   {'[OK]' if ws_ok else '[FAIL]'} {w.get('status')}")

        print("=" * 50)


async def main():
    base = "http://localhost:8000"
    ws = "ws://localhost:8000/ws/video"

    checker = BackendChecker(base, ws)
    data = await checker.run_all()
    checker.print_report(data)

    # 返回非零码如果有失败
    has_error = any("error" in v for v in data.values() if isinstance(v, dict))
    sys.exit(1 if has_error else 0)


if __name__ == "__main__":
    asyncio.run(main())