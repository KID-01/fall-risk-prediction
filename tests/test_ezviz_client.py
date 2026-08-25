"""萤石客户端 Token 与分析流选择测试。"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock

from src.ezviz.client import EzvizClient


def test_normalize_expire_time_formats():
    now = 1_800_000_000.0

    assert EzvizClient._normalize_expire_time(604800, now) == now + 604800
    assert EzvizClient._normalize_expire_time(1_800_604_800, now) == 1_800_604_800
    assert EzvizClient._normalize_expire_time(1_800_604_800_000, now) == 1_800_604_800


def test_token_refresh_is_shared_between_concurrent_callers():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "code": "200",
        "data": {
            "accessToken": "temporary-token",
            "expireTime": int((time.time() + 604800) * 1000),
        },
    }
    http_client = Mock()
    http_client.post = AsyncMock(return_value=response)
    client = EzvizClient("app-key", "app-secret")
    client._get_client = AsyncMock(return_value=http_client)

    async def get_tokens():
        return await asyncio.gather(client.get_token(), client.get_token(), client.get_token())

    tokens = asyncio.run(get_tokens())

    assert tokens == ["temporary-token"] * 3
    http_client.post.assert_awaited_once()


def test_analysis_stream_uses_actual_url_scheme():
    client = EzvizClient("app-key", "app-secret")
    client.get_live_stream = AsyncMock(
        side_effect=[
            "https://example.invalid/live.m3u8",
            "rtmp://example.invalid/live",
        ]
    )

    stream = asyncio.run(client.get_analysis_stream("SERIAL", channel_no=2))

    assert stream == "rtmp://example.invalid/live"
    assert client.get_live_stream.await_args_list[0].kwargs["protocol"] == 3
    assert client.get_live_stream.await_args_list[1].kwargs["protocol"] == 2


def test_analysis_stream_rejects_browser_only_protocols():
    client = EzvizClient("app-key", "app-secret")
    client.get_live_stream = AsyncMock(
        side_effect=[
            "https://example.invalid/live.flv",
            "https://example.invalid/live.m3u8",
            None,
        ]
    )

    assert asyncio.run(client.get_analysis_stream("SERIAL")) is None
