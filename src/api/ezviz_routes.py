"""萤石设备发现、EZOpen 播放授权和一键监控启动接口。"""
from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.api.routes import monitor
from src.ezviz.client import EzvizClient

ezviz_router = APIRouter(prefix="/api/v1/ezviz", tags=["萤石设备"])


class EzvizPlayerRequest(BaseModel):
    device_id: str = Field(min_length=8, max_length=64)
    channel_no: int = Field(default=1, ge=1, le=64)


class EzvizMonitorRequest(EzvizPlayerRequest):
    person_id: str = Field(default="default", min_length=1, max_length=128)


def _device_id(serial: str) -> str:
    """为前端生成稳定的非敏感设备标识。"""
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()[:16]


def _is_online(value: Any) -> bool:
    return value in (1, "1", True)


def _is_encrypted(value: Any) -> bool:
    return value not in (None, 0, "0", False, "")


def _channels(device: dict[str, Any]) -> list[int]:
    values = device.get("channelNos") or device.get("channels") or [device.get("channelNo", 1)]
    if not isinstance(values, (list, tuple)):
        values = [values]
    channels: list[int] = []
    for value in values:
        try:
            number = int(value.get("channelNo", 1) if isinstance(value, dict) else value)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in channels:
            channels.append(number)
    return channels or [1]


def _public_device(device: dict[str, Any]) -> dict[str, Any] | None:
    serial = str(device.get("deviceSerial") or "")
    if not serial:
        return None
    return {
        "device_id": _device_id(serial),
        "display_serial": f"{serial[:2]}****{serial[-4:]}",
        "name": str(device.get("deviceName") or "未命名设备"),
        "online": _is_online(device.get("status")),
        "encrypted": _is_encrypted(device.get("isEncrypt")),
        "channels": _channels(device),
    }


def _client(request: Request) -> EzvizClient:
    client = getattr(request.app.state, "ezviz_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="未配置萤石凭据，请创建本机 configs/ezviz.yaml")
    return client


async def _resolve_device(
    client: EzvizClient, opaque_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        devices = await client.list_devices(page_size=50)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="萤石设备列表获取失败") from exc
    for device in devices:
        public = _public_device(device)
        if public and public["device_id"] == opaque_id:
            return device, public
    raise HTTPException(status_code=404, detail="未找到该萤石设备，请刷新设备列表")


def _player_payload(serial: str, channel_no: int, token: str) -> dict[str, Any]:
    return {
        "accessToken": token,
        "url": f"ezopen://open.ys7.com/{serial}/{channel_no}.live",
    }


@ezviz_router.get("/devices")
async def list_devices(request: Request):
    """返回设备选择信息，不返回完整序列号或 AppSecret。"""
    client = _client(request)
    try:
        devices = await client.list_devices(page_size=50)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="萤石设备列表获取失败") from exc
    return {"devices": [item for device in devices if (item := _public_device(device)) is not None]}


@ezviz_router.post("/player")
async def player_config(
    request: Request, body: EzvizPlayerRequest, response: Response
):
    """为 EZUIKit 获取临时播放配置。"""
    client = _client(request)
    device, public = await _resolve_device(client, body.device_id)
    if not public["online"]:
        raise HTTPException(status_code=409, detail="设备离线，无法播放")
    if body.channel_no not in public["channels"]:
        raise HTTPException(status_code=400, detail="所选通道不属于该设备")
    try:
        token = await client.ensure_token()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="萤石授权获取失败") from exc
    response.headers["Cache-Control"] = "no-store"
    return {
        "device": public,
        "channel_no": body.channel_no,
        **_player_payload(str(device["deviceSerial"]), body.channel_no, token),
    }


@ezviz_router.post("/monitor/start")
async def start_monitor(
    request: Request, body: EzvizMonitorRequest, response: Response
):
    """同时启动 RTMP/RTSP 后端分析，并返回 EZOpen 播放配置。"""
    if monitor.status.is_running:
        raise HTTPException(status_code=409, detail="监控已在运行中，请先停止")
    client = _client(request)
    device, public = await _resolve_device(client, body.device_id)
    if not public["online"]:
        raise HTTPException(status_code=409, detail="设备离线，无法启动监控")
    if body.channel_no not in public["channels"]:
        raise HTTPException(status_code=400, detail="所选通道不属于该设备")

    serial = str(device["deviceSerial"])
    try:
        token = await client.ensure_token()
        analysis_url = await client.get_analysis_stream(serial, body.channel_no)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="萤石直播地址获取失败") from exc
    if not analysis_url:
        raise HTTPException(status_code=502, detail="未获取到可供分析的 RTMP/RTSP 地址")

    audio_source = None
    try:
        rtsp_url = await client.get_rtsp_url(serial, body.channel_no)
        if rtsp_url:
            audio_source = rtsp_url
    except Exception:
        pass
    if not audio_source:
        audio_source = analysis_url

    if not monitor.start(
        source=analysis_url,
        person_id=body.person_id,
        device_id=body.device_id,
        audio_source=audio_source,
    ):
        raise HTTPException(status_code=500, detail="后端监控启动失败")
    response.headers["Cache-Control"] = "no-store"
    return {
        "code": 200,
        "message": "萤石监控已启动",
        "device": public,
        "channel_no": body.channel_no,
        **_player_payload(serial, body.channel_no, token),
    }
