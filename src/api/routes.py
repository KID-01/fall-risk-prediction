"""
路由模块 — 三个 APIRouter 均在此文件定义
  monitor_router — 实时监控相关(启动/停止/风险/基线)
  alerts_router  — 告警相关(查询/确认)
  stats_router   — 统计面板
"""
from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.database import Database
from src.inference.monitor import FallRiskMonitor
from src.notifications.service import NOTIFICATION_POLICY
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── 请求模型 ──

class MonitorStartRequest(BaseModel):
    source: str = "0"
    person_id: str = "default"
    device_id: str = "default"
    audio_source: str | None = None


class PredictRequest(BaseModel):
    video_url: str | None = None


class UploadedStreamStartRequest(BaseModel):
    upload_id: str
    person_id: str = "default"
    device_id: str = "default"
    audio_source: str | None = None


# ── 监控路由 ──

monitor_router = APIRouter(prefix="/api/v1", tags=["监控"])
monitor = FallRiskMonitor()
db = Database()

VIDEO_UPLOAD_MAX_BYTES = 500 * 1024 * 1024
VIDEO_UPLOAD_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VIDEO_UPLOAD_DIR = Path(tempfile.gettempdir()) / "fall-risk-prediction-uploads"
VIDEO_UPLOAD_TTL_SECONDS = 10 * 60


@dataclass
class StagedVideo:
    path: Path
    original_name: str
    expires_at: float


_staged_videos: dict[str, StagedVideo] = {}
_staged_videos_lock = threading.Lock()


def _delete_staged_video(staged: StagedVideo | None) -> None:
    if staged is None:
        return
    try:
        staged.path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning(f"清理暂存视频失败: {exc}")


def cleanup_staged_videos(force: bool = False) -> None:
    """清理过期或服务关闭时仍未确认的上传文件。"""
    now = time.time()
    with _staged_videos_lock:
        expired = [
            upload_id
            for upload_id, staged in _staged_videos.items()
            if force or staged.expires_at <= now
        ]
        stale = [_staged_videos.pop(upload_id) for upload_id in expired]
    for staged in stale:
        _delete_staged_video(staged)


async def _save_uploaded_video(file: UploadFile) -> tuple[Path, str]:
    """保存并验证上传视频，返回临时路径和原始文件名。"""
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if not original_name or suffix not in VIDEO_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="仅支持 .mp4/.avi/.mov/.mkv/.webm 视频文件",
        )

    try:
        VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="创建视频临时目录失败") from exc
    path = VIDEO_UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    total = 0
    try:
        with path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > VIDEO_UPLOAD_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="视频文件超过 500MB 上限")
                output.write(chunk)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="保存视频文件失败") from exc

    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="视频文件为空")

    try:
        capture = cv2.VideoCapture(str(path))
        try:
            decodable = capture.isOpened()
        finally:
            capture.release()
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="视频无法解码，请检查文件是否损坏或格式是否受支持") from exc
    if not decodable:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="视频无法解码，请检查文件是否损坏或格式是否受支持")
    return path, original_name


@monitor_router.post("/stream/start")
async def stream_start(req: MonitorStartRequest):
    """启动实时视频流分析"""
    if monitor.status.is_running:
        raise HTTPException(status_code=409, detail="监控已在运行中,请先停止")
    success = monitor.start(
        source=req.source,
        person_id=req.person_id,
        device_id=req.device_id,
        audio_source=req.audio_source,
    )
    if not success:
        raise HTTPException(status_code=500, detail="启动失败")
    return {
        "code": 200,
        "message": "监控已启动",
        "source": req.source,
        "person_id": req.person_id,
        "device_id": req.device_id,
        "audio_source": monitor.status.audio_source,
    }


@monitor_router.post("/stream/stop")
async def stream_stop():
    """停止视频流分析"""
    monitor.stop()
    return {"code": 200, "message": "监控已停止"}


@monitor_router.post("/stream/upload")
async def stream_upload(
    file: UploadFile = File(...),
    person_id: str = Form("default"),
    device_id: str = Form("default"),
    audio_source: str | None = Form(None),
):
    """上传并暂存本地视频，等待用户确认开始分析。"""
    cleanup_staged_videos()
    path, original_name = await _save_uploaded_video(file)
    # 单用户监控台只保留最后一次待确认上传，避免无主文件堆积。
    cleanup_staged_videos(force=True)
    upload_id = uuid4().hex
    expires_at = time.time() + VIDEO_UPLOAD_TTL_SECONDS
    with _staged_videos_lock:
        _staged_videos[upload_id] = StagedVideo(
            path=path,
            original_name=original_name,
            expires_at=expires_at,
        )
    return {
        "code": 200,
        "message": "视频已上传，请确认开始分析",
        "upload_id": upload_id,
        "source_name": original_name,
        "expires_at": expires_at,
        "expires_in_seconds": VIDEO_UPLOAD_TTL_SECONDS,
        "monitor_running": monitor.status.is_running,
    }


@monitor_router.post("/stream/upload/start")
async def stream_upload_start(req: UploadedStreamStartRequest):
    """确认使用暂存视频，并替换当前监控源。"""
    cleanup_staged_videos()
    with _staged_videos_lock:
        staged = _staged_videos.pop(req.upload_id, None)
    if staged is None:
        raise HTTPException(status_code=404, detail="上传视频不存在或已过期，请重新选择")

    try:
        if monitor.status.is_running:
            monitor.stop()
        success = monitor.start(
            source=str(staged.path),
            person_id=req.person_id or "default",
            device_id=req.device_id or "default",
            audio_source=req.audio_source,
            temporary_source_path=str(staged.path),
        )
    except Exception as exc:
        _delete_staged_video(staged)
        log.exception("启动上传视频监控失败")
        raise HTTPException(status_code=500, detail="启动上传视频监控失败") from exc
    if not success:
        _delete_staged_video(staged)
        raise HTTPException(status_code=409, detail="当前监控无法停止，请稍后重试")
    return {
        "code": 200,
        "message": "上传视频监控已启动",
        "source_type": "uploaded",
        "source_name": staged.original_name,
        "person_id": monitor.person_id,
        "device_id": monitor.device_id,
        "audio_source": monitor.status.audio_source,
    }


@monitor_router.delete("/stream/upload/{upload_id}")
async def stream_upload_cancel(upload_id: str):
    """取消暂存视频并清理文件。"""
    with _staged_videos_lock:
        staged = _staged_videos.pop(upload_id, None)
    if staged is None:
        raise HTTPException(status_code=404, detail="上传视频不存在或已过期")
    _delete_staged_video(staged)
    return {"code": 200, "message": "暂存视频已取消"}


@monitor_router.get("/risk/current")
async def risk_current():
    """获取当前风险状态"""
    return monitor.get_status()


@monitor_router.get("/risk/history")
async def risk_history(
    person_id: str | None = None,
    hours: int = 24,
    limit: int = 100,
    offset: int = 0,
):
    """历史风险记录(分页)"""
    start_time = time.time() - hours * 3600
    records = db.query_risk_records(
        person_id=person_id,
        start_time=start_time,
        limit=limit,
        offset=offset,
    )
    return {"total": len(records), "records": records}


@monitor_router.post("/baseline/reset")
async def baseline_reset(person_id: str | None = None):
    """重置个体化基线"""
    monitor.reset_baseline(person_id)
    return {"code": 200, "message": "基线已重置", "person_id": person_id or monitor.person_id}


# ── 告警路由 ──

alerts_router = APIRouter(prefix="/api/v1", tags=["告警"])


@alerts_router.get("/notifications/policy")
async def get_notification_policy():
    """获取风险等级与通知渠道策略，供客户端初始化。"""
    return monitor.notification_service.policy()


@alerts_router.get("/notifications")
async def get_notifications(
    person_id: str | None = None,
    device_id: str | None = None,
    risk_level: str | None = None,
    acknowledged: int | None = None,
    hours: int = 24,
    limit: int = 100,
    offset: int = 0,
):
    """查询通知事件及各渠道投递状态。"""
    if risk_level and risk_level not in NOTIFICATION_POLICY:
        raise HTTPException(status_code=400, detail="risk_level 仅支持 low/attention/critical")
    start_time = time.time() - hours * 3600
    notifications = db.query_notifications(
        person_id=person_id,
        device_id=device_id,
        risk_level=risk_level,
        acknowledged=acknowledged,
        start_time=start_time,
        limit=limit,
        offset=offset,
    )
    return {"total": len(notifications), "notifications": notifications}


@alerts_router.get("/notifications/{notification_id}")
async def get_notification(notification_id: str):
    """按不透明通知 ID 查询详情。"""
    notification = db.get_notification(notification_id)
    if notification is None:
        raise HTTPException(status_code=404, detail="通知不存在")
    return notification


@alerts_router.get("/alerts")
async def get_alerts(
    level: str | None = None,
    person_id: str | None = None,
    hours: int = 24,
    acknowledged: int | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """查询告警历史"""
    start_time = time.time() - hours * 3600
    alerts = db.query_alert_events(
        alert_level=level,
        person_id=person_id,
        start_time=start_time,
        acknowledged=acknowledged,
        limit=limit,
        offset=offset,
    )
    for alert in alerts:
        notification = db.get_notification_by_alert_id(alert["id"])
        if notification:
            alert["notification"] = notification
        else:
            alert["notification"] = None
    return {"total": len(alerts), "alerts": alerts}


@alerts_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """确认告警"""
    success = db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="告警不存在")
    monitor.notification_service.acknowledge_alert(alert_id)
    return {"code": 200, "message": "告警已确认"}


# ── 统计路由 ──

stats_router = APIRouter(prefix="/api/v1", tags=["统计"])


@stats_router.get("/stats")
async def get_stats(hours: int = 24):
    """统计面板数据"""
    return db.get_stats(hours=hours)


# ── 音频事件路由 ──

audio_events_router = APIRouter(prefix="/api/v1", tags=["音频事件"])


@audio_events_router.get("/audio/events")
async def get_audio_events(
    person_id: str | None = None,
    category: str | None = None,
    hours: int = 24,
    limit: int = 100,
    offset: int = 0,
):
    """查询音频事件历史"""
    start_time = time.time() - hours * 3600
    events = db.query_audio_events(
        person_id=person_id,
        category=category,
        start_time=start_time,
        limit=limit,
        offset=offset,
    )
    return {"total": len(events), "events": events}
