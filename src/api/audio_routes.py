"""音频分析接口 — 上传音频文件进行 PANNs 声音事件识别, 检测跌倒相关声音 (呼救/撞击)。

GET  /api/v1/audio/status   分析器配置与模型状态
POST /api/v1/audio/analyze  上传 wav/flac/ogg 音频, 返回声音事件与 top-k 标签

错误映射: 503 未启用/模型不可用, 413 超大小上限, 415 类型或扩展名不允许, 400 解码失败
"""
from __future__ import annotations

import asyncio
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import APIRouter, HTTPException, UploadFile

from src.alerts.engine import AlertEngine, RiskLevel
from src.api.database import Database
from src.api.websocket import manager
from src.inference.audio_analyzer import AudioAnalyzer
from src.inference.deviation import DeviationResult
from src.notifications.service import NotificationService
from src.utils.logger import get_logger

log = get_logger(__name__)

audio_router = APIRouter(prefix="/api/v1/audio", tags=["音频"])

# 上传限制: 20MB 大小上限 + 扩展名白名单 + Content-Type 需为 audio/*
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg"}

_analyzer: AudioAnalyzer | None = None
# 串行化 PANNs 推理, 见 audio_analyzer 模块文档 "需外部串行化"
_INFERENCE_LOCK = threading.Lock()


def get_analyzer() -> AudioAnalyzer:
    """模块级懒加载单例 — 测试通过 monkeypatch 此函数替换分析器"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioAnalyzer()
    return _analyzer


_alert_engine: AlertEngine | None = None
_notification_service: NotificationService | None = None


def get_alert_engine() -> AlertEngine:
    """模块级懒加载预警引擎 — 测试可 monkeypatch 注入轻量配置实例"""
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertEngine()
    return _alert_engine


def get_notification_service() -> NotificationService:
    """模块级懒加载通知服务 — 测试可通过 monkeypatch 替换"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


def get_monitor():
    """懒加载监控单例 — 延迟 import 避免与 routes 模块的导入顺序耦合。

    routes 模块在 import 时即实例化 FallRiskMonitor() 单例; 此处函数内 import
    保证直接单独导入本模块(如测试)时不触发整条监控流水线导入链。
    """
    from src.api.routes import monitor

    return monitor


def _sync_risk_state(level: RiskLevel, risk_score: float, message: str) -> None:
    """把音频告警同步到监控当前风险状态, 使风险表盘与趋势图随之更新。

    video 流水线在 EngineeringRiskFusion.evaluate() 后写 current_risk_score/level/
    message; 音频上传路径原本只写 alert_events, 不更新这些字段, 导致纯音频上传触发
    告警后表盘/趋势停在原值。此处按同一模式同步: 写 current_* 状态 + 落一条 risk_records
    + 广播 risk_update, 让前端 WS/轮询即时反映音频告警。
    """
    monitor = get_monitor()
    if monitor is None:
        return
    monitor.status.current_risk_score = risk_score
    monitor.status.current_risk_level = level
    monitor.status.current_risk_message = message
    try:
        Database().insert_risk_record(
            risk_score=risk_score,
            risk_level=level.value,
            person_id=monitor.status.person_id or "default",
            device_id=monitor.status.device_id or "default",
            risk_score_source="audio_upload_v0",
            reason_codes=["audio_upload_detection"],
        )
    except Exception as exc:
        log.warning(f"音频风险记录持久化失败: {exc}")
    manager.broadcast_threadsafe(
        {
            "type": "risk_update",
            "level": level.value,
            "score": risk_score,
            "reason_codes": ["audio_upload_detection"],
        }
    )


def evaluate_audio_alerts(
    events: list,
    person_id: str,
    device_id: str,
) -> dict[str, Any] | None:
    """对识别出的声音事件做预警评估并落库、通知与实时广播。

    复用监控主链的 AlertEngine 音频升级逻辑(撞击声→critical, 人声呼救→attention)。
    评估结果低于关注级(LOW)时不产生告警记录, 返回 None; 否则持久化 alert_events 并
    经 NotificationService 分发通知、经 WebSocket manager 广播, 返回告警信息。
    """
    # has_activity=True 使本次评估只由音频事件驱动(上传分析不评估真实静止时间),
    # 避免触发"超过N分钟无活动"的误导性分支从而污染告警信息
    alert = get_alert_engine().evaluate(
        DeviationResult(),
        timestamp=time.time(),
        has_activity=True,
        audio_events=events,
        emit=False,
    )
    if alert.level.priority <= 0:  # LOW 不产生告警记录
        return None

    risk_score = float(alert.level.priority) * 100.0 / 3.0
    reason_codes = ["audio_upload_detection"]
    # 同步当前风险状态(表盘+趋势), 使纯音频上传告警也能驱动前端风险评分转动
    _sync_risk_state(alert.level, risk_score, alert.message)
    alert_id: int | None = None
    notification: dict | None = None
    try:
        alert_id = Database().insert_alert_event(
            alert_level=alert.level.value,
            message=alert.message,
            risk_score=risk_score,
            person_id=person_id,
            device_id=device_id,
            reason_codes=reason_codes,
        )
        notification = get_notification_service().dispatch(
            alert,
            alert_id=alert_id,
            risk_score=risk_score,
            person_id=person_id,
            device_id=device_id,
            reason_codes=reason_codes,
        )
    except Exception as exc:
        log.warning(f"音频告警持久化/通知失败: {exc}")

    manager.broadcast_threadsafe(
        {
            "type": "alert",
            "level": alert.level.value,
            "score": risk_score,
            "message": alert.message,
            "reason_codes": reason_codes,
            "source": "audio_upload",
            "notification_id": (
                notification.get("notification_id") if notification else None
            ),
        }
    )
    return {
        "level": alert.level.value,
        "risk_score": risk_score,
        "message": alert.message,
        "alert_id": alert_id,
        "notification_id": (
            notification.get("notification_id") if notification else None
        ),
        "reason_codes": reason_codes,
    }


@audio_router.get("/status")
async def audio_status() -> dict[str, Any]:
    analyzer = get_analyzer()
    checkpoint_exists = Path(analyzer.checkpoint_path).expanduser().is_file()
    if checkpoint_exists and analyzer.enabled and not analyzer.model_loaded:
        try:
            await asyncio.to_thread(analyzer._ensure_model)
        except Exception as exc:
            log.warning(f"模型预加载失败: {exc}")
    return {
        "enabled": analyzer.enabled,
        "model_type": analyzer.model_type,
        "device": analyzer.device,
        "sample_rate": analyzer.sample_rate,
        "chunk_seconds": analyzer.chunk_seconds,
        "top_k": analyzer.top_k,
        "min_event_score": analyzer.min_event_score,
        "vocal_distress_threshold": analyzer.vocal_threshold,
        "impact_threshold": analyzer.impact_threshold,
        "checkpoint_exists": checkpoint_exists,
        "model_loaded": analyzer.model_loaded,
    }


@audio_router.post("/analyze")
async def analyze_audio(
    file: UploadFile,
    timestamp: float = 0.0,
    person_id: str = "default",
    device_id: str = "default",
) -> dict[str, Any]:
    """分析上传的音频文件, 返回声音事件与全局 top-k 标签

    Args:
        file: multipart 音频文件 (.wav/.flac/.ogg)
        timestamp: 音频起始时间戳(秒), 默认为 0
        person_id: 人员ID (默认 "default", 用于入库归属)
        device_id: 设备ID (默认 "default", 用于入库归属)
    """
    analyzer = get_analyzer()
    if not analyzer.enabled:
        raise HTTPException(status_code=503, detail="音频分析未启用 (audio.enabled=false)")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="音频文件超过 20MB 上限")

    suffix = Path(file.filename or "").suffix.lower()
    content_type = file.content_type or ""
    # octet-stream 是 curl/浏览器对未识别扩展名的通用回退, 仍受扩展名白名单约束
    type_allowed = (
        content_type.startswith("audio/") or content_type == "application/octet-stream"
    )
    if suffix not in ALLOWED_EXTENSIONS or not type_allowed:
        raise HTTPException(
            status_code=415,
            detail="仅支持 .wav/.flac/.ogg 音频文件 (Content-Type: audio/*)",
        )

    try:
        waveform, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="音频解码失败, 文件已损坏或格式不受支持"
        ) from exc

    # 推理在独立线程执行, 避免 CPU 密集计算阻塞事件循环; 线程内加锁串行化共享模型访问
    try:
        with _INFERENCE_LOCK:
            result = await asyncio.to_thread(
                analyzer.analyze_waveform, waveform, int(sample_rate), timestamp
            )
    except Exception as exc:
        log.warning(f"音频分析失败: {exc}")
        raise HTTPException(status_code=503, detail=f"音频分析失败: {exc}") from exc

    alert_payload: dict[str, Any] | None = None
    if result.events:
        try:
            db = Database()
            db.insert_audio_events(
                result.events,
                person_id=person_id,
                device_id=device_id,
                epoch_base=time.time(),
            )
        except Exception as exc:
            log.warning(f"音频事件持久化失败: {exc}")

        # 事件触发声音告警的评估与落库/通知/广播(撞击声/人声呼救达到阈值时)
        try:
            alert_payload = evaluate_audio_alerts(result.events, person_id, device_id)
        except Exception as exc:
            log.warning(f"音频告警评估失败: {exc}")

    payload: dict[str, Any] = {
        "duration_sec": result.duration_sec,
        "elapsed_ms": result.elapsed_ms,
        "events": [
            {
                "category": event.category.value,
                "label": event.label,
                "class_index": event.class_index,
                "score": event.score,
                "timestamp": event.timestamp,
            }
            for event in result.events
        ],
        "top_labels": [[label, score] for label, score in result.top_labels],
    }
    if alert_payload is not None:
        payload["alert"] = alert_payload
    return payload
