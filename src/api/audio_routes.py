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

from src.api.database import Database
from src.inference.audio_analyzer import AudioAnalyzer
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

    return {
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
