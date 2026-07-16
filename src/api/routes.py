"""
路由模块 — 按功能拆分 API 路由
  routes_monitor.py  — 实时监控相关
  routes_alerts.py   — 告警相关
  routes_stats.py    — 统计面板
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.database import Database
from src.inference.monitor import FallRiskMonitor
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── 请求模型 ──

class MonitorStartRequest(BaseModel):
    source: str = "0"
    person_id: str = "default"


class PredictRequest(BaseModel):
    video_url: str | None = None


# ── 监控路由 ──

monitor_router = APIRouter(prefix="/api/v1", tags=["监控"])
monitor = FallRiskMonitor()
db = Database()


@monitor_router.post("/stream/start")
async def stream_start(req: MonitorStartRequest):
    """启动实时视频流分析"""
    if monitor.status.is_running:
        raise HTTPException(status_code=409, detail="监控已在运行中,请先停止")
    success = monitor.start(source=req.source, person_id=req.person_id)
    if not success:
        raise HTTPException(status_code=500, detail="启动失败")
    return {"code": 200, "message": "监控已启动", "source": req.source, "person_id": req.person_id}


@monitor_router.post("/stream/stop")
async def stream_stop():
    """停止视频流分析"""
    monitor.stop()
    return {"code": 200, "message": "监控已停止"}


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
    import time
    start_time = time.time() - hours * 3600
    records = db.query_risk_records(
        person_id=person_id,
        start_time=start_time,
        limit=limit,
        offset=offset,
    )
    return {"total": len(records), "records": records}


@monitor_router.post("/baseline/reset")
async def baseline_reset():
    """重置个体化基线"""
    monitor.reset_baseline()
    return {"code": 200, "message": "基线已重置"}


# ── 告警路由 ──

alerts_router = APIRouter(prefix="/api/v1", tags=["告警"])


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
    import time
    start_time = time.time() - hours * 3600
    alerts = db.query_alert_events(
        alert_level=level,
        person_id=person_id,
        start_time=start_time,
        acknowledged=acknowledged,
        limit=limit,
        offset=offset,
    )
    return {"total": len(alerts), "alerts": alerts}


@alerts_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """确认告警"""
    success = db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="告警不存在")
    return {"code": 200, "message": "告警已确认"}


# ── 统计路由 ──

stats_router = APIRouter(prefix="/api/v1", tags=["统计"])


@stats_router.get("/stats")
async def get_stats(hours: int = 24):
    """统计面板数据"""
    return db.get_stats(hours=hours)
