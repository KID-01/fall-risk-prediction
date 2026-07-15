"""
跌倒风险预测系统 — FastAPI 后端入口
启动: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

API 路由 (RESTful + WebSocket):
  GET  /                              系统信息
  GET  /health                        健康检查
  GET  /config                        系统配置
  GET  /features/info                 四大特征说明
  GET  /alerts/levels                 预警分级规则

  POST /api/v1/stream/start           启动视频流分析
  POST /api/v1/stream/stop            停止视频流分析
  GET  /api/v1/risk/current           当前风险状态
  GET  /api/v1/risk/history           历史风险记录(分页)
  POST /api/v1/baseline/reset         重置基线

  GET  /api/v1/alerts                 告警历史(筛选+分页)
  POST /api/v1/alerts/{id}/acknowledge 确认告警

  GET  /api/v1/stats                  统计面板数据

  WS   /ws/alerts                     WebSocket 实时告警推送
"""
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pathlib import Path

from src.api.routes import alerts_router, monitor_router, stats_router
from src.api.websocket import websocket_endpoint
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logging

# ── 初始化 ──
setup_logging()
log = get_logger(__name__)
config: Any = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时初始化, 关闭时清理"""
    log.info("FastAPI 服务启动")
    yield
    log.info("FastAPI 服务关闭")


# ── 创建应用 ──
app = FastAPI(
    title="跌倒风险预测系统",
    description="基于多模态AI监测的老年人跌倒风险前置防控 API",
    version=str(config.project.version),
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ──
app.include_router(monitor_router)
app.include_router(alerts_router)
app.include_router(stats_router)

# ── WebSocket ──
app.websocket("/ws/alerts")(websocket_endpoint)


# ── 基础端点 ──

@app.get("/")
async def root():
    return {
        "system": config.project.name,
        "version": config.project.version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/config")
async def get_config_endpoint():
    return OmegaConf.to_container(config, resolve=True)


@app.get("/features/info")
async def features_info():
    """四大相对特征说明"""
    return {
        "walking_rhythm": {
            "name": "行走节拍频率",
            "desc": "髋关节y坐标时序FFT主频,反映行走节奏快慢",
            "unit": "Hz",
        },
        "step_amplitude": {
            "name": "步幅相对幅度",
            "desc": "踝关节摆动幅度经躯干高度归一化,反映步幅大小",
            "unit": "归一化值",
        },
        "trunk_stability": {
            "name": "躯干稳定指数",
            "desc": "肩髋连线与垂直方向夹角变化范围,越大越不稳定",
            "unit": "度",
        },
        "activity_density": {
            "name": "活动密度",
            "desc": "单位时间内站立/行走帧占比,反映日常活动水平",
            "unit": "0~1",
        },
    }


@app.get("/alerts/levels")
async def alert_levels():
    """预警分级规则"""
    levels = config.alert.levels
    return OmegaConf.to_container(levels, resolve=True)


# ── 静态文件服务(前端) ──
frontend_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
