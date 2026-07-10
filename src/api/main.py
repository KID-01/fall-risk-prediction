"""
跌倒风险预测系统 — FastAPI 后端入口
启动方式: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf

# ── 加载配置 ──
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
config: Any = OmegaConf.load(_CONFIG_PATH)


# ── 创建 FastAPI 应用 ──
app = FastAPI(
    title="跌倒风险预测系统",
    description="基于多模态AI监测的老年人跌倒风险前置防控 API",
    version=str(config.project.version),
)

# ── CORS 中间件 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 路由 ──

@app.get("/")
async def root():
    """根路径 — 系统信息"""
    return {
        "system": config.project.name,
        "version": config.project.version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/config")
async def get_config():
    """获取当前系统配置"""
    return OmegaConf.to_container(config, resolve=True)


@app.get("/alerts/levels")
async def alert_levels():
    """获取预警分级规则"""
    return {
        "green": {"threshold": config.alert.green_threshold, "desc": "低风险"},
        "yellow": {
            "threshold": config.alert.yellow_threshold,
            "desc": "中风险",
            "cooldown_minutes": config.alert.yellow_cooldown_minutes,
        },
        "orange": {
            "threshold": config.alert.orange_threshold,
            "desc": "高风险",
            "cooldown_minutes": config.alert.orange_cooldown_minutes,
        },
        "red": {"threshold": 100, "desc": "已跌倒/紧急"},
    }


@app.post("/predict")
async def predict(video_url: str | None = None):
    """
    跌倒风险预测接口（占位）
    实际实现需接入姿态估计 + 步态分析 + Transformer 风险模型
    """
    return {
        "code": 200,
        "message": "预测接口已就绪，模型推理模块待实现",
        "data": {
            "risk_score": None,
            "alert_level": "green",
            "video_url": video_url,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
