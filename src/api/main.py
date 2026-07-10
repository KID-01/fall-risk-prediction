"""
跌倒风险预测系统 — FastAPI 后端入口
启动方式: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

API 端点:
  GET  /                    系统信息
  GET  /health              健康检查
  GET  /config              系统配置
  GET  /alerts/levels       预警分级规则
  POST /predict             单次预测(占位)
  POST /monitor/start       启动实时监控
  POST /monitor/stop        停止监控
  GET  /monitor/status      监控状态
  GET  /monitor/alerts      预警历史
  POST /monitor/baseline/reset  重置基线
  GET  /features/info       四大特征说明
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.inference.monitor import FallRiskMonitor
from src.utils.config import get_config

# ── 加载配置 ──
config: Any = get_config()

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

# ── 监控服务单例 ──
monitor = FallRiskMonitor()


# ============================================================
# 请求/响应模型
# ============================================================

class MonitorStartRequest(BaseModel):
    source: str = "0"               # 视频源(RTSP地址/文件/摄像头编号)
    person_id: str = "default"      # 被监测人员ID


class PredictRequest(BaseModel):
    video_url: str | None = None


# ============================================================
# 基础端点
# ============================================================

@app.get("/")
async def root():
    """系统信息"""
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
async def get_config_endpoint():
    """获取系统配置"""
    return OmegaConf.to_container(config, resolve=True)


# ============================================================
# 预警相关
# ============================================================

@app.get("/alerts/levels")
async def alert_levels():
    """预警分级规则"""
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


# ============================================================
# 特征说明
# ============================================================

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


# ============================================================
# 预测端点(占位)
# ============================================================

@app.post("/predict")
async def predict(req: PredictRequest):
    """跌倒风险预测(占位,实际推理需通过监控服务)"""
    return {
        "code": 200,
        "message": "预测接口已就绪,实时推理请使用 /monitor/start",
        "data": {
            "risk_score": None,
            "alert_level": "low",
            "video_url": req.video_url,
        },
    }


# ============================================================
# 实时监控端点
# ============================================================

@app.post("/monitor/start")
async def monitor_start(req: MonitorStartRequest):
    """启动实时监控"""
    if monitor.status.is_running:
        raise HTTPException(status_code=409, detail="监控已在运行中,请先停止")
    success = monitor.start(source=req.source, person_id=req.person_id)
    if not success:
        raise HTTPException(status_code=500, detail="启动失败")
    return {"code": 200, "message": "监控已启动", "source": req.source, "person_id": req.person_id}


@app.post("/monitor/stop")
async def monitor_stop():
    """停止监控"""
    monitor.stop()
    return {"code": 200, "message": "监控已停止"}


@app.get("/monitor/status")
async def monitor_status():
    """获取监控状态"""
    return monitor.get_status()


@app.get("/monitor/alerts")
async def monitor_alerts(level: str | None = None, limit: int = 100):
    """获取预警历史"""
    return monitor.get_alert_history(level=level, limit=limit)


@app.post("/monitor/baseline/reset")
async def monitor_baseline_reset():
    """重置个体化基线"""
    monitor.reset_baseline()
    return {"code": 200, "message": "基线已重置"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
