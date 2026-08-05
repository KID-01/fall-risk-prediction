"""
汇报演示脚本 — 通过 FastAPI 后端跑通完整跌倒风险预警链路

演示完整架构闭环:
  前端/脚本 → FastAPI API → FallRiskMonitor → YOLO-Pose → 特征 → 基线 → 偏离 → 预警
                                                              → WebSocket 推送
                                                              → SQLite 持久化

使用方式:
  1. 先启动 FastAPI 后端:
     uvicorn src.api.main:app --host 0.0.0.0 --port 8000

  2. 运行本脚本（调用 API 控制监控）:
     python scripts/demo_report.py --source 0                    # 摄像头
     python scripts/demo_report.py --source data/raw/test.mp4    # 视频文件
     python scripts/demo_report.py --source "rtsp://..."         # 萤石流

  3. 同时打开前端看板查看实时效果:
     cd frontend && npm run dev
     浏览器打开 http://localhost:5173

汇报要点:
  - 后端 FastAPI 提供 RESTful API + WebSocket
  - YOLOv8n-Pose 一次推理完成人体检测+关键点提取
  - 规则引擎（马氏距离+偏离检测）实现实时风险判断，不依赖训练权重
  - 前端 React+ECharts 看板实时展示风险等级/仪表盘/趋势图/告警列表
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

API_BASE = "http://localhost:8000/api/v1"


def start_monitor(source: str, person_id: str = "demo_user", device_id: str = "demo"):
    """通过 FastAPI 接口启动监控"""
    print("=" * 70)
    print("  跌倒风险预测系统 — 汇报演示（FastAPI + YOLO-Pose + 规则引擎）")
    print("=" * 70)
    print(f"  API 地址: {API_BASE}")
    print(f"  视频源: {source}")
    print(f"  被监测人: {person_id}")
    print()

    with httpx.Client(timeout=30.0) as client:
        # 1. 健康检查
        print("[1] 健康检查...")
        resp = client.get("http://localhost:8000/health")
        print(f"    {resp.json()}")

        # 2. 查看当前配置（确认后端为 yolo_pose）
        print("\n[2] 确认姿态估计后端配置...")
        resp = client.get("http://localhost:8000/config")
        cfg = resp.json()
        backend = cfg.get("pose_estimation", {}).get("backend", "unknown")
        print(f"    姿态后端: {backend}")

        # 3. 重置基线
        print("\n[3] 重置个体化基线...")
        resp = client.post(f"{API_BASE}/baseline/reset", params={"person_id": person_id})
        print(f"    {resp.json()}")

        # 4. 启动监控
        print(f"\n[4] 启动监控（source={source}）...")
        resp = client.post(
            f"{API_BASE}/stream/start",
            json={"source": source, "person_id": person_id, "device_id": device_id},
        )
        result = resp.json()
        print(f"    {result}")

        if resp.status_code != 200:
            print("\n❌ 启动失败，请检查视频源是否可用")
            return

        # 5. 轮询状态（汇报时实时展示）
        print("\n[5] 实时监控状态（每 3 秒刷新）...")
        print("    （同时可打开前端 http://localhost:5173 查看看板）")
        print("-" * 70)

        for i in range(60):  # 演示 3 分钟
            resp = client.get(f"{API_BASE}/risk/current")
            status = resp.json()

            level = status.get("current_risk_label", "未知")
            baseline_ready = status.get("baseline_ready", False)
            baseline_samples = status.get("baseline_samples", 0)
            frames_processed = status.get("frames_processed", 0)
            frames_valid = status.get("frames_valid", 0)
            feature = status.get("last_feature")
            alert = status.get("last_alert")

            # 打印状态
            feature_str = ""
            if feature and len(feature) == 4:
                feature_str = (
                    f"节拍={feature[0]:.2f}Hz  步幅={feature[1]:.3f}  "
                    f"躯干={feature[2]:.1f}°  活动={feature[3]:.2f}"
                )

            baseline_str = (
                f"✅就绪({baseline_samples}样本)"
                if baseline_ready
                else f"⏳采集({baseline_samples}/100)"
            )

            print(
                f"  [{i*3:>3}s] {level}  | 帧:{frames_processed}(有效{frames_valid})  "
                f"| 基线:{baseline_str}  | {feature_str}"
            )

            if alert and alert.get("level") != "low":
                level_emoji = {"attention": "🟡", "warning": "🟠", "critical": "🔴"}
                print(
                    f"         {level_emoji.get(alert['level'], '⚠️')} "
                    f"[{alert.get('message', '')}]"
                )

            # 本地视频读完后后端会将 is_running 置为 False，立即结束轮询。
            if not status.get("is_running", False):
                break
            time.sleep(3)

        # 6. 停止监控
        print("\n[6] 停止监控...")
        resp = client.post(f"{API_BASE}/stream/stop")
        print(f"    {resp.json()}")

        # 7. 查询历史记录和告警
        print("\n[7] 查询数据库记录...")
        resp = client.get(f"{API_BASE}/risk/history?hours=1&limit=5")
        history = resp.json()
        print(f"    风险记录: {history.get('total', 0)} 条")

        resp = client.get(f"{API_BASE}/alerts?hours=1&limit=5")
        alerts = resp.json()
        print(f"    告警事件: {alerts.get('total', 0)} 条")

        resp = client.get(f"{API_BASE}/stats?hours=1")
        stats = resp.json()
        print(f"    统计面板: {stats}")

    # 技术路径说明
    print("\n" + "=" * 70)
    print("  技术路径说明")
    print("=" * 70)
    print("  架构: 前端(React) → FastAPI(REST+WS) → FallRiskMonitor → YOLO-Pose")
    print("  链路: 视频帧 → 人体检测+关键点 → 帧过滤 → 四大特征 → 基线 → 偏离 → 预警")
    print("  持久化: SQLite(risk_records + alert_events)")
    print("  推送: WebSocket → 前端实时刷新")
    print()
    print("  ✅ 当前演示: 规则引擎（马氏距离+偏离检测），不依赖训练权重")
    print("  🔄 升级方向: FallRiskPredictor 深度模型，训练集采集+算法迭代中")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="跌倒风险预测 — 汇报演示（FastAPI）")
    parser.add_argument(
        "--source", type=str, default="0",
        help="视频源: 文件路径 / 摄像头编号(0) / RTSP地址",
    )
    parser.add_argument("--person", type=str, default="demo_user", help="被监测人ID")
    parser.add_argument("--device", type=str, default="demo", help="设备ID")
    parser.add_argument(
        "--api", type=str, default="http://localhost:8000", help="FastAPI 地址"
    )
    args = parser.parse_args()

    API_BASE = f"{args.api}/api/v1"
    start_monitor(source=args.source, person_id=args.person, device_id=args.device)
