# 更新日志

## 2026-07-10 核心算法管线实现

### 概述

基于项目方案（基于多模态AI监测的老年人跌倒风险前置防控），完成了从视频帧到风险预警的完整算法管线，覆盖技术路线全链路：

```
视频拉流 → YOLOv8n人体检测 → MediaPipe Pose关键点提取 → 帧质量过滤
→ 四大相对特征计算 → 个体化基线对比 → 双层偏离检测 → 四级预警输出
```

共新增/修改 **14 个文件**，新增依赖 `mediapipe` + `ultralytics`。

---

### 新增文件

#### 工具模块 `src/utils/`

| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置加载单例，基于 OmegaConf 读取 `configs/base.yaml` |
| `keypoints.py` | MediaPipe Pose 33 个关键点索引定义（`PoseKeypoint` 枚举）、`KeypointFrame` 数据结构、帧质量检查函数（下肢可见性 + 躯干完整性） |

#### 数据管线 `src/data/`

| 文件 | 说明 |
|------|------|
| `video_capture.py` | 视频流采集器，支持 RTSP/本地文件/摄像头，帧采样降频（15fps→10fps），帧缓冲区（用于视频回溯） |
| `human_detector.py` | YOLOv8n 人体检测器，延迟加载模型，返回检测框并判断是否完整人体 |
| `keypoint_extractor.py` | MediaPipe Pose 关键点提取器，输出 33 个 2D 关键点 `[x, y, z, visibility]`，自动执行帧质量检查 |
| `frame_filter.py` | 帧质量过滤器，置信度阈值过滤 + 下肢 6 关键点可见性检查（至少 4 个可见） |

#### 推理引擎 `src/inference/`

| 文件 | 说明 |
|------|------|
| `features.py` | 四大相对特征计算：①行走节拍频率（髋关节 y 坐标 FFT 主频）②步幅相对幅度（踝关节摆动/躯干高度归一化）③躯干稳定指数（肩髋连线与垂直方向夹角变化范围）④活动密度（髋部位移判断运动，统计占比） |
| `baseline.py` | 个体化基线管理：7 天采集期，计算均值/标准差/协方差逆矩阵，支持马氏距离计算和 Z-Score，SQLite 持久化存储 |
| `deviation.py` | 双层偏离检测：第一层短期（5 分钟窗口/30 秒步长/马氏距离/连续 3 窗口触发）；第二层长期（14 天窗口/线性回归斜率/连续 7 天负向变化触发） |
| `monitor.py` | 跌倒风险监控服务（单例），线程化运行，整合全链路：视频→检测→关键点→过滤→特征→基线→偏离→预警 |

#### 预警引擎 `src/alerts/`

| 文件 | 说明 |
|------|------|
| `engine.py` | 四级预警引擎：低风险（持续监测）→ 关注级（短期偏离≥3次/小时，APP 推送）→ 预警级（长期趋势下降，短信通知）→ 高危级（近似跌倒/4 小时无活动，电话通知）。支持注册响应动作、事件日志、视频回溯触发 |

---

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `configs/base.yaml` | 全面重写，对齐方案技术路线：MediaPipe Pose Lite 替换 movenet_thunder、新增 YOLOv8n 人体检测配置、四大特征参数、个体化基线配置、双层偏离检测参数、四级预警规则、萤石平台配置占位 |
| `src/api/main.py` | 从基础脚手架扩展为完整 API（11 个端点）：新增 `/monitor/start`、`/monitor/stop`、`/monitor/status`、`/monitor/alerts`、`/monitor/baseline/reset`、`/features/info` 等端点，集成 `FallRiskMonitor` 单例 |
| `pyproject.toml` | 新增依赖 `mediapipe>=0.10.0`、`ultralytics>=8.1.0` |
| `src/data/__init__.py` | 更新模块文档字符串 |
| `src/inference/__init__.py` | 新增模块文档字符串 |
| `src/alerts/__init__.py` | 新增模块文档字符串 |
| `src/edge/__init__.py` | 新增模块文档字符串 |
| `src/utils/__init__.py` | 新增模块文档字符串 |

---

### 依赖安装

| 包 | 版本 | 用途 |
|----|------|------|
| `mediapipe` | 0.10.35 | MediaPipe Pose 2D 关键点提取 |
| `ultralytics` | 8.4.91 | YOLOv8n 人体目标检测 |

---

### 验证结果

- 全部模块导入成功 ✅
- FastAPI app 正常加载 ✅
- 配置正确读取（MediaPipe Pose Lite / YOLOv8n / 四大特征 / 四级预警）✅
- Lint 检查通过（仅 HINT 级未使用导入，已清理）✅

---

## 2026-07-10 项目部署与文档

### 部署

- 创建 Python 3.14.0 虚拟环境 `venv/`
- 安装全部项目依赖（torch 2.13.0+cpu, fastapi, onnxruntime, omegaconf, loguru 等 20+ 包）
- 环境检查通过 14/15（仅 CUDA 不可用，CPU 模式正常运行）

### 文档

- 新增 `tips.md` — 完整使用方法（环境准备、启动/停止服务、API 说明、命令对照、端口占用排查）
- 修复 `.gitignore` 误加 `src/api/main.py` 忽略规则

### Git

- 提交 `8a19728`：feat: 添加 FastAPI 服务入口和使用说明文档
- 推送至 `origin/master` 成功

---

## 待开发模块

按项目方案研究进度，以下模块待后续实现：

| 优先级 | 模块 | 说明 |
|--------|------|------|
| P0 | 萤石 API 接入 | RTSP 地址获取、设备列表管理、`configs/ezviz.yaml` 配置 |
| P0 | 视频回溯 | 预警时自动保存前后 15 秒视频片段 |
| P1 | 前端界面 | Vue.js + ECharts：实时监控面板、趋势折线图、告警时间线、视频回放 |
| P1 | 推送通道 | 短信 API / APP 推送 / 电话通知对接 |
| P2 | 测试数据 | 采集模拟场景数据（正常行走/缓慢行走/拄拐/坐姿起立等） |
| P2 | 参数调优 | 检测阈值、滑动窗口、特征权重优化 |
| P3 | 适老化前端 | 大字体、高对比色、三步以内操作 |
