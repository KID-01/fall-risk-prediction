# 跌倒风险预测系统 — 完全使用手册

> 基于多模态 AI 监测的老年人跌倒风险前置防控系统
> 挑战杯 · 揭榜挂帅赛道 · 发榜单位: 海康威视/萤石

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [环境准备与安装](#3-环境准备与安装)
4. [项目结构详解](#4-项目结构详解)
5. [配置文件说明](#5-配置文件说明)
6. [启动服务](#6-启动服务)
7. [API 接口文档](#7-api-接口文档)
8. [前端看板](#8-前端看板)
9. [Docker 部署](#9-docker-部署)
10. [开发指南](#10-开发指南)
11. [常见问题](#11-常见问题)

---

## 1. 项目概述

### 1.1 背景

居家养老场景中，老年人跌倒是最常见的安全事故之一。传统方案聚焦于"跌倒后识别"（检测到跌倒再报警），而本系统的核心理念是 **"跌前预判"**——通过对老年人日常活动模式的持续监测，提前发现风险趋势，在跌倒发生之前进行干预。

### 1.2 核心创新

- **从"跌倒后识别"升级为"以跌倒风险前置防控为核心"的全流程方案**
- **个体化基线**：为每位老人建立专属活动模式画像，以"相对于自身的变化"而非"统一标准"作为判断依据
- **双层偏离检测**：同时捕捉短期急性异常（突然行走不稳）和长期渐进衰退（体力逐步下降）
- **四级预警**：低风险→关注级→预警级→高危级，避免过度告警

### 1.3 技术路线

```
视频帧 → YOLOv8n 人体检测 → MediaPipe Pose / YOLOv8-Pose 关键点
→ 帧质量过滤 → 四大相对特征计算 → 个体化基线对比
→ 双层偏离检测 → 四级预警输出
```

### 1.4 适用场景

- 居家养老、社区养老中心的老年人日常活动监测
- 通过萤石 RTSP 摄像头或本地视频文件实时分析
- 家属端实时看板，可远程查看老人风险状态

---

## 2. 技术架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (React + ECharts)                  │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐   │
│  │ 风险等级  │  │ 风险评分  │  │ 风险趋势  │  │ 告警   │   │
│  │ 大卡片    │  │ 仪表盘   │  │ 折线图    │  │ 列表   │   │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬───┘   │
│       └────────────┴──────────────┴──────────────┘       │
│                         │  HTTP / WebSocket               │
├─────────────────────────┼─────────────────────────────────┤
│                    后端 (FastAPI)                          │
│  ┌────────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │ RESTful API│  │ WebSocket │  │ 数据库 (SQLite)      │   │
│  │ 路由       │  │ 实时推送  │  │ risk_records        │   │
│  └─────┬──────┘  └────┬─────┘  │ alert_events        │   │
│        │              │        └─────────────────────┘   │
├────────┼──────────────┼──────────────────────────────────┤
│        ▼              ▼                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │         核心推理引擎 (FallRiskMonitor)              │    │
│  │  ┌──────┐ ┌────────┐ ┌────────┐ ┌─────────────┐   │    │
│  │  │ 特征  │→│ 基线   │→│ 偏离   │→│ 预警引擎     │   │    │
│  │  │ 计算  │ │ 管理   │ │ 检测   │ │ (四级)      │   │    │
│  │  └──────┘ └────────┘ └────────┘ └─────────────┘   │    │
│  └────────────────────────────────────────────────────┘    │
│                         ▲                                   │
│  ┌──────────────────────┼──────────────────────────────┐    │
│  │           数据采集管线                                │    │
│  │  ┌──────────┐ ┌──────────────┐ ┌──────────────┐    │    │
│  │  │ 视频拉流  │→│ 人体检测      │→│ 关键点提取   │    │    │
│  │  │ RTSP/文件│ │ YOLOv8n     │ │ MediaPipe    │    │    │
│  │  └──────────┘ └──────────────┘ └──────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心流程详解

| 步骤 | 模块 | 说明 |
|------|------|------|
| 1. 视频采集 | `src/data/video_capture.py` | 从 RTSP 流/本地文件/摄像头读取帧，降频采样（15→10fps） |
| 2. 人体检测 | `src/data/human_detector.py` | YOLOv8n 检测画面中的人体，无人则跳过避免浪费算力 |
| 3. 关键点提取 | `src/data/keypoint_extractor.py` | MediaPipe Pose Lite 提取 33 个人体关键点坐标 |
| 4. 帧过滤 | `src/data/frame_filter.py` | 丢弃关键点置信度低、下肢不可见的帧 |
| 5. 特征计算 | `src/inference/features.py` | 计算四个相对特征（见 2.3） |
| 6. 基线对比 | `src/inference/baseline.py` | 前 7 天建立个体基线，后续用马氏距离对比 |
| 7. 偏离检测 | `src/inference/deviation.py` | 短期（分钟级）马氏距离 + 长期（天级）斜率双重检测 |
| 8. 预警输出 | `src/alerts/engine.py` | 四级风险判定，触发对应动作 |

### 2.3 四大相对特征

系统设计为**不依赖物理尺度**——单目摄像头无法精确测量步长多少米，因此设计四个相对特征，只关注变化趋势：

| 特征 | 计算方法 | 物理意义 |
|------|---------|---------|
| **行走节拍频率** | 髋关节 y 坐标时序 FFT 主频 | 行走时身体上下律动，频率反映步频快慢 |
| **步幅相对幅度** | 踝关节摆动范围 ÷ 躯干高度 | 用躯干高度归一化，消除距离/视角影响 |
| **躯干稳定指数** | 肩髋连线与垂直方向夹角变化范围 | 正常行走躯干基本垂直，不稳时摇摆加剧 |
| **活动密度** | 窗口内髋关节有位移的帧占比 | 反映日常活动水平，久坐/少动时显著变化 |

### 2.4 双层偏离检测

| 层级 | 时间窗口 | 方法 | 触发条件 | 监测目标 |
|------|---------|------|---------|---------|
| **短期** | 5 分钟窗口/30 秒步长 | 马氏距离 | 连续 3 个窗口距离 > 3.0 | 突然行走不稳、差点绊倒 |
| **长期** | 14 天滑动窗口 | 线性回归斜率 | 斜率 < -0.05 持续 7 天 | 体力逐步下降、衰落趋势 |

### 2.5 四级预警

| 等级 | 名称 | 判定条件 | 推送方式 |
|------|------|---------|---------|
| 🟢 | 低风险 | 所有特征在基线 ±1σ 内 | 不推送，持续监测 |
| 🟡 | 关注级 | 短期偏离 ≥3 次/小时 | APP 推送提醒家属关注 |
| 🟠 | 预警级 | 长期趋势连续 7 天负向变化 | 短信通知家属，建议体检 |
| 🔴 | 高危级 | 近似跌倒动作 / 超 4 小时无活动 | 电话通知家属，立即确认 |

---

## 3. 环境准备与安装

### 3.1 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.10 及以上
- **RAM**: ≥ 4GB（推荐 8GB）
- **硬盘**: ≥ 2GB 可用空间

### 3.2 快速安装

```powershell
# 1. 克隆仓库
git clone <仓库地址>
cd fall-risk-prediction

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

# 4. 安装依赖
pip install -e ".[dev]"

# 5. 检查环境
python scripts/check_env.py
```

### 3.3 环境检查说明

`python scripts/check_env.py` 会逐项检查：

- ✅ Python 版本 ≥ 3.10
- ✅ 所有依赖包已安装及版本
- ✅ 配置文件存在且完整
- ✅ 各模块可成功导入
- ⚠️ CUDA 不可用—当前为 CPU 版 PyTorch，不影响功能，仅推理较慢

**预期输出**（15 项检查中 14 项通过，CUDA 不可用为正常）：

```
[OK] Python 版本: 3.14.0
[OK] PyTorch: 2.13.0+cpu
[OK] MediaPipe: 0.10.35
...
[INFO] CUDA 不可用，使用 CPU 模式
结果: 14/15 通过
```

### 3.4 Makefile 命令速查

| 命令 | 等效命令 | 说明 |
|------|---------|------|
| `make install` | `pip install -e .` | 安装运行依赖 |
| `make install-dev` | `pip install -e ".[dev]"` | 安装开发依赖（含 lint/测试工具） |
| `make check` | `python scripts/check_env.py` | 环境检查 |
| `make serve` | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload` | 启动 API 服务 |
| `make lint` | `ruff check src/` | 代码质量检查 |
| `make format` | `ruff format src/` | 自动格式化代码 |
| `make clean` | 清理 `__pycache__` 和 `*.pyc` | 清理临时文件 |

> ⚠️ Windows 默认没有 `make` 命令。你可以：
> - 通过 Scoop 安装: `scoop install make`
> - 或直接使用上面的"等效命令"

---

## 4. 项目结构详解

```
fall-risk-prediction/
│
├── src/                          # 全部 Python 源码
│   ├── utils/                    # 工具层
│   │   ├── config.py             # 全局配置加载（OmegaConf 单例）
│   │   ├── keypoints.py          # 关键点枚举、数据定义、质量检查
│   │   └── logger.py             # 结构化日志（loguru，按天轮转）
│   │
│   ├── data/                     # 数据采集与处理管线
│   │   ├── video_capture.py      # 视频拉流（RTSP/文件/摄像头）
│   │   ├── human_detector.py     # YOLOv8n 人体检测
│   │   ├── keypoint_extractor.py # MediaPipe Pose 关键点提取
│   │   ├── frame_filter.py       # 帧质量过滤
│   │   ├── preprocess.py         # ROI 裁剪预处理
│   │   ├── keypoint_store.py     # 关键点数据持久化（.npy）
│   │   └── dataset.py            # PyTorch Dataset/DataLoader
│   │
│   ├── inference/                # 推理引擎
│   │   ├── features.py           # 四大相对特征计算
│   │   ├── baseline.py           # 个体化基线管理
│   │   ├── deviation.py          # 双层偏离检测
│   │   └── monitor.py            # 全链路监控服务（单例）
│   │
│   ├── alerts/                   # 预警引擎
│   │   └── engine.py             # 四级预警引擎
│   │
│   ├── api/                      # FastAPI 后端
│   │   ├── main.py               # 应用入口、基础端点
│   │   ├── routes.py             # RESTful API 路由
│   │   ├── database.py           # SQLite 数据库层
│   │   └── websocket.py          # WebSocket 实时推送
│   │
│   ├── models/                   # 深度学习模型（训练用）
│   │   ├── temporal_encoder.py   # Transformer 时序编码器
│   │   ├── multimodal_fusion.py  # 多模态交叉注意力融合
│   │   ├── risk_head.py          # 风险回归+分类头
│   │   └── fall_risk_predictor.py# 完整模型组装
│   │
│   └── edge/                     # 边缘计算（待开发占位）
│
├── configs/                      # 配置文件
│   └── base.yaml                 # 全局配置（OmegaConf YAML）
│
├── scripts/                      # 脚本工具
│   ├── check_env.py              # 环境检查
│   ├── collect_video.py          # 视频采集脚本
│   ├── preprocess_videos.py      # 批量预处理
│   ├── extract_keypoints.py      # 批量关键点提取
│   ├── train.py                  # 模型训练管线
│   ├── test_pipeline.py          # 核心算法测试
│   └── update_task_checklist.py  # 任务清单进度更新
│
├── docs/                         # 文档
│   ├── guide.md                  # 本文件 — 完全使用手册
│   ├── tips.md                   # 详细使用说明
│   ├── log.md                    # 开发日志
│   ├── fall-risk-tech-tasks.html # 技术任务清单
│   └── 挑战杯大纲0.1.md          # 竞赛方案大纲
│
├── frontend/                     # React 前端
│   ├── src/
│   │   ├── App.jsx               # 主仪表盘组件
│   │   ├── index.css             # 适老化样式
│   │   └── main.jsx              # 入口文件
│   ├── package.json              # 前端依赖
│   └── vite.config.js            # Vite 构建配置
│
├── docker/                       # Docker 部署
│   ├── Dockerfile                # 镜像构建
│   ├── docker-compose.yml        # 服务编排
│   └── nginx.conf                # Nginx 反向代理配置
│
├── data/                         # 数据目录（gitignore）
│   └── baseline.db               # 个体基线 SQLite 数据库
│
├── pyproject.toml                # 项目元数据 & 依赖
├── Makefile                      # 常用命令
└── README.md                     # 项目简介
```

---

## 5. 配置文件说明

系统所有配置集中在 `configs/base.yaml`，修改后重启服务生效。

### 5.1 视频采集（`video`）

```yaml
video:
  source_fps: 15        # 摄像头原始帧率
  sample_fps: 10        # 实际采样帧率（从15fps取10fps）
  resolution: [1280, 720]  # 分辨率
  buffer_size: 30       # 帧缓冲区大小（用于预警视频回溯）
```

### 5.2 人体检测（`human_detection`）

```yaml
human_detection:
  model: "yolov8n"              # 轻量模型，首次自动下载（~6MB）
  confidence_threshold: 0.5     # 检测置信度阈值
  device: "cpu"                 # cpu 或 cuda
```

### 5.3 姿态估计（`pose_estimation`）

```yaml
pose_estimation:
  model_type: "mediapipe_pose_lite"  # MediaPipe Pose Lite
  model_complexity: 0                 # 0=Lite, 1=Full, 2=Heavy
  confidence_threshold: 0.5           # 关键点可见性阈值
  min_visible_lower_keypoints: 4      # 下肢6点至少4个可见才保留
```

### 5.4 四大特征（`features`）

```yaml
features:
  walking_rhythm:
    fft_min_freq: 0.5    # FFT 最低频率（过滤直流）
    fft_max_freq: 5.0    # FFT 最高频率（正常步频上限）
  step_amplitude:
    normalization: "torso_height"  # 用躯干高度归一化步幅
  trunk_stability:
    source_keypoints: [11, 12, 23, 24]  # 左右肩 + 左右髋
  activity_density:
    window_seconds: 60         # 统计窗口 60 秒
    motion_threshold: 0.01     # 像素位移阈值
```

### 5.5 个体基线（`baseline`）

```yaml
baseline:
  collection_days: 7    # 基线采集期：7 天
  min_samples: 100      # 最少 100 帧有效样本
  storage: "sqlite"     # 存储方式
```

### 5.6 偏离检测（`deviation`）

```yaml
deviation:
  short_term:
    window_minutes: 5      # 5 分钟滑动窗口
    stride_seconds: 30     # 每 30 秒滑动一次
    method: "mahalanobis"  # 马氏距离
    threshold: 3.0         # 阈值（卡方分布 p<0.01）
    consecutive_windows: 3 # 连续3窗口超阈才触发
  long_term:
    window_days: 14        # 14 天窗口
    method: "linear_regression"
    min_negative_days: 7   # 连续7天负向触发
    slope_threshold: -0.05 # 斜率阈值
```

### 5.7 预警规则（`alert`）

```yaml
alert:
  levels:
    low:       # 低风险 — 持续监测
    attention: # 关注级 — APP 推送
    warning:   # 预警级 — 短信通知家属
    critical:  # 高危级 — 电话通知家属
  short_term_freq_threshold: 3     # 每小时偏离≥3次触发关注
  inactivity_threshold_minutes: 240 # 4小时无活动触发高危
```

---

## 6. 启动服务

### 6.1 启动 API 服务

```powershell
# 确保虚拟环境已激活
.\venv\Scripts\Activate.ps1

# 启动服务
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- **Swagger 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc
- **系统信息**: http://localhost:8000/

`--reload` 参数表示开发模式，代码修改后自动重启。生产环境去掉 `--reload`。

### 6.2 停止服务

按 `Ctrl+C` 终止进程。

如果端口被占用：

```powershell
# 查看 8000 端口占用
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess

# 终止进程（替换 <PID>）
Stop-Process -Id <PID> -Force
```

### 6.3 Windows 编码问题

Windows 控制台默认 GBK 编码，运行含中文输出的脚本前需设置：

```powershell
$env:PYTHONIOENCODING="utf-8"
chcp 65001 > $null
```

---

## 7. API 接口文档

### 7.1 基础端点

| 方法 | 路径 | 说明 | 请求参数 | 返回示例 |
|------|------|------|---------|---------|
| GET | `/` | 系统信息 | 无 | `{"system":"fall-risk-prediction","version":"0.1.0","status":"running","docs":"/docs"}` |
| GET | `/health` | 健康检查 | 无 | `{"status":"healthy"}` |
| GET | `/config` | 获取完整配置 | 无 | OmegaConf 完整配置对象 |
| GET | `/features/info` | 四大特征说明 | 无 | 各特征的名称、描述、单位 |
| GET | `/alerts/levels` | 预警分级规则 | 无 | 四级预警判定条件与动作 |

### 7.2 监控管理

#### 启动视频流分析

```http
POST /api/v1/stream/start
Content-Type: application/json

{
  "source": "0",              // "0"=摄像头, 或RTSP地址, 或视频文件路径
  "person_id": "grandpa"     // 被监测老人标识
}
```

**返回**:
```json
{
  "code": 200,
  "message": "监控已启动",
  "source": "0",
  "person_id": "grandpa"
}
```

**注意**: 如果监控已在运行中，返回 409 冲突。

#### 停止视频流分析

```http
POST /api/v1/stream/stop
```

**返回**:
```json
{ "code": 200, "message": "监控已停止" }
```

### 7.3 风险数据

#### 获取当前风险状态

```http
GET /api/v1/risk/current
```

**返回**:
```json
{
  "is_running": true,
  "current_risk_level": "low",
  "current_risk_label": "低风险",
  "baseline_ready": true,
  "baseline_samples": 150,
  "frames_processed": 5000,
  "frames_valid": 4200,
  "last_feature": {
    "walking_rhythm": 1.2,
    "step_amplitude": 0.35,
    "trunk_stability": 3.8,
    "activity_density": 0.72
  },
  "last_alert": null,
  "deviation_result": {
    "level": "NONE",
    "mahalanobis_distance": 1.5,
    "short_term_triggered": false,
    "long_term_triggered": false
  }
}
```

#### 获取历史风险记录

```http
GET /api/v1/risk/history?person_id=grandpa&hours=24&limit=100&offset=0
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `person_id` | string | 可选 | 人员标识，不传则全部 |
| `hours` | int | 24 | 查询时间范围（小时） |
| `limit` | int | 100 | 每页数量 |
| `offset` | int | 0 | 分页偏移 |

#### 重置个体化基线

```http
POST /api/v1/baseline/reset
```

**返回**:
```json
{ "code": 200, "message": "基线已重置" }
```

重置后系统重新进入 7 天基线采集期。

### 7.4 告警管理

#### 查询告警历史

```http
GET /api/v1/alerts?level=warning&person_id=grandpa&hours=24&acknowledged=0&limit=100&offset=0
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | 可选 | 过滤等级: low/attention/warning/critical |
| `person_id` | string | 可选 | 人员标识 |
| `hours` | int | 24 | 时间范围 |
| `acknowledged` | int | 可选 | 0=未确认, 1=已确认 |
| `limit` | int | 100 | 每页数量 |
| `offset` | int | 0 | 分页偏移 |

**返回**:
```json
{
  "total": 5,
  "alerts": [
    {
      "id": 1,
      "timestamp": 1712345678.0,
      "alert_level": "warning",
      "risk_score": 72.5,
      "message": "长期趋势连续7天负向变化",
      "acknowledged": 0,
      "person_id": "grandpa"
    }
  ]
}
```

#### 确认告警

```http
POST /api/v1/alerts/{alert_id}/acknowledge
```

**返回**:
```json
{ "code": 200, "message": "告警已确认" }
```

### 7.5 统计数据

#### 获取统计面板数据

```http
GET /api/v1/stats?hours=24
```

**返回**:
```json
{
  "hours": 24,
  "total_risk_records": 2880,
  "total_alerts": 3,
  "alerts_by_level": {
    "attention": 2,
    "warning": 1
  },
  "avg_risk_score": 35.2
}
```

### 7.6 WebSocket 实时推送

**端点**: `ws://localhost:8000/ws/alerts`

连接后，后端会在产生新告警时主动推送：

```json
{
  "type": "alert",
  "data": {
    "id": 1,
    "alert_level": "warning",
    "message": "长期趋势连续7天负向变化",
    "timestamp": 1712345678.0
  }
}
```

**心跳机制**: 客户端应定时发送 `{"type": "ping"}`，服务端回复 `{"type": "pong"}`。

### 7.7 使用示例 (PowerShell)

```powershell
$base = "http://localhost:8000"

# 健康检查
Invoke-RestMethod "$base/health"

# 查看特征说明
Invoke-RestMethod "$base/features/info" | ConvertTo-Json

# 启动监控（本地摄像头）
Invoke-RestMethod "$base/api/v1/stream/start" -Method Post `
    -ContentType "application/json" `
    -Body '{"source":"0","person_id":"grandpa"}'

# 查看当前风险
Invoke-RestMethod "$base/api/v1/risk/current" | ConvertTo-Json

# 查看历史告警
Invoke-RestMethod "$base/api/v1/alerts?level=warning&limit=10" | ConvertTo-Json

# 确认告警
Invoke-RestMethod "$base/api/v1/alerts/1/acknowledge" -Method Post

# 停止监控
Invoke-RestMethod "$base/api/v1/stream/stop" -Method Post
```

---

## 8. 前端看板

### 8.1 启动前端（开发模式）

```powershell
cd frontend
npm install
npm run dev
```

前端默认运行在 http://localhost:3000，Vite 已配置代理转发 API 请求到 http://localhost:8000。

### 8.2 前端功能

前端是一个**家属端实时风险看板**，包含：

| 区域 | 功能 |
|------|------|
| 风险等级卡片 | 大号彩色卡片显示当前风险等级（绿/黄/橙/红） |
| 控制按钮 | 启动监控、停止监控、重置基线 |
| 风险评分仪表盘 | ECharts 仪表盘，0-100 分 |
| 风险趋势折线图 | 近 24 小时风险评分变化曲线 |
| 告警列表 | 最新告警记录，按时间倒序 |
| 实时连接状态 | WebSocket 连接指示灯 |

### 8.3 构建生产版本

```powershell
cd frontend
npm run build
```

构建产物在 `frontend/dist/`，API 服务启动后会自动挂载为静态文件服务。

---

## 9. Docker 部署

### 9.1 构建与启动

```powershell
# 构建并启动全部服务（API + Redis + Nginx）
docker-compose -f docker/docker-compose.yml up -d

# 查看运行状态
docker-compose -f docker/docker-compose.yml ps

# 查看日志
docker-compose -f docker/docker-compose.yml logs -f api

# 停止服务
docker-compose -f docker/docker-compose.yml down
```

### 9.2 服务架构

| 服务 | 镜像 | 端口 | 说明 |
|------|------|------|------|
| `api` | 本地构建 | 8000 | FastAPI 应用 |
| `redis` | redis:7-alpine | 6379 | 缓存（预留，当前代码未使用） |
| `nginx` | nginx:alpine | 80 | 反向代理（前端静态文件 + API + WebSocket） |

### 9.3 数据持久化

docker-compose.yml 配置了以下卷挂载：

- `../data` → `/app/data` — 数据库文件
- `../logs` → `/app/logs` — 日志文件
- `../configs` → `/app/configs` — 配置文件（可热更新）

---

## 10. 开发指南

### 10.1 代码风格

| 规则 | 标准 |
|------|------|
| 语言 | Python 3.10+，强制类型注解 |
| 行宽 | 100 字符 |
| 引号 | 双引号 |
| Linter | ruff (规则: E/F/W/I/N/UP/B) |
| Formatter | ruff format |
| 导入排序 | isort 风格（ruff I 规则） |

```bash
# 检查代码
ruff check src/

# 自动格式化
ruff format src/
```

### 10.2 核心算法测试

系统提供了一个模拟测试脚本，**无需摄像头和视频文件**即可验证全链路算法：

```powershell
$env:PYTHONIOENCODING="utf-8"; chcp 65001 > $null
python scripts/test_pipeline.py
```

该脚本会：
1. 生成模拟的人体关键点数据（正常行走→异常变化→恢复）
2. 计算四大相对特征
3. 建立模拟基线
4. 测试马氏距离偏离检测
5. 测试四级预警引擎
6. 输出各步骤的验证结果

### 10.3 模块导入规范

项目采用 src-layout，所有导入以 `src.` 开头：

```python
from src.utils.config import get_config
from src.inference.monitor import FallRiskMonitor
from src.alerts.engine import AlertEngine, RiskLevel
```

不要在模块内使用相对导入（`from ..utils`）。

### 10.4 日志规范

使用项目统一的日志系统：

```python
from src.utils.logger import get_logger

log = get_logger(__name__)
log.info("信息日志")
log.warning("警告")
log.error("错误")
```

日志输出：
- 控制台：彩色，仅 INFO 及以上
- 文件：`logs/app_YYYY-MM-DD.log`，包含 DEBUG 级别，每天轮转，保留 30 天
- 错误日志单独文件：`logs/error_YYYY-MM-DD.log`

### 10.5 配置加载

```python
from src.utils.config import get_config, reload_config

config = get_config()           # 获取配置（单例）
device = config.inference.device  # 读取配置值
reload_config()                 # 修改 YAML 后热重载
```

### 10.6 测试数据采集

使用采集脚本收集训练数据：

```powershell
# 单段采集（RTSP 流）
python scripts/collect_video.py --source "rtsp://admin:pass@192.168.1.1/stream" --scene "normal_walk" --duration 120

# 批量采集（读取 CSV 采集计划）
python scripts/collect_video.py --batch采集计划.csv

# 预处理（人体检测 + ROI 裁剪）
python scripts/preprocess_videos.py

# 批量关键点提取
python scripts/extract_keypoints.py
```

---

## 11. 常见问题

### 11.1 启动服务报 "WinError 10013"

端口被占用。释放端口：

```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess
Stop-Process -Id <PID> -Force
```

或改用其他端口：

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### 11.2 CUDA 不可用，推理很慢

当前安装的是 CPU 版 PyTorch。如需 GPU 加速：

```powershell
# 卸载 CPU 版
pip uninstall torch torchvision

# 安装 CUDA 版（以 CUDA 12.1 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

然后在 `configs/base.yaml` 中设置：

```yaml
inference:
  device: "cuda"
```

### 11.3 首次运行自动下载模型

- YOLOv8n 权重（~6MB）：首次调用 `HumanDetector` 时自动下载
- MediaPipe Pose 模型：随 pip 包安装，无需额外下载

### 11.4 基线采集期不产生预警

系统部署后前 7 天为基线采集期。在此期间只采集正常活动数据，**不产生任何预警**。满足以下条件后基线就绪：

- 采集天数 ≥ 7 天
- 有效样本数 ≥ 100 帧
- 四个特征数据均收集完成

基线就绪后，状态接口返回 `"baseline_ready": true`。

### 11.5 如何重置基线？

基线建立后如需重新采集（如老人身体状况发生重大变化）：

```powershell
Invoke-RestMethod "http://localhost:8000/api/v1/baseline/reset" -Method Post
```

重置后重新进入 7 天采集期。

### 11.6 如何接入萤石摄像头？

1. 在 `configs/` 目录下创建 `ezviz.yaml`（已 gitignore）：

```yaml
ezviz:
  app_key: "你的AppKey"
  app_secret: "你的AppSecret"
  device_serial: "设备序列号"
```

2. 使用 RTSP 地址作为 `source` 参数启动监控：

```powershell
Invoke-RestMethod "http://localhost:8000/api/v1/stream/start" -Method Post `
    -ContentType "application/json" `
    -Body '{"source":"rtsp://username:password@192.168.1.100:554/stream","person_id":"elder"}'
```

### 11.7 前端页面无法连接后端

默认开发配置：
- 前端（Vite）运行在 **http://localhost:3000**
- 后端（FastAPI）运行在 **http://localhost:8000**
- Vite 配置了 proxy，`/api/*` 和 `/ws/*` 请求会自动转发到 8000 端口

确保先启动后端服务，再启动前端。

### 11.8 如何贡献代码？

1. 遵循 Python 3.10+ 类型注解
2. 运行 `ruff check src/` 确保无 lint 错误
3. 运行 `ruff format src/` 格式化代码
4. 运行 `python scripts/test_pipeline.py` 确保核心算法正常
5. 提交前阅读 `docs/log.md` 了解项目历史

---

> **更多资料**：
> - `docs/tips.md` — 详细使用步骤与命令对照
> - `docs/log.md` — 开发日志与版本变更记录
> - `docs/fall-risk-tech-tasks.html` — 完整技术任务清单
