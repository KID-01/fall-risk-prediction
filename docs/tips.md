# 跌倒风险预测系统 — 使用方法

---

## 一、环境准备（首次使用）

```powershell
# 1. 进入项目目录
cd d:\tiaozhanbei\fall-risk-prediction【改成你自己的路径】

# 2. 创建虚拟环境
py -3.14 -m venv venv

# 3. 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 4. 安装依赖
pip install -e ".[dev]"

# 5. 环境检查（需设置 UTF-8 编码以支持中文和特殊符号输出）
$env:PYTHONIOENCODING="utf-8"; chcp 65001 > $null
python scripts/check_env.py
```

> **说明**：当前系统未安装 `make`，所有 `make xxx` 命令需用 venv 中的 python 直接执行等效命令。

---

## 二、启动 / 停止 API 服务

### 启动服务

```powershell
cd d:\tiaozhanbei\fall-risk-prediction
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- **Swagger 文档**：http://localhost:8000/docs
- **ReDoc 文档**：http://localhost:8000/redoc

### 停止服务 & 端口占用排查

```powershell
# 查看占用 8000 端口的进程
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess

# 终止占用进程（替换 <PID>）
Stop-Process -Id <PID> -Force
```

> 若报 `WinError 10013`，说明端口被占用，按上述步骤释放后重启。或改用 8080 端口。

---

## 三、系统架构总览

系统采用 **"视频帧输入 → 人体检测 → 关键点提取 → 帧过滤 → 特征计算 → 基线对比 → 偏离检测 → 分级预警"** 的完整闭环处理流程：

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│ 视频拉流  │───▶│ YOLOv8n  │───▶│ MediaPipe    │───▶│ 帧质量   │
│ RTSP/文件 │    │ 人体检测  │    │ Pose关键点   │    │ 过滤     │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘
                                                          │
                                                          ▼
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│ 四级预警  │◀───│ 双层偏离  │◀───│ 个体化基线   │◀───│ 四大相对 │
│ 引擎     │    │ 检测     │    │ 对比         │    │ 特征计算 │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘
```

### 项目目录结构

```
src/
├── utils/                  # 工具层
│   ├── config.py           # 配置加载
│   └── keypoints.py        # 关键点定义
├── data/                   # 数据层
│   ├── video_capture.py    # 视频拉流
│   ├── human_detector.py   # 人体检测
│   ├── keypoint_extractor.py  # 关键点提取
│   └── frame_filter.py     # 帧质量过滤
├── inference/              # 推理层
│   ├── features.py         # 四大特征计算
│   ├── baseline.py         # 个体化基线
│   ├── deviation.py        # 双层偏离检测
│   └── monitor.py          # 监控服务（整合全链路）
├── alerts/                 # 预警层
│   └── engine.py           # 四级预警引擎
├── api/                    # 接口层
│   └── main.py             # FastAPI 服务
└── edge/                   # 边缘计算（待开发）

configs/base.yaml           # 全局配置
scripts/test_pipeline.py    # 核心算法测试脚本
```

---

## 四、各模块详解

### 4.1 配置加载 — `src/utils/config.py`

**功能**：全局配置加载单例，读取 `configs/base.yaml`。

**原理**：使用 OmegaConf 库加载 YAML 配置文件，采用单例模式确保全局只加载一次，支持热重载。

**使用方法**：
```python
from src.utils.config import get_config, reload_config

config = get_config()           # 获取配置
device = config.inference.device  # 读取推理设备
reload_config()                 # 修改yaml后重新加载
```

---

### 4.2 关键点定义 — `src/utils/keypoints.py`

**功能**：定义 MediaPipe Pose 的 33 个人体关键点索引，提供关键点数据结构和帧质量检查。

**原理**：MediaPipe Pose 输出 33 个关键点（鼻、眼、耳、肩、肘、腕、髋、膝、踝等），每个点包含 `[x, y, z, visibility]` 四个值。本项目重点关注下肢（髋/膝/踝）和躯干（肩/髋）关键点。

**核心内容**：
- `PoseKeypoint` 枚举：33 个关键点的索引常量（如 `LEFT_HIP = 23`）
- `KeypointFrame` 数据类：单帧关键点数据，包含时间戳、坐标数组、有效性标记
- `check_frame_quality()` 函数：检查下肢 6 个关键点至少 4 个可见 + 躯干 4 个关键点至少 3 个可见

**使用方法**：
```python
from src.utils.keypoints import KeypointFrame, PoseKeypoint, check_frame_quality

# 获取某关键点的 2D 坐标
hip_xy = frame.get_xy(PoseKeypoint.LEFT_HIP)  # [x, y]

# 检查关键点是否可见
if frame.is_visible(PoseKeypoint.LEFT_ANKLE, threshold=0.5):
    ...

# 检查帧质量
is_valid, reason = check_frame_quality(frame, confidence_threshold=0.5, min_visible_lower=4)
```

---

### 4.3 视频拉流 — `src/data/video_capture.py`

**功能**：从 RTSP 视频流、本地视频文件或摄像头采集视频帧。

**原理**：封装 OpenCV 的 `VideoCapture`，支持帧采样降频（如从 15fps 采样到 10fps 以降低计算负载），维护一个帧缓冲区用于视频回溯（预警时截取前后帧）。

**使用方法**：
```python
from src.data.video_capture import VideoCapture

# 方式1: 上下文管理器（自动释放）
with VideoCapture(source="rtsp://admin:pass@192.168.1.1/stream") as cap:
    for video_frame in cap.frames():
        # video_frame.frame 是 BGR 图像 (numpy数组)
        # video_frame.timestamp 是时间戳(秒)
        process(video_frame)

# 方式2: 手动控制
cap = VideoCapture(source="0")  # "0" = 默认摄像头
cap.open()
frame = cap.read_frame()
cap.close()

# 获取缓冲区帧（用于视频回溯）
recent_frames = cap.get_buffer_frames(before_seconds=15)
```

**配置项**（`configs/base.yaml` → `video`）：
- `source_fps: 15` — 摄像头原始帧率
- `sample_fps: 10` — 实际采样帧率
- `buffer_size: 30` — 帧缓冲区大小

---

### 4.4 人体检测 — `src/data/human_detector.py`

**功能**：使用 YOLOv8n 轻量模型检测画面中的人体，作为关键点提取的前置触发器。

**原理**：YOLOv8n 是 YOLOv8 系列最轻量的模型，仅当检测到完整人体（边界框高宽比 > 1.2）时才触发后续关键点提取，避免无人画面浪费计算资源。模型延迟加载（首次调用时才下载）。

**使用方法**：
```python
from src.data.human_detector import HumanDetector

detector = HumanDetector()
boxes = detector.detect(frame)           # 检测所有人体
best = detector.detect_best(frame)       # 返回最佳（置信度最高的完整人体）
if best:
    print(f"人体位置: ({best.x1}, {best.y1}) - ({best.x2}, {best.y2}), 置信度: {best.confidence}")
```

**配置项**（`configs/base.yaml` → `human_detection`）：
- `model: "yolov8n"` — 模型名称
- `confidence_threshold: 0.5` — 置信度阈值
- `device: "cpu"` — 推理设备

> **注意**：首次使用时会自动下载 YOLOv8n 权重文件（约 6MB）。

---

### 4.5 关键点提取 — `src/data/keypoint_extractor.py`

**功能**：使用 MediaPipe Pose Lite 从视频帧中提取人体 33 个 2D 关键点坐标。

**原理**：MediaPipe Pose 是 Google 开源的实时人体姿态估计方案，Lite 版本在普通 CPU 上即可达到 30fps+。输出每个关键点的归一化坐标 `[x, y, z, visibility]`（x/y 归一化到 0~1，z 为相对深度）。提取后自动执行帧质量检查。

**使用方法**：
```python
from src.data.keypoint_extractor import KeypointExtractor
from src.data.video_capture import VideoFrame

extractor = KeypointExtractor()
kp_frame = extractor.extract(video_frame)  # 返回 KeypointFrame 或 None

if kp_frame and kp_frame.is_valid:
    # 关键点有效，可用于特征计算
    hip = kp_frame.get_xy(PoseKeypoint.LEFT_HIP)
else:
    # 关键点无效或未检测到人体
    print(kp_frame.invalid_reason if kp_frame else "未检测到人体")
```

**配置项**（`configs/base.yaml` → `pose_estimation`）：
- `model_complexity: 0` — 0=Lite（最快）, 1=Full, 2=Heavy
- `confidence_threshold: 0.5` — visibility 低于此值的关键点被丢弃
- `min_visible_lower_keypoints: 4` — 下肢 6 个关键点至少 4 个可见

---

### 4.6 帧质量过滤 — `src/data/frame_filter.py`

**功能**：过滤质量不足的关键点帧，仅保留可靠数据用于后续分析。

**原理**：两层过滤——①置信度过滤：visibility < 0.5 的关键点不可信；②下肢可见性检查：行走分析依赖下肢关键点，6 个下肢点（左右髋/膝/踝）至少 4 个可见才保留该帧。不可用的帧被标记为 `is_valid=False` 并跳过。

**使用方法**：
```python
from src.data.frame_filter import FrameFilter

filter = FrameFilter(confidence_threshold=0.5, min_visible_lower=4)
filter.filter(kp_frame)                    # 过滤单帧（更新 is_valid 标记）
valid_frames = filter.filter_batch(frames)  # 批量过滤，返回有效帧列表
```

---

### 4.7 四大相对特征计算 — `src/inference/features.py`

**功能**：从关键点序列中计算四个不依赖物理尺度的相对特征，反映老人的活动模式。

**原理**：单目摄像头无法精确测量物理尺度（如步长多少米），因此设计四个相对特征，只关注变化趋势而非绝对数值：

| 特征 | 计算方法 | 原理 |
|------|---------|------|
| **行走节拍频率** | 对髋关节 y 坐标时序做 FFT，提取主频 | 行走时身体上下律动，髋关节 y 坐标呈周期性波动，主频即为步频 |
| **步幅相对幅度** | 踝关节 x 坐标摆动范围 / 躯干高度 | 踝关节前后摆动反映步幅，用躯干高度归一化消除距离/视角影响 |
| **躯干稳定指数** | 肩髋连线与垂直方向夹角的变化范围 | 正常行走躯干基本垂直，不稳时摇摆加剧，角度变化范围增大 |
| **活动密度** | 时间窗口内髋关节有位移的帧占比 | 髋关节位移 > 阈值判定为运动，统计运动帧占比反映活动水平 |

**使用方法**：
```python
from src.inference.features import FeatureCalculator, FeatureVector

calc = FeatureCalculator()
feature = calc.calculate(keypoint_frames)  # 传入一帧关键点列表

print(feature.walking_rhythm)      # 行走节拍频率 (Hz)
print(feature.step_amplitude)      # 步幅相对幅度
print(feature.trunk_stability)     # 躯干稳定指数 (度)
print(feature.activity_density)    # 活动密度 (0~1)

# 转为 numpy 数组（用于马氏距离计算）
arr = feature.to_array()  # [节拍, 步幅, 稳定性, 密度]
```

---

### 4.8 个体化基线 — `src/inference/baseline.py`

**功能**：为每位老人建立专属的活动模式基线，以"相对于自身的变化"而非"是否达到绝对标准"作为判断依据。

**原理**：系统部署后前 7 天为基线采集期，持续采集正常状态下的特征样本，计算各特征的均值、标准差和协方差矩阵。后续将实时特征与基线对比，计算马氏距离（考虑特征间相关性的多维偏离度量）判断是否异常。数据存储在 SQLite 数据库中。

**核心概念**：
- **基线采集期**：前 7 天正常活动数据形成个人画像
- **马氏距离**：多维空间中点到分布中心的距离，考虑特征间相关性，比简单 Z-Score 更准确
- **个性化**：80 岁老人步速慢是正常状态，不与年轻人统一标准对比

**使用方法**：
```python
from src.inference.baseline import BaselineManager
from src.inference.features import FeatureVector

manager = BaselineManager()

# 采集期：持续添加样本
manager.add_sample("老人A", feature)

# 计算基线（样本数 >= 100 后就绪）
baseline = manager.compute_baseline("老人A")
print(f"基线就绪: {baseline.is_ready}, 样本数: {baseline.sample_count}")

# 计算马氏距离（基线就绪后）
distance = baseline.mahalanobis_distance(new_feature)
if distance > 3.0:
    print("偏离基线！")

# 重置基线（重新采集）
manager.reset_baseline("老人A")
```

**配置项**（`configs/base.yaml` → `baseline`）：
- `collection_days: 7` — 基线采集天数
- `min_samples: 100` — 最少样本数
- `storage: "sqlite"` — 存储方式

---

### 4.9 双层偏离检测 — `src/inference/deviation.py`

**功能**：同时捕捉短期急性异常和长期渐进趋势，综合判断风险。

**原理**：跌倒风险有两种模式——①急性变化（突然行走不稳）；②渐进衰退（体力逐步下降）。分别用不同时间尺度检测：

| 层级 | 时间尺度 | 方法 | 触发条件 |
|------|---------|------|---------|
| **短期异常检测** | 5 分钟窗口 / 30 秒步长 | 马氏距离 | 连续 3 个窗口距离 > 3.0 |
| **长期趋势分析** | 14 天窗口 | 线性回归斜率 | 斜率 < -0.05 持续 7 天 |

**使用方法**：
```python
from src.inference.deviation import DeviationDetector

detector = DeviationDetector()

# 每次计算完特征后检测
result = detector.check(feature, baseline)

print(result.level)                  # NONE / SHORT_TERM / LONG_TERM / BOTH
print(result.mahalanobis_distance)   # 马氏距离
print(result.short_term_triggered)   # 短期是否触发
print(result.long_term_triggered)    # 长期是否触发
print(result.z_scores)               # 各特征 Z-Score
print(result.trend_slopes)           # 各特征趋势斜率
print(result.detail)                 # 详细描述
```

**配置项**（`configs/base.yaml` → `deviation`）：
- 短期：`window_minutes: 5`, `stride_seconds: 30`, `threshold: 3.0`, `consecutive_windows: 3`
- 长期：`window_days: 14`, `min_negative_days: 7`, `slope_threshold: -0.05`

---

### 4.10 四级预警引擎 — `src/alerts/engine.py`

**功能**：根据偏离检测结果划分风险等级，触发对应响应措施。

**原理**：四级风险从低到高，避免一刀切式频繁告警：

| 风险等级 | 判定条件 | 响应措施 |
|---------|---------|---------|
| **低风险** | 所有特征在基线 ±1 标准差内 | 持续监测，不推送 |
| **关注级** | 短期偏离频繁出现（≥3 次/小时） | APP 推送提醒 |
| **预警级** | 长期趋势连续 7 天负向变化 | 短信通知家属 |
| **高危级** | 近似跌倒动作 / 超 4 小时无活动 | 电话通知家属 |

**使用方法**：
```python
from src.alerts.engine import AlertEngine, RiskLevel

engine = AlertEngine()

# 注册响应动作（如推送通知）
def send_sms(event):
    print(f"发送短信: {event.message}")
engine.register_action(RiskLevel.WARNING, send_sms)

# 评估风险
alert = engine.evaluate(deviation_result, timestamp=123.4, has_activity=True)
print(alert.level)        # RiskLevel.ATTENTION
print(alert.level.label)  # "关注级"
print(alert.message)      # "短期偏离频繁(3次/小时)"

# 查询历史
events = engine.get_events(level=RiskLevel.WARNING, limit=50)
current = engine.get_current_level()
```

---

### 4.11 监控服务 — `src/inference/monitor.py`

**功能**：整合全链路的监控服务单例，在后台线程中持续运行完整的跌倒风险预测流程。

**原理**：`FallRiskMonitor` 是单例模式，启动后在独立线程中循环执行：读取视频帧 → 人体检测 → 关键点提取 → 帧过滤 → 特征计算 → 基线采集/对比 → 偏离检测 → 预警评估。全程通过 `MonitorStatus` 暴露状态。

**使用方法**（通常通过 API 调用，也可直接代码调用）：
```python
from src.inference.monitor import FallRiskMonitor

monitor = FallRiskMonitor()

# 启动监控
monitor.start(source="0", person_id="grandpa")  # "0"=摄像头, 也可传RTSP地址

# 查看状态
status = monitor.get_status()
print(status["current_risk_level"])   # "low" / "attention" / "warning" / "critical"
print(status["baseline_ready"])       # 基线是否就绪
print(status["baseline_samples"])     # 已采集样本数

# 查看预警历史
alerts = monitor.get_alert_history(level="warning", limit=50)

# 停止监控
monitor.stop()
```

---

## 五、API 接口说明

服务启动后，浏览器打开 **http://localhost:8000/docs** 可使用 Swagger UI 交互测试。

### 全部端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 系统信息（名称、版本、状态） |
| `/health` | GET | 健康检查 |
| `/config` | GET | 获取完整系统配置 |
| `/alerts/levels` | GET | 预警分级规则 |
| `/features/info` | GET | 四大相对特征说明 |
| `/predict` | POST | 单次预测（占位接口） |
| `/monitor/start` | POST | 启动实时监控 |
| `/monitor/stop` | POST | 停止监控 |
| `/monitor/status` | GET | 监控状态（风险等级/基线/特征/偏离） |
| `/monitor/alerts` | GET | 预警历史（可按等级筛选） |
| `/monitor/baseline/reset` | POST | 重置个体化基线 |

### 接口调用示例

```powershell
$base = "http://localhost:8000"

# 启动监控（用本地摄像头）
Invoke-RestMethod "$base/monitor/start" -Method Post `
    -ContentType "application/json" `
    -Body '{"source":"0","person_id":"grandpa"}'

# 查看监控状态
Invoke-RestMethod "$base/monitor/status"

# 查看预警历史（只看预警级）
Invoke-RestMethod "$base/monitor/alerts?level=warning&limit=50"

# 重置基线
Invoke-RestMethod "$base/monitor/baseline/reset" -Method Post

# 停止监控
Invoke-RestMethod "$base/monitor/stop" -Method Post
```

---

## 六、测试方法

### 6.1 核心算法测试（无需摄像头）

使用模拟关键点数据测试特征计算 → 基线建立 → 偏离检测 → 预警引擎全链路：

```powershell
cd d:\tiaozhanbei\fall-risk-prediction
.\venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"; chcp 65001 > $null
python scripts/test_pipeline.py
```

预期输出：四大特征计算通过、基线建立通过、马氏距离偏离检测通过、四级预警引擎通过。

### 6.2 API 端点测试

```powershell
# 启动服务后，浏览器打开 Swagger UI
start http://localhost:8000/docs
```

### 6.3 实时监控测试（需摄像头或视频文件）

```powershell
# 用本地摄像头
Invoke-RestMethod "http://localhost:8000/monitor/start" -Method Post `
    -ContentType "application/json" -Body '{"source":"0","person_id":"test"}'

# 用视频文件（替换为实际路径）
Invoke-RestMethod "http://localhost:8000/monitor/start" -Method Post `
    -ContentType "application/json" -Body '{"source":"C:/path/to/video.mp4","person_id":"test"}'

# 查看状态
Invoke-RestMethod "http://localhost:8000/monitor/status"
```

---

## 七、常用命令对照（Makefile → 等效命令）

系统中未安装 `make`，以下是等效命令（需先激活虚拟环境）：

| Make 命令 | 等效命令 | 说明 |
|-----------|---------|------|
| `make install` | `pip install -e .` | 安装运行依赖 |
| `make install-dev` | `pip install -e ".[dev]"` | 安装开发依赖 |
| `make check` | `python scripts/check_env.py` | 环境检查 |
| `make serve` | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload` | 启动 API 服务 |
| `make lint` | `ruff check src/ tests/` | 代码检查 |
| `make format` | `ruff format src/ tests/` | 自动格式化 |
| `make test` | `python scripts/test_pipeline.py` | 核心算法测试 |

### 清理临时文件

```powershell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

---

## 八、配置文件说明 — `configs/base.yaml`

所有参数集中配置，修改后重启服务生效。主要配置区块：

| 区块 | 说明 | 关键参数 |
|------|------|---------|
| `video` | 视频采集 | `sample_fps: 10`（采样帧率） |
| `human_detection` | 人体检测 | `model: yolov8n`, `confidence_threshold: 0.5` |
| `pose_estimation` | 姿态估计 | `model_complexity: 0`（Lite）, `confidence_threshold: 0.5` |
| `features` | 四大特征 | 各特征的计算参数（FFT频率范围、运动阈值等） |
| `baseline` | 个体化基线 | `collection_days: 7`, `min_samples: 100` |
| `deviation` | 偏离检测 | 短期`threshold: 3.0`，长期`slope_threshold: -0.05` |
| `alert` | 四级预警 | 各等级判定条件和响应措施 |
| `inference` | 推理服务 | `device: cpu`, `inference_interval_ms: 200` |
| `ezviz` | 萤石平台 | RTSP地址模板（密钥在 `configs/ezviz.yaml`） |

---

## 九、注意事项

1. **编码问题**：Windows 控制台默认 GBK 编码，运行含中文/特殊符号的脚本前需设置：
   ```powershell
   $env:PYTHONIOENCODING="utf-8"; chcp 65001 > $null
   ```

2. **CUDA 不可用**：当前安装的是 CPU 版 PyTorch（`torch 2.13.0+cpu`），推理速度较慢。如需 GPU 加速，需安装对应 CUDA 版本的 PyTorch。

3. **首次运行模型下载**：YOLOv8n 首次使用时自动下载权重（约 6MB）；MediaPipe 模型随包安装。

4. **基线采集期**：系统部署后前 7 天为基线采集期，期间不产生预警（`baseline.is_ready = False`）。样本数达到 100 后基线就绪。

5. **端口占用**：启动服务前确保 8000 端口未被占用，排查方法见「二、启动/停止服务」。

6. **待开发模块**：萤石 API 接入、视频回溯、前端界面、推送通道尚未实现，详见 `log.md`。
