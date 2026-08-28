# wwh 模型结果与可实现功能说明

## 1. 文档定位

本文档说明 `wwh` 交付目录提供的视觉模型、工程扩展和实时输出数据，以及它们在当前项目中的使用方式。

需要区分两类结果：

- **识别结果**：人体、姿态、环境目标和光照等直接从视频帧得到的数据。
- **工程风险结果**：motion、environment、fusion 及 v0.3 扩展分数。这些分数用于工程规则和页面展示，不等同于校准后的跌倒概率。

## 2. 模型与文件

| 模型/模块 | 文件或目录 | 主要用途 | 输出 |
|---|---|---|---|
| YOLO Pose | `checkpoints/yolo26n-pose.pt` | 人体姿态识别 | 人体框、关键点、姿态质量 |
| YOLO Detect | `checkpoints/yolo26n.pt` | 环境目标识别 | 家具、包、球、桌椅等目标框和置信度 |
| v0.2 MVP | `wwh/competition_handoff_v0.2/mvp_source` | person/motion/environment/fusion 基础工程管线 | 逐帧 JSON、摘要和演示视频 |
| v0.3 扩展 | `wwh/competition_handoff_v0.3/mvp_source/risk_extensions` | 低光、障碍物、轨迹、交互扩展 | 扩展状态和工程风险指标 |

当前两份 YOLO 权重已经与项目 `checkpoints` 中的文件核对一致：pose 模型任务为 `pose`，环境模型任务为 `detect`。

## 3. 可实现功能

实时视频逐帧处理可以完成以下工作：

1. 检测画面中的人体并定位人体框。
2. 提取人体姿态关键点，支持运动特征计算。
3. 根据人体时序变化计算 motion 工程分数。
4. 识别环境中的家具和常见 COCO 目标。
5. 统计目标数量、类别和置信度。
6. 根据目标类别权重和与人体的图像邻近关系计算 environment 工程分数。
7. 输出危险目标排序 `top_hazards`。
8. 根据平均亮度判断低光状态。
9. 在数据足够时提供轨迹和交互状态。
10. 通过 fusion 将人体运动和环境信息合并为综合风险状态。
11. 在前端显示 AI 视频、姿态骨架、风险卡片和环境检测侧栏。

## 4. 常见输出字段

### 人体和姿态

| 字段 | 含义 |
|---|---|
| `person.present` | 当前帧是否检测到人体 |
| `person.bbox` | 人体框 `[x1, y1, x2, y2]`，单位为像素 |
| `person.keypoints_present` | 是否获得有效姿态关键点 |
| `motion.score` | 人体运动工程分数 |
| `motion.state` | `LOW/MEDIUM/HIGH/UNKNOWN` 等状态 |

### 环境目标

| 字段 | 含义 |
|---|---|
| `environment.objects` | 当前帧识别到的环境目标列表 |
| `objects[].label` | COCO 类别名，例如 `chair`、`couch`、`backpack` |
| `objects[].confidence` | 目标识别置信度 |
| `environment.score` | 环境工程风险分数，范围通常为 `0-1` |
| `environment.state` | 环境状态 `LOW/MEDIUM/HIGH` |
| `environment.top_hazards` | 按风险贡献排序的目标 |
| `top_hazards[].risk_contribution` | 类别权重、置信度和邻近因素计算出的贡献值 |

### v0.3 扩展

| 字段 | 含义 |
|---|---|
| `low_light` | 平均亮度和低光状态：`LOW/NORMAL/HIGH` |
| `obstacle` | 障碍物通道当前状态 |
| `trajectory` | 历史轨迹数据是否足够、是否可以外推 |
| `interaction` | 人体与环境目标是否建立空间交互关系 |
| `fusion` | 人体和环境结果的综合状态及来源 |
| `reason_codes` | 触发综合状态的原因列表 |

## 5. 实时帧 JSON 示例

以下数据为脱敏示例，不代表准确率、概率或真实实验结果：

```json
{
  "person": {
    "present": true,
    "bbox": [120, 80, 410, 620],
    "keypoints_present": true
  },
  "motion": {
    "source": "motion_heuristic_v0",
    "score": 0.42,
    "state": "MEDIUM"
  },
  "environment": {
    "source": "env_risk_v0",
    "score": 0.58,
    "state": "MEDIUM",
    "objects": [
      {"label": "chair", "confidence": 0.88}
    ],
    "top_hazards": [
      {"label": "chair", "confidence": 0.88, "risk_contribution": 0.45}
    ]
  },
  "low_light": {"brightness": 74, "state": "NORMAL"},
  "obstacle": {"state": "AVAILABLE"},
  "trajectory": {"state": "AVAILABLE"},
  "interaction": {"state": "AVAILABLE"},
  "fusion": {
    "overall_state": "MEDIUM",
    "reason": ["motion_medium", "environment_medium"]
  }
}
```

## 6. 页面展示对应关系

| 页面区域 | 主要数据 |
|---|---|
| AI 分析视频 | 姿态骨架、风险等级、基线状态、处理帧数 |
| 环境检测侧栏 | 环境目标类别、置信度、目标数量、`top_hazards` |
| 光照信息 | `low_light.brightness` 和低光状态 |
| 风险卡片 | 兼容字段 `current_risk_level`、偏离距离和告警原因 |
| 诊断状态 | 模型加载、数据不足、检测失败、扩展状态 |

## 7. 本机验证

### 检查模型文件

```powershell
Get-FileHash checkpoints\yolo26n.pt -Algorithm SHA256
Get-FileHash checkpoints\yolo26n-pose.pt -Algorithm SHA256
```

### 检查模型任务类型

```powershell
.\venv\Scripts\python.exe -c "from ultralytics import YOLO; print(YOLO('checkpoints/yolo26n.pt').task); print(YOLO('checkpoints/yolo26n-pose.pt').task)"
```

预期输出为 `detect` 和 `pose`。

### 启动服务

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```powershell
cd frontend
npm.cmd run dev
```

### 启动本地视频并检查结果

```powershell
$body = @{
  source = "D:\TIAOZHANBEI\fall-risk-prediction\data\raw\test.mp4"
  person_id = "local-test"
  device_id = "local-video"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/stream/start `
  -ContentType "application/json" -Body $body

Invoke-RestMethod http://127.0.0.1:8000/api/v1/risk/current
```

重点检查 `environment_boxes`、`environment_count`、`illumination`、`environment`、`low_light` 和 `fusion` 是否持续更新。

## 8. 限制与边界

- `yolo26n.pt` 是 COCO 环境目标预训练模型，不是专门的跌倒环境分类器。
- motion、environment 和 fusion 分数是工程指标，不是校准跌倒概率。
- v0.3 是兼容工程扩展；当前交付材料没有单一冻结的最终部署预测器。
- 不能仅凭一次视频演示报告准确率、误报率、漏报率或临床效果。
- 单目图像中的目标距离是近似的图像平面距离，不是真实米制距离。
- 多人跟踪、复杂遮挡和复杂人体-物体交互仍有边界。
- 当前页面展示的是实时推理输出，不应直接把演示目录中的 JSON 或 `summary.json` 当作在线结果。
