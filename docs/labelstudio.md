# LabelStudio 跌倒风险标注工作流

> 配套脚本: `scripts/labelstudio_import.py` / `scripts/labelstudio_export.py` / `scripts/labelstudio_agreement.py`
> 目的: 为 `src/models/*` 的监督训练收集带标注的关键点序列数据

---

## 目录

1. [项目背景与标注目的](#1-项目背景与标注目的)
2. [标注对象与任务定义](#2-标注对象与任务定义)
3. [LabelStudio 标注配置](#3-labelstudio-标注配置)
4. [数据导入](#4-数据导入)
5. [标注规范](#5-标注规范)
6. [数据导出](#6-数据导出)
7. [标注一致性评估](#7-标注一致性评估)
8. [审核工作流](#8-审核工作流)

---

## 1. 项目背景与标注目的

跌倒风险预测系统通过视频帧 → YOLOv8n 人体检测 → MediaPipe Pose 关键点提取，
最终由 `src/models/*` 的时序模型输出跌倒风险等级。当前模型缺少**有监督训练数据**，
本工作流即用于收集带标注的关键点序列，作为后续训练 `src/models/fall_risk_predictor.py`
的监督数据集。

人工标注的价值：

- 提供高质量的关键点可见性/位置真值，用于监督关键点质量与预标注纠错
- 提供序列级跌倒风险等级真值，用于风险分类头的监督训练
- 通过多标注者一致性评估控制数据质量

## 2. 标注对象与任务定义

### 2.1 帧级任务 (frames 模式)

每个任务 = **一帧图片**，标注内容包括：

| 标注项 | 类型 | 说明 |
|--------|------|------|
| 33 个 MediaPipe 关键点 | keypointlabels | 判断每个关键点是否可见，可见则放置到对应关节位置 |
| 跌倒风险等级 | choices | 单帧/时刻的跌倒风险判定（四选一） |

关键点与 `src/utils/keypoints.py` 的 `PoseKeypoint` 枚举一一对应，标签名使用枚举小写名：

| 索引 | 标签名 | 索引 | 标签名 |
|------|--------|------|--------|
| 0 | nose | 17 | left_pinky |
| 1 | left_eye_inner | 18 | right_pinky |
| 2 | left_eye | 19 | left_index |
| 3 | left_eye_outer | 20 | right_index |
| 4 | right_eye_inner | 21 | left_thumb |
| 5 | right_eye | 22 | right_thumb |
| 6 | right_eye_outer | 23 | left_hip |
| 7 | left_ear | 24 | right_hip |
| 8 | right_ear | 25 | left_knee |
| 9 | mouth_left | 26 | right_knee |
| 10 | mouth_right | 27 | left_ankle |
| 11 | left_shoulder | 28 | right_ankle |
| 12 | right_shoulder | 29 | left_heel |
| 13 | left_elbow | 30 | right_heel |
| 14 | right_elbow | 31 | left_foot_index |
| 15 | left_wrist | 32 | right_foot_index |
| 16 | right_wrist | | |

### 2.2 片段级任务 (clips 模式)

每个任务 = **一段短片段视频**（默认 30 帧），标注内容仅为跌倒风险等级。
片段级标注用于学习时序上下文下的风险判定，适合从日常监控视频中快速筛选高危/正常样本。

### 2.3 风险等级定义

风险等级与 `src/alerts/engine.py` 的 `RiskLevel` 对齐（`priority` 0-3）：

| 等级 | 数值 | 判定参考（来自 configs/base.yaml） |
|------|------|------|
| 低风险 | 0 | 所有特征正常，无异常动作 |
| 关注级 | 1 | 出现短暂不平衡/小幅度踉跄等短期异常苗头 |
| 预警级 | 2 | 明显行走不稳、步伐紊乱、长期功能衰退迹象 |
| 高危级 | 3 | 近似跌倒动作、站立困难或长时间无活动 |

## 3. LabelStudio 标注配置

在 LabelStudio 项目 Settings → Labeling Interface 中粘贴以下 XML 配置
（也可运行 `python scripts/labelstudio_import.py --emit-label-config configs/labelstudio_config.xml` 生成）。

### 3.1 帧级关键点标注配置

```xml
<View>
  <Header value="跌倒风险关键点标注"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <KeyPointLabels name="pose" toName="image" strokeWidth="3" pointSize="small" opacity="0.9">
    <Label value="nose" background="#e6194b"/>
    <Label value="left_eye_inner" background="#3cb44b"/>
    <Label value="left_eye" background="#ffe119"/>
    <Label value="left_eye_outer" background="#4363d8"/>
    <Label value="right_eye_inner" background="#f58231"/>
    <Label value="right_eye" background="#911eb4"/>
    <Label value="right_eye_outer" background="#46f0f0"/>
    <Label value="left_ear" background="#f032e6"/>
    <Label value="right_ear" background="#bcf60c"/>
    <Label value="mouth_left" background="#fabebe"/>
    <Label value="mouth_right" background="#008080"/>
    <Label value="left_shoulder" background="#e6beff"/>
    <Label value="right_shoulder" background="#9a6324"/>
    <Label value="left_elbow" background="#fffac8"/>
    <Label value="right_elbow" background="#800000"/>
    <Label value="left_wrist" background="#aaffc3"/>
    <Label value="right_wrist" background="#808000"/>
    <Label value="left_pinky" background="#ffd8b1"/>
    <Label value="right_pinky" background="#000075"/>
    <Label value="left_index" background="#808080"/>
    <Label value="right_index" background="#ffffff"/>
    <Label value="left_thumb" background="#000000"/>
    <Label value="right_thumb" background="#e6194b"/>
    <Label value="left_hip" background="#3cb44b"/>
    <Label value="right_hip" background="#ffe119"/>
    <Label value="left_knee" background="#4363d8"/>
    <Label value="right_knee" background="#f58231"/>
    <Label value="left_ankle" background="#911eb4"/>
    <Label value="right_ankle" background="#46f0f0"/>
    <Label value="left_heel" background="#f032e6"/>
    <Label value="right_heel" background="#bcf60c"/>
    <Label value="left_foot_index" background="#fabebe"/>
    <Label value="right_foot_index" background="#008080"/>
  </KeyPointLabels>
  <Choices name="fall_risk" toName="image" choice="single" showInLine="true" required="true">
    <Choice value="低风险"/>
    <Choice value="关注级"/>
    <Choice value="预警级"/>
    <Choice value="高危级"/>
  </Choices>
</View>
```

### 3.2 片段级风险标注配置

```xml
<View>
  <Header value="跌倒风险片段标注"/>
  <Video name="video" value="$video"/>
  <Choices name="fall_risk" toName="video" choice="single" showInLine="true" required="true">
    <Choice value="低风险"/>
    <Choice value="关注级"/>
    <Choice value="预警级"/>
    <Choice value="高危级"/>
  </Choices>
</View>
```

### 3.3 配置说明

- `KeyPointLabels name="pose"` — 关键点标注区域，`toName="image"` 绑定图片；导入时预填充的
  `predictions` 也使用同一 `from_name="pose"`，标注者只需核对/修正位置，无需从零标注。
- `Choices name="fall_risk"` — 风险等级单选，`required="true"` 强制标注，防止漏标。
- 三个脚本硬编码了与上述 XML 一致的结构常量，修改需保持同步
  （`FALL_RISK_CHOICES` 与 `KEYPOINT_LABEL_NAMES`）。

## 4. 数据导入

### 4.1 准备关键点 JSON

`scripts/labelstudio_import.py` 接受两种关键点 JSON 结构：

```json
{
  "source": "video_001.mp4",
  "fps": 10.0,
  "frames": [
    {"timestamp": 0.0, "is_valid": true,
     "keypoints": [[x, y, z, visibility] x 33]},
    {"timestamp": 0.1, "is_valid": true,
     "keypoints": [[x, y, z, visibility] x 33]}
  ]
}
```

或数组格式（等价于 `KeypointStore` 保存的 `.npy` 转 JSON）：

```json
{
  "source": "video_001.mp4",
  "fps": 10.0,
  "timestamps": [0.0, 0.1],
  "keypoints": [[[x, y, z, visibility] x 33] x T]
}
```

关键点数组必须为 `(33, 4)`，第 4 列为 MediaPipe 可见性分数 (0-1)。

### 4.2 图片引用方式

LabelStudio 显示任务图片有两种方式：

1. **URL 方式（推荐）**：将帧图片部署到静态服务器，导入时用 `--image-url-base` 指定前缀。
   图片命名约定为 `frame_{帧号:06d}.jpg`，与 `--mode video` 的输出一致。
2. **本地文件方式**：用 `--image-dir` 指定本地目录，并以下列命令启动 LabelStudio
   （需先在本机 `pip install label-studio`）：

```powershell
label-studio start --local-files-document-root D:\WorkSpace\coding\fall-risk-prediction\data\labelstudio
```

### 4.3 导入命令

```powershell
# frames 模式: 每帧一个任务, 预填关键点标注 (可见性 > 0.5 的关键点生成 predictions)
python scripts/labelstudio_import.py --input data/keypoints/video_001.json --output data/labelstudio/tasks_video_001.json --mode frames --image-url-base http://localhost:8080/images

# 批量: 输入为目录时处理其中全部 *.json
python scripts/labelstudio_import.py --input data/keypoints --output data/labelstudio/tasks_all.json --mode frames --image-url-base http://localhost:8080/images

# clips 模式: 每 30 帧一个片段任务 (片段级风险标注)
python scripts/labelstudio_import.py --input data/keypoints/video_001.json --output data/labelstudio/tasks_clips.json --mode clips --clip-len 30

# video 模式: 从视频抽样帧生成任务 (默认每 5 帧取 1 帧), 可选 --keypoints 预填充
python scripts/labelstudio_import.py --input data/videos/video_001.mp4 --output data/labelstudio/tasks_video.json --mode video --image-dir data/labelstudio/images
```

输出为 LabelStudio 可直接导入的任务 JSON 数组（`--output` 指定文件），
在项目 Data Manager 中点击 Import 上传即可。任务包含 `data.keypoints`
（原始 33x4 数组，导出时还原）与 `predictions`（预填充关键点标注）。

## 5. 标注规范

### 5.1 关键点标注规则

- **可见性**：关键点在当前帧中位置明确、未被遮挡时放置标注；被遮挡、出画或无法判断时不放置。
- **位置**：尽量贴合关节中心，与预填充位置偏差明显时才需调整。
- **下肢关键点优先**：`left_hip / right_hip / left_knee / right_knee / left_ankle / right_ankle`
  是本项目四大特征计算的核心（见 `src/utils/keypoints.py` 的 `LOWER_BODY_KEYPOINTS`），
  必须优先保证其标注质量。

### 5.2 跌倒风险等级判定规则

参照 [2.3 节](#23-风险等级定义)，结合**当前帧/片段的动作表现**判定：

- 正常站立、行走、坐姿 → 低风险
- 轻微摇晃、需要扶墙、脚步不稳但未失去平衡 → 关注级
- 明显踉跄、步伐紊乱、跌倒后缓慢站起 → 预警级
- 跌倒在地、无法自行站起、持续无活动 → 高危级

> 原则：拿不准时取较低等级；同一片段内风险变化时，以最高风险为准。

## 6. 数据导出

### 6.1 导出命令

在 LabelStudio 项目 Settings → Export 中导出 **JSON** 格式标注文件，
然后运行：

```powershell
python scripts/labelstudio_export.py --input data/labelstudio/export.json --output data/labelstudio/annotations.json
```

### 6.2 训练标注格式

输出的 `annotations.json` 为训练样本列表，每帧一条：

```json
[
  {
    "frame_id": 0,
    "timestamp": 0.0,
    "source": "video_001.mp4",
    "keypoints": [[x, y, z, visibility] x 33],
    "fall_risk_label": 1
  }
]
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | int | 帧序号（clips 模式按 `frame_ids` 展开） |
| `timestamp` | float | 帧时间戳（秒） |
| `source` | str | 数据来源视频标识 |
| `keypoints` | list[list[float]] | `(33, 4)` 关键点数组 |
| `fall_risk_label` | int | 0=低风险, 1=关注级, 2=预警级, 3=高危级；未标注为 `null` |

导出逻辑要点：

- `keypoints` 以导入时预填充的 `data.keypoints` 为基底；
  标注了关键点labels 的关键点位置以标注坐标为准（百分比 → 0-1），可见性置 1.0；
  未标注的关键点可见性置 0.0。
- `fall_risk_label` 从标注的 choices 结果映射，兼容中文标签/英文标签/数字字符串。

## 7. 标注一致性评估

对同一批任务由**两名标注者**独立标注后，各导出一份训练标注文件，
用 `scripts/labelstudio_agreement.py` 计算一致性：

```powershell
python scripts/labelstudio_agreement.py --a data/labelstudio/annotator_a.json --b data/labelstudio/annotator_b.json --output data/labelstudio/agreement.json
```

评估指标（输出同时打印到控制台与 JSON 报告）：

- **fall_risk_label**：原始一致率（`raw_agreement`）+ Cohen's kappa + 混淆矩阵
  （kappa 优先使用 `sklearn.metrics.cohen_kappa_score`，未安装时回退纯 Python 实现）
- **逐关键点可见性**：33 个关键点各自的可见性一致率与 kappa
  （可见性判定：`visibility > 0.5`）

质量门槛建议：

| 指标 | 达标线 | 说明 |
|------|--------|------|
| 风险等级 kappa | ≥ 0.6 | 低于则需统一标注口径后重标 |
| 风险等级原始一致率 | ≥ 0.75 | 见 5.2 判定规则 |
| 下肢关键点可见性一致率 | ≥ 0.8 | 关键特征输入，必须可靠 |

## 8. 审核工作流

```
采集视频 → 提取关键点 → 导入 LabelStudio → 双人独立标注
    → 导出两份标注 → 一致性评估
        ├─ 达标 → 合并为训练集 (不一致样本由第三人仲裁)
        └─ 未达标 → 复盘规则 → 重标或调整任务
    → 训练标注格式 (annotations.json) → 送入 src/models 训练
```

1. **采集**：`scripts/collect_video.py` + `scripts/preprocess_videos.py` + `scripts/extract_keypoints.py`
   产出关键点 JSON 序列。
2. **导入**：见 [4. 数据导入](#4-数据导入)，建议按 7 天窗口/单视频文件分批导入。
3. **双人标注**：每个任务至少分配两名标注者，互不可见对方结果。
4. **一致性评估**：见 [7. 标注一致性评估](#7-标注一致性评估)。
5. **仲裁与合并**：一致性达标的样本直接合并；`fall_risk_label` 不一致的样本由第三人仲裁；
   可见性不一致的关键点按"标注则可见"优先，或按仲裁结果处理。
6. **训练**：`annotations.json` 直接作为 `src/data/dataset.py` 的数据来源（按需扩充 Dataset 读取逻辑）。
