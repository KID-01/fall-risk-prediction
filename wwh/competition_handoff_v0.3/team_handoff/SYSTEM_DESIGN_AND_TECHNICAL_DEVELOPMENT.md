# 系统设计与技术开发文档

## 1. 文档目的

本文档描述当前竞赛交付系统的真实工程实现、模块边界、运行方式、接口契约、配置项和已知限制。
本文档描述的是 `fall_mvp` v0.1 工程原型，不把工程启发式风险分数表述为校准跌倒概率，也不替代 `experiments/competition_sprint/` 中的科研实验记录。

## 2. 系统目标与范围

### 2.1 当前目标

系统从视频或 RGB 帧目录中逐帧提取：

1. 人体/运动信息；
2. 环境物体及其与人的图像平面邻近关系；
3. 运动风险与环境风险；
4. 可解释的 `LOW / MEDIUM / HIGH / UNKNOWN` 工程状态；
5. JSONL、摘要、配置快照和带 overlay 的演示视频。

### 2.2 当前不包含

- 临床跌倒概率；
- 经过校准的概率输出；
- 已验证的工业部署模型；
- webcam 输入的正式验证；
- 独立的 post-impact confirmation 部署器；
- 已证明的 pre-impact 性能提升模型。

## 3. 总体架构

```text
Video / RGB frame directory
            |
            +--------------------+
            |                    |
      Visual Adapter       YOLO COCO Detector
      Pose/Motion branch   Environment branch
            |                    |
   motion_heuristic_v0     objects/person boxes
            |                    |
            +------ person matching / sync ------+
                                                   |
                                     env_risk_v0
                            object weight + proximity
                                                   |
                         risk_fusion_v0 rule table
                                                   |
                  JSONL + summary + config + demo.mp4
```

主入口：`fall_mvp/run.py`。

## 4. 数据流

对每个输入帧：

1. `_iter_frames()` 读取视频帧或 RGB 目录，并生成时间戳；目录输入按 30fps 计算，视频输入读取视频 FPS。
2. Pose/Motion 分支使用 `weights/pose/yolo26n-pose.pt`，获得人体框并计算 bbox 中心垂直运动启发式分数。
3. 环境分支使用根目录 `yolo26n.pt` 的 COCO 检测结果，区分 `person` 与环境物体。
4. `choose_primary()` 选择主人物：优先与上一帧有 IoU 的候选，否则选择最大 bbox。
5. `person_match_pair()` 对 Pose 人框和环境分支人框做 IoU/中心距离一致性检查。
6. `compute_env_risk()` 对已配置危险类别计算人与物体的图像平面邻近风险。
7. `fuse()` 根据运动状态和环境状态查确定性规则表；环境单独不能触发 HIGH。
8. `causal_persist()` 使用当前帧和历史帧做有限帧多数稳定化，不读取未来帧。
9. 写出逐帧 JSONL，更新 summary，并把 overlay 写入 MP4。

## 5. 核心模块

### 5.1 Motion / Pose 分支

文件：`fall_mvp/visual_adapter.py`、`fall_mvp/run.py`。

当前实现使用 Pose 模型人体框计算工程运动分数：

```text
normalized vertical velocity = center_y velocity / previous bbox height
motion score = clipped normalized velocity / 5
```

该分数来源标记为 `motion_heuristic_v0`，不是 frozen v1.0 Logistic Regression predictor，也不是概率。

### 5.2 环境识别

文件：`fall_mvp/yolo_detector.py`、`fall_mvp/env_risk.py`。

环境模型为 COCO 预训练检测器。当前配置中的危险物权重包括 chair、couch、bed、dining table、backpack、suitcase、sports ball 和 laptop；tv、remote、dog 为中性类别。

### 5.3 人-物邻近风险

环境风险计算步骤：

1. 以人物 bbox 底边中心作为脚部代理点；
2. 计算该点到物体 bbox 的最近图像平面距离；
3. 除以人物 bbox 高度得到 `d_norm`；
4. 按 near/far 阈值计算邻近因子；
5. 乘以物体类别权重、检测置信度和脚部带重叠因子；
6. 多个物体贡献求和并 clip 到 `[0, 1]`；
7. 输出 top hazards。

`d_norm` 是单目图像平面代理，不是米制物理距离。

### 5.4 风险融合

融合规则由 `FUSION_TABLE` 定义：

- LOW + LOW/MEDIUM/HIGH → LOW；环境只做上下文升高；
- MEDIUM + LOW/MEDIUM → MEDIUM；
- MEDIUM + HIGH → HIGH；
- HIGH + 任意已知环境状态 → HIGH；
- 人缺失 → UNKNOWN；
- Pose/Motion 缺失 → UNKNOWN；
- 分支人物不匹配 → UNKNOWN。

因此 `UNKNOWN` 不等于 LOW，也不等于跌倒概率为零。

## 6. 输入输出契约

### 6.1 输入

已验证：

- 视频文件（mp4/avi 等）；
- RGB 帧目录（PNG/JPG）。

未正式验证：

- webcam `--input 0`；
- 多路实时摄像头；
- 真实米制距离标定。

### 6.2 输出

指定 `--output <dir>` 后生成：

- `frames.jsonl`：每帧包含 timestamp、person、motion、environment、fusion、quality；
- `summary.json`：帧数、有效人物帧、各状态帧数、最高环境风险、危险物计数；
- `config_snapshot.yaml`：本次实际配置快照；
- `demo.mp4`：overlay 演示视频。

典型 JSON 结构：

```json
{
  "timestamp": 0.0,
  "person": {"present": true, "match": true},
  "motion": {"source": "motion_heuristic_v0", "score": 0.0, "state": "LOW"},
  "environment": {"source": "env_risk_v0", "score": 0.52, "state": "MEDIUM", "top_hazards": []},
  "fusion": {"source": "risk_fusion_v0", "overall_state": "LOW", "reason": ["motion_low"]},
  "quality": {"sync_delta_sec": 0.0, "status": "OK"}
}
```

## 7. 配置与部署

配置集中在 `fall_mvp/contract.py::default_config()`，运行时写入 `config_snapshot.yaml`。主要配置：

- Pose/环境模型路径、conf、imgsz；
- motion window；
- 环境类别权重、near/far 阈值、对象置信度；
- person matching IoU/center ratio；
- fusion motion/env thresholds；
- causal persistence frames。

运行命令：

```bash
python -m fall_mvp.run --input <video_or_rgb_frame_dir> --output <output_dir>
```

当前交付形式是 Python 源码入口，不是独立 `.exe`。队友机器需要自行准备 Python、Ultralytics 源码、模型权重和依赖；本包不包含权重或原始数据。

## 8. 工程异常处理

- 输入帧无法读取：跳过不可读图片或视频读取结束；
- 无人物：fusion 输出 `UNKNOWN/person_missing`；
- Pose 与环境人物不匹配：输出 `UNKNOWN/person_branch_mismatch`；
- Motion 缺失：输出 `UNKNOWN/motion_missing`，不伪造 LOW；
- 没有危险物：环境状态为 LOW，表示当前没有检测到配置中的危险物，不代表没有跌倒风险；
- 输出目录不存在时自动创建。

## 9. 当前验证结论与限制

工程 smoke 测试已在 3 个正确 RGB 样本上通过：无 crash、无 missing-file、JSON 可解析、schema 一致、演示视频生成成功。

真实样本中 `HIGH=0`，原因包括 Pose 在跌倒/躺地过渡段退化以及运动启发式没有持续得到有效信号；没有通过单样本调参修复这一结果。

科学 frozen v1.0 的 `EWR@0.5=0.233` 属于独立科研评估，不等于本工程 MVP 的 `motion_heuristic_v0`，二者不可混用。

## 10. 扩展接口

未来替换 motion provider 时保持以下接口语义：

```text
{timestamp, score/state, person/bbox, availability, source}
```

Environment Engine、Fusion、JSONL 和 UI 不应依赖某个特定 Motion Provider。Audio 可作为 post-impact confirmation 插件接入，但主系统不能依赖 Audio 才能运行。

## 11. v0.3 可选环境/轨迹交互扩展（兼容增量）

> v0.2 已发给队友并保持有效；本节描述后续可选工程增强。扩展默认关闭，不修改 frozen baseline 或旧 fusion 语义。

启用方式：

```bash
python -m fall_mvp.run --input <source> --output <dir> --enable-risk-extensions
```

新增 provider：

- `lighting_v0`：亮度、暗像素比例和对比度工程风险；
- `clutter_v0`：人物脚部通道与 COCO 障碍物 bbox 的几何重叠；
- `causal_linear_trajectory_v0`：仅基于历史脚点线性外推未来 1 秒；
- `interaction_v0`：预测路径与障碍物/危险区域的交互风险；
- `wet_floor_unavailable`：水渍模型未满足接入条件时明确 UNKNOWN。

开启后只追加 `risk_extensions` 字段；旧字段保持不变。新指数是 0–100 工程指数，不是概率。

详细路线、兼容约束和后续执行纪律见 `team_handoff/ENVIRONMENT_INTERACTION_EXTENSION_PLAN.md`。
