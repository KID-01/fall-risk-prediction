# 01 — Frozen Baseline 系统架构（按真实代码）

> 本文件严格依据 `fall_mvp/` 现行源码 `run.py / env_risk.py / fusion.py / contract.py` 总结。
> **不虚构任何模块。** 若某模块未在代码中出现，即为不存在。
> 运行入口：`python -m fall_mvp.run --input <video_or_frames> --output <out_dir>`（需在含 `fall_mvp/` 的仓库根或把根加入 PYTHONPATH）。

## 0. 逐帧处理模型

`run.py` 的主循环 `main()` 对输入（视频或 RGB 帧目录）逐帧迭代，每帧依次执行：

```
frame (同源帧)
   ├────▶ 分支 A：visual / motion（Pose）
   └────▶ 分支 B：environment（YOLO COCO 预训练）
                     │
        person 一致性 (IoU / center ratio) 检查
                     │
                     ▼
        motion_heuristic_v0（bbox 体心垂直速度）
        env_risk_v0（Environment Risk Engine）
                     │
                     ▼
        risk_fusion_v0（确定性规则表）
                     │
                     ▼
        causal_persist（3 帧多数稳定化，因果，仅过去+当前）
                     │
                     ▼
        overall = LOW / MEDIUM / HIGH / UNKNOWN
                     │
                     ▼
        frames.jsonl + demo.mp4 + summary.json + config_snapshot.yaml
```

## 1. 输入（Input）

- 支持：**视频文件（mp4/avi/…）** 或 **RGB 帧目录（PNG/JPG 序列）**。
- 帧来源：`run.py::_iter_frames()`。目录 → 递归收集 `*.png`；视频 → `cv2.VideoCapture`。
- 时间戳：帧序号 × `1000/fps`（目录按 30fps 估算，视频按 `CAP_PROP_FPS`）。`t=ts_ms/1000` 秒。
- 单一载体为逐帧 `image`（numpy array），两分支 **同源同帧** 输入（`run.py` 注释明确「同源帧（同一帧依次喂两分支）」）。
- 注意：真实科研 pipeline 的**精确同步时间戳未在此入口导出**（此处用帧号估计）；精准 HR 级时序需回填 metadata timestamp_ms（在 README/HANDOFF 有说明，属已知局限）。

## 2. Pose / person perception（分支 A：visual / motion）

- 模型：`config["visual"]["model"] = weights/pose/yolo26n-pose.pt`，`conf=0.25`，`imgsz=640`（`contract.py`）。
- `run.py` 用 `pose_model.predict(source=frame, ...)` 得到检测框。
- `_parse_boxes()`：把检测框按 COCO label 分成 `person` 与 `object`（`run.py` 此处 pose 分支只取 person 框，用于 motion）。
- **主人物选择** `choose_primary()`（`env_risk.py`，两分支共用）：优先与上一帧主人物 IoU>0 的；否则**最大 bbox**；否则最高 conf。这是确定性主人物（v0.1 简化，非多目标跟踪复杂度）。
- **motion_heuristic_v0**：`run.py` 用主人物 bbox **体心（cy）的垂直速度**（除以 bbox 高归一化，向下为正），`mscore = clamp(vy/5.0, [0,1])`。**这是工程启发，不是校准概率，也不是 frozen v1.0 LR predictor。**
- Python 环境无 CUDA（`torch 2.13.0+cpu`），CPU 推理。

## 3. Environment perception（分支 B：YOLO COCO）

- 模型：`config["yolo"]["model"] = yolo26n.pt`（COCO 预训练），`conf=0.25`，`imgsz=640`。
- `env_model.predict()` 检出的框同 `_parse_boxes()` 分成 person + objects。
- `choose_primary()` 选一个主人物（env 支），用于 env 风险计算的参考人。

## 4. Temporal / risk pipeline（风险引擎）

### 4.1 env_risk_v0（`env_risk.py::compute_env_risk`）
- **foot reference**：主人物 bbox 底边中心（图像平面近似）。
- **proximity**：foot 点到每个 object bbox 的最近图像平面距离 `d_px`，除以 person bbox 高 => `d_norm`（单目邻近代理，**非米制距离**）。
- **proximity_factor()**：分段线性，`d<=near_thr(0.4)`→1，`d>=far_thr(1.5)`→0，中间线性。
- **foot_band_factor()**：物体与人下半带（`[mid_y, y2]`）垂直重叠比例；纯横向无重叠→抑制。
- **contribution** `= class_weight × proximity_factor × min(conf,1) × foot_band_factor`。
- **environment_risk_score** = sum(contributions)，clip 到 [0,1]。`top_hazards` 取贡献前 k。
- object risk 权重表在 `contract.py`（chair=0.6, couch=0.4, bed=0.3, dining table=0.5, backpack=0.5, suitcase=0.7, sports ball=0.7, laptop=0.3；neutral：tv/remote/dog）。**工程启发，非科研优化。**

### 4.2 主人物跨分支一致性 / person match
- `run.py::person_match_pair()`：pose 主人物 bbox vs YOLO 主人物 bbox 的 IoU（min 0.15）或中心距离（≤0.7×bbox高）判定是否一致。
- `person_match = None`（两人支都无→无人）/ `False`（一个有人一个无→不匹配）/ `True`。

### 4.3 risk_fusion_v0（`fusion.py::fuse`）
- 确定性规则表 `FUSION_TABLE`：
  - `(LOW,LOW)=LOW`、`(LOW,MEDIUM)=LOW`、`(LOW,HIGH)=LOW`（**环境单独不能 HIGH**，motion LOW 时给 `context_elevated`）、
  - `(MEDIUM,*)=MEDIUM`、`(MEDIUM,HIGH)=HIGH`、
  - `(HIGH,*)=HIGH`。
- `state_from_score()`：`score < thr_low`→LOW，`>= thr_high`→HIGH，否则 MEDIUM；`None`→UNKNOWN。motion_thr=[0.3,0.6]、env_thr=[0.4,0.7]。
- 缺失语义（**UNKNOWN ≠ LOW**）：
  - 无人 `person_present=False` → `UNKNOWN(person_missing)`；
  - `person_match=False` → `UNKNOWN(person_branch_mismatch)`；
  - motion UNKNOWN、env 有人 → `UNKNOWN(motion_missing)`（环境仍可报 context）；
  - 环境单独（motion LOW + env HIGH）→ `LOW + context_elevated`，**不触发 HIGH**。

### 4.4 causal_persist（`fusion.py::causal_persist`）
- 最近 n 帧（exposure，n=persistence_frames=3）多数稳定化；**因果**（仅过去+当前），无未来帧。
- `n<=1` 关闭；并列偏好最近。

## 5. Risk output（输出）

`run.py` 每帧写一行 `frames.jsonl`：
```json
{"timestamp":0.0,
 "person":{"present":true,"match":true},
 "motion":{"source":"motion_heuristic_v0","score":0.0,"state":"LOW"},
 "environment":{"source":"env_risk_v0","score":0.0,"state":"LOW",
                "top_hazards":[{"class":"chair","confidence":0.88,
                                "normalized_distance":0.57,"risk_contribution":0.45}]},
 "fusion":{"source":"risk_fusion_v0","overall_state":"LOW",
           "reason":["motion_low","environment_medium","context_elevated"]},
 "quality":{"sync_delta_sec":0.0,"status":"OK"}}
```
- `summary.json`：帧计数、LOW/MEDIUM/HIGH/UNKNOWN 计数、max_env_risk、top hazards。
- `config_snapshot.yaml`：本次运行 config 快照。

## 6. Visualization

- `run.py::_draw()`：`cv2` 绘制 person 框（绿）、object 框（蓝）+label+conf、person 底边 foot 点（黄）。
- 文本 overlay：`t=... Motion=.../state`、`Env=.../state`、`OVERALL=... (reason...)`、`top hazard ...`。
- 颜色映射：LOW=绿、MEDIUM=黄、HIGH=红、UNKNOWN=灰。
- 输出 `demo.mp4`（`cv2.VideoWriter`，mp4v，30fps，frame 元尺寸）。

## 7. Deployment / Demo

- 运行：`python -m fall_mvp.run --input <video_or_frames> --output <out_dir>`。
- 已 smoke 验证输入：`E:\ur_fall_rgb\fall-01/02/03` 三个正确 RGB 帧目录，一条命令跑通、JSON 可解析、schema 一致。见 `artifacts/fall/mvp_final/smoke_test_report.md`。
- **真实样本 HIGH=0**（因跌倒/躺地过渡区 Pose 退化，motion heuristic 无法持续获得有效人体运动信号）。**不是 bug，未通过调参"修复"**；HIGH 分支仅由 synthetic/unit 证明逻辑可达，**非真实性能证据**。

## 8. 数据流汇总

```
帧 → [Pose branch] → person bbox → motion_heuristic_v0(垂直速度) → motion score/state
   → [YOLO branch] → person bbox + objects → env_risk_v0(邻近+权重) → env score/state
   → person 一致性(IoU/center) → risk_fusion_v0(规则表) → causal_persist(3帧)
   → overall state → frames.jsonl / summary.json / config_snapshot / demo.mp4
```

## 9. 模块不存在 / 未实现（避免队友误解）

- **无** 训练代码、无调参过程（本项目是确定性规则 + 预训练模型）。
- **无** 多目标复杂跟踪；主人物为确定性简化。
- **无** 真正跌倒分类器 / 校准概率。
- **无** audio 模块（接口预留见 HANDOFF §9，非 critical path，本版不实现）。
- **无** ST-GCN / Mamba / TCN / Transformer（这些仍在长期科研路线，deferred）。
- **无** depth / 真实米制距离（`d_norm` 为单目近似）。
