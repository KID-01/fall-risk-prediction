# 03 — Confirmed Findings：Pose Temporal Dropout Diagnostic

> **状态标记：engineering diagnostic / preliminary finding。**
> 本文件内容为一次只读、一次性的工程诊断，**不是 proven improvement**。
> 不声称已提升 Precision / Recall / F1 / EWR。
> 数据来源：`experiments/competition_sprint/scripts/temporal_dropout_diagnostic.py` 及其产物
> （`dropout_temporal_aggregate.csv`、`dropout_temporal_seqlevel.csv`、`dropout_anomaly_seq_overview.csv`、`dropout_temporal_summary.json`、`pose_dropout_summary.json`）。

## 1. Hypothesis

> **（待验证假设）** Pose 在跌倒关键窗口（尤其是 early-warning 时可捕捉的 pre-impact 阶段）出现显著 dropout（检测丢失 / 低置信），因此「Pose dropout 是 early-warning 主要瓶颈」。

## 2. Method

- 模型：`weights/pose/yolo26n-pose.pt`，ByteTrack，conf=0.25，imgsz=640，device=cpu。
- 数据：正确 RGB 帧目录 `E:\ur_fall_rgb\<seq>\<seq>-cam0-rgb\*.png`（30 fall + 2 混淆 ADL，hardset，**非完整 test set**）。
- 时间轴：以人工 `t_impact`（impact_frame_cam0）为 0 点，`秒 = 帧/30`（30fps）；一阶诊断，非精确同步时间戳。
- 统计分桶：`pre[-2,-0.5]`、`pre[-0.5,-0.3]`、`pre[-0.3,-0.2]`、`pre[-0.2,0]`、`post[0,0.2]`、`post[0.2,0.5]`、`post[0.5,1]`、`post[1,2]`。
- 每帧判零态：
  - **zero-pose**：`person_conf < 0.25` 或 有效关键点（conf≥0.5 数）< 4；
  - **low-conf**：平均关键点置信 < 0.5（**DIAGNOSTIC 阈值，不冻结，不是已验证的 fallback trigger**）。
- 修正：关键点 CSV 仅记录检测到人的行；**完全无检测帧已从 frame_idx 连续性还原并计入 zero-pose**。

## 3. Evidence（数值）

### 3.1 Aggregate pooling（全序列分窗，相对 impact）

| 窗口(相对 impact) | 帧数 | zero_rate | lowconf_rate |
|---|---|---|---|
| pre[-2,-0.5] | 994 | 0.0 | 0.015 |
| pre[-0.5,-0.3] | 180 | **0.0** | 0.0 |
| pre[-0.3,-0.2] | 90 | **0.0** | 0.0 |
| pre[-0.2,0] | 180 | 0.117 | 0.122 |
| post[0,0.2] | 163 | 0.166 | 0.233 |
| post[0.2,0.5] | 194 | 0.443 | 0.495 |
| post[0.5,1] | 254 | **0.50** | 0.614 |
| post[1,2] | 237 | 0.274 | 0.435 |

### 3.2 异常序列在 EWR horizon 的 zero_rate（关键）

异常序列 fall-01/03/07/10/15/22/29（`dropout_temporal_seqlevel.csv`）在 `pre[-0.5,-0.3]`、`pre[-0.3,-0.2]` 全部为 **0.0**；`pre[-0.2,0]` 仅有部分序列出现（如 fall-15/17/19/21/23/25/27/29 有非零，其余为 0）。degradation 主要集中在 `post[0.2,0.5]` 之后。

### 3.3 单序列 zero_pose_ratio 概览（`pose_dropout_summary.json`）

- 高 zero_pose_ratio 序列（如 fall-22=0.2727, fall-01=0.0894, fall-15=0.0926, fall-29=0.1463）其 degradation 大多出现在 post-impact / 躺地段。
- 全体序列 mean_person_conf ≈ 0.71–0.91，mean_kpt_conf ≈ 0.65–0.93（多数序列关键点质量尚可）。

## 4. Conclusion（结论，务必按此表述）

1. **在正式 early-warning horizon（-0.5 / -0.3 / -0.2 秒）没有观察到明显 zero-pose dropout**（`zero_rate=0.0`）。
2. **Pose degradation 主要集中在紧邻 impact 的 `pre[-0.2,0]` 以及 post-impact 阶段**（post-impact 后陡升，+0.5~1s 达 0.50）。
3. **因此「Pose dropout 是 early-warning 主要瓶颈」这一假设目前未被证实。**
4. **当前暂不进行 YOLO Pose s/m A/B**（决策门：需先在 t_impact 前见到确认的 degradation，未满足则不下载 s/m、不做 A/B）。
5. `lowconf=0.5` 只是 **diagnostic threshold**，**不是已经验证的 fallback trigger**。

## 5. Limitation（局限，必须如实说明）

- **仅 CPU、单模型（yolo26n-pose）**；未做 s/m/l/x A/B（被否决/暂缓）。
- **hardset，非完整 test set**（32 序列，其中 2 个为混淆 ADL），不可泛化代表全量性能。
- **未覆盖 MCFD / 外部数据**（MCFD 未下载）。
- **时间轴为帧/30 一阶估计**，非 metadata 精确同步时间戳；HR 级时序精度需回填。
- **只考察了 pose dropout 这一单一瓶颈**，未系统评估环境、融合、阈值等其它因素。
- 未做统计显著性 / 校准；结论为描述性诊断。
- **关键区分**：
  - **pre-impact early warning**（EWR horizon -0.5/-0.3/-0.2s，本诊断结论为「无显著 dropout」）；
  - **impact / post-impact fall confirmation**（+0.2s 之后，degradation 明显）。

## 6. 产物（已归档）

- 图：`experiments/competition_sprint/figures/fig_dropout_temporal.png`、`fig_dropout_by_seq.png`。
- CSV/JSON：`dropout_temporal_aggregate.csv`、`dropout_temporal_seqlevel.csv`、`dropout_anomaly_seq_overview.csv`、`dropout_temporal_summary.json`、`pose_dropout_summary.json`。
- 记录：`experiments/competition_sprint/experiment_log.md`（SPRINT-001 / SPRINT-002）。
