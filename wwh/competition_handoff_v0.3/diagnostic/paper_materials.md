# Competition Sprint — Paper Materials（科研素材固化）

> 本文件固化 competition sprint 期间的**系统性分析结论**（SPRINT-002–011），作为论文/答辩素材。
> 全部为**诚实的诊断与负结果**，来源为只读重放 + 冻结评估，无任何伪造指标。
> 实验日志全文见本目录 `experiment_log.md`；脚本见 `scripts/`；数据见 `results/`、`figures/`。

---

## 0. 系统性定位（必须先声明）

本项目是 **pose-guided fall early warning**。核心研究问题是「跌倒**提前预警**的可行性与瓶颈」。
本冲刺不做"正面刷分"，而是用多轮独立诊断**逐层定位** pre-impact 预警与 post-impact 确认的真实瓶颈，
产出**严谨的证据链**。结论诚实：主要发现是**负结果 + 瓶颈分析**，而非性能提升。

**可复现基线（frozen v1.0 PREIMPACT）**：EWR@0.5 = 0.233（7/30）、FA/min = 6.81、lead median 0.567s。

---

## 1. 快速参考：本次冲刺的完整证据链

| # | 阶段 | 核心结论 | 关键数字 |
|---|---|---|---|
| SPRINT-001 | 正确RGB Pose dropout 初证 | 主假设首证 | fall-01 fall-win zero=0.141 |
| SPRINT-002 | temporal dropout 诊断（t_impact=0） | **EWR horizon(-0.5/-0.3/-0.2s) Pose 无 dropout**；degradation 在 post-impact | pre[-0.5,-0.3]=0.0; post[0.5,1]=0.50 |
| SPRINT-003/004 | early-timing 判别力诊断 | **MISS 不是单一问题，分三类（真过晚/alarm<onset/never-alarm）** | HIT=7; 真过晚=10; never=6 |
| SPRINT-005/006 | B: threshold sweep | **B3：阈值不是根本解法** | 降 θ 至 50% EWR 仍≤0.2, FA 爆炸 |
| SPRINT-007/008 | A: 早期判别特征 LR | **无提升（负结果）** | A2 EWR0.5=0.167/0.200 < 0.233 |
| SPRINT-009 | P-2 post-impact 诊断 | **Pose degradation 非 fall-specific** | fall post-zero 0.343 vs ADL 0.235 |
| SPRINT-010/011 | P-3 post-impact 确认 | **综合 NO-GO** | combined rec 0.633 / adlFC 0.455 |

---

## 2. 可写入论文的表

### 表 A：Pose 可靠性随时间分布（SPRINT-002，aggregate，30 fall）

| 窗口（相对 t_impact=0） | zero-pose 率 | low-conf 率 |
|---|---|---|
| pre[-2,-0.5] | 0.000 | 0.015 |
| pre[-0.5,-0.3] | **0.000** | 0.000 |
| pre[-0.3,-0.2] | **0.000** | 0.000 |
| pre[-0.2,0] | 0.117 | 0.122 |
| post[0,0.2] | 0.166 | 0.233 |
| post[0.2,0.5] | 0.443 | 0.495 |
| post[0.5,1] | **0.50** | 0.614 |
| post[1,2] | 0.274 | 0.435 |

> 注意：修正口径（含完全无检测帧计入 zero-pose）。时间轴为 frame/30（一阶诊断）。

### 表 B：pre-impact 早期时序判别（SPRINT-003/004，theta=冻结值）

| 类别 | 数量 | 说明 |
|---|---|---|
| HIT（满足 EWR@0.5） | 7 | alarm ∈ [onset, impact-500ms] |
| 真过晚（delay>deadline） | 10 | median delay 267ms > deadline 201ms |
| alarm<onset（正常段误报） | 7 | 被 D-012 排除，非 qualifying |
| never-alarm | 6 | risk 从未达 θ |

### 表 C：threshold sweep（SPRINT-006，proxy/Arm R 口径）

| θ percentile | EWR@0.5 | FA/min | never-alarm |
|---|---|---|---|
| 99 | 0.067 | 3.08 | 13 |
| 90 | 0.200 | 11.43 | 3 |
| 50 | 0.167 | 22.6 | 0 |

> 结论：即使降到 50 percentile，EWR@0.5 ≤0.2（<0.233），FA 爆炸。→ **阈值不是解**。

### 表 D：早期判别特征（SPRINT-007，A-1 诊断）

| 特征 | AUC | Cohen's d |
|---|---|---|
| hip_mid_y_min | 0.771 | 1.15 |
| rh_y_min | 0.711 | 0.89 |
| body_h_min | 0.694 | 0.67 |
| body_h_delta | 0.680 | 0.29 |
| nose_y_min | 0.668 | 0.82 |

> 早期判别信号集中在 y（垂直）方向的 _min/_delta 特征，onset 后 0.3s 即显著区分。

### 表 E：早期判别 LR（SPRINT-008，A-2，对比 frozen v1.0）

| 模型 | EWR@0.5 | FA/min | lead |
|---|---|---|---|
| frozen v1.0 | **0.233** | **6.81** | 0.567 |
| A2 判别子集(11) | 0.167 | 6.35 | 0.600 |
| A2 全特征(72) | 0.200 | **10.07** | 0.651 |

> 结论：早期判别特征 + 早期目标 LR 未能提升 EWR@0.5。

### 表 F：post-impact 确认（SPRINT-010/011，P-3 v2，30 fall + 11 targeted hard-ADL）

| 模式 | fall recall | hard-ADL FC | med delay(s) |
|---|---|---|---|
| A terminal-only | 0.667 (20/30) | 0.909 (10/11) | 0.45 |
| B descent-only | 0.933 (28/30) | 0.909 (10/11) | 0.067 |
| C combined | 0.633 (19/30) | 0.455 (5/11) | 0.467 |

> **hard-ADL 为 targeted 子集（坐/蹲/躺等高混淆 ADL），非完整 ADL test-set，不能表述为完整 specificity。**

---

## 3. 核心结论（可写入论文，严格按措辞纪律）

### 3.1 Pose 可靠性分析
- **降级主要发生在 post-impact（+0.5~1s 达 zero=0.50），而非 pre-impact EWR horizon**（-0.5/-0.3/-0.2s 为 0）。
- 因此：**Pose degradation 不是 pre-impact 早预警失败的原因**（EWR horizon 内 Pose 可靠）。

### 3.2 pre-impact 早预警瓶颈（B3 成立）
- 多轮独立证据：v1.1 运动特征失败、v1.2 Arm O 全零、threshold sweep（B3）、早期判别特征 LR（A 无提升）。
- 结论：**在 v1.0 LR 家族 + 1.0s 因果窗 + 当前数据下，pre-impact EWR@0.5≈0.2–0.23 已接近该类线性因果方法的实际上限。**
- 证据链：不是缺信号 / 不是 Pose dropout / 不是阈值 —— 而是**早期窗口的线性判别置信时序不足**。

### 3.3 post-impact 确认（诚实负结果）
- **Pose degradation 非 fall-specific**：hard ADL（坐/蹲/躺）同样大规模 Pose 丢失。
- **bbox-height 下降也非 fall-specific**：rapid descent 对 fall 28/30 但对 hard ADL 10/11。
- combined 相对 terminal-only 有 real specificity gain（adlFC 0.909→0.455），但达不到工程门槛（≤0.273）；纯规则/几何线索在单目 2D 层面无法可靠区分 fall vs sit/squat/lie。

### 3.4 措辞纪律（允许 / 禁止）
- ✅ 允许写：
  - "Pose degradation is not fall-specific in the evaluated hard-ADL subset."
  - "Simple pose degradation and bbox-height based geometric cues exhibit substantial confusion between falls and fall-like ADLs."
- ❌ 禁止写（未证实）："Monocular 2D vision is inherently unable to distinguish falls from sitting/squatting/lying."

---

## 4. 图清单（`figures/`，已生成，可直接用于论文/PPT）

| 文件 | 内容 | 备注 |
|---|---|---|
| `fig_dropout_temporal.png` | Pose dropout 按时间窗（t_impact=0） | EWR horizon 与 post-impact 对照 |
| `fig_dropout_by_seq.png` | 每序列 post[0.5,1] 窗 dropout | 双峰现象可视化 |
| `pareto_proxy_ewr_vs_fa.png` | EWR@0.5 vs FA/min（θ sweep） | proxy 口径，标 frozen 点 |
| `pareto_proxy_p_sweep.png` | θ→EWR/FA 双轴 | proxy 口径 |

> proxy/Arm R 口径图需标注「ENGINEERING DIAGNOSTIC; proxy estimator」，不冒充正式基线。

---

## 5. 产物清单（`results/`，全部可复现）

- Pose dropout：`pose_dropout_summary.{csv,json}`、`dropout_temporal_*.csv`、`dropout_anomaly_seq_overview.csv`
- early-timing：`early_timing_diagnostic.csv`、`early_timing_margin_seqlevel.csv`
- sweep：`sweep_proxy_percentile.csv`、`sweep_proxy_detail.csv`
- A-1/A-2：`early_discrimination.csv`、`a2_early_lr_result.json`、`a2_early_lr_allfeats_result.json`
- post-impact：`postimpact_diag.json`、`p3_ablation.csv`、`p3_sequence_results.csv`、`p3_failure_cases.csv`、`p3_v2/*`
- 关键点缓存：`pose_export/{fall-01..30, adl-*}/*.csv`（30 fall + 11 ADL 已提取，可复用）
- 重放输入：`replayed/`（cam0_models + features.tsv + 评估归档）

---

## 6. 方法论要点（供 Method 章节）

- **数据**：UR-Fall，70 序列（30 fall + 40 ADL），正确 cam0 RGB 640×480 30fps；人工 t_impact/onset 标注。
- **Pose 特征提取**：yolo26n-pose.pt + ByteTrack（CPU），逐帧 17 关键点 + bbox + person_conf。
- **评估协议**：事件级 EWR@0.5/0.3/0.2、FA/min、lead（D-012）；sequence-level stratified 5-fold（6 fall/8 ADL per fold，D-015）；因果窗口。
- **可复现**：`sp_pose_export_batch.py` / `sp_pose_export_adl.py` + 各诊断脚本，全部单命令可跑。

---

## 7. 未来方向（论文 Discussion / limitation）

- pre-impact 预警在**线性因果 LR + 当前 pose 几何特征**下达到实际上限（≈0.2）；
- post-impact 确认受单目 2D 特征本征混淆限制；
- **真正突破候选**（未验证，仅备注）：非线性时序模型（在 72h 内重训风险高）、深度/多视角信息、cross-view 特征。
