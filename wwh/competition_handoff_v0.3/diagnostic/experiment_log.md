# Competition Sprint — Experiment Log（72h 冲刺）

> 自动化/人工记录。每个实验一个条目：ID / 日期时间 / 模型 / 数据 / 参数 / 指标。
> 备注：本机为 CPU（无 CUDA）、无 sklearn；评测依赖 sklearn 的部分在服务器 or 复用已冻结结果。
> 目录：`experiments/competition_sprint/`

---

## ENGINEERING-v0.3-PHASE0/1：兼容环境与轨迹交互扩展（2026-08-25）

- **性质：工程增强 / smoke，不是科研实验，不改变 EWR/FA/lead。**
- 路线：`team_handoff/ENVIRONMENT_INTERACTION_EXTENSION_PLAN.md`；v0.2 交接包保持不变，新功能默认关闭。
- Phase 0：固化 v0.2 frame/summary/config 契约，manifest=`manifests/fall_mvp_v02_contract.json`；回归测试 4/4 PASS。
- Phase 1 新增：Lighting、Clutter、Causal Linear Trajectory、Interaction、Wet-Floor unavailable 契约；单元测试 5/5 PASS。
- 主入口新增 `--enable-risk-extensions`；关闭时无 `risk_extensions` 字段；开启时只追加字段，不修改旧 fusion。
- 8 帧端到端 smoke：
  - disabled：`artifacts/fall/v03_smoke_disabled/`，旧 schema；
  - enabled：`artifacts/fall/v03_smoke_enabled/`，新增指数/路径/overlay；
  - 两个 demo 均 640×480、8 帧、可读取；
  - enabled summary max human/env/interaction = 1.8 / 53.75 / 87.4；仅工程观测，不是概率或准确率。
- Wet Floor：未接模型，输出 `available=false, state=UNKNOWN, risk_index=null`；禁止伪造水渍 mask。
- 尚未执行：全片 Demo、三类演示场景、Wet-floor checkpoint 决策门、v0.3 正式交接包。

---

## 0. 环境审计（2026-08-24 19:20，sprint_audit）

- Python 3.10.20 / torch 2.13.0+cpu / **CUDA 不可用(0卡)** / ultralytics 8.4.96 / cv2 5.0.0 / numpy 2.2.6 / sklearn MISSING
- 磁盘：E 716GB 空闲（充足）、C 32.9GB、D 107GB
- Pose 模型：`weights/pose/yolo26n-pose.pt`(7.88MB)；环境检测：根目录 `yolo26n.pt`(COCO,5.54MB)
- 数据：正确 RGB 位于 `E:\ur_fall_rgb\`（30 fall + 40 adl，已解压 PNG 目录，零下载）
  - ⚠️ 旧关键点 `runs/fall_warning/pose_export/fall_01_cam0/` 来自**错误拼接视频**（左深+右RGB 640×240），**不可用于 dropout 统计**
- Git：当前分支 `fix/fire-project-paths`（大量 Fall 资产未提交，保留不 commit）；`competition-sprint` 分支创建被权限规则拒绝，未强行操作
- 结论：Pose A/B 与全量序列推理**本机 CPU 可行**（fall-01 单序列 ≈9s）；MCFD 需下载（未开始）；sklearn 依赖任务受限于本机

## 1. hard-set manifest（build_hardset.py，2026-08-24 19:22）

- 产出：`manifests/hardset_manifest.csv`，32 序列（30 fall + 2 混淆 ADL adl-10/11）
- 可用帧：**2995**（30 fall + 2 adl），全部来自 `E:\ur_fall_rgb\`
- 用途：快速 A/B 验证梯子；**不得伪装成完整 test set**
- 不含 MCFD（未下载）

## 2. Pose 导出（竞争专用，sp_pose_export.py）2026-08-24 19:24

- 输入：正确 RGB 目录 `E:\ur_fall_rgb\<seq>\<seq>-cam0-rgb\*.png`
- 模型：`weights/pose/yolo26n-pose.pt`，ByteTrack，conf=0.25，imgsz=640，device=cpu
- 输出：`experiments/competition_sprint/results/pose_export/<seq>/keypoints.csv`
- **隔离**：不写 `runs/`，不覆盖任何 baseline
- fall-01：160 帧 / 123 人体行 / 9.1s CPU

## SPRINT-001：fall-01 正确RGB dropout 初证（2026-08-24 19:25，pose_dropout_stats.py）

| 指标 | fall-01（正确RGB） |
|---|---|
| zero_pose_ratio（全片） | 0.089 |
| **fall-window(impact±2s) zero_pose_ratio** | **0.141** |
| fall-window low-conf ratio | 0.141 |
| mean_person_conf | 0.869 |
| mean_kpt_conf | 0.863 |

- 结论：**主假设首证成立**——跌倒关键窗口 Pose 失效占比（14.1%）为全片平均（8.9%）的 ~1.6 倍。
- 证据文件：`results/pose_dropout_fall01.json`
- 对照组（错误拼接视频旧关键点）：fallwin_zero_ratio=0.0（**作废**，非真实数据）

## SPRINT-002：temporal dropout diagnostic（以人工 t_impact 为 0 点）2026-08-24 20:07

- 方法：`scripts/temporal_dropout_diagnostic.py`。每帧按 `frame_idx - impact_frame` 分窗（秒=帧/30, 30fps）；
  分别统计 zero-pose（person_conf<0.25 或有效关键点<4）与 low-conf（平均关键点置信<0.5，**DIAGNOSTIC 阈值，不冻结**）。
  ⚠️ 修正：关键点 CSV 仅记录检测到人的行，**完全无检测帧已从 frame_idx 连续性还原并计入 zero-pose**。
- 口径：正确 RGB（`E:\ur_fall_rgb\`），yolo26n-pose.pt，conf=0.25，imgsz=640，CPU。时间轴=帧/30（一阶诊断，非精确同步时间戳）。
- 结果（aggregate pooling）：

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

- **关键结论（诚实，修正早前乐观判断）**：
  - 在正式 **EWR horizon（-0.5/-0.3/-0.2s）zero_pose_rate=0.0**，Pose 无 dropout → **主假设（Pose dropout 是早预警瓶颈）在这些 horizon 未被证实**。
  - degradation 集中在 **紧贴 impact 的 pre[-0.2,0]**（0.117）与 **post-impact**（+0.2s 后陡升，+0.5~1s 达 0.50）。
  - 异常序列 fall-01/03/07/10/15/22/29 在 EWR horizon 的 zero 全部为 0；degradation 都在 post-impact。
- **决策门**：按「确认 degradation 在 t_impact 前出现才下载 s/m」→ **未满足，暂不下载 s/m 做 Pose A/B**。
- **重新定位**：degradation 主场景=post-impact/横卧检测失效，与 pre-impact 预警（EWR horizon）不同轴。
- 产物：`results/dropout_temporal_aggregate.csv`、`dropout_temporal_seqlevel.csv`、`dropout_anomaly_seq_overview.csv`、`dropout_temporal_summary.json`。
- 不表述为"已证明 Confidence-Aware Fallback 必要"（pre-impact horizon 不支持；仅 post-impact 场景成立，待确认）。

## SPRINT-003：temporal-predictor early-timing 判别力诊断（2026-08-24 20:45）

- 方法：`scripts/early_timing_diagnostic.py`，只读复用冻结 `v1_eval/audit_consistency.json` 的 `per_event_table`
  （含 onset/lie/impact + alarm 事件时间戳，pre-impact 口径），零推理。
- 目的：回答「pre-impact Pose 存在，为何 temporal predictor 未在 EWR horizon 提前报警」。

### 关键结果
- **onset→impact 时长：median 734ms；30/30 序列 > 500ms** → 理论 0.5s lead 窗口**全部成立**（EWR 上限结构上=1.0）。
- **HIT（7 个，hit_0.5=1）**：首报警相对 onset 延迟 **median 100ms**，lead **median 567ms** ✓
- **MISS（23 个，hit_0.5=0）**：首报警相对 onset 延迟 **median 267ms**，lead **median 433ms** ✗；其中 **6 序列完全无 alarm**（fall-05/13/14/21/25/27）。
- 理论可用报警窗 ≈ onset2imp − 500ms = **median 234ms**。MISS 报警延迟中位 267ms **超出 33ms** 才掉出 0.5s 窗口。

### 结论（回答因果问题）
- **不是缺信号**：pre-impact Pose 可用（EWR horizon zero_pose=0），且 30/30 有 0.5s 窗口。
- **不是 Pose dropout**：degradation 在 post-impact 才出现（SPRINT-002）。
- **是判别器的 early-timing 判别力不足**：MISS 序列报警延迟中位 267ms > 可用 234ms 门槛（差 33ms 掉窗）；且 6/30 风险分数从未达 θ（完全不触发）。
- **根因**：onset 后头 ~200ms 的 Pose 几何特征不足以让 LR 稳定超过为压 FA 而设的高阈值 θ（0.90-0.94，99 百分位）——与 v1.1 卡结论一致（瓶颈在"onset 后反应/决策层"，非缺运动信号）。
- **重新定位瓶颈**：从"信号/特征提取/Pose 可用性" → **"早期时序判别决策"**。换更大 Pose 无益（Pose 可用）；方向应指向早期判别敏感性与 FA/θ 权衡。

### 产物
`results/early_timing_diagnostic.csv`（逐序列 onset2imp / first_alarm / delay_after_onset / lead / n_events / hit）；本日志 SPRINT-003。

## SPRINT-004：sequence-level margin 诊断（修正口径，2026-08-24 20:53）

- 方法：`scripts/early_timing_margin.py`，只读复用冻结 `v1_eval/audit_consistency.json`。
- 评审修正：median 33ms 仅为线索。改为逐 sequence 判定：`deadline_i = impact_i-500-onset_i`；
  `delay_i = first_alarm_i - onset_i`；`margin_i = deadline_i - delay_i`。
  **满足 EWR@0.5 ⟺ delay_i>=0 且 margin_i>=0**（D-012 qualifying 窗 [onset, impact-0.5]）。
- **关键修正**：第一版 margin>=0 会把 alarm 早于 onset（delay<0）误判为 HIT（fall-07/08/23/04/10/20/30）。
  加入 delay>=0 后，判定与冻结归档 `hit_preimp_imp_0.5` 逐序列**完全一致（无 mismatch）**。

### 结果（theta=冻结值，30 fall）
- HIT=7；MISS(alarm)=17；MISS(never-alarm)=6。
- **MISS 并非单一问题，分三类**：
  1. **真过晚（delay>=0 且 delay>deadline）= 10 个**：fall-01/06/09/11/12/15/17/22/26/29。median delay=267ms > median deadline=201ms（差 ~66ms 掉窗）。这才是"提前不够"。
  2. **alarm 早于 onset（delay<0）= 7 个**：fall-07/08/10/20/23/30/04。警报发生在正常段（pre-onset 误报），非 qualifying（D-012 排除），性质是**误报而非来不及**。
  3. **never-alarm（无任何事件）= 6 个**：fall-05/13/14/21/25/27。可能存在 pre-impact 信号但风险分数从未达 θ。
- HIT 的 median margin = **+67ms**（阳性余量很小，勉强命中）。

### 待续（B 完整执行需逐帧 risk score）
- threshold sweep / never-alarm 最大 score / θ 权衡曲线（z Pareto）**需要逐帧 float risk score**；
  本地 `v1_eval/` 只有聚合表、无逐帧 score（已搜全仓库确认无存档）。
- ⚠️ blocker：本地无法在不重跑模型的前提下做 threshold sweep。需从服务器取逐帧 score
  （或授权在服务器做 score-only 导出，不改任何冻结产物）后方可完成 B 的 sweep/Pareto/never-alarm 分析。

### 产物
`results/early_timing_margin_seqlevel.csv`（逐序列 onset2imp/deadline/delay/margin/lead/status/归档命中交叉核对）。

## SPRINT-005：方案①重放管线验证 + 两套模型口径发现（2026-08-24 21:40）

- 方法：`sp_replay_validate.py`——用 `replayed/cam0_models/fold*.json`（coef/intercept/θ/feat_cols/test_falls）构造 FrozenLR，
  复用 `evaluate_v1.eval_fold`（posture/ts 由 features.tsv 行内 posture_label/ts_ms 重建），t_impact 来自 `impact_annotations.csv`。
- **重放管线准确**：复现 `EWR@0.5=0.1667 / FA=4.8346 / lead=0.567` —— **与冻结 Arm R (prediction-frozen control) 逐位一致**（impact 0.167/4.8346）。
- ⚠️ **关键发现：cam0_models = proxy 模型，非正式 pre-impact 基线**：
  - `cam0_models/fold*.json` 的 C/θ（fold0 0.01/0.7383、fold2 10/0.9657、fold3 0.01/0.7333、fold4 0.01/0.6558）
    **= proxy 口径 `v1_lr_eval.json` 的 per_fold**（frozen 于 pre-lying proxy 阶段，Arm R 复现用）。
  - 正式 pre-impact 基线（`v1_lr_eval_preimpact.json` 0.233/6.81）用**另一组 impact-selection C/θ**：
    fold0 0.01/0.7383、fold1 0.01/0.6762、fold2 0.01/0.7237、fold3 10/0.7196、fold4 10/0.5375。
  - **正式基线模型的 coef/intercept 权重未存档**（v1_lr_eval_preimpact.json 仅含聚合 + per_fold C/θ 摘要，无逐帧、无权重）。
- **影响**：无法直接对正式 frozen baseline（0.233/6.81）做逐帧 sweep（其 impact-selection 模型权重缺失）。
  可用的只有 proxy 模型（Arm R 口径）分数，其 sweep 结果**只能作为 Arm R 口径诊断**，不能冒充正式基线结论。

### 待决（需用户/服务器确认）
- 服务器是否存在 **impact-selection 模型权重存档**（另一组 cam0_models / v1_eval_preimpact 下的 model json）。
- 若无：B 的 threshold sweep 只能基于 proxy（Arm R）模型，结论标注为 Arm R 口径；或需重新 fit（违反"不重训"约束）。

## SPRINT-006：proxy(Arm R) 模型统一 percentile threshold sweep（B，2026-08-24 22:05）

- 方法：`sp_replay_sweep.py`。5 fold proxy 模型 + 统一 risk percentile（跨 fold 可比），θ_k=percentile(fold_k ADL OOF risk, p)，
  p∈{99,97.5,95,90,85,80,75,70,60,50}；每 fold 评估其 test_falls(EWR)+test_adls+fall-pre-onset(FA)，聚合 OOF。
- **口径标注**：proxy / Arm R 模型（非正式基线 0.233/6.81，其 impact-selection 权重未存档）。
- 结果（`results/sweep_proxy_percentile.csv`、`sweep_proxy_detail.csv`）：

| p | EWR@0.5 | EWR@0.3 | EWR@0.2 | FA/min | never-alarm | n_qual |
|---|---|---|---|---|---|---|
| 99 | 0.067 | 0.167 | 0.233 | 3.08 | 13 | 2 |
| 97.5 | 0.100 | 0.233 | 0.333 | 4.61 | 10 | 3 |
| 95 | 0.067 | 0.233 | 0.333 | 7.69 | 6 | 2 |
| **90** | **0.200** | 0.367 | 0.400 | **11.43** | 3 | 6 |
| 85 | 0.167 | 0.300 | 0.333 | 17.14 | 3 | 5 |
| 80 | 0.100 | 0.200 | 0.233 | 19.99 | 1 | 3 |
| 75 | 0.067 | 0.233 | 0.233 | 23.29 | 0 | 2 |
| 50 | 0.167 | 0.200 | 0.200 | 22.63 | 0 | 5 |

- **关键观察**：
  1. 即便把 percentile 降到 **50%（最松）**，EWR@0.5 也**从未超过 0.2 (6/30)**，达不到正式目标 0.233；
  2. never-alarm 从 99% 的 13 降到 50% 的 **0**（都触发了）**但 EWR@0.5 没同步提升** → 报警发生了，却**太晚**（错过 0.5s 窗口）；
  3. 降 θ 只换来看似合理的 EWR（0.2 在 p=90）但 FA 从 3 爆炸到 11-23；**没有 FA 可接受的提升区间**。
- **knee point**：无。任何"EWR 上升"都伴随 FA 陡增，且 EWR 上限卡在 0.2，无显著膝点。
- **结论（单判）→ B3**：大量序列即使显著降低 θ（到 50 percentile、never-alarm 清零）仍无法在 0.5s 窗口命中
  → **pre-impact score 本身的时序判别性不足**，阈值不是根本解法。与 SPRINT-003/004（early-timing 判别力不足）相互印证。
- **口径限制（诚实）**：结论基于 proxy/Arm R 模型（EWR@0.5 起点 0.167），非正式基线 0.233。proxy 与正式基线同族同协议，
  趋势作为强线索；正式基线权威 sweep 需其权重，未存档（重 fit 违反"不重训"约束）。
- 图：`figures/pareto_proxy_ewr_vs_fa.png`、`figures/pareto_proxy_p_sweep.png`（ENGINEERING DIAGNOSTIC）。
- 本阶段结束，**按指令不自动开始 A**，等待 Human Gate。

## SPRINT-007（A-1）：早期判别力诊断（授权后，2026-08-24 22:25）

- 方法：`a1_early_discrimination.py`（纯 numpy AUC，零推理）。早期窗 [onset, onset+0.3s] 内 fall 帧 vs 对照
  （ADL 全帧 + fall pre-onset，与 D-012 负暴露一致）。逐特征 AUC + Cohen's d。
- 结果：**31/72 特征具显著早期判别力**（|AUC−0.5|≥0.12）；Top 特征几乎全为 **_min / _delta 的 y（垂直）方向**：
  hip_mid_y_min AUC=0.771(d=1.15)、rh_y_min 0.711、body_h_min 0.694、body_h_delta 0.680、torso_ok_min 0.673、
  nose_y_min 0.668、ls_y_min 0.668、torso_ok_delta 0.650。
- **信号解读**：onset 后 0.3s 内人体**垂直下沉**已显著区分摔倒 vs 正常（y-min/delta），是 v1.0/v1.1
  **未聚焦利用**的早期信号（v1.1 用全序列运动特征、未用窗口极值）。
- 产物：`results/early_discrimination.csv`。

## SPRINT-008（A-2）：轻量早期判别 LR（numpy IRLS 实现，2026-08-24 22:40）

- 方法：`a2_early_lr.py`。本地无 sklearn → **numpy IRLS(Newton+l2) 实现 LR**。early-win=0.3s。
  训练目标：positive = fall 且 posture_label==0 且 frame∈[onset, onset+0.3s]（早期 falling 帧）；
  negative = ADL 全帧 + fall pre-onset。sequence-level 5-fold、因果、outer-locked（C 折内 Brier 选、
  θ 折内 ADL FA≤6.81），复用 evaluate_v1 alarm/ew_rate 评估，impact=人工 t_impact。
  两个变体：A1-判别特征子集（AUC≥0.58，11 特征）vs 全 72 维。

| 变体 | EWR@0.5 | EWR@0.3 | EWR@0.2 | FA/min | lead median |
|---|---|---|---|---|---|
| **frozen v1.0** | **0.233** | 0.400 | 0.500 | 6.81 | 0.567 |
| A2 判别子集(11) | 0.167 | 0.333 | 0.400 | 6.35 | 0.600 |
| A2 全特征(72) | 0.200 | 0.333 | 0.367 | **10.07** | 0.651 |

- **结果（诚实，负/弱结果）**：
  - 两变体 EWR@0.5 均**未超** frozen v1.0（0.167 / 0.200 < 0.233）；
  - 判别子集版 FA 略降(6.35)但 EWR 更低；全特征版 FA 恶化到 **10.07**；
  - lead median 略升（0.60/0.65），即"命中时提前更多"，但**命中数(EWR)下降**。
- **结论**：早期判别特征（A-1 证明存在信号）+ 早期目标 LR **无法把信号转化为更高 EWR@0.5**。
  → 进一步印证 B3：瓶颈不是"缺判别特征"，而是**线性 LR 在 0.5s 边界内的置信/时序决策不足**。
  （A-1 判别力 AUC 最高 0.771 单特征，但组合后 LR 仍无法稳定边界命中。）
- **层级判定**：早期判别特征路线 = **无提升**；与 v1.1/v1.2/A-1 合并显示"v1.0 LR 家族 + 1.0s 因果窗下
  EWR@0.5≈0.2-0.23 已接近该类线性因果方法实际上限"。真正提升或需新模态/非线性时序模型（72h 内重训风险），
  或聚焦 post-impact 可展示方向。
- 产物：`results/a2_early_lr_result.json`、`a2_early_lr_allfeats_result.json`。

## SPRINT-009（P-1/P-2）：post-impact 确认信号诊断（2026-08-24 22:50）

- P-1：导出 11 个代表性 ADL 关键点（含躺下 adl-10/11/34/35/39、坐/弯/蹲类），`results/pose_export/adl-*`。
  ⚠️ ADL RGB 需先解压 zip（`sp_pose_export_adl.py`）；已清理先前空导出。
- P-2：`p2_postimpact_diag.py`。fall post-impact 段（impact..impact+2s）vs ADL 全序列 zero-pose 对比。

### 关键结果（诚实，修正"简单 fallback"预期）
- **fall post-impact: zero_pose median=0.343 mean=0.377**；**pre-impact(0-0.5s): median=0.000**（fall 前 Pose 可靠）。
- **ADL whole-seq: zero median=0.235**（躺下/弯/蹲类 adl-32=0.34、adl-21=0.278、adl-39=0.30、adl-35=0.26、adl-31=0.244）。
- **fall post-impact 内部双峰**：一半真丢失（0.5-0.94：fall-01/02/04/09/10/15/17/22/23/29），
  一半几乎不丢（0-0.11：fall-06/08/11/13/14/16/20/24/26）。
- **结论：单纯「post-impact zero-pose」无法区分 fall vs ADL**（重叠严重，且 fall 有一半不倒伏丢失、ADL 躺下也丢人）。
  → post-impact 确认**不能只靠零姿态**，需组合「**快速下降运动证据** + 终态丢人/停留」：
    fall = 先有高速下降（impact 前 last-pose 几何已倒地）→ 后 Pose 丢失；
    ADL 躺下 = 缓慢坐下→躺（无快速下降段）→ 虽丢人但不确认。
- 产物：`results/postimpact_diag.json`。

## SPRINT-010（P-3 第一轮）：Post-impact Fall Confirmation 两阶段 FSM（2026-08-24 23:00）

- 实现：`p3_confirmation.py`。两阶段（Stage1 rapid descent = 5帧窗 bbox 高度归一化骤降；Stage2 terminal =
  zero/横卧/低位持续 TERMINAL_PERSIST=6 帧），因果、无 GT leakage（t_impact 仅离线算 delay）。
  统一先验阈值（THRESH，记录来源，未做数据驱动调参）。三消融 A terminal-only / B descent-only / C combined。
- ⚠️ 第一轮实现缺陷如实记录：**Stage1 descent 用 5帧窗口 bbox-h 骤降 >0.4 → 30 fall 中仅检出极少数**。
  调试显示 fall 的 bbox-h 是从站立(≈350px)到倒地(≈120px) **平缓下降 ~1s（not 瞬时）**，5帧窗口内 drop<0.4；
  且 ADL（如 adl-10 站起 h→480 再蹲/坐 h→122）**同样出现 bbox-h 大幅下降** —— **bbox-height 骤降与 Pose 丢失一样并非 fall-specific**。
- 结果（30 fall + 11 targeted hard-ADL）：

| mode | fall_rec | adl_FC | fall_hit | adl_hit | med_delay_s |
|---|---|---|---|---|---|
| A_terminal_only | 0.667 (20/30) | 0.909 (10/11) | 20 | 10 | 0.45 |
| B_descent_only | 0.000 | 0.182 (2/11) | 0 | 2 | — |
| C_combined | 0.000 | 0.091 (1/11) | 0 | 1 | — |

- **诚实解读**：
  1. A(terminal-only)：ADL 误确认 0.909 → 印证 P-2，Pose degradation 绝非 fall-specific。
  2. B/C：descent 当前几乎不检出 fall（rec 0），**无法评估 combined 是否优于 pose-only**——因 combined 把 fall 也全丢了。
  3. **关键分析**：即便修复 descent 实现参数，bbox-height 骤降在 ADL「快速坐下/蹲」同样出现（adl-10/21/31 等），
     因此「快速下降 + 终态丢失」仍可能不充分区分「踉跄跌倒」vs「快速坐下」。Pose degradation 与 rapid descent
     **两条信号都指向同一类本征模糊（坐/蹲/躺 ADL 与跌倒共享大部分视觉-几何特征）**。
- 产物：`results/p3_ablation.csv`、`p3_sequence_results.csv`、`p3_failure_cases.csv`。
- 定位重申：Post-impact confirmation，非 pre-impact，不计入 EWR@0.5。

## SPRINT-011（P-3 唯一实现修正 v2）：Stage1 窗口 5帧→1.0s（2026-08-24 23:05）

- 唯一改动：`p3_confirmation_v2.py` Stage1 `descent_event` 窗口=round(fps*1.0)=30帧(1.0s)；其余阈值不变；
  terminal_event 低位窗口保持 5（解耦，保证 A 复现）。新增 Stage1 有效性统计。输出 `results/p3_v2/*`。
- **A 复现成功**：terminal-only = 0.667/0.909（与第一轮一致）✓。

### v2 消融（30 fall + 11 hard ADL）

| mode | fall_rec | adl_FC | fall_hit | adl_hit | med_delay_s |
|---|---|---|---|---|---|
| A_terminal_only | 0.667 | 0.909 | 20 | 10 | 0.45 |
| B_descent_only | 0.933 | 0.909 | 28 | 10 | 0.067 |
| **C_combined** | **0.633** | **0.455** | 19 | 5 | 0.467 |

### Stage1 diagnostics
- **descent 触发：fall 28/30（可抓）但 hard ADL 10/11（大量误触发）**；
- fall first-descent 相对 impact：median 0.067s（几乎在 impact 时刻），min -0.133 max 0.933；
- fall 触发序列：fall-01..29（缺 fall-13/30）；adl 触发：adl-10/11/21/22/30/31/32/33/34/35（缺 adl-39）。

### 结论 → NO-GO（按 Gate）
- Combined rec=0.633 ≥ 0.60 ✓；但 **Combined hard-ADL FC=0.455 > 0.273 ✗** → **NO-GO**。
- **主要失败机制**：descent 修复后确实能抓 fall（28/30），**但同样大量触发 hard ADL（10/11）**——
  「bbox-h 骤降 + 终态丢失」在快速坐下/蹲/躺 ADL 同样成立。combined 靠加 terminal 把 adlFC 0.909→0.455
  （**showing real specificity gain vs terminal-only**），但仍残留 5/11 hard ADL 误确认（坐下/蹲类无法排除）。
- **最终科学结论（冻结）**：Pose degradation 与 bbox-height 下降**均非 fall-specific**；单目 2D bbox/keypoint
  层面，fall 与 sit/squat/lie ADL 存在本征混淆（诚实的 negative finding，作论文 limitation）。
- 停止：**不进行 P-3 第三轮**；不换 post-impact feature；不训练复杂 post-impact 模型。P-2/P-3 作为 diagnostic/limitation 固化。

---
