# 04 — 论文写作边界（Paper Writing Boundaries）

> 本文件在写论文/报告时**必须遵守**。它防止把 preliminary diagnostic 写成 proven 结论。
> 三条核心红线见底部；每条陈述都按其可信度归入四个档次。

## A. 现在可以写进论文（已确认、可直接引用）

1. **Frozen v1.0 PREIMPACT baseline**（科学口径）：`EWR@0.5 = 0.233`、`FA/min = 6.81`、`lead median = 0.567 s`（t_impact 参考；URI：FALL-EX-001_PREIMPACT_RESULT_CARD.md，hash `c14176647e9997d1`）。
2. **系统性架构**：`fall_mvp` v0.1 的输入 / Pose(motion) / YOLO(env) / env_risk / risk_fusion / causal_persist / 输出（见 `01`）。
3. **确定性融合规则**（motion 是主赢警信号；环境单独不能 HIGH；UNKNOWN ≠ LOW）作为工程实现事实可描述。
4. **诊断方法学**：基于人工 `t_impact` 的 temporal-dropout 分窗统计方法，可作为方法描述（`pre[-0.5,-0.3]` 等窗口定义）。
5. **缺陷/局限作为工程事实**：real-sample HIGH=0、deployment-model gap、单目 `d_norm` 非米制、帧/30 一阶时间轴。
6. **数据来源**：UR Fall Detection Dataset（30 fall + 40 ADL），camera-0，评价口径 t_impact pre-impact。
7. **数据处理事实**：正确 RGB 目录 `E:\ur_fall_rgb\`；旧关键点（错误拼接视频）**作废，不可用作证据**。

## B. 只能作为 Preliminary Observation（preliminary finding）

1. **Pose temporal dropout 的时间分布**，且**仅作为描述性观察**：
   - 在正式 EWR horizon（-0.5/-0.3/-0.2s）`,`zero_rate=0.0`；
   - degradation 集中在 `pre[-0.2,0]` 与 post-impact（+0.5~1s 达 0.50）。
   - 可写为「我们观察到……的初步迹象」，**不可**写为「证明」或「原因」。
2. **Pose 无法在 EWR horizon 解释 early-warning 失败**——这是「单模型、hardset、单瓶颈」的观察，**不是**最终结论。
3. **post-impact pose degradation** 是值得进一步研究的现象，但当前只作为 **observation / future work**，不写成系统验证结果。
4. 任何「confidence-aware fallback 可能有用」的表述，**只能用于 post-impact confirmation 语境**，并注明「仅在 post-impact 场景观察到，pre-impact horizon 不支持，待确认」。

## C. 等待最终实验后再写（不可提前写）

1. **Precision / Recall / F1 / EWR / FA/min / lead time** 的最终值 —— 全来自主科研 Session，**尚未产生**，见 `05_PENDING_RESULTS.md`。
2. **Ablation / A/B 对比**（尤其 YOLO Pose s/m A/B：**当前被否决/暂缓，未做**）。
3. **更大模型（yolo26s/m/l/x）是否改善** —— **未做**。
4. **MCFD 上评估** —— **MCFD 尚未下载**，无任何 MCFD 结果。
5. **Pre-impact early-warning 是否被显著改善** —— 无最终实验支撑。

## D. 禁止声称（绝对红线）

1. **禁止声称：Pose dropout 已被证明是 early-warning failure 的主要原因。**（诊断结论：EWR horizon zero_rate=0.0，假设未被证实。）
2. **禁止声称：更大的 YOLO（或任何 s/m/l/x 模型）已提高效果。**（A/B 未做。）
3. **禁止写不存在的 A/B/C/D 数字。**（不得臆造 EWR/FA/Precision/Recall/F1 对比表。）
4. **禁止声称已提升 Precision / Recall / F1 / EWR。**（诊断未做任何这类提升验证。）
5. **禁止把 `fall_mvp` 的 motion_environment/overall score 当作校准跌倒概率。**
6. **禁止把 proxy（t_lie）口径 0.433/4.83 与 pre-impact（t_impact）0.233/6.81 混用。**
7. **禁止把 old keypoints（错误拼接视频）当作证据。**
8. **禁止把 engineering smoke / synthetic 单测当作真实性能。**

## E. 三句话记牢（队友务必传阅）

- **「这是 Frozen Engineering Baseline + Preliminary Diagnostic，不是最终比赛模型。」**
- **「Pose dropout 尚未被证明是 early-warning 失败主因，更大的 YOLO 尚未被证明更优。」**
- **「凡是主科研 Session 尚未产出的指标（P/R/F1/EWR/FA/lead/ablation/final demo），一律不能写进论文，只能等待。」
