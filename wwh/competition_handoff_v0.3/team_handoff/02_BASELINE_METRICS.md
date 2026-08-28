# 02 — Frozen Baseline 指标

> **诚实原则：没有的数据写 TBD / N/A，不虚构。**
> 此处区分两类对象，避免队友混淆：
> - **(A) 科学 frozen baseline**（v1.0 PREIMPACT）：有指标，见下方 §1。
> - **(B) `fall_mvp` 工程原型**（motion_heuristic_v0）：**无科研指标**，见 §5。

## 1. Baseline 名称

- **v1.0 PREIMPACT**（科学 frozen baseline，engineering evaluation 正式口径）。
- 来源卡：`results/fall_warning/FALL-EX-001_PREIMPACT_RESULT_CARD.md`（hash `c14176647e9997d1`）。
- 归档：`artifacts/fall/V1_0_ARCHIVE.md`。
- 注：这不是 `fall_mvp` 工程原型；`fall_mvp` 用 `motion_heuristic_v0` 并非代表 v1.0 被否定，而是「scientific evaluation artifact ≠ deployment artifact」（see `fall_mvp/HANDOFF.md §7`）。

## 2. 数据集（Dataset）

- **UR Fall Detection Dataset**（University of Rzeszow）。
- 30 fall（demo 用）、40 ADL（活动）序列（数据集完整清单见 `artifacts/fall/UR_FALL_DATASET_MANIFEST.md`）。
- 评价采用 camera-0（`--camera0-only` 等协议定义见 v1.2 协议，PREIMPACT 卡为主）。
- 外部补充数据（如 MCFD）**未**包含在本 baseline 指标内（MCFD 未下载，见 `05` / MCFD 交接）。

## 3. Evaluation protocol

- **仅使用当前帧与历史帧（pre-impact / 因果）**；禁止未来帧泄漏。
- 事件参考时间：人工标注 `t_impact`（impact frame），非 `t_lie`（proxy 为 t_lie 口径，见 V1_0_ARCHIVE）。
- 主评价协议：D-012 定义（EWR@0.5s + FA events/min + lead median/IQR）。
- 前置条件：参考耦合超参选择 / cascade-forecast 等严格 protocol 见 `FALL_V11_PROTOCOL.md` / `FALL_V12_PROTOCOL.md`。
- 评价方式：sequence-level nested/5-fold 模型 + fold-specific thresholds（见 §6 deployment gap）。
- 说明：**这是 v1.0 的科学评价协议**；`fall_mvp` 工程原型未走此协议（见 §5）。

## 4. 已确认指标（Confirmed metrics）

| 指标 | 值 | 单位/口径 | 来源 |
|---|---|---|---|
| EWR@0.5 | **0.233** | pre-impact，t_impact 参考，qualifying alarm ∈ [onset, t_impact−0.5s] | V1_0_ARCHIVE / 冻结卡 |
| False Alarm | **6.81** | **FA/min**（{FA} 事件每分钟，31/4.5506） | METRIC_CONSISTENCY_AUDIT |
| lead median | 0.567 s | pre-impact lead（t_impact − t_alarm），median | V1_0_ARCHIVE |
| EWR@0.3 / EWR@0.2 | 见 note | 附加报告 | V1_0_ARCHIVE |

- **EWR@0.5 = 0.233**：即 30 个 fall 中 7 个命中（7/30=0.2333）。
- **FA = 6.81**：单位是 **events / min**（每分钟误报事件数），**不是**事件总数。总数 = 31，分母 4.5506 min。
- EWR@0.3、EWR@0.2 的对应值：v1.2/诊断链中引用（EWR@0.3=0.4、EWR@0.2=0.5 在 METRIC_CONSISTENCY_AUDIT 对账出现），但以各自卡为准；本表只冻结主指标 EWR@0.5=0.233 与 FA=6.81。

## 5. `fall_mvp` 工程原型（B）的指标状态

- **EWR / FA / Precision / Recall / F1 / lead time**：**N/A**（未按科学协议评价，属于 engineering prototype）。
- 工程 smoke（非科研）：`artifacts/fall/mvp_final/smoke_test_report.md` —— 仅证明 3 个输入可跑通、JSON 可解析、schema 一致；**不含任何科研指标声明**。
- 真实样本 **HIGH=0**：为如实结果，未通过调参修复；HIGH 分支仅 synthetic/unit 证明可达。

## 6. UNKNOWN / 尚未确认（TBD）

- **Deployment-model gap**：frozen v1.0 科学评价经 sequence-level 5-fold 模型 + fold-specific thresholds，**不存在单一冻结部署预测器**；构建 all-data deployment/ensemble 属新artifact，需独立验证。**当前 TBD / 未解决。**
- **Precision / Recall / F1**：本 baseline 未以这些指标报告（EWR/FA/lead 为项目协议）；如比赛需要，TBD（等主 Session 提供）。
- **真实同步时间戳**：`fall_mvp` 入口用帧号×1000/fps 估计；HR 级 precision 需回填 metadata timestamp_ms（见 METADATA_TIMEALIGN_AUDIT），**TBD**。

## 7. 已知局限（Known limitations）

1. **Real-sample HIGH=0**：真实 RGB 序列上工程原型未触发 HIGH；仅 synthetic/unit 证明分支逻辑可达。
2. **Deployment gap**：无单一冻结部署预测器（见 §6）。
3. **不是校准概率**：motion / environment / overall risk 均为工程启发分。
4. **单目近似**：`d_norm` 非米制距离；env 分数对家具密集场景敏感。
5. **多人物简化**：确定性主人物。
6. **v0.1 输入**：video / RGB 帧目录；webcam `--input 0` 未验证。

## 8. 历史更精确定义来源（记录，不重新计算）

- FA/min 数值 4.83 vs 6.81 的差异来源：`METRIC_CONSISTENCY_AUDIT_2026-08-22.md` —— **4.83 (Arm R diagnostic, 22/4.5506) 与 6.81 (v1.0 PREIMPACT, 31/4.5506) 均为正确 FA/min**，差异完全来自参考耦合超参再选择（t_lie→t_impact 使 θ 下降 → FA 事件 22→31）。**请勿混用两者**。
- proxy 口径（t_lie）EWR@0.5=0.433 / FA=4.83 是不同口径，**不等于** pre-impact 0.233 / 6.81。
