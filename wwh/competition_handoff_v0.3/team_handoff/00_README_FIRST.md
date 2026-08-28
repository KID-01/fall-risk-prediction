# 00 — 队友先读：Frozen Engineering Baseline + Preliminary Diagnostic

> 本目录是 **竞赛交付套件** 的队友入口。请先读本文件，再按编号顺序阅读其余文档。

## 这是什么（重要）

**这是「Frozen Engineering Baseline + Preliminary Diagnostic」，不是最终比赛模型。**

- **Frozen Baseline**：`fall_mvp/` v0.1 工程原型（可运行、可演示的 engineering prototype）。
- **Preliminary Diagnostic**：`experiments/competition_sprint/` 中已完成的一次 **engineering diagnostic**（Pose temporal dropout 时间分布分析）。

两者都不是「最终比赛模型」，也都不代表「科研上已证明的跌倒预警改进」。

## 为什么交付它

- 当前**主科研 Session 仍在运行**，真正的 final model / 指标 / ablation **尚未产生**。
- 本套件只移交「当前已冻结、且可交付给队友」的工程资产与诊断记录，**不代替**最终比赛结果。
- 主 Session 的后续结果一旦就绪，会另行追加；本套件 **不覆盖、不重复** 其工作。

## 文件清单与阅读顺序

| 编号 | 文件 | 内容 |
|---|---|---|
| 00 | `00_README_FIRST.md` | 本文件：总览与边界 |
| 01 | `01_SYSTEM_ARCHITECTURE.md` | 按真实代码总结的系统架构与数据流 |
| 02 | `02_BASELINE_METRICS.md` | frozen baseline 已确认指标与协议（含 TBD） |
| 03 | `03_CONFIRMED_FINDINGS.md` | Pose temporal dropout diagnostic：假设/方法/证据/结论/局限 |
| 04 | `04_PAPER_WRITING_BOUNDARIES.md` | 论文写作边界（能写 / 只能 preliminary / 等待最终实验 / 禁止声称） |
| 05 | `05_PENDING_RESULTS.md` | 仍等待主科研 Session 提供的结果 |
| 06 | `ENVIRONMENT_INTERACTION_EXTENSION_PLAN.md` | v0.2 之后的兼容环境/轨迹交互 v0.3 路线；后续 AI 实现前必读 |

## 三个必须记住的硬边界

1. **这是 engineering baseline，不是最终模型、不是科学验证系统。** 所有 score（motion / environment / overall risk）均为**工程启发分，不是校准跌倒概率**。
2. **诊断结论目前只是 preliminary finding，不是 proven improvement。** 不得声称已提升 Precision / Recall / F1 / EWR。
3. **科学 frozen 指标（EWR@0.5=0.233、FA/min=6.81）属于 v1.0 PREIMPACT 科学 baseline**，与 `fall_mvp` 工程原型（motion_heuristic_v0）是两个不同事物；后者**不等于**前者，见 `02` 与 `fall_mvp/HANDOFF.md §7`。

## 本套件的生成信息

- 生成时间：见 `artifacts/handoff/competition_handoff_v0.1/README.md`。
- 来源 commit/hash：`fall_mvp/` 未纳入 git（untracked），无 commit hash；仓库 HEAD = `dcc3a47`（分支 `fix/fire-project-paths`）。科学卡 hash = `c14176647e9997d1`（v1.0 PREIMPACT 卡）。
