# Fall Risk Competition Handoff v0.2

> 这是当前竞赛冲刺的统一交接包：**可运行工程原型 + 冻结科学 baseline + SPRINT-002–011 诊断素材**。
> 它不是已经工业验证的跌倒预测模型，也不包含任何校准跌倒概率。

## 先看什么

1. `team_handoff/00_README_FIRST.md`
2. `team_handoff/01_SYSTEM_ARCHITECTURE.md`
3. `team_handoff/02_BASELINE_METRICS.md`
4. `team_handoff/04_PAPER_WRITING_BOUNDARIES.md`
5. `diagnostic/paper_materials.md`
6. `diagnostic/experiment_log.md`
7. `submission_docs/SYSTEM_DESIGN_AND_TECHNICAL_DEVELOPMENT.md`
8. `submission_docs/FUNCTIONAL_TEST_REPORT.md`

本 README 是 v0.2 包的总入口；如果旧文档中写着“主 Session 尚未完成”，以本 README 和 `diagnostic/` 为准。

## 能演示什么

`demo/demo_baseline.mp4` 展示完整工程链路：

```text
RGB/video -> Pose/Motion -> YOLO environment objects
         -> person-object proximity risk -> deterministic fusion
         -> LOW / MEDIUM / HIGH / UNKNOWN + overlay video
```

运行源码（需要队友机器自行准备权重）：

```bash
python -m fall_mvp.run --input <video_or_rgb_frame_dir> --output <output_dir>
```

工程输出不是跌倒概率。`UNKNOWN` 表示信息不足，不等于 LOW。当前真实 MVP 样本的 HIGH=0 已在 smoke report 中如实记录。

## 当前科学基线与结论

- Frozen v1.0 PREIMPACT：EWR@0.5=`0.233 (7/30)`、FA/min=`6.81`、lead median=`0.567s`。
- P0-A IMU-only：NO-GO；不进入 fusion。
- SPRINT-002–008：Pose dropout、early-timing、threshold、early-feature LR 的系统诊断；没有宣称性能提升。
- P-3 v2 post-impact confirmation：combined recall=`19/30=0.633`，targeted hard-ADL false-confirmation=`5/11=0.455`，未达到预设门槛，NO-GO。
- hard-ADL 是 targeted 子集，不是完整 ADL test-set specificity。

**答辩必须区分**：pre-impact early warning 与 post-impact confirmation；P-3 结果不能计入 EWR@0.5。

## 适合论文和答辩的主线

不要声称“我们准确率最高”。当前最可信的差异化是：

1. 姿态 + 环境危险物 + 人物邻近关系 + 可解释风险融合的完整系统；
2. 严格区分提前预警和跌倒后确认，没有把事后检测冒充提前预警；
3. 用人工 `t_impact`、因果窗口、EWR/FA/lead 对瓶颈进行可复核分析；
4. 明确报告负结果和限制，而不是隐藏 UNKNOWN、误报或 hard-ADL 混淆；
5. 模块化接口允许队友继续接入 audio/depth/更强时序模型。

推荐答辩句式：

> 我们不仅做了一个能运行的演示系统，还验证了它在什么时间段、什么类型的行为上可靠，以及为什么当前规则不能把跌倒与坐下、蹲下、躺下完全分开。

## 外部依赖（未打包）

以下内容因体积/科研资产保护未放入 ZIP：

- `weights/pose/yolo26n-pose.pt`
- 根目录 `yolo26n.pt`
- `E:\ur_fall_rgb` / UR-Fall 原始帧和 CSV
- `experiments/competition_sprint/results/pose_export/` 大型逐帧关键点缓存
- 服务器上的 `features.tsv`、模型权重和原始数据

队友需要按实际机器准备权重；不得把原始数据集或模型权重提交 Git。

## 资产边界

- `mvp_source/`：工程代码，可作为队友开发入口。
- `diagnostic/`：论文/答辩素材；所有 proxy sweep 必须标为 engineering diagnostic/proxy 口径。
- `demo/`：可播放的工程演示与 smoke evidence。
- `submission_docs/`：面向赛题提交的正式系统设计/技术开发文档、功能测试报告和机器可读测试结果。
- 不要把 `motion_heuristic_score`、`environment_risk_score`、`overall_risk_state` 说成校准概率。
- 不要把 hard-ADL 结果说成完整 ADL specificity。
- 不要声称 Pose dropout 已被证明是 pre-impact 失败主因；诊断显示正式 EWR horizon 内 zero-pose=0。
- 不要声称更大 Pose、MCFD 或 audio 已经带来改进；这些没有正式结果。
