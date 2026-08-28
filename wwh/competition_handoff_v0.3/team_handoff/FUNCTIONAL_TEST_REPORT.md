# 系统功能测试报告

## 1. 报告目的

本报告验证当前 `fall_mvp` 工程原型的部署、运行、输入处理、模块输出、融合分支和产物生成能力，支撑赛题要求中的“可实现、可运行、可验证”。

本报告是**工程功能测试**，不是科研性能报告，不把 smoke/单元测试当作 Precision、Recall、F1、EWR 或临床有效性证据。

## 2. 测试对象

- 入口：`python -m fall_mvp.run --input <source> --output <dir>`；
- Pose 模型：`weights/pose/yolo26n-pose.pt`；
- 环境模型：`yolo26n.pt`（COCO 预训练）；
- 输入：正确 UR-Fall RGB 帧目录；
- 输出：`frames.jsonl`、`summary.json`、`config_snapshot.yaml`、`demo.mp4`。

## 3. 测试环境

| 项目 | 实测值 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.13.0+cpu |
| CUDA | 不可用；本次工程 smoke 使用 CPU |
| Ultralytics | 8.4.96 |
| OpenCV | 5.0.0 |
| NumPy | 2.2.6 |
| 输入分辨率 | 640×480 正确 RGB |
| 模型推理尺寸 | `imgsz=640` |
| 输出视频 | MP4，overlay |

## 4. 测试方法与命令

### 4.1 端到端运行

```bash
E:\anaconda\conda_envs\yolo\python.exe -m fall_mvp.run \
  --input E:\ur_fall_rgb\fall-01 \
  --output artifacts\fall\mvp_final
```

同样流程在 fall-02、fall-03 上执行，分别生成独立输出目录。

### 4.2 输出结构检查

对每个输出目录检查：

- `frames.jsonl` 是否存在且每行可解析为 JSON；
- 顶层字段是否包含 `timestamp/person/motion/environment/fusion/quality`；
- `summary.json` 是否可解析且结构一致；
- `config_snapshot.yaml` 是否存在；
- `demo.mp4` 是否存在且可读取。

### 4.3 模块 sanity 测试

`fusion.fuse` 的确定性规则分支检查：

| 输入 | 预期输出 |
|---|---|
| motion=0.8, env=0.9 | HIGH |
| motion=0.5, env=0.9 | HIGH |
| motion=0.1, env=0.9 | LOW + context_elevated |
| motion=None, env=0.9 | UNKNOWN + motion_missing |

这些是逻辑 sanity，不是真实样本性能。

## 5. 测试结果

### 5.1 端到端样本结果

| 样本 | 总帧数 | 有效人物帧 | LOW | MEDIUM | HIGH | UNKNOWN | 最高环境风险 | 主要危险物 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fall-01 | 160 | 150 | 111 | 4 | 0 | 45 | 0.935 | chair(282), couch(9) |
| fall-02 | 110 | 86 | 72 | 4 | 0 | 0 | 0.997 | chair(171), suitcase(2), couch(1) |
| fall-03 | 215 | 215 | 215 | 0 | 0 | 0 | 1.000 | chair(436), suitcase(4), couch(2) |

### 5.2 测试结论

| 测试项 | 结果 | 说明 |
|---|---|---|
| 输入帧读取 | PASS | 3 个正确 RGB 样本均完成处理 |
| 视频/帧目录入口 | PASS | 本次验证覆盖帧目录；视频入口由代码实现支持 |
| Pose/Motion 分支 | PASS | 输出可写入 JSONL，缺失状态可传播 |
| 环境物体识别 | PASS | 检测到 chair、couch、suitcase 等 COCO 类别 |
| 人物-环境匹配 | PASS | 输出 person match 和 quality 状态 |
| 邻近风险计算 | PASS | 输出 environment score、normalized distance、top hazards |
| 风险融合 | PASS | LOW/MEDIUM/UNKNOWN 分支及 synthetic HIGH sanity 均可达 |
| JSONL schema | PASS | 3 个样本顶层字段一致、可解析 |
| summary 生成 | PASS | 帧统计和危险物统计可读取 |
| config 快照 | PASS | 实际配置写入输出目录 |
| demo 视频 | PASS | 3 个样本均生成 overlay MP4 |
| 连续运行稳定性 | PASS | 3 个样本无 crash、无 missing-file |
| 真实 HIGH 触发 | OBSERVED ZERO | 当前 3 个真实样本均 HIGH=0；这是观测结果，不是测试失败修复项 |

## 6. 异常与边界测试状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 空输入目录 | NOT EXECUTED | 尚未形成独立测试证据 |
| 不存在输入路径 | NOT EXECUTED | 尚未形成独立测试证据 |
| 损坏图片 | PARTIAL | `_iter_frames` 会跳过无法读取图片；未做单独报告样本 |
| 缺失 Pose | PASS BY LOGIC | `person_missing/motion_missing` 进入 UNKNOWN 分支，真实比例见 summary |
| 分支人物不匹配 | PASS BY LOGIC | `person_branch_mismatch` 分支已由代码实现 |
| 缺失模型权重 | NOT EXECUTED | 代码会在模型加载阶段报错；未执行破坏性测试 |
| webcam `--input 0` | NOT VERIFIED | 不列为当前交付能力 |
| 多人物复杂场景 | LIMITED | 当前为 v0.1 确定性主人物策略 |
| FPS/实时吞吐 | NOT FORMALLY BENCHMARKED | 当前报告不虚构 FPS/latency |

## 7. 输出样例与可复核文件

- 主 demo：`artifacts/fall/mvp_final/demo.mp4`；
- 主摘要：`artifacts/fall/mvp_final/summary.json`；
- 逐帧输出：`artifacts/fall/mvp_final/frames.jsonl`；
- 配置快照：`artifacts/fall/mvp_final/config_snapshot.yaml`；
- 原始 smoke 记录：`artifacts/fall/mvp_final/smoke_test_report.md`。

## 8. 工程限制与科研边界

1. `motion_heuristic_score`、`environment_risk_score` 和 `overall_risk_state` 都是工程输出，不是校准跌倒概率；
2. 当前真实样本 HIGH=0，不能宣传为真实跌倒预测已经可靠；
3. smoke 测试证明的是工程链路可运行和 schema 稳定，不证明科研泛化；
4. frozen v1.0 科学 baseline（EWR@0.5=0.233、FA/min=6.81）来自独立 causal sequence-level 评估，不等同于 `fall_mvp` 的 motion heuristic；
5. 终端 `UNKNOWN` 是信息不足状态，不等于 LOW 或“安全”；
6. 单目 `d_norm` 是图像平面邻近代理，不是物理米制距离；
7. targeted hard-ADL post-impact 结果不是完整 ADL specificity。

## 9. 总结

当前版本已证明：

- 工程代码可以运行；
- 正确 RGB 输入可以处理；
- Pose/Motion、环境识别、邻近风险和融合链路可以协同工作；
- 输出 schema、摘要、配置快照和演示视频能够稳定生成；
- 缺失信息可以显式输出 UNKNOWN；
- 测试数据和结果文件可以复核。

当前版本尚未证明：

- 工业部署级跌倒预测；
- 校准风险概率；
- pre-impact 性能提升；
- 完整数据集上的 specificity、Precision、Recall 或 F1；
- webcam、复杂多人场景和正式实时性能。

## 10. v0.3 可选扩展补充测试（2026-08-25）

> 本节是 v0.2 之后的兼容工程 smoke，不改变上述 v0.2 结论或 frozen 科研指标。

### 10.1 回归与单元测试

- v0.2 契约回归：4/4 PASS；
- v0.3 provider 单元测试：5/5 PASS；
- Python compile：PASS；
- extensions disabled 8帧 smoke：PASS，逐帧无 `risk_extensions`，旧 schema 保持；
- extensions enabled 8帧 smoke：PASS，新增可选字段且旧 fusion 状态不变。

### 10.2 完整 v0.3 Demo

输入：`E:\ur_fall_rgb\fall-01`（160 帧正确 RGB）。

命令：

```bash
python -m fall_mvp.run --input E:\ur_fall_rgb\fall-01 \
  --output artifacts\fall\mvp_v03_final --enable-risk-extensions
```

结果：

| 指标 | 值 |
|---|---:|
| 总帧数 | 160 |
| 扩展字段帧数 | 160 |
| interaction 可用帧 | 147 |
| max human risk index | 46.92 |
| max environment risk index | 93.48 |
| max interaction risk index | 87.40 |
| 旧 fusion HIGH | 0 |
| 新 overall engineering HIGH | 40 |
| Wet Floor | 160/160 UNKNOWN（detector unavailable） |
| 输出视频 | 640×480、30fps、160帧、可读取 |

**语义纪律**：新 engineering HIGH 由环境/路径交互指数触发，不是跌倒 HIGH，不计入 EWR@0.5，也不是跌倒概率。

### 10.3 Wet-floor 决策

SynSpill 只读审计结论为 NO-GO for automatic detector integration：未确认直接可用 checkpoint、实时推理入口和 release，当前也无真实 wet/dry sanity 集。v0.3 使用明确 UNKNOWN 契约，不伪造水渍检测。详见 `WET_FLOOR_INTEGRATION_DECISION.md`。
