# Fall Risk MVP v0.1 — Engineering Smoke Test Report

> **Engineering smoke（非科研实验）。** 目的仅为发现 crash / path / JSON / schema / missing-file 问题；**不**报告科研 EWR/FA，**不**为 demo 优化阈值。

## 输入（全部正确 RGB 帧目录，本地工程样本）

| sample | frames | valid_person | LOW | MEDIUM | HIGH | UNKNOWN | max_env | top hazards |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `E:\ur_fall_rgb\fall-01`（最终 demo） | 160 | 150 | 111 | 4 | **0** | 45 | 0.935 | chair(282), couch(9) |
| `E:\ur_fall_rgb\fall-02` | 110 | 86 | 72 | 4 | **0** | 34 | 0.997 | chair(171), suitcase(2), couch(1) |
| `E:\ur_fall_rgb\fall-03` | 215 | 215 | 215 | 0 | **0** | 0 | 1.000 | chair(436), suitcase(4), couch(2) |

## 结果

- **运行稳定性**：3 个输入全部一条命令跑通，无 crash / 无 missing-file / JSON 可解析。
- **schema 一致性**：3 个 `frames.jsonl` 顶层 key 一致（timestamp/person/motion/environment/fusion/quality）；`summary.json` 结构一致。
- **实际观察到 HIGH=0**：所有真实序列均无 HIGH。原因包括：跌倒/躺地过渡区 Pose 分支退化、motion heuristic 无法持续获得有效人体运动信号（如 fall-01/02 躺地段），或该序列运动未超 HIGH 阈值（如 fall-03 全程 low）。
- **未通过调参"修复" HIGH=0**（Day-4 feature freeze）。
- **HIGH 分支逻辑存在性（synthetic / unit sanity，非真实性能）**：对 `fusion.fuse` 直接单测：
  - `motion 0.8 + env 0.9 → HIGH`；`motion 0.5 + env 0.9 → HIGH`；`motion 0.1 + env 0.9 → LOW(context_elevated)`；`motion None → UNKNOWN(motion_missing, environment_high)`。

## 产物
- `artifacts/fall/mvp_final/`：demo.mp4 / frames.jsonl / summary.json / config_snapshot.yaml
- `artifacts/fall/mvp_smoke_fall02/`、`artifacts/fall/mvp_smoke_fall03/`

## 边界声明
本报告仅证明 MVP 工程路径可运行且 schema 稳定；**不含任何科研指标声明**。