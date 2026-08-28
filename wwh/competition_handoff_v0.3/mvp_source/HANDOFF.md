# Fall Risk MVP v0.1 — Teammate HANDOFF

> 面向接手队友的工程交接；非科研长文。
> 输出为 engineering prototype；**任何输出都不是校准跌倒概率/临床跌倒预测/科学验证风险概率**。

## 1. 如何运行

```bash
E:\anaconda\conda_envs\yolo\python.exe -m fall_mvp.run --input <video_or_frames> --output <out_dir>
```

- 需在仓库根目录（含 `fall_mvp/` 包）运行，或把仓库根加入 `PYTHONPATH`。
- 输入：视频文件，或 RGB 帧目录（PNG/JPG）。
- 输出：`frames.jsonl`、`summary.json`、`config_snapshot.yaml`、`demo.mp4`。

## 2. 输入是什么

video / RGB frame directory（v0.1 verified）。webcam `--input 0` 未验证。

## 3. 输出是什么 & 如何读取 JSONL

每行一个 JSON（`frames.jsonl`）：

```json
{
  "timestamp": 0.0,
  "person": {"present": true, "match": true},
  "motion": {"source": "motion_heuristic_v0", "score": 0.0, "state": "LOW"},
  "environment": {"source": "env_risk_v0", "score": 0.0, "state": "LOW",
                  "top_hazards": [{"class": "chair", "confidence": 0.88,
                                   "normalized_distance": 0.57, "risk_contribution": 0.45}]},
  "fusion": {"source": "risk_fusion_v0", "overall_state": "LOW",
             "reason": ["motion_low", "environment_medium", "context_elevated"]},
  "quality": {"sync_delta_sec": 0.0, "status": "OK"}
}
```

读取示例（python）：

```python
import json
rows = [json.loads(l) for l in open("out/frames.jsonl", encoding="utf-8")]
print([r["fusion"]["overall_state"] for r in rows[:10]])
```

## 4. 哪些 config 可以改

- 集中在 `fall_mvp/contract.py::default_config()`；运行会写出 `config_snapshot.yaml`。
- 可改：motion LOW/HIGH 阈值、environment LOW/HIGH 阈值、object risk weights、neutral classes、near/far 邻近阈值、person-match 阈值（IoU/center ratio）、sync tolerance、persistence frames。
- **注意**：这些是 engineering heuristics，不是科学优化参数；请勿为让某个 demo 好看而改，除非你有独立验证目标。
- 如需修改 `frames.jsonl` schema：在 HANDOFF 记录变更（本版 schema 已固定）。

## 5. 哪些东西不能被误解

- `motion_heuristic_score` / `environment_risk_score` / `overall_risk_state` **都不是**跌倒概率。
- **UNKNOWN ≠ LOW**：UNKNOWN=信息不足不能判断（person_missing/motion_missing/person_branch_mismatch/sync_failed）；LOW=看到有效信息且风险低。
- **环境单独不能触发 HIGH**（v0.1 规则：motion LOW + env HIGH → LOW + context_elevated）。
- 真实样本 `HIGH=0`（因跌倒/躺地过渡 Pose 退化）——**不是 bug，未通过调参"修复"**。

## 6. 如何以后替换 motion predictor

**关键接口（冻结，便于替换 motion_heuristic_v0 → future_causal_predictor_v1，不重写 Environment/Fusion/UI）：**

```text
Motion Provider
     ↓
{ timestamp, score/state, person/bbox, availability }
     ↓
Environment Engine
     ↓
Fusion
```

- `run.py` 中 motion 分支只产生 `{score, state(经 fusion 判定), person bbox, availability}`；`fusion.py` 消费 `score` + 状态。
- 替换时保持：timestamp 对齐、`None` 表示缺失（→ motion UNKNOWN 语义）、person bbox 结构（`x1,y1,x2,y2,conf`）。
- 建议新 provider 也输出 `source` 标记以利于追溯。

## 7. Deployment-model gap（必须记录）

> The frozen v1.0 scientific evaluation does not currently correspond to a single frozen deployable predictor artifact.
> v1.0 was evaluated through sequence-level nested/5-fold models and fold-specific thresholds.
> Creating an all-data deployment model or ensemble would constitute a new engineering/scientific artifact requiring its own validation.

因此 MVP 用 `motion_heuristic_v0`，**不是因为 frozen baseline 被否定**，而是 `scientific evaluation artifact ≠ deployment artifact`。这是未来必须解决的工程任务。

## 8. 依赖环境（当前实际验证；勿升级依赖）

| 依赖 | 版本 | 备注 |
|---|---|---|
| Python | 3.10.20 | 验证解释器：`E:\anaconda\conda_envs\yolo\python.exe`（队友机器路径可不同） |
| ultralytics | 8.4.96 | 本地仓库源码 |
| torch | 2.13.0+cpu | 当前 backend=cpu |
| NumPy | 2.2.6 | |
| OpenCV | 5.0.0 | |
| Pillow | 12.3.0 | |
| PyYAML | 6.0.3 | |

模型文件（参考路径，队友机器需按实际放置）：
- pose：`weights/pose/yolo26n-pose.pt`
- env：`yolo26n.pt`（COCO 预训练）

## 9. Audio 未来接口（不在 MVP critical path）

```text
candidate_event
      ↓
audio_confirmation  (post-impact confirmation plugin，可拆给队友独立开发)
      ↓
{ timestamp, event_type, confidence }
```

- 定位：**post-impact confirmation**，不是 pre-impact predictor。
- 主系统**不得依赖 audio 才能运行**。
- 本版不实现。

## 10. 已知局限（v0.1）

- 真实样本 HIGH=0（过渡区 Pose 退化；synthetic/unit 仅证明分支逻辑可达）。
- 多人物简化；`d_norm` 单目近似非米制；环境分数对家具密集场景敏感；v0.1 输入=video/帧目录。
- 长期科研路线不变：MVP → P0-A IMU-only causal protocol → IMU baseline → GO/NO-GO → visual+IMU fusion → post-impact confirmation → depth feasibility → complex temporal（ST-GCN/TCN/Transformer/Mamba 仍 deferred）。