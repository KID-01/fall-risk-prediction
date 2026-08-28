# Fall Risk MVP v0.1（Environment-Aware）

> **工程/产品原型，非科研验证系统。**
> 不得称为：validated fall prediction model / scientifically validated fall probability system / clinical fall predictor。
>
> **Critical warning**：所有 score 均为 **engineering heuristic score，不是校准跌倒概率**。
> 科研参考（frozen）：v1.0 PREIMPACT EWR@0.5=0.233、FA/min=6.81（无单一可部署模型 artifact，见 HANDOFF §deployment-model gap）。

## 核心组成

```text
motion_heuristic_v0
        +
env_risk_v0
        +
risk_fusion_v0
        ↓
LOW / MEDIUM / HIGH / UNKNOWN
```

- `motion_heuristic_v0`：工程身体运动行为指标（非 frozen v1.0 LR baseline）。
- `env_risk_v0`：工程上下文危险指标。
- `risk_fusion_v0`：确定性规则融合输出。

## Quick Start

一条命令：

```bash
E:\anaconda\conda_envs\yolo\python.exe -m fall_mvp.run \
  --input <video_or_frames> --output <output_dir>
```

支持输入：视频（mp4/avi…）、**RGB 帧目录**（PNG/JPG 序列）。

## Inputs / Outputs

**Input**：Camera/video/RGB frame directory。

**Outputs**（写到 `--output`）：
- `frames.jsonl` —— 每帧融合结果
- `summary.json` —— 工程概要统计
- `config_snapshot.yaml` —— 使用的配置快照
- `demo.mp4` —— 带 overlay 的演示视频

## Module architecture

```text
Input
  ├─ Pose/Motion -> motion_heuristic_v0
  └─ YOLO(COCO预训练) -> env_risk_v0
                ↓
           risk_fusion_v0
                ↓
      LOW / MEDIUM / HIGH / UNKNOWN
```

## Output semantics

- **Motion Score（motion_heuristic_score）**：工程身体垂直运动启发。**非校准概率**。
- **Environment Risk Score（environment_risk_score）**：工程上下文危险指示（邻近物体加权，单目图像平面近似）。**非跌倒概率**。
- **Overall Risk State（LOW/MEDIUM/HIGH/UNKNOWN）**：确定性规则融合。**环境单独不能触发 HIGH imminent-fall 警告**。
- **UNKNOWN**：信息不足，不能判断（≠ LOW）。保留 `person_missing` / `motion_missing` / `person_branch_mismatch` / `sync_failed`。
- **reason codes**：可解释触发原因（`motion_low/medium/high`、`environment_low/medium/high`、`context_elevated` 等）。

## Known Limitations（v0.1）

1. **真实样本 HIGH=0 为如实结果，未通过调参"修复"**：air sample 上跌倒/躺地过渡区 Pose 分支退化，motion heuristic 无法持续获得有效人体运动信号 → HIGH 不出现。单测（synthetic/unit）证明 HIGH 分支逻辑可达，但**仅是 synthetic logic sanity，非真实性能证据**。
2. **deployment-model gap**：frozen v1.0 科学评估经由 sequence-level nested/5-fold 模型与 fold-specific thresholds，**不存在单一冻结部署预测器**；构建 all-data deployment model/ensemble 属新工程/科研 artifact，需独立验证。MVP 用 `motion_heuristic_v0` 不代表否定 frozen baseline。
3. 多人物处理为 v0.1 简化（确定性主人物）。
4. `d_norm` 为单目图像平面邻近代理，**不代表物理米制距离**。
5. 环境分数对家具密集场景敏感；权重为工程师启发（config 集中、未科学优化）。
6. v0.1 verified input = **video / RGB frame directory**（webcam `--input 0` 不在本版验证范围）。

## Config

集中管理于 `fall_mvp/contract.py::default_config()`（运行时会写 `config_snapshot.yaml`）：
motion LOW/HIGH 阈值、environment LOW/HIGH 阈值、object risk weights、neutral classes、proximity thresholds（near/far）、person-match 阈值、sync tolerance、persistence frames。

> 这些参数是 engineering heuristics，不是 scientifically optimized parameters。
> 请勿在未做新验证前按单一样本调整它们。