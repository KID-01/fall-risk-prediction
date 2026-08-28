# v0.3 环境与交互风险兼容扩展路线

> 状态：工程路线已认可，待按本计划实施。
> 基线关系：`Fall Risk Competition Handoff v0.2` 已发给队友，视为历史冻结交付；v0.3 只能做兼容增量，不覆盖或重写 v0.2。
> 科研边界：本计划不修改 frozen v1.0 PREIMPACT baseline，不提高或重新解释 EWR@0.5，不重新开启已关闭的 Pose/threshold/P-3 研究路线。

## 1. 目标

把当前工程原型从：

```text
Pose/Motion + 物体邻近环境风险
```

兼容扩展为：

```text
Human Risk Index
        +
Environment Risk Index
        +
Causal Trajectory Interaction Risk Index
        ↓
Unified Engineering Risk Dashboard
```

目标是提升可解释性、环境风险覆盖范围、答辩可视化，以及 wet-floor/audio/depth provider 的后续扩展能力。

所有指数均为 `0–100 engineering risk index`，不是概率，也不得表述为临床跌倒概率。

## 2. v0.2 兼容性硬约束

### 2.1 必须保持不变

- `python -m fall_mvp.run --input ... --output ...` 原命令继续可用；
- `frames.jsonl` 现有顶层字段 `timestamp/person/motion/environment/fusion/quality` 不删除、不改名；
- `summary.json` 现有字段不删除、不改变语义；
- `motion_heuristic_v0`、`env_risk_v0`、`risk_fusion_v0` 保留；
- 默认配置下，扩展关闭时输出与 v0.2 行为等价；
- 不修改 `fall_risk_mvp_competition_v0.2.zip`，新包另命名 v0.3；
- 不修改 frozen v1.0、P0-A、SPRINT-002–011 和 P-3 的科研数字或结论。

### 2.2 允许新增

- 新配置块 `risk_extensions`，默认 `enabled: false`；
- 新顶层可选字段 `risk_extensions`；
- 新 summary 字段 `risk_extensions_summary`；
- 新 provider、新测试、新 overlay；
- 新演示视频和 v0.3 交接包。

旧消费者忽略新字段即可继续工作。

## 3. 推荐代码布局

不重构现有文件，新增独立目录：

```text
fall_mvp/
├─ risk_extensions/
│  ├─ __init__.py
│  ├─ contract.py
│  ├─ lighting.py
│  ├─ clutter.py
│  ├─ trajectory.py
│  ├─ interaction.py
│  └─ wet_floor.py
├─ contract.py          # 只追加默认配置块
├─ run.py               # 只增加可选调用和 overlay
└─ ...                  # v0.2 其余模块不动
```

新增模块不得反向依赖 frozen 科研脚本。

## 4. 统一 Provider 契约

每个环境 provider 输出：

```json
{
  "source": "lighting_v0",
  "available": true,
  "risk_index": 42.0,
  "state": "MEDIUM",
  "evidence": {},
  "reason_codes": []
}
```

没有模型或信号时必须输出：

```json
{
  "source": "wet_floor_unavailable",
  "available": false,
  "risk_index": null,
  "state": "UNKNOWN",
  "regions": [],
  "reason_codes": ["detector_unavailable"]
}
```

禁止用手绘 mask 或人工指定结果冒充自动检测。

## 5. 模块设计

### 5.1 Lighting Risk Provider

输入当前帧，输出灰度平均亮度、暗像素比例、灰度标准差/对比度和欠曝状态。

定位：工程视觉质量/环境危险指标，不代表跌倒概率。

### 5.2 Clutter / Obstacle Provider

复用现有 YOLO COCO 检测，不新增模型，覆盖 chair、suitcase、backpack、sports ball、dining table 等配置类别。

由“最近距离”扩展为：人物前方通道内障碍物数量、bbox 与通道重叠面积、最近障碍物归一化距离和检测置信度。

### 5.3 Causal Trajectory Provider

只使用 `<=t` 的历史人物脚点/bbox bottom-center：

1. 保存最近 N 帧脚点；
2. 剔除 missing/outlier；
3. 用统一线性最小二乘拟合速度；
4. 外推未来 1.0 秒的路径点；
5. 生成固定宽度路径 corridor；
6. 输出轨迹质量状态。

名称必须为 `causal_linear_trajectory_v0`，不能称为深度学习轨迹预测模型。

### 5.4 Interaction Risk Engine

计算预测路径 corridor 与障碍物 bbox、wet-floor mask（如可用）和其他危险区域的相交程度及 time-to-interaction。

示例输出：

```json
{
  "risk_index": 86.0,
  "state": "HIGH",
  "time_to_interaction_s": 0.7,
  "intersections": ["chair"],
  "reason_codes": ["predicted_path_intersects_obstacle"]
}
```

### 5.5 Wet-Floor Provider 决策门

SynSpill 当前只确认提供合成数据/训练方法和 MIT 代码参考，未确认可直接部署的 wet-floor checkpoint。

在接入前一次性审计：

1. 是否有可下载 checkpoint；
2. 是否有独立推理入口；
3. 权重与代码许可证是否允许比赛分发；
4. 是否能在当前环境运行；
5. 是否有真实湿地面样本做 false-positive sanity；
6. 输出是 detection 还是 segmentation，如何映射到地面区域。

若任一关键项不满足，v0.3 使用 `wet_floor_unavailable`，UI 显示 `UNKNOWN`，不阻塞其他模块。

不得临时训练后直接宣称水渍准确率提升；正式水渍识别需要独立数据、split、标签和评估协议。

## 6. 风险显示与融合纪律

推荐 UI：

```text
PERSON RISK INDEX        31 / 100
ENVIRONMENT RISK INDEX   78 / 100
INTERACTION RISK INDEX   86 / 100

OVERALL ENGINEERING STATE: HIGH
```

角落固定显示：

```text
Engineering risk indices, not calibrated probabilities
```

v0.3 第一阶段不修改 v0.2 的 `fusion.overall_state`；新增 dashboard state 单独命名：

```text
overall_engineering_state_v0_3
```

## 7. 实施顺序（1–2 天）

### Phase 0：回归护栏（1–2 小时）

- 保存 v0.2 代表性输出摘要；
- 增加 `extensions disabled` 回归测试；
- 验证旧命令和旧字段不变；
- 创建 `CHANGELOG_v0.3.md` 和 `CONTRACT_DIFF_v0.2_to_v0.3.md`。

### Phase 1：不依赖新模型的环境增强（3–5 小时）

- Lighting provider；
- Clutter provider；
- Causal trajectory provider；
- Obstacle interaction engine；
- JSON 输出和 overlay；
- 单元测试与三个演示场景。

### Phase 2：Wet-floor 决策门（最多 2–3 小时）

- 只做 checkpoint/推理/许可证/样本 sanity 审计；
- 满足条件才接插件；
- 不满足立即转 `UNKNOWN`，不继续消耗时间。

### Phase 3：交付与文档（2–3 小时）

- Demo：安全走路、路径与障碍物相交、低光/杂乱；
- 更新系统设计文档和功能测试报告；
- 生成 v0.3 交接包；
- v0.2 保留不动；
- 给队友一页变更说明。

## 8. 测试与验收

### 8.1 必须回归

- v0.2 命令可运行；
- extensions disabled 时旧 JSON 字段和值语义不变；
- 原 demo 仍可生成；
- 不读取未来帧；
- provider unavailable 不导致主程序崩溃。

### 8.2 新增功能测试

- 低光/正常光 lighting 状态；
- 无障碍物/通道内障碍物；
- 静止人物轨迹质量 UNKNOWN 或 LOW；
- 移动人物 1s 外推路径；
- 路径与物体相交；
- wet-floor available/unavailable 两种契约；
- JSON schema 和 overlay 不越界。

### 8.3 不能宣称的指标

- 不宣称 EWR@0.5 提升；
- 不宣称跌倒分类准确率提升；
- 不把 hard-coded demo mask 当 wet-floor accuracy；
- 不把 trajectory extrapolation 当真实未来轨迹 GT；
- 不把 risk index 当概率。

## 9. Go / No-Go

### GO

- v0.2 回归通过；
- Lighting/Clutter/Trajectory/Interaction 稳定输出；
- 至少两个真实演示场景可复现；
- 交互风险能解释路径与障碍物关系；
- 无科研口径混淆。

### NO-GO / 降级

- wet-floor 无可用模型或 false positive 明显：降级 UNKNOWN；
- trajectory 质量不足：Interaction 输出 UNKNOWN，不强行给分；
- 新字段破坏旧消费者：回退为独立 JSON/overlay，不改旧 schema；
- 新功能影响 v0.2 主链稳定性：关闭扩展开关，保留 v0.2。

## 10. 队友沟通模板

```text
v0.2 已发出的代码、文档和科学结论保持有效。
后续 v0.3 是可选的兼容工程扩展：新增 Lighting、Clutter、因果轨迹外推和路径交互风险；
不修改 frozen v1.0 baseline，不声称提高 EWR，也不替换原风险融合。
旧命令继续可用，新模块默认关闭或通过新字段输出。
如果 wet-floor 没有可靠模型，会显示 UNKNOWN，不会伪造检测。
```

## 11. 后续 AI 执行纪律

任何后续 Agent 开始实现前必须：

1. 阅读本文件；
2. 阅读 `fall_mvp/README.md`、`HANDOFF.md`、`contract.py`；
3. 检查 git status，保护用户未提交改动；
4. 先写 v0.2 回归测试；
5. 只做新增模块，不重构主链；
6. 遇到 schema 破坏、依赖升级、模型训练、水渍数据标签等事项时停止到 Human Gate；
7. 每阶段记录 Created / Modified / Validated / Deferred / Risks；
8. 新包命名 v0.3，不覆盖 v0.2。
