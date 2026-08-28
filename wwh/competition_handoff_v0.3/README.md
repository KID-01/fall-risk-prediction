# Fall Risk Competition Handoff v0.3

> v0.3 是 v0.2 之后的**向后兼容工程扩展**。v0.2 仍然有效，未被覆盖。
> v0.3 新增 Lighting、Clutter、因果线性轨迹、路径交互风险和 Wet-Floor UNKNOWN 契约；不修改 frozen v1.0 或 EWR/FA/lead。

## 先读

1. `team_handoff/ENVIRONMENT_INTERACTION_EXTENSION_PLAN.md`
2. `team_handoff/CONTRACT_DIFF_v0.2_to_v0.3.md`
3. `team_handoff/CHANGELOG_v0.3.md`
4. `team_handoff/SYSTEM_DESIGN_AND_TECHNICAL_DEVELOPMENT.md`
5. `team_handoff/FUNCTIONAL_TEST_REPORT.md`
6. `team_handoff/WET_FLOOR_INTEGRATION_DECISION.md`

## 运行

v0.2 兼容模式（默认关闭扩展）：

```bash
python -m fall_mvp.run --input <source> --output <dir>
```

v0.3 扩展模式：

```bash
python -m fall_mvp.run --input <source> --output <dir> --enable-risk-extensions
```

## 新增能力

- `lighting_v0`：低光、暗像素、低对比度工程风险；
- `clutter_v0`：人物脚部通道与 COCO 障碍物 bbox 的重叠；
- `causal_linear_trajectory_v0`：只使用历史脚点，外推未来 1 秒；
- `interaction_v0`：预测路径与障碍物/危险区域的交互风险；
- `wet_floor_unavailable`：无可靠模型时明确 UNKNOWN。

所有指数是 0–100 engineering risk index，不是概率。新 `overall_engineering_state_v0_3` 不替换旧 `fusion.overall_state`。

## 完整 Demo

`demo/v03/demo.mp4`：160帧、640×480、30fps。

- 旧 fusion HIGH=0；
- 新 engineering HIGH=40帧；
- 后者表示环境/路径交互风险高，不表示跌倒预测 HIGH；
- Wet Floor 始终 UNKNOWN（无 detector）。

## 测试

- v0.2 compatibility tests：4/4 PASS；
- v0.3 provider unit tests：5/5 PASS；
- disabled/enabled 8帧 smoke：PASS；
- full 160帧 demo：PASS。

## 不包含

数据集、模型权重、API 密钥、原始服务器特征和大型 Pose 缓存均未打包。
