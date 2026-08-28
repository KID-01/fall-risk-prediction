# Fall MVP v0.3 Changelog（规划/实施记录）

## 版本关系

- v0.2：已经发给队友的历史冻结交付包，保持不变。
- v0.3：可选、向后兼容的工程扩展；不覆盖 v0.2，不修改 frozen 科研结论。

## 当前完成（Phase 0）

- 新增 `ENVIRONMENT_INTERACTION_EXTENSION_PLAN.md`；
- 新增 v0.2 契约 manifest；
- 新增 v0.2 回归测试；
- 确定 risk extensions 默认关闭；
- 确定新字段只能追加，不删除/改名旧字段。

## 当前完成（Phase 1 最小链，2026-08-25）

- Lighting Risk Provider；
- Clutter / Obstacle Provider；
- Causal Linear Trajectory Provider；
- Interaction Risk Engine；
- Wet-Floor Provider unavailable 契约（无可靠模型时 UNKNOWN）；
- `--enable-risk-extensions` 可选开关，默认关闭；
- `risk_extensions` 可选逐帧字段和 `risk_extensions_summary`；
- v0.3 overlay（风险指数、工程状态、预测路径）；
- 扩展单元测试 5/5 PASS；
- v0.2 契约回归 4/4 PASS；
- 8 帧端到端 smoke（关闭/开启）均 PASS；
- 关闭扩展时逐帧 key 仍为 timestamp/person/motion/environment/fusion/quality；
- 开启扩展时旧 fusion 状态不变，新状态独立命名 `overall_engineering_state_v0_3`；
- Wet Floor 输出 `available=false/state=UNKNOWN/risk_index=null`。

## 当前 smoke 证据

- 关闭输出：`artifacts/fall/v03_smoke_disabled/`；
- 开启输出：`artifacts/fall/v03_smoke_enabled/`；
- 两个 demo 均为 640×480、8 帧、可读取；
- 开启扩展 summary：human max=1.8、environment max=53.75、interaction max=87.4；
- 这些是工程 smoke，不是准确率/EWR/跌倒概率证据。

## 尚未完成

- 低光/障碍物/安全走路三个正式演示样本；
- v0.3 正式功能测试报告和新交接包。

## 完整 v0.3 Demo（2026-08-25）

- 输出：`artifacts/fall/mvp_v03_final/`；
- 输入：fall-01 正确 RGB 160帧；
- 旧 v0.2 summary 数字完全复现：LOW111/MED4/HIGH0/UNKNOWN45；
- 新指数 max human/env/interaction=46.92/93.48/87.4；
- 新 engineering HIGH=40帧（环境/路径交互语义，非跌倒 HIGH）；
- interaction available=147/160；Wet Floor=160/160 UNKNOWN；
- demo.mp4 640×480、30fps、160帧可读取。

## Wet-floor 决策门（2026-08-25）

- 结论：NO-GO for automatic detector integration；
- 原因：SynSpill 未确认直接可用 checkpoint/实时推理入口/GitHub release，且当前无真实 wet-floor sanity 数据；
- 许可证：数据与 associated materials 仅限 non-commercial research/education 并要求引用；
- v0.3 行为：明确输出 UNKNOWN，不伪造检测；
- 详细证据：`WET_FLOOR_INTEGRATION_DECISION.md`。

## 不会改变

- `motion_heuristic_v0`；
- `env_risk_v0`；
- `risk_fusion_v0` 的旧输出语义；
- frozen v1.0 EWR/FA/lead；
- P0-A、SPRINT-002–011、P-3 结论；
- v0.2 ZIP 内容与 hash。
