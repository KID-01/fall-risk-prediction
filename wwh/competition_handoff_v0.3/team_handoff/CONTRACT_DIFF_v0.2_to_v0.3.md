# Fall MVP v0.2 → v0.3 契约差异

## v0.2 原有逐帧字段（必须保留）

```text
timestamp
person
motion
environment
fusion
quality
```

## v0.3 可选新增字段

扩展关闭：无新增字段，v0.2 消费者不受影响。

扩展打开：只追加：

```json
{
  "risk_extensions": {
    "human_risk_index": {},
    "lighting": {},
    "clutter": {},
    "trajectory": {},
    "wet_floor": {},
    "interaction": {},
    "overall_engineering_state_v0_3": "LOW"
  }
}
```

## 兼容规则

1. 旧字段不删除、不改名、不重新解释；
2. `fusion.overall_state` 仍是 v0.2 规则融合结果；
3. v0.3 的 `overall_engineering_state_v0_3` 是单独工程状态；
4. 所有 index 是 0–100 工程指数，不是概率；
5. provider 不可用时 `available=false, state=UNKNOWN, risk_index=null`；
6. extensions 默认关闭；
7. v0.3 输出消费者必须容忍可选 provider 缺失。
