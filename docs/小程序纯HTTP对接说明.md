# 微信小程序纯 HTTP 对接说明（方案 A）

版本：`risk-notification-v2-http`
更新时间：2026-08-30

本说明以 `fallAlarmPush` 公网网关为小程序唯一对外接口。小程序不连接 Python 后端 WebSocket，也不调用 Python 后端 REST；告警由 Python 后端推送到云函数，小程序通过云函数轮询和确认。

## 1. 数据流

```text
Python 风险后端
  -> HTTPS POST fallAlarmPush(action=push)
  -> CloudBase fall_alerts 集合
  -> 小程序每 3 秒调用 fallAlarmPush(action=pull)
  -> 用户调用 fallAlarmPush(action=confirm)
```

公网网关：

```text
https://cloud1-d2gl1av2eb6e440e-1477389215.ap-shanghai.app.tcloudbase.com/fallAlarmPush
```

## 2. 后端上报契约

后端在关注级 `attention` 和高危级 `critical` 产生通知时发送：

```json
{
  "action": "push",
  "data": {
    "risk_label": "跌倒高危",
    "title": "跌倒高危告警",
    "message": "人体异常与环境风险叠加，请立即确认现场情况",
    "risk_level": "critical",
    "risk_score": 86.4,
    "isRead": false
  }
}
```

- `low` 仅保留本地记录，不调用云函数。
- 内部 `warning` 对外映射为 `attention`。
- `risk_score` 是 0–100 工程风险指数，不是医学概率。
- 云函数返回 `code=0` 时，本地通知的 `cloud_push.status` 与 `cloud_function` 投递记录为 `sent`；失败时为 `failed`，不影响本地风险记录或高危电话兜底。

## 3. 小程序工作

- 页面进入后每 3 秒调用 `fallAlarmPush(action=pull)`，按 `createTime` 渲染最新十条记录。
- 当 `data.risk_level === "critical" && data.isRead === false` 时展示确认按钮。
- 点击确认后调用 `fallAlarmPush(action=confirm, id=<记录_id>)`，成功后刷新列表。
- 关注级只展示普通提醒；高危级展示强提醒。
- 不再连接 `/ws/alerts`，不需要 `ping/pong`、重连或 `notification_id` 去重逻辑。

## 4. 后端启动配置

启用云函数上报时，在启动后端前设置：

```powershell
$env:WECHAT_FALL_ALARM_PUSH_ENABLED = "1"
$env:WECHAT_FALL_ALARM_PUSH_URL = "https://cloud1-d2gl1av2eb6e440e-1477389215.ap-shanghai.app.tcloudbase.com/fallAlarmPush"
venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

云函数 HTTP 网关返回 `code=0` 前，系统会显示 `cloud_push.status=failed`，并保留网关返回的错误码用于排查。

## 5. 当前联调状态

2026-08-30 已按本文档使用 `action=push` 实测公网网关，仍返回：

```json
{"code":"INVALID_ENV","message":"Env invalid"}
```

因此 Python 后端的请求格式已经切换完成，但 CloudBase 环境或公网网关地址仍需云函数团队修复。对方应提供一次返回 `code=0` 的 `action=push` 测试结果后再进行小程序验收。
