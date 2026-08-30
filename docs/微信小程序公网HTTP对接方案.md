# 微信小程序公网 HTTP 对接方案

版本：`risk-notification-public-http-v1`  
更新时间：2026-08-30

本文是 Python 风险后端与微信小程序团队的最终联调交接文档。小程序只访问公网 HTTPS 网关，不连接开发机的 `127.0.0.1`，也不依赖 Python 后端 WebSocket。

## 1. 目标架构

```text
本地/服务器 Python 风险后端
        │ 仅出站 HTTPS POST action=push
        ▼
CloudBase 公网网关 /fallAlarmPush
        │ 写入云数据库 fall_alerts
        ▼
微信小程序 HTTPS 轮询 action=pull
        │ 用户确认 action=confirm
        └── 同一公网网关
```

公网地址（由云函数团队部署并确认可用）：

```text
https://cloud1-d2gl1av2eb6e440e-1477389215.ap-shanghai.app.tcloudbase.com/fallAlarmPush
```

该地址必须由 CloudBase HTTP 触发器实际绑定到有效环境和已发布的 `fallAlarmPush` 函数。Python 后端无法修复 CloudBase 返回的 `INVALID_ENV`；出现该错误时应先由云函数团队重新部署或提供新地址。

## 2. 双格式兼容的推送请求

后端支持环境变量 `WECHAT_FALL_ALARM_PUSH_PAYLOAD_MODE`：

- `legacy`：仅发送旧云函数使用的 `action` + `data` 结构。
- `hybrid`：推荐公网联调，同时发送顶层字段和 `action` + `data`，兼容两种云函数实现。
- `flat`：仅发送对方方案中约定的三个顶层字段。

一键启动脚本已使用 `hybrid`。推荐请求如下：

```http
POST /fallAlarmPush
Content-Type: application/json
```

```json
{
  "action": "push",
  "elderId": "E001",
  "riskLevel": "critical",
  "riskScore": 86.4,
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

云函数必须接受 `action=push` 并优先使用 `data` 写入记录；若采用扁平方案，也必须读取 `elderId`、`riskLevel`、`riskScore`。建议保存以下字段，便于小程序显示和追溯：`elderId`、`riskLevel`、`riskScore`、`risk_label`、`title`、`message`、`isRead`、`createTime`。

风险等级只允许：

| 值 | 含义 | 小程序行为 |
| --- | --- | --- |
| `low` | 低风险 | 不主动推送，可不入云端告警列表 |
| `attention` | 关注级 | 普通提醒 |
| `critical` | 高危级 | 强提醒，显示确认按钮 |

`riskScore` 是 0-100 的工程风险指数，不是医学概率。

## 3. 小程序必须实现的接口

小程序只调用上述公网网关，不调用本地 Python 地址。

### 3.1 拉取告警

```json
POST /fallAlarmPush
{"action":"pull"}
```

成功响应：

```json
{
  "code": 0,
  "data": [
    {
      "_id": "cloud-document-id",
      "data": {
        "elderId": "E001",
        "riskLevel": "critical",
        "riskScore": 86.4,
        "title": "跌倒高危告警",
        "message": "请立即确认现场",
        "isRead": false
      },
      "createTime": "2026-08-30T12:00:00.000Z"
    }
  ]
}
```

页面进入、回到前台和下拉刷新时调用一次；前台轮询建议每 3-5 秒一次。按 `createTime` 倒序展示，使用 `_id` 去重。

### 3.2 确认高危告警

```json
POST /fallAlarmPush
{"action":"confirm","id":"cloud-document-id"}
```

确认成功必须返回 `code=0`，并将对应记录的 `isRead` 更新为 `true`。按钮提交期间禁用，重复确认应保持幂等。

### 3.3 健康检查

云函数团队必须提供以下最小自测：

```json
POST /fallAlarmPush
{"action":"pull"}
```

返回 HTTP 200 且 JSON `code=0` 才表示网关、环境、函数和数据库链路可用。禁止用浏览器 GET 访问该地址代替 POST 测试。

## 4. 云函数团队需要完成的工作

1. 在 CloudBase 中确认环境 `cloud1-d2gl1av2eb6e440e` 有效。
2. 部署并发布 `fallAlarmPush`，打开 HTTP 公网触发器，路径固定为 `/fallAlarmPush`。
3. 函数解析 JSON 请求体，兼容本文第 2 节的 hybrid 请求。
4. 创建并授权访问 `fall_alerts` 集合；`pull` 按 `createTime` 倒序返回最新 10 条。
5. `confirm` 校验 `_id` 存在后更新 `data.isRead=true`，不存在返回明确的 404 或业务错误码。
6. 为生产网关增加鉴权、限流、HTTPS 和来源校验；密钥只能由后端服务端和云函数环境变量保存，不能放入小程序代码。
7. 提供一次 `action=pull` 和一次 `action=push` 的完整响应、函数版本、环境 ID 和部署时间。

建议错误码：`0` 成功，`-1` 参数错误，`-100` 风险等级非法，`-99` 云函数内部错误。HTTP 层错误（404、5xx、网关超时）也必须返回可诊断的 JSON 或在联调记录中说明。

## 5. Python 后端需要完成的工作

后端已实现风险通知生成、本地落库和出站 HTTP 适配器。部署前设置：

```powershell
$env:WECHAT_FALL_ALARM_PUSH_ENABLED = "1"
$env:WECHAT_FALL_ALARM_PUSH_URL = "https://cloud1-d2gl1av2eb6e440e-1477389215.ap-shanghai.app.tcloudbase.com/fallAlarmPush"
$env:WECHAT_FALL_ALARM_PUSH_PAYLOAD_MODE = "hybrid"
venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

后端只向网关发起出站请求，不要求公网访问本地 8000 端口。网关失败时，本地风险记录仍保留，通知详情中的 `cloud_push.status` 为 `failed` 并带有返回码；修复网关后重新触发风险即可验证。

## 6. 联调顺序与验收标准

1. 云函数团队先用 `{"action":"pull"}` 验证公网地址返回 HTTP 200、`code=0`。
2. 使用 hybrid 示例执行一次 `action=push`，确认记录写入 `fall_alerts`。
3. 后端设置三个环境变量并重启，触发一次 `attention`，确认后端通知详情 `cloud_push.status=sent`。
4. 小程序轮询，在 5 秒内看到关注级记录。
5. 触发 `critical`，小程序显示强提醒和确认按钮。
6. 调用 `confirm` 后刷新列表，确认 `isRead=true`，重复调用仍返回幂等结果。
7. 模拟网关 404、`INVALID_ENV`、超时和非 0 业务码；小程序显示“服务暂不可用”，不得伪造告警成功，后端本地告警不得丢失。
8. 真机验收时只使用 HTTPS 域名；不使用 `127.0.0.1`、局域网 IP 或 Python WebSocket。

## 7. 常见问题定位

| 现象 | 责任方 | 处理 |
| --- | --- | --- |
| `INVALID_ENV` | CloudBase 环境/触发器 | 重新确认环境 ID、函数发布状态和公网触发器地址 |
| HTTP 404 | CloudBase 路径配置 | 确认路径包含 `/fallAlarmPush` 且使用 POST |
| HTTP 200 但 `code` 非 0 | 云函数代码/数据库 | 查看函数日志、集合权限和请求字段解析 |
| 后端 `cloud_push=failed`，小程序无记录 | 网关或后端配置 | 检查 URL、三个环境变量和后端重启时间 |
| 小程序请求 `127.0.0.1` 失败 | 小程序配置 | 改为公网 CloudBase URL，不让小程序直连本地服务 |

## 8. 交接物

小程序团队交付：公网 URL 可用性截图、函数版本和环境 ID、`pull/push/confirm` 三组响应、数据库集合结构、轮询和确认代码、真机验收记录。后端团队交付：当前文档、风险字段说明、一次 `cloud_push.status=sent` 的通知详情和错误场景日志。

