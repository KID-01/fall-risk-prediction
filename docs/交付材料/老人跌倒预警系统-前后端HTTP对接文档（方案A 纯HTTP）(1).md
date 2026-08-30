# 老人跌倒预警系统‑前后端HTTP对接文档（方案A 纯HTTP）
>
> 文档版本：V1.0
> 对接对象：后端/萤石平台推送服务
> 技术栈：微信小程序云开发 + 云函数
> 云环境ID：cloud1‑d2gl1av2eb6e440e
> 云函数名称：fallAlarmPush

## 1 整体交互说明

整体数据流：
`后端/萤石平台 → 调用云函数fallAlarmPush(action=push) → 写入云数据库fall_alerts集合 → 小程序前端轮询云函数fallAlarmPush(action=pull) → 渲染告警列表 → 用户点击确认 → 调用云函数fallAlarmPush(action=confirm) 修改isRead状态`

> ✅ 所有接口统一通过云函数交互，不再使用http域名接口，避免url截断404问题

## 2 云函数接口规范

云函数名称：`fallAlarmPush`
入参格式：JSON对象，必须携带`action`字段区分能力

统一返回格式：

```json
{
  "code": 0,
  "msg": "提示文字",
  "data": {}
}
```

- `code=0`：代表成功
- `code=-1`：代表参数错误/失败
- `data`：非必返，根据action不同返回不同内容

### 2.1 【后端推送告警】action: push

- 用途：后端/萤石检测到跌倒风险时，推送告警数据入库

**请求入参示例**

```json
{
  "action": "push",
  "data": {
    "risk_label": "跌倒高危",
    "title": "老人疑似跌倒",
    "message": "卧室检测到跌倒行为",
    "risk_level": "critical",
    "isRead": false
  }
}
```

**字段说明**

| 字段 | 类型 | 必填 | 说明 |
| ------ | ------ | ------ | ------ |
| risk_label | string | 是 | 告警标签，如：跌倒高危/异常逗留 |
| title | string | 是 | 告警标题 |
| message | string | 是 | 告警详情描述 |
| risk_level | string | 是 | 告警等级，critical=高危（可确认），其他可自行扩展 |
| isRead | boolean | 是 | 是否已确认，新告警固定false |

**返回示例**

```json
{
  "code": 0,
  "msg": "告警已入库"
}
```

### 2.2 【小程序拉取告警列表】action: pull

- 用途：小程序前端定时拉取最新告警列表

**请求入参**

```json
{
  "action": "pull"
}
```

**返回示例**

```json
{
  "code": 0,
  "msg": "",
  "data": [
    {
      "_id": "数据库自动生成主键",
      "data": {
        "risk_label": "跌倒高危",
        "title": "老人疑似跌倒",
        "message": "卧室检测到跌倒行为",
        "risk_level": "critical",
        "isRead": false
      },
      "createTime": "2026-08-30T12:00:00.000Z"
    }
  ]
}
```

> 排序规则：按`createTime`倒序，最多返回10条

### 2.3 【小程序确认告警】action: confirm

- 用途：用户在前端点击【确认】按钮，标记告警已读

**请求入参**

```json
{
  "action": "confirm",
  "id": "数据库记录的_id"
}
```

**返回示例**

```json
{
  "code": 0,
  "msg": "告警已确认"
}
```

## 3 数据库设计

集合名称：`fall_alerts`

| 字段 | 类型 | 说明 |
| ------ | ------ | ------ |
| _id | string | 文档唯一主键，自动生成 |
| data | object | 告警业务对象（上面push接口里的data结构） |
| createTime | Date | 告警产生时间，入库自动填充 |

## 4 前端行为说明

1. 页面进入后，每3秒轮询一次`action=pull`，自动刷新告警列表
2. 只有`risk_level = critical`且`isRead=false`的告警，才展示【确认】按钮
3. 点击确认后调用`action=confirm`，将该条记录`data.isRead`置为`true`，按钮消失

## 5 部署&环境信息

- 云环境：`cloud1‑d2gl1av2eb6e440e`
- 云函数：`fallAlarmPush`（已部署上线）
- 数据库集合：`fall_alerts`
- 前端页面路径：`pages/index/index`

## 6 测试方式

### 方式1：后端调用push写入测试数据

入参：

```json
{
  "action": "push",
  "data": {
    "risk_label": "跌倒高危",
    "title": "老人疑似跌倒",
    "message": "卧室检测到跌倒行为",
    "risk_level": "critical",
    "isRead": false
  }
}
```

调用成功后，小程序页面3s内自动加载出告警，可点击确认。

### 方式2：云开发控制台手动新增记录

直接在`fall_alerts`集合新增文档，结构同上面示例。

## 7 异常约定

1. `action`不存在/不识别：返回`code:-1, msg:"action参数错误"`
2. `confirm`传入的`id`不存在：可后续扩展增加异常提示（当前版本直接静默）

## 8 可扩展预留

- 可新增`action: delete` 删除告警
- 可扩展更多`risk_level`等级样式
- 可增加分页、时间筛选

## 附录：当前完整云函数代码（供后端核对）

```javascript
const cloud = require('wx-server-sdk')
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })
const db = cloud.database()
const ALERT_COLLECTION = 'fall_alerts'

exports.main = async (event, context) => {
  const { action, data, id } = event
  if (action === 'push') {
    await db.collection(ALERT_COLLECTION).add({
      data,
      createTime: new Date()
    })
    return { code: 0, msg: '告警已入库' }
  }
  if (action === 'pull') {
    const res = await db.collection(ALERT_COLLECTION)
      .orderBy('createTime', 'desc')
      .limit(10)
      .get()
    return { code: 0, data: res.data }
  }
  if (action === 'confirm') {
    await db.collection(ALERT_COLLECTION)
      .doc(id)
      .update({ data: { isRead: true } })
    return { code: 0, msg: '告警已确认' }
  }
  return { code: -1, msg: 'action参数错误' }
}
```

> 如果你需要，我可以输出极简版本（只保留接口，去掉多余文字，直接发给后端），或者补充萤石平台webhook对接版本。# 老人跌倒预警系统‑前后端HTTP对接文档（方案A 纯HTTP）
> 文档版本：V1.0
> 对接对象：后端/萤石平台推送服务
> 技术栈：微信小程序云开发 + 云函数
> 云环境ID：cloud1‑d2gl1av2eb6e440e
> 云函数名称：fallAlarmPush

## 1 整体交互说明

整体数据流：
`后端/萤石平台 → 调用云函数fallAlarmPush(action=push) → 写入云数据库fall_alerts集合 → 小程序前端轮询云函数fallAlarmPush(action=pull) → 渲染告警列表 → 用户点击确认 → 调用云函数fallAlarmPush(action=confirm) 修改isRead状态`

> ✅ 所有接口统一通过云函数交互，不再使用http域名接口，避免url截断404问题

## 2 云函数接口规范

云函数名称：`fallAlarmPush`
入参格式：JSON对象，必须携带`action`字段区分能力

统一返回格式：

```json
{
  "code": 0,
  "msg": "提示文字",
  "data": {}
}
```

- `code=0`：代表成功
- `code=-1`：代表参数错误/失败
- `data`：非必返，根据action不同返回不同内容

### 2.1 【后端推送告警】action: push

- 用途：后端/萤石检测到跌倒风险时，推送告警数据入库

**请求入参示例**

```json
{
  "action": "push",
  "data": {
    "risk_label": "跌倒高危",
    "title": "老人疑似跌倒",
    "message": "卧室检测到跌倒行为",
    "risk_level": "critical",
    "isRead": false
  }
}
```

**字段说明**

| 字段 | 类型 | 必填 | 说明 |
| ------ | ------ | ------ | ------ |
| risk_label | string | 是 | 告警标签，如：跌倒高危/异常逗留 |
| title | string | 是 | 告警标题 |
| message | string | 是 | 告警详情描述 |
| risk_level | string | 是 | 告警等级，critical=高危（可确认），其他可自行扩展 |
| isRead | boolean | 是 | 是否已确认，新告警固定false |

**返回示例**

```json
{
  "code": 0,
  "msg": "告警已入库"
}
```

### 2.2 【小程序拉取告警列表】action: pull

- 用途：小程序前端定时拉取最新告警列表

**请求入参**

```json
{
  "action": "pull"
}
```

**返回示例**

```json
{
  "code": 0,
  "msg": "",
  "data": [
    {
      "_id": "数据库自动生成主键",
      "data": {
        "risk_label": "跌倒高危",
        "title": "老人疑似跌倒",
        "message": "卧室检测到跌倒行为",
        "risk_level": "critical",
        "isRead": false
      },
      "createTime": "2026-08-30T12:00:00.000Z"
    }
  ]
}
```

> 排序规则：按`createTime`倒序，最多返回10条

### 2.3 【小程序确认告警】action: confirm

- 用途：用户在前端点击【确认】按钮，标记告警已读

**请求入参**

```json
{
  "action": "confirm",
  "id": "数据库记录的_id"
}
```

**返回示例**

```json
{
  "code": 0,
  "msg": "告警已确认"
}
```

## 3 数据库设计

集合名称：`fall_alerts`

| 字段 | 类型 | 说明 |
| ------ | ------ | ------ |
| _id | string | 文档唯一主键，自动生成 |
| data | object | 告警业务对象（上面push接口里的data结构） |
| createTime | Date | 告警产生时间，入库自动填充 |

## 4 前端行为说明

1. 页面进入后，每3秒轮询一次`action=pull`，自动刷新告警列表
2. 只有`risk_level = critical`且`isRead=false`的告警，才展示【确认】按钮
3. 点击确认后调用`action=confirm`，将该条记录`data.isRead`置为`true`，按钮消失

## 5 部署&环境信息

- 云环境：`cloud1‑d2gl1av2eb6e440e`
- 云函数：`fallAlarmPush`（已部署上线）
- 数据库集合：`fall_alerts`
- 前端页面路径：`pages/index/index`

## 6 测试方式

### 方式1：后端调用push写入测试数据

入参：

```json
{
  "action": "push",
  "data": {
    "risk_label": "跌倒高危",
    "title": "老人疑似跌倒",
    "message": "卧室检测到跌倒行为",
    "risk_level": "critical",
    "isRead": false
  }
}
```

调用成功后，小程序页面3s内自动加载出告警，可点击确认。

### 方式2：云开发控制台手动新增记录

直接在`fall_alerts`集合新增文档，结构同上面示例。

## 7 异常约定

1. `action`不存在/不识别：返回`code:-1, msg:"action参数错误"`
2. `confirm`传入的`id`不存在：可后续扩展增加异常提示（当前版本直接静默）

## 8 可扩展预留

- 可新增`action: delete` 删除告警
- 可扩展更多`risk_level`等级样式
- 可增加分页、时间筛选

## 附录：当前完整云函数代码（供后端核对）

```javascript
const cloud = require('wx-server-sdk')
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })
const db = cloud.database()
const ALERT_COLLECTION = 'fall_alerts'

exports.main = async (event, context) => {
  const { action, data, id } = event
  if (action === 'push') {
    await db.collection(ALERT_COLLECTION).add({
      data,
      createTime: new Date()
    })
    return { code: 0, msg: '告警已入库' }
  }
  if (action === 'pull') {
    const res = await db.collection(ALERT_COLLECTION)
      .orderBy('createTime', 'desc')
      .limit(10)
      .get()
    return { code: 0, data: res.data }
  }
  if (action === 'confirm') {
    await db.collection(ALERT_COLLECTION)
      .doc(id)
      .update({ data: { isRead: true } })
    return { code: 0, msg: '告警已确认' }
  }
  return { code: -1, msg: 'action参数错误' }
}
```

> 如果你需要，我可以输出极简版本（只保留接口，去掉多余文字，直接发给后端），或者补充萤石平台webhook对接版本。
