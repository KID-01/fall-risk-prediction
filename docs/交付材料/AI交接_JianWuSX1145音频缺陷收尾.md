# JianWuSX1145 音频缺陷收尾交接

> 用途：交给负责维护《系统设计与技术开发文档》和《验证材料与功能测试报告》的 AI。
>
> 代码基线：`master` 以合并提交 `ef63ddb` 为起点，包含云端 `91b9e5d`（双模式音频监测）；本轮缺陷收尾代码和测试分别提交为 `1023424`、`c6e723f`、`9610a04`。

## 1. 结论

系统主线仍是视觉姿态、环境风险和确定性融合；音频是可选辅助分支。音频已经接入采集、PANNs 分析、告警、SQLite、API、前端和萤石收音入口，但不能写成“音频识别完成”或“生产级融合完成”。本轮完成的是工程链路修复和可复现性补强，不代表真实音频分类效果已经验收。

`91b9e5d` 已修复或补齐的内容包括：音频块使用 Unix 时间戳、上传音频默认使用当前时间、纯音频告警写入 `alert_events`、音频事件直接写入 `audio_events`、音视频并发事件日志加锁、原始视频 WebSocket，以及浏览器收音/视频源收音双模式。

## 2. 本轮实际修改

### 后端

- `src/inference/monitor.py`
  - 统一 `off`、`mic`、`video_source`、兼容 `camera/auto` 和直接流 URL 的解析。
  - `mic` 不再被错误改写为视频源；本地视频选择视频源收音时返回明确不支持结果。
  - 视频停止事件和音频停止事件拆分，新增 `start_audio()` / `stop_audio()`。
  - 新增 `audio_status`：`DISABLED`、`STARTING`、`RUNNING`、`UNAVAILABLE`。
  - 音频资源缺失时主视频链仍可运行，并在状态中保留缺失原因。
  - 网络音频中断最多按配置重连 3 次，默认间隔 2 秒。
  - 基线未就绪时直接处理纯音频告警，不再把同一事件留给视频线程重复处理；高等级音频告警会更新 `last_alert` 和当前风险等级。
  - 基线就绪时只将事件放入最多 100 条的联合评估队列，并按默认 15 秒时间窗口筛选。

- `src/api/routes.py`
  - 新增 `POST /api/v1/stream/audio/start`，视频监控运行中单独启动音频。
  - 新增 `POST /api/v1/stream/audio/stop`，只停止音频，不停止视频。
  - 原有 `/stream/start`、`/stream/stop` 和 `audio_source` 字段保持兼容。

- `src/api/audio_routes.py`、`src/inference/audio_analyzer.py`
  - 音频状态查询不再为了展示状态强制加载模型。
  - 状态增加 `labels_exist`、`resources_ready`、`model_status`。
  - `model_status` 可能为 `DISABLED`、`UNAVAILABLE`、`READY`、`LOADED`。

- `src/alerts/engine.py`
  - 对达到阈值的同类音频告警增加默认 30 秒冷却。
  - 冷却只抑制重复升级和通知，原始声音事件仍应由监控层写入 `audio_events`。

- `src/data/audio_capture.py`
  - `close(signal_stop=False)` 支持网络流重连时释放旧进程而不终止整个音频循环。
  - 默认仍按最多约 200ms 子块检查停止事件；ffmpeg stdout 在终止前主动关闭。

- `src/api/ezviz_routes.py`
  - 萤石监控请求默认 `audio_source=off`。
  - 选择 `video_source` 时优先尝试 RTSP 音频地址，失败后回退分析流。

### 配置与前端

- `configs/base.yaml`
  - 移除 JianWuSX1145 机器上的 ffmpeg 绝对路径，默认从 `PATH` 查找。
  - 音频默认源改为 `off`，但 `audio.enabled` 仍可通过资源准备后启用。
  - 增加音频告警冷却、音视频合并窗口和网络重连参数。

- `frontend/src/App.jsx`、`frontend/src/AudioMonitor.jsx`
  - 视频源收音的停止按钮调用独立音频停止接口，不再只改变前端文字状态。
  - 增加真实音频状态展示；模型资源缺失显示为不可用，未加载显示为待首次分析。
  - 浏览器录音上传使用 Unix 时间戳，避免把相对录音时长写入实时历史查询。

## 3. 测试结果

- 音频专项及萤石回归：`106 passed, 2 skipped`。
- 全量 Python 测试：`253 passed, 2 skipped`。
- 两个跳过项属于需要真实外部 PANNs 资源的模型测试，不代表模型已通过实机验收。
- 前端 `npm run build`：成功，Vite 生成生产包；仍有大于 500 kB 的 chunk 警告，不影响本次功能验收。
- `git diff --check`：通过；未发现 Git 冲突标记。
- 测试运行建议设置 `MPLBACKEND=Agg`，否则无界面环境下绘图测试可能尝试启动 Tk。

## 4. 外部前置条件与未完成项

以下事项不能由本轮代码修改替代，技术文档必须继续标记为“待提供/待验证”：

1. PANNs `Cnn14_mAP=0.431.pth` 和 `Path.home()/panns_data/class_labels_indices.csv` 不在 Git 仓库中，需要模型/部署负责人提供来源、版本、许可证和 SHA-256。
2. 需要安装可执行的 ffmpeg，并验证真实 RTSP/RTMP 流包含音频轨道；麦克风模式需要系统录音权限和可用设备。
3. 尚无带标注真实音频集上的准确率、召回率、误报率、阈值标定和端到端告警时延，不能填写“误报率<5%”或“500 ms 内完成”等指标。
4. 音频分数是 PANNs 输出分数，不是经过校准的概率；音频不是跌倒结论，只能作为呼救/撞击等辅助事件。
5. 音频队列仍是进程内最多 100 条；进程重启可能丢失尚未完成联合处理的事件。真实生产部署仍需继续验证异常恢复、长时间运行和音视频时钟误差。
6. `yolo26n.pt` 与 `yolo26n-pose.pt` 是 COCO/COCO-Pose 通用预训练权重，不是项目最终跌倒风险模型；环境检测 MVP 与主监控统一部署仍需单独说明。
7. Docker、多人身份独立跟踪、边缘隐私、人脸模糊、声光报警、监护人 APP 推送和急救平台联动，只有存在现场证据时才能写“已验收”；当前代码和测试不能替代这些验收。

## 5. 两份技术文档的修改口径

- 文档基线更新为 `ef63ddb` 及本轮 `1023424`、`c6e723f`、`9610a04`，不要继续只写 `17f641a`。
- 音频源统一描述为：浏览器麦克风（上传分析）和视频源收音（后端音频线程）；后端仍兼容 `mic`、`camera`、`auto` 和直接流 URL。
- 将旧的 `audio_status=UNAVAILABLE` 单一描述改为当前 `audio_status` 四态和 `/audio/status` 的 `model_status` 资源状态。
- 已解决的问题写成“代码已修复，待回归/实机验证”：Unix 时间戳、纯音频告警历史入库、并发事件日志、ffmpeg 停止释放。
- 仍需保留：真实数据指标、阈值标定、音视频时间同步、队列持久化、异常恢复和重复告警现场验证。
- 删除或改写与当前证据矛盾的表述，例如“Docker、多人场景、边缘隐私、声光告警和移动端推送已完成现场验收”。
- 音频章节必须标注“部分完成/辅助模态/待真实资源和数据验证”，不能把 `Shout`、`Slam`、`Thump` 等标签写成“检测到跌倒”。

本文件只作为交接材料，不直接修改两份现有 `.docx`；文档 AI 应依据上述代码路径、接口和测试结果更新正式文档。
