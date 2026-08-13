# MXJ个人汇报总结

**汇报人：** MXJ（Git 提交者 `MengXJ0410`）  
**项目：** 跌倒风险预测系统  
**Git 依据：** 20 次提交（含 1 次合并）。重点工作集中于 `c7d3ee6`（2026-07-10，核心算法管线）、`3fc98c6`（2026-07-15，系统工程化）和 `80082a9`（2026-08-05，联调修正）。

## 一、个人贡献定位

我在项目中承担的核心工作不是单独实现某一个算法或页面，而是将“跌倒风险前置预警”拆解为可运行的完整链路，并完成首版工程化落地：

```text
视频流 -> 人体/姿态关键点 -> 步态特征 -> 个人正常基线
     -> 短期/长期偏离检测 -> 四级预警 -> API/看板展示
```

其中，贡献最集中的部分为：

1. **核心风险识别方法的设计与实现**：将原始关键点时序转化为可解释的步态指标，并以个体自身历史数据为基准识别异常；
2. **端到端实时监控服务的搭建**：把多模块组织为可启动、可查询、可演示的监控系统；
3. **工程化架构与关键联调修正**：实现首版 API、持久化、前端和容器化交付，并解决连续运行时会影响结果正确性的状态问题。

下文重点说明这三项工作。模型训练、数据脚本、部署和文档等作为支撑工作在最后简述。

---

## 二、重点贡献一：个体化步态风险识别闭环

### 1. 要解决的问题

不同老人的正常走路速度、步幅和身体姿态差异很大。直接使用统一阈值容易将“个体正常差异”误判为风险。因此，我实现的核心思路是：**不只判断动作是否符合统一标准，而是先建立个人正常步态基线，再判断当前状态是否偏离该基线。**

这条链路由“特征提取 -> 基线建模 -> 双层偏离检测 -> 分级预警”构成。

### 2. 从关键点到四类可解释步态特征

我将姿态估计输出的关键点序列统一封装为四维 `FeatureVector`，避免后续模块直接处理高维、噪声较大的原始坐标。四项特征均采用相对值或角度变化，减少摄像头距离、拍摄角度和人体身高带来的影响。

| 特征 | 实现方法 | 风险含义 | 代码对应 |
| --- | --- | --- | --- |
| 行走节拍频率 | 对左右髋关节平均 y 坐标的时序信号去均值后做 FFT，在 0.5-5Hz 范围内提取主频。 | 反映行走节奏变化。 | `src/inference/features.py`：`WalkingRhythmCalculator.calculate()` |
| 步幅相对幅度 | 分别计算左右踝关节 x 坐标摆动范围，再用躯干高度归一化。 | 反映步幅缩小或步态不对称。 | `src/inference/features.py`：`StepAmplitudeCalculator.calculate()` |
| 躯干稳定指数 | 计算肩部中点到髋部中点连线与竖直方向的夹角，并取时间窗口内的变化范围。 | 角度波动越大，表示躯干摇晃越明显。 | `src/inference/features.py`：`TrunkStabilityCalculator._trunk_angle()`、`calculate()` |
| 活动密度 | 基于相邻帧髋部中点位移，统计超过运动阈值的帧占比。 | 反映活动水平下降或运动不足。 | `src/inference/features.py`：`ActivityDensityCalculator.calculate()` |

四类计算器由 `FeatureCalculator.calculate()` 统一调度，输出的 `FeatureVector.to_array()` 是后续统计检测的唯一输入接口。这样做使“姿态提取模块”和“风险检测模块”解耦，后续切换姿态后端时无需重写风险算法。

### 3. 个体化基线：从“统一阈值”改为“与自己比较”

我实现了以 `person_id` 为索引的基线管理器。系统在基线采集期保存该对象的有效特征样本，基线成熟后计算：

- 四项特征的均值 `mean`，描述该对象的常态；
- 标准差 `std`，用于识别单项指标的异常程度；
- 协方差逆矩阵 `cov_inv`，用于计算同时考虑特征相关性的马氏距离；
- 基线状态 `is_ready` 和样本量 `sample_count`，阻止样本不足时误进入风险判断。

配置中将基线采集设为 **7 天、至少 100 个有效样本**；当样本不足时，仅显示“基线采集中”，不对用户输出不可靠的风险结论。

| 关键能力 | 代码对应 | 说明 |
| --- | --- | --- |
| 特征样本按人员持久化 | `src/inference/baseline.py`：`BaselineManager.add_sample()`、`get_samples()` | SQLite 表以 `person_id` 区分对象，避免不同人员的数据混用。 |
| 基线参数计算 | `BaselineManager.compute_baseline()` | 计算均值、标准差和协方差；对协方差矩阵加入 `1e-6` 正则项，降低奇异矩阵导致计算失败的风险。 |
| 多维异常度量 | `IndividualBaseline.mahalanobis_distance()` | 计算 `sqrt((x-mean)^T cov_inv (x-mean))`，兼顾多项特征的联合变化。 |
| 单项异常解释 | `IndividualBaseline.z_scores()` | 生成四个维度的 Z-Score，便于定位是节拍、步幅、躯干还是活动量异常。 |
| 生命周期管理 | `load_baseline()`、`reset_baseline()` | 支持重新启动后加载既有基线，也支持按人员重建。 |

### 4. 双层偏离检测与四级预警

基线建成后，系统不以单帧异常立即报警，而是分开处理短时波动和长期退化：

| 层级 | 判定逻辑 | 默认配置 | 代码对应 |
| --- | --- | --- | --- |
| 短期偏离 | 在滑动窗口内计算特征均值与个人基线的马氏距离；连续超阈值才触发。 | 5 分钟窗口、30 秒步长、距离阈值 3.0、连续 3 窗口。 | `src/inference/deviation.py`：`ShortTermDetector.add_and_check()` |
| 长期趋势 | 保存每日特征均值，对每个特征做线性回归，识别持续负向斜率。 | 14 天窗口、至少 7 天负向变化、斜率阈值 -0.05。 | `LongTermDetector.add_daily_mean()`、`check_trend()` |
| 综合判定 | 合并短期和长期结果，输出无异常、短期异常、长期下降或两者同时发生。 | `DeviationResult` 带距离、Z-Score、斜率和文字说明。 | `DeviationDetector.check()` |
| 风险升级 | 将综合结果映射为低风险、关注级、预警级、高危级；支持注册不同等级的通知动作。 | 严重短期异常、短期+长期同时触发、持续无活动可升至高危。 | `src/alerts/engine.py`：`AlertEngine.evaluate()`、`register_action()` |

这部分的价值在于：既避免把短暂的检测噪声当作风险，也能捕捉到日常步态逐渐退化这一“跌倒前兆”。算法闭环测试由 `scripts/test_pipeline.py` 的 `test_features()`、`test_baseline()` 和 `test_deviation_and_alerts()` 覆盖。

---

## 三、重点贡献二：端到端实时监控服务

### 1. 将算法模块组织为可运行服务

上述算法如果仅以独立脚本存在，无法接入视频、展示状态或支持实际演示。因此我主导建立 `FallRiskMonitor` 作为运行总控：使用单例避免并发重复启动，以后台线程持续处理视频流，并对外暴露状态、风险和告警历史。

**核心代码：** `src/inference/monitor.py` 的 `FallRiskMonitor`。

`FallRiskMonitor._run()` 将完整流程固化为八个明确阶段：

| 阶段 | 执行内容 | 核心代码 |
| --- | --- | --- |
| 1 | 从摄像头或本地文件读取视频帧，并进行人体检测；YOLO-Pose 后端可跳过独立人体检测。 | `VideoCapture.frames()`；`HumanDetector.detect_best()` |
| 2 | 提取人体关键点。 | `create_keypoint_extractor()`；`KeypointExtractor.extract()` |
| 3 | 按关键点置信度、下肢和躯干可见性过滤低质量帧。 | `FrameFilter.filter()` |
| 4 | 在最近 30 帧缓存上计算步态特征。 | `FeatureCalculator.calculate()` |
| 5 | 基线未就绪时采集样本，达到条件后构建个人基线。 | `BaselineManager.add_sample()`、`compute_baseline()` |
| 6 | 基线就绪后运行短期/长期偏离检测。 | `DeviationDetector.check()` |
| 7 | 生成四级风险事件与预警文本。 | `AlertEngine.evaluate()` |
| 8 | 写入风险记录与非低风险告警事件。 | `Database.insert_risk_record()`、`insert_alert_event()` |

这种组织方式把每个模块的输入输出固定下来：上游只需要提供有效的 `KeypointFrame`，下游统一消费 `FeatureVector` 和 `DeviationResult`，使算法替换、接口扩展和演示脚本能够独立演进。

### 2. 服务接口与可观测状态

为让监控服务可被前端和演示脚本调用，我在 FastAPI 首版工程化中完成了路由拆分和关键接口。核心路由位于 `src/api/routes.py`：

| 接口 | 对应函数 | 对监控服务的作用 |
| --- | --- | --- |
| `POST /api/v1/stream/start` | `stream_start()` | 传入视频源、`person_id` 和 `device_id`，启动后台监控线程。 |
| `POST /api/v1/stream/stop` | `stream_stop()` | 停止线程并关闭视频资源。 |
| `GET /api/v1/risk/current` | `risk_current()` | 返回帧处理数、基线进度、最新特征、偏离结果和风险等级。 |
| `POST /api/v1/baseline/reset` | `baseline_reset()` | 重置指定人员基线，支持重新采集。 |
| `GET /api/v1/risk/history` | `risk_history()` | 查询持久化风险记录。 |
| `GET /api/v1/alerts` | `get_alerts()` | 查询历史告警，支持等级、人员、时间和确认状态筛选。 |

其中，`FallRiskMonitor.get_status()` 特别区分“基线采集中”和“风险可评估”状态：基线未就绪时返回 `risk_evaluable=False`，避免前端将默认低风险误读为真实评估结果。

### 3. 首版可演示系统工程化

2026-07-15 的 `3fc98c6` 提交中，我将算法链路扩展为首版可部署系统：

- `src/api/main.py`：注册监控、告警、统计三类路由，提供健康检查、配置和特征说明接口；
- `src/api/database.py`：建立风险记录和告警事件的 SQLite 数据访问层；
- `src/api/websocket.py`：实现 `ConnectionManager` 与 `/ws/alerts` 实时推送入口；
- `frontend/src/App.jsx`：实现实时状态轮询、风险仪表盘、趋势图、告警列表和基线控制；
- `docker/Dockerfile`、`docker/docker-compose.yml`、`docker/nginx.conf`：提供前后端容器化与反向代理配置。

这部分的个人贡献定位是**主导首版架构和功能框架搭建**。其中数据库持久化以及监控模块后续有其他成员继续参与测试加固和修改，具体演进以 Git 历史为准。

---

## 四、重点贡献三：联调中发现并修正结果正确性问题

2026-08-05 的 `80082a9` 不是简单的界面调整，而是围绕“系统连续运行时的结果是否可信”进行修正。相关问题和处理如下。

### 1. 防止跨视频状态污染

**问题：** 上一次视频留下的关键点缓存、短期偏离窗口和告警计数会进入下一次运行，导致第二段视频在初始阶段就可能被错误判断。

**修正：** 在 `FallRiskMonitor.start()` 中清空 `_keypoint_buffer`，并调用 `DeviationDetector.reset()` 清空短期窗口/长期数据、调用 `AlertEngine.reset()` 清空告警计数与事件。

**代码对应：**

- `src/inference/monitor.py`：`FallRiskMonitor.start()`
- `src/inference/deviation.py`：`DeviationDetector.reset()`
- `src/alerts/engine.py`：`AlertEngine.reset()`

**结果：** 每一次启动都从独立、干净的运行状态开始，避免不同视频和不同演示轮次之间相互影响。

### 2. 防止异常行为污染“正常基线”

**问题：** 原监控循环会持续将当前特征写入基线样本。基线建立后，异常步态也会被吸收进“正常范围”，削弱后续风险识别能力。

**修正：** 将样本写入限制为 `if not baseline.is_ready`；基线准备完成后只做偏离检测，不再改写正常参照。

**代码对应：** `src/inference/monitor.py`：`FallRiskMonitor._run()` 中阶段 5 的基线采集条件。

**结果：** 保证基线代表初始正常状态，而不是被后续异常数据逐渐拉偏。

### 3. 严重短期异常的及时升级

**问题：** 仅把短期偏离标为“关注级”会延后极端异常的处置，即使马氏距离或单项 Z-Score 已明显超出正常范围。

**修正：** 在 `AlertEngine.evaluate()` 中增加高危直升规则：马氏距离达到短期阈值的两倍，或任一特征绝对 Z-Score 达到 6 时，直接判定为 `CRITICAL`。

**代码对应：** `src/alerts/engine.py`：`severe_deviation_distance`、`severe_deviation_z`、`AlertEngine.evaluate()`。

**结果：** 在高强度异常场景下，风险分级能够优先考虑及时性，而不是等待频次累积。

### 4. 监控结束状态收敛与多对象基线重置

**问题：** 本地视频读完或线程异常时，运行状态可能仍显示为启动；同时，基线重置无法明确指定对象。

**修正：**

- 用 `try/except/finally` 包裹监控循环，在退出时统一写入 `is_running=False` 并释放视频引用；
- `scripts/demo_report.py` 检测到停止状态后退出轮询；
- `/baseline/reset` 接口增加 `person_id` 参数，`reset_baseline()` 仅更新当前目标对象的状态。

**代码对应：**

- `src/inference/monitor.py`：`FallRiskMonitor._run()`、`reset_baseline()`
- `src/api/routes.py`：`baseline_reset()`
- `scripts/demo_report.py`：`start_monitor()`

**结果：** 演示流程可自然结束；在多对象场景下可准确控制基线，不会误清除当前以外对象的状态。

---

## 五、支撑性工作

除上述重点工作外，我还完成以下首版工程化支撑：

| 工作 | 代码/文档对应 | 作用 |
| --- | --- | --- |
| 训练数据与模型管线 | `src/data/dataset.py`；`scripts/train.py`；`src/models/temporal_encoder.py`、`multimodal_fusion.py`、`risk_head.py`、`fall_risk_predictor.py` | 提供关键点时序建模、多模态融合和联合风险预测能力。 |
| 视频和数据处理脚本 | `scripts/collect_video.py`、`extract_keypoints.py`、`preprocess_videos.py` | 支持视频采集、批量关键点提取和预处理。 |
| 可复现演示数据 | `scripts/generate_test_fixtures.py`、`data/keypoints/fixture_*.npy` | 在缺少真实数据或训练权重时仍可完成链路验证。 |
| 交付文档 | `docs/功能性测试指南0805.md`、`官方测试集使用方法.md`、`数据安全与合规说明.md`、`进度汇报0805.md` | 支撑测试、演示、合规说明和项目过程留痕。 |

## 六、总结

我的主要贡献可概括为：**以个体化步态基线为核心，完成从视频关键点到风险预警的算法闭环；将该闭环组织为可通过 API、前端和容器环境运行的实时监控系统；并在联调阶段解决了影响风险结果可信度的状态污染、基线污染和异常升级问题。**

关于协作边界：监控主流程、核心算法管线和首版服务架构由本人创建并持续联调；数据库持久化、YOLO-Pose 后端和监控模块的部分后续改动由团队成员协作完善，本总结未将其后续贡献归为本人独立工作。
