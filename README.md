# 跌倒风险预测系统

> 基于多模态AI监测的老年人跌倒风险前置防控
>
> 挑战杯 · 揭榜挂帅赛道 · 发榜单位: 海康威视/萤石

---

## 项目简介

本项目针对居家养老场景中老年人跌倒风险，实现 **"跌前预判 → 跌时识别 → 跌后响应"** 三级防护体系。

核心创新：**从"跌倒后识别"升级为"以跌倒风险前置防控为核心"的全流程方案**。

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 深度学习 | PyTorch, ONNX Runtime |
| 姿态估计 | MoveNet / RTMPose |
| 后端 | FastAPI + Redis + SQLite |
| 前端 | React + ECharts |
| 部署 | Docker + Nginx |
| 平台 | 萤石开放平台 |

## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd fall-risk-prediction

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
make install-dev

# 4. 检查环境
make check

# 5. 启动服务
make serve
```

## 项目结构

```
fall-risk-prediction/
├── src/
│   ├── data/          # 数据采集、预处理、Dataset定义
│   ├── models/        # 所有模型定义
│   ├── inference/     # 推理引擎封装
│   ├── alerts/        # 分级预警引擎
│   ├── api/           # FastAPI 后端
│   ├── edge/          # 边缘计算 & 隐私保护
│   └── utils/         # 通用工具函数
├── configs/           # YAML 配置文件
├── scripts/           # 采集、训练、部署脚本
├── tests/             # 单元测试 & 集成测试
├── notebooks/         # Jupyter 探索分析
├── frontend/          # React 前端
├── docker/            # Dockerfile & compose
├── checkpoints/       # 模型权重 (gitignore)
├── data/              # 数据集 (gitignore)
├── pyproject.toml     # 项目配置 & 依赖
└── Makefile           # 常用命令
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `make check` | 检查开发环境 |
| `make train` | 训练风险预测模型 |
| `make evaluate` | 评估模型性能 |
| `make serve` | 启动后端API服务 |
| `make test` | 运行所有测试 |
| `make lint` | 代码检查 |
| `make format` | 自动格式化 |
| `make docker-up` | 启动Docker服务 |

## 许可证

MIT