# 跌倒风险预测系统 — 使用方法

## 一、环境准备（首次使用）

```powershell
# 1. 进入项目目录
cd d:\tiaozhanbei\fall-risk-prediction【改成你自己的路径】

# 2. 创建虚拟环境
py -3.14 -m venv venv

# 3. 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 4. 安装依赖
pip install -e ".[dev]"

# 5. 环境检查（需设置 UTF-8 编码以支持中文和特殊符号输出）
$env:PYTHONIOENCODING="utf-8"; chcp 65001 > $null
python scripts/check_env.py
```

> **说明**：当前系统未安装 `make`，所有 `make xxx` 命令需用 venv 中的 python 直接执行等效命令。

---

## 二、启动 API 服务

### 方式一：命令行启动（推荐开发使用）

```powershell
cd d:\tiaozhanbei\fall-risk-prediction
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 方式二：Python 直接启动

```powershell
cd d:\tiaozhanbei\fall-risk-prediction
.\venv\Scripts\python.exe src\api\main.py
```

### 方式三：后台启动（无窗口）

```powershell
cd d:\tiaozhanbei\fall-risk-prediction
Start-Process -FilePath ".\venv\Scripts\python.exe" `
    -ArgumentList "-m","uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000" `
    -WindowStyle Hidden
```

启动后访问：
- **API 根路径**：http://localhost:8000
- **Swagger 文档**：http://localhost:8000/docs
- **ReDoc 文档**：http://localhost:8000/redoc

---

## 三、停止服务

```powershell
# 停止所有 python 进程（会同时停止服务）
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# 或按端口查找并停止
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess |
    ForEach-Object { Stop-Process -Id $_ -Force }
```

---

## 四、API 接口说明

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 系统信息（名称、版本、状态） |
| `/health` | GET | 健康检查 |
| `/config` | GET | 获取完整系统配置（来自 `configs/base.yaml`） |
| `/alerts/levels` | GET | 预警分级规则（绿/黄/橙/红） |
| `/predict` | POST | 跌倒风险预测（模型推理待实现，当前为占位接口） |
| `/docs` | GET | Swagger 交互式 API 文档 |

### 接口调用示例

```powershell
$base = "http://localhost:8000"

# 健康检查
Invoke-RestMethod "$base/health"

# 获取预警分级
Invoke-RestMethod "$base/alerts/levels"

# 预测接口（占位）
Invoke-RestMethod "$base/predict?video_url=rtsp://example/stream" -Method Post
```

---

## 五、常用命令对照（Makefile → 等效命令）

系统中未安装 `make`，以下是等效命令（需先激活虚拟环境）：

| Make 命令 | 等效命令 | 说明 |
|-----------|---------|------|
| `make install` | `pip install -e .` | 安装运行依赖 |
| `make install-dev` | `pip install -e ".[dev]"` | 安装开发依赖 |
| `make check` | `python scripts/check_env.py` | 环境检查 |
| `make serve` | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload` | 启动 API 服务 |
| `make lint` | `ruff check src/ tests/` | 代码检查 |
| `make format` | `ruff format src/ tests/` | 自动格式化 |
| `make test` | `pytest tests/ -v --tb=short` | 运行测试 |
| `make clean` | 见下方清理命令 | 清理临时文件 |

### 清理临时文件

```powershell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

---

## 六、注意事项

1. **编码问题**：Windows 控制台默认 GBK 编码，运行含中文/特殊符号的脚本前需设置：
   ```powershell
   $env:PYTHONIOENCODING="utf-8"; chcp 65001 > $null
   ```

2. **CUDA 不可用**：当前安装的是 CPU 版 PyTorch（`torch 2.13.0+cpu`），推理速度较慢。如需 GPU 加速，需安装对应 CUDA 版本的 PyTorch。

3. **配置文件**：系统配置位于 `configs/base.yaml`，修改后重启服务生效。

4. **项目状态**：当前仅实现了 API 脚手架，模型训练、推理引擎、预警引擎等核心模块待开发。`/predict` 接口目前为占位返回。
