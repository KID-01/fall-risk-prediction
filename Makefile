# ============================================================
# 跌倒风险预测系统 - 常用命令
# 使用方法: make <目标>
# 例如: make install   (安装依赖)
#       make train     (训练模型)
#       make serve     (启动API服务)
# ============================================================

.PHONY: help install install-dev setup check test lint format clean train serve docker-build docker-up docker-down

# ── 默认目标：显示帮助 ──
help:
	@echo "============================================="
	@echo " 跌倒风险预测系统 - 可用命令"
	@echo "============================================="
	@echo ""
	@echo " 环境准备:"
	@echo "   make install      安装运行依赖"
	@echo "   make install-dev  安装开发依赖(含测试/代码检查)"
	@echo "   make setup        完整初始化(推荐首次使用)"
	@echo "   make check        检查环境是否就绪"
	@echo ""
	@echo " 代码质量:"
	@echo "   make test         运行所有测试"
	@echo "   make lint         代码检查 (ruff)"
	@echo "   make format       自动格式化代码"
	@echo ""
	@echo " 训练:"
	@echo "   make train        训练模型"
	@echo ""
	@echo " 运行:"
	@echo "   make serve        启动后端API服务"
	@echo ""
	@echo " Docker:"
	@echo "   make docker-build 构建Docker镜像"
	@echo "   make docker-up    启动Docker服务"
	@echo "   make docker-down  停止Docker服务"
	@echo ""
	@echo " 清理:"
	@echo "   make clean        清理临时文件"
	@echo ""

# ── 安装依赖 ──
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ── 完整初始化 ──
setup: install-dev
	@echo ">>> 初始化完成！运行 make check 检查环境"

# ── 环境检查 ──
check:
	python scripts/check_env.py

# ── 代码检查 ──
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# ── 测试 ──
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html

# ── 训练模型 ──
train:
	python scripts/train.py

# ── 启动API服务 ──
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Docker ──
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# ── 清理 ──
clean:
	@echo "清理 __pycache__ 和临时文件..."
	@powershell -Command "Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force"
	@powershell -Command "Get-ChildItem -Recurse -Filter '*.pyc' | Remove-Item -Force"
	@echo "清理完成"