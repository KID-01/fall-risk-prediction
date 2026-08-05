# ============================================================
# 跌倒风险预测系统 — Dockerfile
# 构建: docker build -t fall-risk-prediction .
# 运行: docker run -p 8000:8000 fall-risk-prediction
# ============================================================

FROM python:3.12-slim

# 系统依赖 (OpenCV + MediaPipe 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖 (利用 Docker 层缓存)
COPY pyproject.toml Makefile ./
RUN pip install --no-cache-dir -e "."

# 拷贝源码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 创建数据目录
RUN mkdir -p data/raw data/processed data/keypoints checkpoints logs

# 暴露端口
EXPOSE 8000

# 环境变量
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
