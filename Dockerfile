# Enterprise-QA-Agent 生产镜像（M5）
# 多阶段：builder 用 uv 装依赖 → runtime 精简；非 root 运行。
FROM python:3.13-slim AS builder

ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

# 安装 uv（官方安装脚本）
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app
# 先复制依赖清单，利用缓存层。
# 注意：本项目 .gitignore 忽略 uv.lock（不提交锁文件），故这里不能 COPY uv.lock
# 也不能用 --frozen（缺锁文件会直接构建失败）。uv sync 会在构建时解析依赖并在
# 镜像内生成 uv.lock。若日后要可复现构建：把 uv.lock 移出 .gitignore 并提交，
# 再把下面两处改为 `COPY pyproject.toml uv.lock ./` + `--frozen`。
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# 复制源码
COPY app ./app
COPY scripts ./scripts
RUN uv sync --no-dev

# ── runtime ──────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# 运行依赖（Chroma/PyMuPDF 需要的运行时库）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app ./app
COPY --from=builder /app/scripts ./scripts

# data 目录（运行态，挂卷覆盖；保证容器首次启动可写）
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser
EXPOSE 8000

# 健康检查：依赖 /health 语义（503=degraded 也视为存活，故检查 2xx/503 均算通过）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/health || \
        curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/v1/health | grep -q '503'

# 多 worker 部署：生产用 uvicorn 单进程多 worker 会各自建 Chroma 连接（并发写风险），
# 默认单进程；扩容请走多副本（见部署手册 §容量规划）。
CMD ["/app/.venv/bin/uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
