"""FastAPI 应用组装（M4）：lifespan 构建共享服务 + 中间件 + 错误处理 + 路由。"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import BASE_DIR

from app.api.deps import Services
from app.api.errors import register_error_handlers
from app.api.middleware import register_middleware
from app.api.routes import chat, debug, documents, health, tasks, threads
from app.core.config import settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = Services()
        app.state.services = services
        app.state.inflight = asyncio.Semaphore(settings.api_max_inflight)
        logger.info("API 启动: version=0.1.0 max_inflight=%s tenant=%s",
                    settings.api_max_inflight, settings.default_tenant_id)
        yield
        services.close()
        logger.info("API 关闭")

    app = FastAPI(
        title="Enterprise-QA-Agent API",
        description="混合检索 + LangGraph 多 Agent 企业问答（契约 docs/api-contract.md）",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # 本地/内网部署；生产按域名收紧
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )

    register_middleware(app)
    register_error_handlers(app)

    api_prefix = "/api/v1"
    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)
    app.include_router(tasks.router, prefix=api_prefix)
    app.include_router(threads.router, prefix=api_prefix)
    app.include_router(debug.router, prefix=api_prefix)
    app.include_router(health.router, prefix=api_prefix)

    # ── 前端静态页（M7）：根路径 → 提问页；/web 下挂载静态资源 ──
    web_dir = BASE_DIR / "web"
    if web_dir.is_dir():
        @app.get("/", include_in_schema=False)
        async def _root():
            return RedirectResponse(url="/index.html")

        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = create_app()
