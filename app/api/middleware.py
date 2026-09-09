"""请求上下文中间件（契约 §1.3/§1.4 + 可选鉴权）。

- X-Request-Id：客户端透传或服务端生成（req_ 前缀），贯穿日志与错误体；响应头回显；
- 鉴权（可选，配置 SERVICE_API_KEY 后启用）：Authorization: Bearer <key> 校验；
  未配置 → 跳过（本地/内网 demo 部署），日志提示一次；
- X-Tenant-Id 解析放入 request.state（路由 deps 读取），不强制（默认租户兜底）。
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.observability import TimedSpan
from app.models.api import ErrorBody

logger = logging.getLogger(__name__)
_warned_no_auth = False


def _gen_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def register_middleware(app: FastAPI) -> None:
    global _warned_no_auth

    if not settings.service_api_key and not _warned_no_auth:
        logger.warning("未配置 SERVICE_API_KEY → 鉴权关闭（仅限本地/内网 demo；生产必须配置）")
        _warned_no_auth = True

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-Id") or _gen_request_id()
        request.state.request_id = rid
        request.state.tenant_id = request.headers.get(
            "X-Tenant-Id", settings.default_tenant_id)

        if settings.service_api_key:
            auth = request.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else ""
            if token != settings.service_api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": ErrorBody(
                        code="AUTH_MISSING_KEY" if not token else "AUTH_INVALID_KEY",
                        message="鉴权失败", request_id=rid).model_dump()},
                    headers={"X-Request-Id": rid})

        try:
            span = TimedSpan(name="http_request")
            response = await call_next(request)
            duration_ms = span.stop(log_slow=False)
            # 结构化请求日志（req_id 贯穿，观测/链路追踪用）
            logger.info(
                "http_request method=%s path=%s status=%s duration_ms=%.1f",
                request.method, request.url.path, response.status_code, duration_ms)
        except Exception:  # noqa: BLE001 — 未捕获异常由 error_handlers 统一兜底
            raise
        response.headers["X-Request-Id"] = rid
        return response
