"""API 异常体系（契约 §1.5 / §3 错误码）：统一 {error:{code,message,request_id,details}}。

规则：
- 业务错误抛 `ApiError`（HTTP + code + 用户可读 message，禁止堆栈）；
- 未捕获异常 / 请求校验失败 → 全局 handler 转 INTERNAL_ERROR / VALIDATION_INVALID_ARGUMENT；
- request_id 贯穿（从 request.state 读取，见 middleware）。
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.api import ErrorBody

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """业务错误：调用方应只读 code/message/status_code（安全），细节入 details。"""

    def __init__(self, code: str, message: str, status_code: int = 400,
                 details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def _body(code: str, message: str, request_id: str, details: dict | None = None) -> dict:
    return ErrorBody(code=code, message=message, request_id=request_id,
                     details=details).model_dump()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": _body(exc.code, exc.message, _request_id(request), exc.details)},
            headers={"X-Request-Id": _request_id(request)},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = {}
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            details[loc or "body"] = err.get("msg", "")
        return JSONResponse(
            status_code=422,
            content={"error": _body(
                "VALIDATION_INVALID_ARGUMENT", "请求参数校验失败", _request_id(request), details)},
            headers={"X-Request-Id": _request_id(request)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # 兜底（如路由未匹配 404 / 方法不允许 405）
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
                401: "AUTH_INVALID_KEY", 403: "AUTH_FORBIDDEN"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": _body(code, str(exc.detail), _request_id(request))},
            headers={"X-Request-Id": _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未捕获异常 request=%s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": _body(
                "INTERNAL_ERROR", "服务内部错误，请携带 request_id 反馈", _request_id(request))},
            headers={"X-Request-Id": _request_id(request)},
        )
