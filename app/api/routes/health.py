"""GET /v1/health（契约 §2.7）健康检查。

语义（v1.3 明确）：核心依赖（vector_store/bm25）不可用 → HTTP 503 status=degraded；
Redis checkpointer 不可用时系统按设计降级内存继续服务 → checks.redis=down 但 HTTP 200
status=degraded（核心链路仍可用）。llm 存在即视为 up（provider=stub 降级属功能可用）。
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.deps import get_services
from app.models.api import HealthResponse

router = APIRouter(tags=["health"])

_VERSION = "0.1.0"


@router.get("/health")
async def health(request: Request):
    services = get_services(request)
    checks = {
        "vector_store": "up" if services.vector_store.health() else "down",
        "bm25": "up" if services.bm25_store.health() else "down",
        "redis": "down" if services.agent._ckpt_degraded else "up",
        "llm": "up",
    }
    core_down = checks["vector_store"] == "down" or checks["bm25"] == "down"
    status = "ok" if (not core_down and checks["redis"] == "up") else "degraded"
    body = HealthResponse(status=status, version=_VERSION, checks=checks)
    return JSONResponse(
        status_code=503 if core_down else 200, content=body.model_dump())
