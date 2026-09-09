"""GET /v1/threads/{thread_id}/history（契约 §2.5）对话历史。

注意：history 接口返回条数与 Agent 上下文窗口（F4.3 最近 10 轮）无关。
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.api.deps import check_thread_tenant, get_services
from app.api.errors import ApiError
from app.models.api import HistoryResponse

router = APIRouter(prefix="/threads", tags=["threads"])


@router.get("/{thread_id}/history")
async def thread_history(
    thread_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    services = get_services(request)
    tenant_id = request.state.tenant_id
    check_thread_tenant(thread_id, tenant_id)
    msgs = services.agent.history(thread_id, limit=limit)
    return HistoryResponse(thread_id=thread_id, messages=msgs)
