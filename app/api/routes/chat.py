"""POST /v1/chat（契约 §2.1）：SSE 流式问答 + stream=false 非流式。

SSE 事件序列按契约：ready → (token)* → (citation)* → done | error；空闲超时 ping 保活。
客户端断开 → StreamingResponse 停止消费（worker 线程 daemon，不再转发）；
并发在途闸（Semaphore，默认 50）满 → 429 RATE_LIMITED（契约错误码表）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from app.api.deps import check_thread_tenant, get_request_id, get_services
from app.api.errors import ApiError
from app.core.config import settings
from app.models.api import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _fmt(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def chat(req: ChatRequest, request: Request, response: Response):
    services = get_services(request)
    tenant_id = request.state.tenant_id
    check_thread_tenant(req.thread_id, tenant_id)
    rid = get_request_id(request)
    inflight: asyncio.Semaphore = request.app.state.inflight

    if inflight.locked():
        raise ApiError(
            "RATE_LIMITED", "并发请求已满，请稍后重试", 429,
            {"inflight_limit": settings.api_max_inflight})

    # ── stream=false：单次回复（非 SSE，契约 §2.1）──────────
    if not req.stream:
        t0 = time.perf_counter()
        async with inflight:
            reply = await asyncio.to_thread(
                services.agent.reply, req.question, req.thread_id,
                include_expired=req.include_expired)
        reply.request_id = rid
        reply.latency_ms = int((time.perf_counter() - t0) * 1000)
        response.headers["X-Request-Id"] = rid
        return reply

    # ── stream=true：SSE 流式 ─────────────────────────────
    _END = object()

    def _anext(it):
        # 将同步生成器逐事件推进到线程；StopIteration 经 to_thread 的 Future 不能
        # 直接传播（Py3.12+ 会转 RuntimeError）→ 用哨兵收尾。
        try:
            return next(it)
        except StopIteration:
            return _END

    async def gen():
        # 图在 worker 线程执行 + token_sink 跨线程回传（stream_events 内部处理）
        it = services.agent.stream_events(
            req.question, req.thread_id,
            include_expired=req.include_expired, request_id=rid)
        async with inflight:
            try:
                while True:
                    ev = await asyncio.to_thread(_anext, it)
                    if ev is _END:
                        return
                    yield _fmt(ev["type"], ev["data"])
            except Exception as exc:  # noqa: BLE001 — 流中异常转 error 事件（HTTP 保持 200）
                logger.exception("SSE 流异常 thread=%s", req.thread_id)
                yield _fmt("error", {
                    "code": "INTERNAL_ERROR", "message": "生成过程异常",
                    "request_id": rid, "details": {"hint": str(exc)[:200]}})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": rid,
            "Connection": "keep-alive",
        })
