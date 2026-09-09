"""POST /v1/debug/retrieve（契约 §2.6，P1 仅供开发）：双路明细 + RRF 融合。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import get_services
from app.models.api import (DebugFusedItem, DebugRetrieveRequest,
                            DebugRetrieveResponse)

router = APIRouter(prefix="/debug", tags=["debug"])


def _lanes_detail(res, fused: list[dict]) -> list[DebugFusedItem]:
    vec_rank = {h["chunk_id"]: (i + 1, h.get("score")) for i, h in enumerate(
        res.detail.get("baseline", {}).get("vector_hits", []) or [])}
    bm_rank = {h["chunk_id"]: (i + 1, h.get("bm25_score")) for i, h in enumerate(
        res.detail.get("baseline", {}).get("bm25_hits", []) or [])}
    out: list[DebugFusedItem] = []
    for rank, it in enumerate(fused, start=1):
        m = it.get("metadata", {})
        vr, vs = vec_rank.get(it["chunk_id"], (None, None))
        br, bs = bm_rank.get(it["chunk_id"], (None, None))
        out.append(DebugFusedItem(
            chunk_id=it["chunk_id"],
            doc_title=m.get("doc_title", ""),
            validity=it.get("validity", "valid"),
            fused_rank=rank,
            fused_score=float(it.get("score", 0.0)),
            vector_rank=vr, vector_score=float(vs) if vs is not None else None,
            bm25_rank=br, bm25_score=float(bs) if bs is not None else None,
            content=(it.get("content") or "")[:200],
        ))
    return out


@router.post("/retrieve")
async def debug_retrieve(req: DebugRetrieveRequest, request: Request):
    from app.retrieval.hybrid import allowed_permissions

    services = get_services(request)
    res = services.retriever.retrieve(
        req.query, department=req.department or None)
    resp = DebugRetrieveResponse(
        fused=_lanes_detail(res, res.items),
        filters={
            "is_valid": True,
            "tenant_id": services.retriever.tenant_id,
            "permission": allowed_permissions("internal"),
            "expired_allowed": req.include_expired,
            "department": req.department,
        },
        degraded=list(res.degraded),
        notes=list(res.notes),
    )
    # 过期候选：显式请求放宽 或 现行不足自动兜底
    if req.include_expired or not res.items:
        relaxed = services.retriever.relaxed_retrieve(req.query, top_n=req.top_k)
        resp.expired_candidates = _lanes_detail(relaxed, relaxed.expired_candidates)
        resp.degraded.extend(relaxed.degraded)
        resp.notes.extend(relaxed.notes)
    return resp
