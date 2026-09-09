"""GET /v1/tasks/{task_id}（契约 §2.3）任务状态查询。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import get_services
from app.api.errors import ApiError

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}")
async def task_status(task_id: str, request: Request):
    services = get_services(request)
    rec = services.task_manager.get(task_id)
    if rec is None:
        raise ApiError("INGEST_NOT_FOUND", f"任务 {task_id} 不存在", 404)
    return rec.to_status()
