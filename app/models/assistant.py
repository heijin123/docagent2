"""契约 4.1 对话/引用模型（api-contract.md §4.1 的权威实现，M3+ API 复用）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """单条引用（SSE citation 事件 / done.citations 项；契约 §2.1）。"""

    chunk_id: str
    doc_title: str = ""
    page_num: int = 0
    validity: str = "valid"               # valid / expired（枚举 §5）
    expired_at: int | None = None         # expired 必有
    doc_date: str | None = None           # F2.9 文档期次（年份错位自查）
    image_ids: list[str] = Field(default_factory=list)  # F1.10 图片引用（M1 占位）


class AssistantReply(BaseModel):
    """done 事件 data / stream=false 响应体（契约 §2.1 AssistantReply）。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    degraded: bool = False
    intent: str = "kb_qa"
    request_id: str = ""
    latency_ms: int = 0
    notes: list[str] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)   # 该轮链路 token 用量（贯穿成本核算，M3+）
