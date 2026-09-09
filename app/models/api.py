"""API 请求/响应模型（api-contract.md §2 的权威实现，M4）。

与契约对应关系：
- ChatRequest      ← §2.1 POST /v1/chat 请求体
- UploadResponse   ← §2.2 POST /v1/documents 三态响应体
- TaskStatus       ← §2.3 GET /v1/tasks/{id} 响应体
- HistoryResponse  ← §2.5 对话历史响应体
- DebugRetrieve    ← §2.6 POST /v1/debug/retrieve 请求体
- HealthResponse   ← §2.7 GET /v1/health 响应体
- ErrorBody        ← §1.5 统一错误体（非 SSE）
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ── §1.5 统一错误体 ─────────────────────────────────────────
class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str = ""
    details: dict | None = None


# ── §2.1 /v1/chat ──────────────────────────────────────────
class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128,
                           description="checkpointer key 维度，建议 {tenant}:{user}")
    question: str = Field(min_length=1, max_length=4000)
    stream: bool = True
    include_expired: bool = False

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question 不能为空白")
        return v


# ── §2.2 /v1/documents 上传 ────────────────────────────────
class DocumentMeta(BaseModel):
    """meta 表单 JSON。permission 刻意不入模型 → 客户端传入被忽略（防提权，§1.3）。"""
    doc_key: str | None = None
    department: str | None = None
    category: str | None = None


class UploadResponse(BaseModel):
    task_id: str
    doc_id: str
    version: int
    status: str = "pending"          # pending / processing / done / failed
    duplicated: bool = False


# ── §2.3 任务状态 ──────────────────────────────────────────
class TaskProgress(BaseModel):
    phase: str = "parse"             # parse / chunk / embed / index
    percent: int = 0
    chunks_done: int = 0


class TaskWarning(BaseModel):
    type: str = "parse_warning"
    page: int | None = None
    detail: str = ""


class TaskStatus(BaseModel):
    task_id: str
    doc_id: str | None = None
    status: str = "pending"          # pending / processing / done / failed
    progress: TaskProgress = Field(default_factory=TaskProgress)
    warnings: list[TaskWarning] = Field(default_factory=list)
    error: dict | None = None        # failed 时 {code, message}
    duplicated: bool = False
    created_at: int = 0
    updated_at: int = 0


# ── §2.4 DELETE 幂等 ───────────────────────────────────────
class DeleteResult(BaseModel):
    doc_id: str
    deleted_blocks: int = 0
    message: str = "ok"


# ── §2.5 对话历史 ──────────────────────────────────────────
class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[dict] = Field(default_factory=list)


# ── §2.6 debug/retrieve ────────────────────────────────────
class DebugRetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    include_expired: bool = False
    department: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _strip_q(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query 不能为空白")
        return v


class DebugFusedItem(BaseModel):
    chunk_id: str
    doc_title: str = ""
    validity: str = "valid"
    fused_rank: int = 0
    fused_score: float = 0.0
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    content: str = ""


class DebugRetrieveResponse(BaseModel):
    fused: list[DebugFusedItem] = Field(default_factory=list)
    expired_candidates: list[DebugFusedItem] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)
    degraded: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ── §2.7 health ────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"               # ok / degraded
    version: str = "0.1.0"
    checks: dict = Field(default_factory=dict)   # {redis|vector_store|bm25|llm: up|down}
