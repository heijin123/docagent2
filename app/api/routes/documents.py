"""POST/DELETE /v1/documents（契约 §2.2/§2.4）。

上传三态：
- 202 新文档 / 内容变更（doc_key 不存在 | 异 content_hash | 已软删同 hash）→ 入队后台任务；
- 200 幂等命中（同 doc_key 同 hash 已完成且未软删）→ duplicated=true，不入队；
- 409 同 doc_key 处理中 → INGEST_IN_PROGRESS。

meta.permission 不接受客户端传入（模型未定义该字段 → 自动忽略，防提权）。
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi import status as http_status

from app.api.deps import get_request_id, get_services
from app.api.errors import ApiError
from app.core.config import settings
from app.models import SUPPORTED_EXTENSIONS, make_doc_id
from app.models.api import DocumentMeta, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# MIME 白名单（契约 §2.2 扩展名 + MIME 双校验；application/octet-stream 兜底放行）
_MIME_OK = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              "application/msword", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".txt": {"text/plain", "text/markdown", "application/octet-stream"},
}


def _validate_ext(filename: str, content_type: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ApiError("DOC_UNSUPPORTED_TYPE",
                       f"不支持的文件类型 {ext or '（无扩展名）'}，支持: "
                       + ", ".join(sorted(SUPPORTED_EXTENSIONS)), 415,
                       {"filename": filename})
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct and ct not in _MIME_OK.get(ext, set()) and not ct.startswith("text/"):
        raise ApiError("DOC_UNSUPPORTED_TYPE",
                       f"MIME {ct!r} 与扩展名 {ext} 不匹配", 415)
    return ext


def _normalize_doc_key(filename: str, custom: str | None) -> str:
    if custom and custom.strip():
        key = custom.strip().replace("\\", "/").lstrip("/")
        key = re.sub(r"[:\x00-\x1f]", "_", key)
        return key or Path(filename).name
    return Path(filename).name


def _safe_tenant_dir(tenant_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", tenant_id) or "default"
    d = settings.upload_dir / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("")
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    meta: str | None = Form(None),
):
    services = get_services(request)
    tenant_id = request.state.tenant_id
    rid = get_request_id(request)

    ext = _validate_ext(file.filename or "", file.content_type or "")
    raw = await file.read()
    if not raw:
        raise ApiError("VALIDATION_INVALID_ARGUMENT", "上传文件为空", 422)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        mb = len(raw) / (1024 * 1024)
        raise ApiError("DOC_TOO_LARGE",
                       f"文件大小 {mb:.1f}MB 超过上限 {settings.max_upload_mb}MB", 413,
                       {"max_upload_mb": settings.max_upload_mb})
    content_hash = hashlib.sha256(raw).hexdigest()

    parsed_meta = DocumentMeta()
    if meta:
        import json

        try:
            parsed_meta = DocumentMeta.model_validate(json.loads(meta))
        except Exception as exc:  # noqa: BLE001
            raise ApiError("VALIDATION_INVALID_ARGUMENT",
                           f"meta 不是合法 JSON/DocumentMeta: {exc}", 422) from exc

    doc_key = _normalize_doc_key(file.filename or "upload", parsed_meta.doc_key)
    doc_id = make_doc_id(tenant_id, doc_key)

    existing = services.registry.get(tenant_id, doc_key)
    registry_busy = existing is not None and existing.status == "processing"
    inflight_busy = services.task_manager.is_inflight(tenant_id, doc_key)

    # ── 409：同 doc_key 处理中 ────────────────────────────
    if registry_busy or inflight_busy:
        raise ApiError("INGEST_IN_PROGRESS",
                       f"同文档（doc_key={doc_key}）正在处理中，请稍后查询任务状态", 409,
                       {"doc_id": doc_id})

    # ── 200：幂等命中（同 hash done 且未软删）──────────────
    if (existing is not None and not existing.is_deleted
            and existing.status == "done"
            and existing.content_hash == content_hash):
        response.status_code = http_status.HTTP_200_OK
        response.headers["X-Request-Id"] = rid
        return UploadResponse(task_id="", doc_id=doc_id, version=existing.version,
                              status="done", duplicated=True)

    # ── 202：新文档 / 变更 / 删后重传 → 后台任务 ──────────
    file_path = _safe_tenant_dir(tenant_id) / f"{uuid.uuid4().hex}{ext}"
    file_path.write_bytes(raw)
    try:
        rec = services.task_manager.submit(
            tenant_id=tenant_id, doc_key=doc_key, doc_id=doc_id, file_path=file_path)
    except ApiError:
        file_path.unlink(missing_ok=True)  # 提交冲突（并发窗口）→ 清理孤儿文件
        raise

    # 预公布 version：update（含删后重传）→ +1；retry(failed 同 hash) → 不变；new → 1
    if existing is not None:
        announce = existing.version + 1 if (
            existing.content_hash != content_hash or existing.is_deleted
            or existing.status == "failed") else existing.version
        # failed 同 hash 重试不递增
        if existing.status == "failed" and existing.content_hash == content_hash:
            announce = existing.version
    else:
        announce = 1

    response.status_code = http_status.HTTP_202_ACCEPTED
    response.headers["X-Request-Id"] = rid
    logger.info("上传入队 tenant=%s doc_key=%s task=%s hash=%s…", tenant_id,
                doc_key, rec.task_id, content_hash[:8])
    return UploadResponse(task_id=rec.task_id, doc_id=doc_id, version=announce,
                          status="pending", duplicated=False)


@router.delete("/{doc_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, request: Request, response: Response):
    """F5.7 软删除：置全部有效 chunk is_valid=false（F1.7），不物理删除；重复删除幂等 204。"""
    services = get_services(request)
    tenant_id = request.state.tenant_id
    rec = services.registry.by_doc_id(tenant_id, doc_id)
    if rec is None:
        raise ApiError("DOC_NOT_FOUND", f"doc_id {doc_id} 不存在", 404)
    vec_flipped = services.vector_store.soft_delete_doc(doc_id, rec.version)
    bm_flipped = services.bm25_store.soft_delete_doc(doc_id, rec.version)
    services.registry.mark_deleted(tenant_id, rec.doc_key)
    logger.info("软删除 doc_id=%s v%s: vector=%s bm25=%s（重复删除幂等 204）",
                doc_id, rec.version, vec_flipped, bm_flipped)
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)
