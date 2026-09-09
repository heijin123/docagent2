"""app.models：数据契约（契约文档 4.1 / 4.2 / 7.1 的权威实现）。

- Block：解析器统一输出（契约 4.1）
- DocumentType / 文档级 DocMeta
- ChunkRecord：与 7.1 Chunk 同构 + content（契约 4.2）
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

BlockType = Literal["heading", "paragraph", "table", "image", "code", "figure_transcript"]
BlockTypeValue = Literal["heading", "paragraph", "table", "image", "code", "figure_transcript"]

# 块长常量（字符级护栏，用于超长 table/code 保留后的二次切分参考；F1.2 以 token 计）
CHUNK_TARGET_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 80


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    MD = "md"
    TXT = "txt"
    UNSUPPORTED = "unsupported"

    @property
    def is_supported(self) -> bool:
        return self != DocumentType.UNSUPPORTED


# 支持格式（需求 F1.1：PDF / DOCX / MD / TXT）
SUPPORTED_EXTENSIONS: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".md": DocumentType.MD,
    ".markdown": DocumentType.MD,
    ".txt": DocumentType.TXT,
}


def make_doc_id(tenant_id: str, doc_key: str) -> str:
    """doc_id = sha256(tenant:doc_key)[:12]，内容无关、跨版本稳定（F1.8 / 契约 §2.2）。"""
    digest = hashlib.sha256(f"{tenant_id}:{doc_key}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:12]}"


def normalize_doc_key(file_path: Path) -> str:
    """默认 doc_key = 规范化后的相对路径形式（POSIX 风格、小写化可选）。"""
    return file_path.name


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


# ── 4.1 Block（解析器统一输出）──────────────────────────────────
class Block(BaseModel):
    """解析中间结构：table 为 Markdown 表示；image 为占位描述。"""

    model_config = {"extra": "allow"}

    block_type: BlockTypeValue
    text: str = ""
    page: int | None = None
    heading_level: int | None = None  # 1~3，section 层级追踪
    metadata: dict = Field(default_factory=dict)  # 图片路径、表格行列数等（内部使用）


# ── 文档级元数据 ──────────────────────────────────────────────
class DocMeta(BaseModel):
    """文档级信息（7.1 文档级标识 + F1.8 身份字段）。"""

    tenant_id: str = "tenant_demo"
    doc_key: str = ""                # 逻辑文档身份（默认规范化文件名，可显式指定）
    doc_id: str = ""                 # 由 make_doc_id 派生（可后填）
    doc_title: str = ""              # 展示用标题
    source: str = "pdf"              # wiki / pdf / markdown / database / web
    file_path: str = ""
    version: int = 1
    author: str | None = None
    doc_date: str | None = None      # YYYY-MM-DD 或 YYYY（F2.9，文件名解析，可选）
    doc_year: int | None = None      # 供 where 过滤（F2.9）
    content_hash: str = ""           # 文件内容 SHA-256（变更指纹，不入 chunk payload）
    permission: str = "internal"     # public / internal / secret
    category: str = "general"
    department: str | None = None


# ── 4.2 / 7.1 ChunkRecord ─────────────────────────────────────
class ChunkRecord(BaseModel):
    """带正文的 chunk 记录；字段与契约 7.1 Chunk 一一对应。"""

    # 文档级（共享）
    doc_id: str
    doc_title: str = ""
    source: str = "pdf"
    file_path: str = ""
    version: int = 1
    author: str | None = None
    doc_date: str | None = None
    doc_year: int | None = None
    # chunk 位置
    chunk_id: str = ""               # f"{doc_id}_{version:04d}_{index:05d}"
    chunk_index: int = 0
    page_num: int = 0
    start_offset: int = 0
    end_offset: int = 0
    image_ids: list[str] = Field(default_factory=list)  # F1.10（M1 仅占位）
    # 业务过滤
    category: str = "general"
    department: str | None = None
    permission: str = "internal"
    effective_time: int | None = 0   # 0 = 永久有效（F2.8，字段不可缺失）
    is_valid: bool = True
    # 运维
    create_time: int = 0
    update_time: int = 0
    embedding_model: str = ""
    chunk_size: int = 0              # 实际 token 数（估算）
    # 扩展
    tenant_id: str = "tenant_demo"
    block_type: str = "paragraph"    # 溯源：来自哪种 Block（调试用）
    content: str = ""
    embedding: list[float] | None = None  # 仅内存传递，不入 payload 冗余


def chunk_to_metadata(rec: ChunkRecord) -> dict:
    """ChunkRecord → 存储层 metadata（单一事实源；Chroma / BM25 共用，保证两路过滤键一致）。

    只收标量：list 转逗号串；`doc_year` 缺失写 -1、`department` 缺失写空串（where 可匹配）。
    """
    return {
        "doc_id": rec.doc_id,
        "doc_title": rec.doc_title,
        "source": rec.source,
        "file_path": rec.file_path,
        "version": rec.version,
        "chunk_index": rec.chunk_index,
        "page_num": rec.page_num,
        "start_offset": rec.start_offset,
        "end_offset": rec.end_offset,
        "image_ids": ",".join(rec.image_ids) if rec.image_ids else "",
        "category": rec.category,
        "department": rec.department or "",
        "permission": rec.permission,
        "effective_time": rec.effective_time or 0,
        "is_valid": rec.is_valid,
        "create_time": rec.create_time,
        "update_time": rec.update_time,
        "embedding_model": rec.embedding_model,
        "chunk_size": rec.chunk_size,
        "tenant_id": rec.tenant_id,
        "doc_date": rec.doc_date or "",
        "doc_year": rec.doc_year or -1,
        "author": rec.author or "",
    }


_DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})?[-_]?(\d{2})?")


def extract_doc_date_from_filename(filename: str) -> tuple[str | None, int | None]:
    """从文件名解析 doc_date / doc_year（F2.9）。返回 (doc_date, doc_year)。"""
    m = _DATE_RE.search(filename)
    if not m:
        return None, None
    year = int(m.group(1))
    if m.group(2) and m.group(3):
        return f"{year}-{m.group(2)}-{m.group(3)}", year
    return str(year), year
