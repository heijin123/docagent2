"""格式探测：扩展名主判 + 内容嗅探兜底（需求 F1.1）。

- 扩展名主判：命中即信任，最快路径；
- 内容嗅探兜底：%PDF 头、ZIP 容器目录、OLE 头（明确拒绝旧版 Office）、文本解码。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from app.models import DocumentType, SUPPORTED_EXTENSIONS


def detect_document_type(file_path: Path) -> tuple[DocumentType, str | None]:
    """返回 (DocumentType, note)。note 为嗅探说明或拒绝原因。

    语义（doc-agent 同款）：未知扩展名的可解码文本 = 当 TXT 收下；
    要触发"无法识别拒绝"需不可解码的二进制。
    """
    if not file_path.exists():
        return DocumentType.UNSUPPORTED, "文件不存在"

    # 1) 扩展名主判
    ext = file_path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext], None

    # 2) 内容嗅探兜底
    head = _read_head(file_path)
    if head is None:
        return DocumentType.UNSUPPORTED, "文件不可读"

    if head.startswith(b"%PDF"):
        return DocumentType.PDF, "嗅探: %PDF 头"

    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        doc_type = _sniff_zip(file_path)
        if doc_type is not None:
            return doc_type, f"嗅探: ZIP 容器 → {doc_type.value}"
        return DocumentType.UNSUPPORTED, "ZIP 容器但无法判定为 Office 新版格式"

    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return DocumentType.UNSUPPORTED, "OLE 旧版 Office 格式（.doc/.xls/.ppt），请转存为 docx/xlsx/pptx"

    if _looks_like_text(head):
        return DocumentType.TXT, "嗅探: 可解码文本 → txt"
    if _looks_like_markdown(file_path):
        return DocumentType.MD, "嗅探: 行首 # 标题 → md"

    return DocumentType.UNSUPPORTED, "不可解码二进制，无法识别格式"


def _read_head(file_path: Path, n: int = 1024) -> bytes | None:
    try:
        with open(file_path, "rb") as f:
            return f.read(n)
    except OSError:
        return None


def _sniff_zip(file_path: Path) -> DocumentType | None:
    try:
        with zipfile.ZipFile(file_path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return None
    if "[Content_Types].xml" not in names:
        return None
    if any(n.startswith("word/") for n in names):
        return DocumentType.DOCX
    if any(n.startswith("xl/") for n in names):
        return DocumentType.DOCX  # 与扩展名无关时保守处理（xlsx 属电子表格，M1 未支持，提示如下）
    if any(n.startswith("ppt/") for n in names):
        return DocumentType.DOCX
    return None


def _looks_like_text(head: bytes) -> bool:
    """可解码文本判定：UTF-8 可解码 + 控制字符占比低（NUL/二进制信号直接拒）。"""
    if not head:
        return False
    if b"\x00" in head:
        return False  # 含 NUL → 二进制
    for enc in ("utf-8", "gbk"):
        try:
            decoded = head.decode(enc)
        except UnicodeDecodeError:
            continue
        ctrl = sum(1 for ch in decoded if ord(ch) < 32 and ch not in "\t\n\r")
        if ctrl / len(decoded) < 0.1:
            return True
    return False


def _looks_like_markdown(file_path: Path) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.lstrip()
                if stripped.startswith("# "):
                    return True
                if len(stripped) > 2000:  # 大段无标题 → 判 txt
                    return False
    except OSError:
        return False
    return False
