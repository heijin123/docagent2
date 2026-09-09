"""Chunking（需求 F1.2）：基于 Block 边界递归切分。

规则：
- chunk_size=512 tokens、overlap=80 tokens（按字符粗估：CJK≈1 token/字，ASCII≈1 token/3 字符）；
- **table / code / image 块整块保留不切断**（作者切好的业务边界）；
- heading/paragraph 可切：递归优先按段落/句子边界切，避免语义撕裂；
- chunk 携带完整溯源元数据（契约 7.1）。
"""
from __future__ import annotations

import re
from pathlib import Path

from app.core.config import settings
from app.ingestion.parsers import ParsedDocument
from app.models import ChunkRecord, DocMeta, extract_doc_date_from_filename, now_ts

# 估算 token：CJK 字 ≈1 token，ASCII 每 3 字符 ≈1 token
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    ascii_chars = sum(len(w) for w in _ASCII_WORD_RE.findall(text))
    return cjk + max(1, ascii_chars // 3)


def _slice_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """段落级递归切分：按换行块/句子切，带 overlap。"""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    # 优先在换行处分
    units = [u for u in re.split(r"(\n)", text)]  # 保留换行便于拼接
    lines = text.split("\n")
    if len(lines) > 1:
        # 按行累积
        pieces: list[str] = []
        current = ""
        for line in lines:
            candidate = line if not current else current + "\n" + line
            if estimate_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                # 超长单行：句子级切
                if estimate_tokens(line) > max_tokens:
                    pieces.extend(_split_long_line(line, max_tokens, overlap_tokens))
                    current = ""
                else:
                    current = line
        if current:
            pieces.append(current)
        return _apply_overlap(pieces, overlap_tokens)

    return _split_long_line(text, max_tokens, overlap_tokens)


def _split_long_line(line: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """超长行：按句子边界切（中文句号/分号/逗号，英文句点）。"""
    sentences = re.split(r"(?<=[。；！？!?;])\s*|(?<=[.])\s+", line)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 1:
        # 无句读：硬切
        pieces = []
        current = ""
        for ch in line:
            if estimate_tokens(current + ch) <= max_tokens:
                current += ch
            else:
                pieces.append(current)
                current = ch
        if current:
            pieces.append(current)
        return _apply_overlap(pieces, overlap_tokens)

    pieces: list[str] = []
    current = ""
    for s in sentences:
        if estimate_tokens(current + s) <= max_tokens:
            current += s
        else:
            if current:
                pieces.append(current)
            if estimate_tokens(s) > max_tokens:
                pieces.extend(_split_long_line(s, max_tokens, overlap_tokens))
                current = ""
            else:
                current = s
    if current:
        pieces.append(current)
    return _apply_overlap(pieces, overlap_tokens)


def _apply_overlap(pieces: list[str], overlap_tokens: int) -> list[str]:
    """相邻片断追加 overlap：取前片尾部 N token 拼到后片头部（保证检索不丢衔接）。"""
    if overlap_tokens <= 0 or len(pieces) <= 1:
        return pieces
    result = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = _tail_by_tokens(pieces[i - 1], overlap_tokens)
        result.append((prev_tail + pieces[i]) if prev_tail else pieces[i])
    return result


def _tail_by_tokens(text: str, tokens: int) -> str:
    """从文本末尾截取约 tokens 的尾巴。"""
    if estimate_tokens(text) <= tokens:
        return ""
    chars = list(text)
    acc = ""
    for ch in reversed(chars):
        if estimate_tokens(ch + acc) > tokens + 5:
            break
        acc = ch + acc
    return acc


def chunk_document(parsed: ParsedDocument, doc_meta: DocMeta) -> list[ChunkRecord]:
    """纯函数：ParsedDocument + DocMeta → list[ChunkRecord]（含 version/index 编排）。"""
    records: list[ChunkRecord] = []
    chunk_index = 0

    # 逐 Block 累积，形成逻辑段落单元
    current_text = ""
    current_blocks: list = []  # 该 chunk 的源 Block 引用
    current_start_offset = 0
    offset_cursor = 0

    def flush():
        nonlocal current_text, current_blocks, chunk_index, offset_cursor
        if not current_text.strip():
            current_text = ""
            current_blocks = []
            return
        page_nums = [b.page for b in current_blocks if b.page is not None]
        block_types = set(b.block_type for b in current_blocks)
        start_off = current_start_offset
        end_off = start_off + len(current_text)
        chunk_index += 1
        records.append(
            ChunkRecord(
                doc_id=doc_meta.doc_id,
                doc_title=doc_meta.doc_title,
                source=doc_meta.source,
                file_path=doc_meta.file_path,
                version=doc_meta.version,
                author=doc_meta.author,
                doc_date=doc_meta.doc_date,
                doc_year=doc_meta.doc_year,
                chunk_id=f"{doc_meta.doc_id}_{doc_meta.version:04d}_{chunk_index:05d}",
                chunk_index=chunk_index,
                page_num=page_nums[0] if page_nums else 0,
                start_offset=start_off,
                end_offset=end_off,
                category=doc_meta.category,
                department=doc_meta.department,
                permission=doc_meta.permission,
                effective_time=0,  # F2.8：永久有效（M1 全量）
                create_time=now_ts(),
                update_time=now_ts(),
                embedding_model="",
                chunk_size=estimate_tokens(current_text),
                tenant_id=doc_meta.tenant_id,
                block_type=",".join(sorted(block_types)),
                content=current_text,
            )
        )
        # 更新游标
        offset_cursor = end_off + 2  # 模拟 "\n\n" 间隔
        current_text = ""
        current_blocks = []

    # 计算每 Block 文本长度用于 offset 近似定位
    for block in parsed.blocks:
        text = block.text.strip()
        if not text:
            offset_cursor += len(block.text) + 2
            continue

        atomic = block.block_type in {"table", "code", "image"}

        if atomic:
            # table/code/image：整块保留，即使超长也先 flush 再单独成块（不切断）
            if current_text:
                flush()
            current_start_offset = offset_cursor
            current_text = text
            current_blocks = [block]
            flush()
            offset_cursor = current_start_offset + len(text) + 2
            continue

        # heading/paragraph：可累积可切
        candidate = text if not current_text else current_text + "\n\n" + text
        if not current_text:
            # 空缓冲：直接开始
            current_start_offset = offset_cursor
            current_text = text
            current_blocks = [block]
            offset_cursor += len(text) + 2
            continue

        if estimate_tokens(candidate) <= settings.chunk_size_tokens:
            current_text = candidate
            current_blocks.append(block)
            offset_cursor += len(text) + 2
        else:
            # 缓冲满 → 若当前文本本身超长则先切分
            if estimate_tokens(current_text) > settings.chunk_size_tokens:
                pieces = _slice_by_tokens(current_text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
                for i, piece in enumerate(pieces):
                    chunk_index += 1
                    page_nums = [b.page for b in current_blocks if b.page is not None]
                    start_off = current_start_offset
                    records.append(
                        ChunkRecord(
                            doc_id=doc_meta.doc_id, doc_title=doc_meta.doc_title,
                            source=doc_meta.source, file_path=doc_meta.file_path,
                            version=doc_meta.version, author=doc_meta.author,
                            doc_date=doc_meta.doc_date, doc_year=doc_meta.doc_year,
                            chunk_id=f"{doc_meta.doc_id}_{doc_meta.version:04d}_{chunk_index:05d}",
                            chunk_index=chunk_index,
                            page_num=page_nums[0] if page_nums else 0,
                            start_offset=start_off,
                            end_offset=start_off + len(piece),
                            category=doc_meta.category, department=doc_meta.department,
                            permission=doc_meta.permission, effective_time=0,
                            create_time=now_ts(), update_time=now_ts(),
                            embedding_model="", chunk_size=estimate_tokens(piece),
                            tenant_id=doc_meta.tenant_id, block_type="paragraph",
                            content=piece,
                        )
                    )
                offset_cursor = current_start_offset + len(current_text) + 2
                current_text = ""
                current_blocks = []
                # 新块入新缓冲
                current_start_offset = offset_cursor
                current_text = text
                current_blocks = [block]
                offset_cursor += len(text) + 2
            else:
                # 缓冲未超长但加不下新块 → flush 缓冲，新块开新缓冲
                flush()
                current_start_offset = offset_cursor
                current_text = text
                current_blocks = [block]
                offset_cursor += len(text) + 2

    flush()
    return records


def build_doc_meta(parsed: ParsedDocument, *, tenant_id: str, doc_key: str,
                   doc_title: str | None = None, version: int = 1,
                   content_hash: str = "", source: str | None = None,
                   department: str | None = None, category: str | None = None) -> DocMeta:
    """组装 DocMeta（doc_id 派生 + 文件名日期注入 F2.9）。

    department/category：业务过滤字段（F2.7），上传任务由 API meta 注入
    （契约 §2.2 meta）；缺省保持 DocMeta 默认（general / None）。
    """
    from app.models import make_doc_id, normalize_doc_key

    if not doc_key:
        doc_key = normalize_doc_key(Path(parsed.file_path))
    doc_id = make_doc_id(tenant_id, doc_key)
    date, year = extract_doc_date_from_filename(doc_key)
    fmt = parsed.doc_type.value
    return DocMeta(
        tenant_id=tenant_id,
        doc_key=doc_key,
        doc_id=doc_id,
        doc_title=doc_title or Path(parsed.file_path).stem,
        source=source or fmt,
        file_path=parsed.file_path,
        version=version,
        doc_date=date,
        doc_year=year,
        content_hash=content_hash,
        department=department,
        category=category or "general",
    )
