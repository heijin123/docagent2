"""解析层（需求 F1.1，契约 4.1）：统一接口 parse(file_path) -> ParsedDocument。

可插拔：parse_file() 按探测结果路由到各格式实现，新增格式只加实现不动下游。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.detection import detect_document_type
from app.ingestion.pdf_quality import PageQuality
from app.models import Block, DocumentType

# 各解析器注册表：格式 -> 实现函数 parse -> list[Block]
_PARSER_REGISTRY: dict[DocumentType, callable] = {}


def register_parser(doc_type: DocumentType):
    def decorator(fn):
        _PARSER_REGISTRY[doc_type] = fn
        return fn
    return decorator


@dataclass
class ParsedDocument:
    """统一解析产物：Block 序列 + 逐页质量判级 + 警告。"""

    doc_type: DocumentType
    file_path: str
    blocks: list[Block] = field(default_factory=list)
    page_qualities: list[PageQuality] = field(default_factory=list)  # 仅 PDF
    warnings: list[str] = field(default_factory=list)
    # 解析出的图片（F1.10 占位）：{block_index 或 None: 图片信息}，M1 仅登记元数据
    extracted_images: list[dict] = field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def parse_file(file_path: str | Path) -> ParsedDocument:
    """探测 + 解析 + 清洗，返回统一 ParsedDocument。不支持格式抛 UnsupportedFormatError。"""
    path = Path(file_path)
    doc_type, note = detect_document_type(path)
    if not doc_type.is_supported:
        raise UnsupportedFormatError(path, note or "无法识别格式")

    parser = _PARSER_REGISTRY.get(doc_type)
    if parser is None:
        raise UnsupportedFormatError(path, f"格式 {doc_type.value} 尚无解析器")

    parsed = parser(path)
    parsed.doc_type = doc_type
    return parsed


class UnsupportedFormatError(Exception):
    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"不支持的文件格式 {path.name}: {reason}")


# ── TXT：段落切（空行分隔），编码探测 utf-8 → gbk → replace ──
@register_parser(DocumentType.TXT)
def _parse_txt(path: Path) -> ParsedDocument:
    text = _read_text(path)
    parsed = ParsedDocument(doc_type=DocumentType.TXT, file_path=str(path))
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        if _looks_garbled(para):
            parsed.warnings.append(f"段落疑似乱码（含替换符），长度 {len(para)}")
        parsed.blocks.append(Block(block_type="paragraph", text=para))
    return parsed


# ── MD：标题层级 + 段落 + 代码块整块 + 图片占位 ──
@register_parser(DocumentType.MD)
def _parse_markdown(path: Path) -> ParsedDocument:
    text = _read_text(path)
    parsed = ParsedDocument(doc_type=DocumentType.MD, file_path=str(path))
    lines = text.split("\n")
    i = 0
    n = len(lines)
    buf: list[str] = []
    in_code = False
    code_buf: list[str] = []
    code_lang = ""

    def flush_paragraph():
        nonlocal buf
        text_piece = "\n".join(buf).strip()
        buf = []
        if text_piece:
            parsed.blocks.append(Block(block_type="paragraph", text=text_piece))

    while i < n:
        line = lines[i]

        # 代码围栏
        if line.lstrip().startswith("```"):
            if not in_code:
                in_code, code_lang = True, line.strip()[3:].strip()
                code_buf = []
            else:
                in_code = False
                code_text = "\n".join(code_buf)
                parsed.blocks.append(
                    Block(block_type="code", text=code_text, metadata={"language": code_lang})
                )
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 标题
        stripped = line.lstrip()
        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if title:
                parsed.blocks.append(
                    Block(block_type="heading", text=title,
                          heading_level=min(level, 6), metadata={"line_no": i + 1})
                )
            i += 1
            continue

        # 图片语法 ![alt](url)
        if stripped.startswith("![") and "](http" in stripped:
            alt = stripped[2:].split("](")[0]
            parsed.blocks.append(
                Block(block_type="image", text=alt,
                      metadata={"alt": alt, "src": stripped.split("](")[1].rstrip(")")})
            )
            i += 1
            continue

        # 表格：连续 | 行，且含分隔行 |---|---| → 整块 table
        if stripped.startswith("|"):
            table_lines, i = _collect_table(lines, i)
            if table_lines is not None:
                flush_paragraph()
                parsed.blocks.append(
                    Block(block_type="table", text="\n".join(table_lines),
                          metadata={"md_table": True})
                )
                continue
            # 非表格（单行管道文本）→ 落到普通行处理

        # 普通行（含空行分段）
        if not line.strip():
            flush_paragraph()
        else:
            buf.append(line)
        i += 1

    flush_paragraph()
    if in_code:  # 未闭合围栏：按代码块收下并警告
        parsed.blocks.append(Block(block_type="code", text="\n".join(code_buf),
                                   metadata={"language": code_lang, "unclosed": True}))
        parsed.warnings.append("代码围栏未闭合")
    return parsed


# ── DOCX：正文归一化为段落/表格/图片，标题样式映射 heading_level ──
@register_parser(DocumentType.DOCX)
def _parse_docx(path: Path) -> ParsedDocument:
    import docx  # python-docx

    parsed = ParsedDocument(doc_type=DocumentType.DOCX, file_path=str(path))
    document = docx.Document(str(path))

    # 段落与表格按文档流顺序遍历（WML 顺序）
    from docx.oxml.ns import qn

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag
        if tag == qn("w:p"):
            para = _paragraph_from_xml(child, document)
            if para is not None:
                parsed.blocks.append(para)
        elif tag == qn("w:tbl"):
            table_block = _table_from_xml(child, document)
            if table_block is not None:
                parsed.blocks.append(table_block)

    return parsed


def _paragraph_from_xml(p_element, document):
    from docx.text.paragraph import Paragraph

    para = Paragraph(p_element, document)
    style_name = (para.style.name or "") if para.style else ""
    text = para.text.strip()
    if not text:
        return None

    # Heading 1~3 样式映射 section 层级（中文 Office 样式名也处理）
    lowered = style_name.lower()
    if "heading" in lowered or "标题" in style_name:
        level = None
        for candidate in (style_name, lowered):
            for ch in candidate:
                if ch.isdigit():
                    level = int(ch)
                    break
            if level is not None:
                break
        if level is None:
            level = 1
        if level <= 3:
            return Block(block_type="heading", text=text, heading_level=level)
        # 4+ 级标题当段落
    # 检测是否只含图片（inline shape）
    if para._p.findall(".//" + "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
        return Block(block_type="image", text="[图片]", metadata={"has_inline_image": True})
    return Block(block_type="paragraph", text=text)


def _table_from_xml(tbl_element, document):
    from docx.table import Table

    table = Table(tbl_element, document)
    rows = []
    for row in table.rows:
        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
        rows.append(" | ".join(cells))
    if not rows:
        return None
    header = rows[0]
    sep = " | ".join(["---"] * (header.count(" | ") + 1))
    md_table = "\n".join([header, sep] + rows[1:])
    return Block(block_type="table", text=md_table,
                 metadata={"rows": len(table.rows), "cols": len(table.columns)})


# ── PDF：逐页取文本（PyMuPDF），版面质量门判级，图登记 ──
@register_parser(DocumentType.PDF)
def _parse_pdf(path: Path) -> ParsedDocument:
    import pymupdf

    parsed = ParsedDocument(doc_type=DocumentType.PDF, file_path=str(path))
    doc = pymupdf.open(str(path))

    if doc.needs_pass:
        raise UnsupportedFormatError(path, "加密 PDF（DOC_ENCRYPTED）")

    # 页眉页脚剔除：同一行文本连续 >=3 页出现在页首/页尾 → 剔除
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}

    page_texts: list[str] = []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text")
        page_texts.append(text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
            last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1

    repeat_first = {k for k, v in first_lines.items() if v >= 3}
    repeat_last = {k for k, v in last_lines.items() if v >= 3}

    for page_no, text in enumerate(page_texts, start=1):
        # 图片计数（页内 image xobjects 去重）
        try:
            images = doc[page_no - 1].get_images(full=True)
            image_count = len(images)
        except Exception:  # noqa: BLE001
            image_count = 0

        quality = _grade_from_text(page_no, text, image_count)
        parsed.page_qualities.append(quality)

        if quality.level == "red":
            # 红页原文不可信：先保留并打标 quality_level=red，由 pipeline 决定
            # 转录成功→剔除原文块、追加 figure_transcript；转录失败→降级原文入库（R7 不崩）
            page_blocks = _page_text_to_blocks(text, page_no, repeat_first, repeat_last)
            for b in page_blocks:
                b.metadata["quality_level"] = "red"
            parsed.blocks.extend(page_blocks)
            continue

        page_blocks = _page_text_to_blocks(text, page_no, repeat_first, repeat_last)
        parsed.blocks.extend(page_blocks)

    doc.close()
    return parsed


def _grade_from_text(page_no: int, text: str, image_count: int):
    """M1 页级文本质量判级。红页文本不可信 → 不产块；黄页加 warning。"""
    from app.ingestion.pdf_quality import grade_page

    return grade_page(page_no, text, image_count)


def _page_text_to_blocks(text: str, page_no: int, repeat_first: set, repeat_last: set):
    """整页文本 → 1 个 paragraph Block（剔除页眉页脚；超长由 chunking 二次切分）。"""
    lines = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped in repeat_first or stripped in repeat_last:
            continue  # 页眉页脚
        lines.append(stripped)
    if not lines:
        return []
    return [Block(block_type="paragraph", text=_join_lines(lines), page=page_no)]


def _join_lines(lines: list[str]) -> str:
    joined = lines[0]
    for ln in lines[1:]:
        if joined and joined[-1].isascii() and ln and ln[0].isascii():
            joined += " " + ln
        else:
            joined += ln
    return joined


# ── 工具 ──────────────────────────────────────────────
def _collect_table(lines: list[str], i: int):
    """收集从 i 开始的连续 | 行。若含 md 分隔行（|--|--|）判定为表格，否则 None。"""
    import re

    j = i
    while j < len(lines) and lines[j].strip().startswith("|"):
        j += 1
    block_lines = [ln.strip() for ln in lines[i:j]]
    if not block_lines:
        return None, i
    sep_re = re.compile(r"^\|?[\s:\-|]+\|?\s*$")
    is_table = len(block_lines) >= 2 and bool(sep_re.fullmatch(block_lines[1]))
    return (block_lines if is_table else None), j


def _read_text(path: Path) -> str:
    """编码探测：utf-8 → gbk → utf-8 replace；换行统一为 \\n（CRLF/CR → LF）。

    换行归一化是必要的：下游 `_parse_txt` 按 "\\n\\n" 切段落，Windows 编辑器产出的
    纯文本是 "\\r\\n\\r\\n"，不归一化会导致**整份文件被当成一个段落**；且残留的 "\\r"
    是 C0 控制符，还会触发 `_looks_garbled` 误报"疑似乱码"。
    """
    raw = path.read_bytes()
    text: str | None = None
    for enc in ("utf-8", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _looks_garbled(text: str) -> bool:
    return "\ufffd" in text or sum(1 for c in text if ord(c) < 32 and c not in "\n\t") > len(text) * 0.01
