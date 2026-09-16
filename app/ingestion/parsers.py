"""解析层（需求 F1.1，契约 4.1）：统一接口 parse(file_path) -> ParsedDocument。

可插拔：parse_file() 按探测结果路由到各格式实现，新增格式只加实现不动下游。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
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
    blocks = _parse_markdown_text(text, DocumentType.MD)
    parsed.blocks = blocks
    if any(b.metadata.get("unclosed") for b in blocks):
        parsed.warnings.append("代码围栏未闭合")
    return parsed


def _is_cjk_heading(line: str) -> bool:
    """中文章节/条目标题判定（markitdown 0.1.7 对「第X章」不输出 #，需补回）。"""

    s = line.strip()
    if not s or len(s) > 40:
        return False
    pat = re.compile(
        r"^(第[0-9一二三四五六七八九十百千]+章"
        r"|第[0-9一二三四五六七八九十百千]+节"
        r"|[一二三四五六七八九十]+、"
        r"|[0-9]+(\.[0-9]+)*[.\、]\s*[\u4e00-\u9fff])"
    )
    return bool(pat.match(s))


def _cjk_heading_level(line: str) -> int:
    """中文标题 → heading_level（章=1，节/条=2，其余=3；编号 1.1=2）。"""
 
    s = line.strip()
    if "章" in s:
        return 1
    if re.match(r"^[0-9]+(\.[0-9]+)*[.\、]", s):
        dots = s.split(" ")[0].count(".")
        return min(dots + 1, 3)
    if "节" in s or "条" in s:
        return 2
    return 3


def _parse_markdown_text(text: str, doc_type: DocumentType,
                         *, recover_cjk_headings: bool = False) -> list[Block]:
    """Markdown 文本 → Block 序列（heading/table/code/image/paragraph）。

    抽出为纯函数供 PDF 路径复用：markitdown 把单页 PDF 转成 Markdown 后，PDF 解析器
    按页调用本函数得到带 page 的结构块，从而拿到页内结构 + 准确页码（block.page
    由调用方回填，本函数不负责页码）。

    recover_cjk_headings：仅 PDF 路径开启。markitdown 0.1.7 对中文「第X章/第X节」
    不输出 `#` 标题，此处按章节模式补回 heading 块，使 PDF 也具备标题结构。
    """
    blocks: list[Block] = []
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
            blocks.append(Block(block_type="paragraph", text=text_piece))

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
                blocks.append(
                    Block(block_type="code", text=code_text, metadata={"language": code_lang})
                )
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 标题（# 前缀；PDF 路径可选 recover_cjk_headings 补中文「第X章/节」标题）
        stripped = line.lstrip()
        heading_text = None
        heading_level = None
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            heading_level = min(len(stripped) - len(stripped.lstrip("#")), 6)
        elif recover_cjk_headings and _is_cjk_heading(stripped):
            heading_text = stripped
            heading_level = _cjk_heading_level(stripped)
        if heading_text:
            flush_paragraph()
            blocks.append(
                Block(block_type="heading", text=heading_text,
                      heading_level=heading_level, metadata={"line_no": i + 1})
            )
            i += 1
            continue

        # 图片语法 ![alt](url)
        if stripped.startswith("![") and "](http" in stripped:
            alt = stripped[2:].split("](")[0]
            blocks.append(
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
                blocks.append(
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
    if in_code:  # 未闭合围栏：按代码块收下（警告由调用方按 parsed 加）
        blocks.append(Block(block_type="code", text="\n".join(code_buf),
                            metadata={"language": code_lang, "unclosed": True}))
    return blocks


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


# ── PDF：markitdown 出页内结构 + PyMuPDF 保质量门与页码 ──
@register_parser(DocumentType.PDF)
def _parse_pdf(path: Path) -> ParsedDocument:
    """PDF → 整文档 markitdown 转 Markdown（\\f 分隔页）→ 逐页切片 → 结构块（heading/table/paragraph）。

    设计取舍（对比原 PyMuPDF 整页 1 块）：
    - 整文档转一次，再按 \\f 切成逐页片段，每段独立走 MD 解析并打 PyMuPDF 页码
      （页码权威来源 = PyMuPDF 页序，不依赖 markitdown 对单页 PDF 的页码判断，更稳）；
      每块天然带正确 page 页码（解决原「1 页=1 块、引用指错页」）；
    - markitdown 0.1.7 对中文「第X章」不输出 `#`，故 PDF 路径开启 recover_cjk_headings
      把章节行补成 heading 块；
    - 页眉页脚按「块文本跨 >=3 页出现在页首/页尾，或每页都出现」剔除；
    - 红页（扫描/乱码）markitdown 亦无能为力，退回 PyMuPDF 原文降级入库（VLM 桩未接）。
    """
    import pymupdf
    from markitdown import MarkItDown

    parsed = ParsedDocument(doc_type=DocumentType.PDF, file_path=str(path))
    doc = pymupdf.open(str(path))

    if doc.needs_pass:
        raise UnsupportedFormatError(path, "加密 PDF（DOC_ENCRYPTED）")

    md = MarkItDown()

    # 整文档转一次 Markdown（markitdown 用 \f 分隔页）；再按 \f 切成逐页片段，
    # 每段独立走 MD 解析并打 PyMuPDF 页码（页码权威来源 = PyMuPDF 页序，不依赖
    # markitdown 对单页 PDF 的页码判断，更稳）。
    try:
        whole = md.convert(str(path))
        whole_md = getattr(whole, "markdown", "") or ""
    except Exception:  # noqa: BLE001 — 整文档转换失败则退回逐页原文
        whole_md = ""

    page_md_segments = _split_markdown_by_page(whole_md, doc.page_count)

    # 每页块区间 + 首/尾块文本，供页眉页脚剔除
    page_ranges: list[tuple[int, int]] = []

    for page_no in range(1, doc.page_count + 1):
        page = doc[page_no - 1]
        try:
            image_count = len(page.get_images(full=True))
        except Exception:  # noqa: BLE001
            image_count = 0
        raw_text = page.get_text("text")
        quality = _grade_from_text(page_no, raw_text, image_count)
        parsed.page_qualities.append(quality)

        start_idx = len(parsed.blocks)

        if quality.level == "red":
            # 红页：markitdown 救不了，保留 PyMuPDF 原文（质量门已打标）
            blocks = _page_text_to_blocks(raw_text, page_no, set(), set())
            for b in blocks:
                b.metadata["quality_level"] = "red"
            parsed.blocks.extend(blocks)
        else:
            seg = page_md_segments[page_no - 1] if page_md_segments else ""
            if not seg.strip():
                # markitdown 该页空输出兜底：退回 PyMuPDF 原文段落
                blocks = _page_text_to_blocks(raw_text, page_no, set(), set())
            else:
                blocks = _parse_markdown_text(seg, DocumentType.PDF,
                                              recover_cjk_headings=True)
                for b in blocks:
                    b.page = page_no
            parsed.blocks.extend(blocks)

        if len(parsed.blocks) > start_idx:
            page_ranges.append((start_idx, len(parsed.blocks) - 1))
        else:
            page_ranges.append((start_idx, start_idx - 1))  # 空页

    # 页眉页脚剔除（基于每页首/尾块文本：跨 >=3 页重复，或每页都出现）
    _strip_repeating_headers_footers(parsed.blocks, page_ranges)

    doc.close()
    return parsed


def _split_markdown_by_page(markdown: str, page_count: int) -> list[str]:
    """markitdown 整文档 Markdown 按 \f 分页符切成逐页片段。

    markitdown 把多页 PDF 连成一串、用 \\f 分隔；按页切后每段对应一页，
    便于各自打页码。不足 page_count 段时（末段可能含多余 \\f 或缺失），
    末段兜底合并剩余内容。
    """
    if not markdown:
        return ["" for _ in range(page_count)]
    parts = markdown.split("\f")
    # 去掉每段首尾空白
    segs = [p.strip("\n") for p in parts]
    if len(segs) < page_count:
        # 补齐空串，保证索引对齐页号
        segs = segs + ["" for _ in range(page_count - len(segs))]
    elif len(segs) > page_count:
        # 段数多于页数（异常）：保留前 page_count-1 段，末段合并剩余
        segs = segs[: page_count - 1] + ["\f".join(segs[page_count - 1:])]
    return segs


def _strip_repeating_headers_footers(blocks: list, page_ranges: list[tuple[int, int]]) -> None:
    """删除重复出现在页首/页尾的块（页眉/页脚）。原地修改 blocks。

    判定：某块文本作为某页首/尾块出现 >=3 次，或出现在每一页（>=2 页文档）。
    后者覆盖「短文档每页都有页脚」的情形。
    """
    from collections import Counter

    real_pages = [r for r in page_ranges if r[1] >= r[0]]
    n_pages = len(real_pages)
    heads = [blocks[s].text for s, _ in real_pages]
    tails = [blocks[e].text for _, e in real_pages]

    def _repeats(counter: Counter) -> set[str]:
        out: set[str] = set()
        for t, c in counter.items():
            if t and (c >= 3 or (c == n_pages and n_pages >= 2)):
                out.add(t)
        return out

    repeat_head = _repeats(Counter(heads))
    repeat_tail = _repeats(Counter(tails))
    if not repeat_head and not repeat_tail:
        return

    drop: set[int] = set()
    for s, e in real_pages:
        if blocks[s].text in repeat_head:
            drop.add(s)
        if blocks[e].text in repeat_tail:
            drop.add(e)
    for i in sorted(drop, reverse=True):
        blocks.pop(i)


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
