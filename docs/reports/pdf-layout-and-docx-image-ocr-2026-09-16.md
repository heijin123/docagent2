# PDF 图文版面对齐 + DOCX 内嵌图 OCR 报告

- 日期：2026-09-16
- 需求：F1.9（PDF 质量门 / VLM 转录）· M2 残留项收口（①②）
- 结论：两项均落地并真实跑通；`verify_vlm_ocr.py` **26/26 PASS**，全量无 Key 回归全绿。

## 背景

上一轮已接「PDF 红页整页 OCR + 非红页内嵌图 OCR」，但留了两个洞：

1. **图文版面对齐**：内嵌图 OCR 块一律追加到**该页末尾**，未与邻近段落对齐——多图/图文交替的页面里，
   图内文字与图注/正文错位，chunk 上下文被割裂。
2. **DOCX 内嵌图未 OCR**：`_parse_docx` 只在「段落同时含文本+图」时产一个 `[图片]` 占位块（**文本被丢弃**），
   纯图段落**直接不产块**；且完全没有图片 OCR。

## 一、PDF 图文版面对齐

### 方案：按图纵坐标锚定插入位

不再一律页尾，改为把图内 OCR 块插到「该图在阅读顺序中的位置」：

1. `page.get_image_rects(xref)` → 取图的 y0（页内纵坐标）；
2. `page.get_text("blocks")` → 该页文本块 (y0, y1, text) 列表（`_page_text_spans`）；
3. 取**图上方最近**的文本块（`y1 <= 图y0` 中 y1 最大者），其前 16 字为**锚句**；
4. 在页内 markdown 块中定位**含该锚句**的块 → 插到它后面；
5. 兜底：图在页首（上方无文本）→ 插最前（位序 0）；锚不到（无文本层/文本不进 markdown 块）→ 页尾（等价旧行为）。

### 代码

- `_transcribe_inline_images` 返回类型由 `dict[int, list[Block]]` 改为
  **`dict[int, list[tuple[int, Block]]]`**（`(插入位序, 块)`；位序=插在该页第几个已有块之前）。
- 新增 `_page_text_spans(page)`（静态）、`_inline_insert_pos(page, xref, page_blocks, spans)`（静态）。
- `_merge_vlm_text_blocks` 改为**按位序分桶插入**；红页整页转录仍留页尾（红页原文块已被剔除）。

## 二、DOCX 内嵌图 OCR

### 方案：原位替换占位块（DOCX 块序=段落顺序，天然对齐）

- **解析层** `_parse_docx` 重构：`_paragraph_from_xml` → **`_paragraph_blocks_from_xml`（返回 list）**
  - ① 文本 + 图共存时**文本不再丢**（旧版整段退化成一个 `[图片]` 块）；
  - ② 纯图段落也产占位块（旧版直接丢弃）；
  - ③ 每个内联图占位块带 **`img_rid`**（DrawingML `blip` 的 `r:embed`），供 pipeline 取图字节。
- **pipeline** 新增 `_transcribe_docx_images(path, parsed)`：
  - 从 `document.part.related_parts[rid].blob` 取图片字节（MIME 取 `content_type`，限 png/jpeg/gif/webp/bmp）；
  - Qwen-VL OCR → **原地把占位块改写为 paragraph**（`source=vlm_ocr_docx`），位置不变；
  - 无文字 / 取不到字节 → **丢弃占位块**（`[图片]` 无检索价值，留着只是噪声）。
- `run_document` 增 `elif parsed.doc_type == DocumentType.DOCX:` 分支（`vlm_note` 报 `N 个内嵌图已 VLM OCR 为文本块`）。

## 验证

### `scripts/verify_vlm_ocr.py` — 26/26 PASS

| 链路 | 离线（FakeTranscriber） | 真实（Qwen-VL） |
|---|---|---|
| PDF 红页整页 OCR | ✓ 块/page/source | ✓ 还原「保养/每季度」 |
| PDF 内嵌图 OCR | ✓ 块/page/source | ✓ 还原「保养/每季度」 |
| **版面对齐** | ✓ 插入位 pos=1；合并后「上方段 < 图OCR < 下方段」 | ✓ pos=1 |
| **DOCX 内嵌图 OCR** | ✓ 占位块替换、位置在「说明段」与「结尾段」之间 | ✓ 还原「保养/每季度」 |

- 合成混排页：上/下文本层（`ABOVEMARKER`/`BELOWMARKER`）+ 中间带字图 → 校验 OCR 块落位。
- 合成 DOCX：标题 + 说明段 + 纯内联图段 + 结尾段 → 校验占位块被原位替换。

### 全量回归

- `verify_pdf_structure` **PASS**、`verify_golden_anchors` **55/55**、`verify_cli_ingest` **14/0**、`verify_token_accounting` 通过、`compileall` OK。
- 全量 `app.cli ingest data/samples --rebuild`：**38/38 ok、0 失败**、`provider=dashscope`、锚点词表 **4058 词（与改前一致）** → 说明既有语料（无内嵌图）块结构未变，改动零影响。

## 现状（PDF / DOCX 全内容类型）

| 内容 | 状态 |
|---|---|
| 纯文字 / 表格 | ✅ |
| PDF 整页扫描/红页 | ✅ 整页 OCR |
| PDF 非红页内嵌图 | ✅ OCR + **版面对齐** |
| **DOCX 内嵌图** | ✅ **原位替换占位块** |
| 像素级版面还原（图注—正文严格顺序等） | ⏳ 可再细化 |

## 残留

1. **真实语料无图**：5 个 sample PDF `get_images=0`、6 个 sample DOCX `blips=0/media=0`，两路内嵌图 OCR 目前**仅合成 fixture 验证**；建议后续补一个带内嵌位图的 PDF/DOCX 样本做端到端。
2. **对齐粒度**：当前为「锚句→邻近块」级对齐，非像素级版面还原；对图文严格交错排版的文档可再细化（如按 bbox 精确插空）。
3. **DOCX 页号**：DOCX 无页概念，OCR 块 `page=None`（页号显示「—」），与既有行为一致。
