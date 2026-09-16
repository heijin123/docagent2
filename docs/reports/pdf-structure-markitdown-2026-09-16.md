# PDF 结构改造：markitdown 接入 + 页码回填（2026-09-16）

## 1. 背景与问题

原 PDF 解析走 PyMuPDF 逐页 `get_text("text")`，每页产 **1 个 paragraph 块**（"1 页 = 1 block"），带来三个连锁问题：

| 问题 | 影响 |
|------|------|
| 页内结构被压平 | 标题/表格混进一个段落，检索粒度粗、答案定位差 |
| `page_num` 全 0 | chunk 的 `page_num` 存储值为 0，引用层渲染出"第 0 页"误导用户（2026-09-15 仅用显示层「—」打补丁，根因未解） |
| 中文标题丢失 | 「第X章」等不进结构化块，章节级召回与可回查失败 |

用户决策：**走微软开源库 markitdown**（"造 vs 买"对比后选买），但必须保住**页码溯源**——这是原"自己写解析"方案的核心卖点，不能因换库而丢。

## 2. 方案

`markitdown[pdf]>=0.1.7`（MIT，`microsoft/markitdown`）：PDF 后端为 `pdfplumber` + `pdfminer.six`，整文档输出 Markdown，并用 **`\f`（form feed, chr(12)）分页符**分隔页。

关键设计：**整文档转一次 → 按 `\f` 切逐页片段 → 每段复用现有 `_parse_markdown_text` 出结构块 → 打 PyMuPDF 权威页码**。

- 页码权威源 = **PyMuPDF 页序**（`doc[page_no-1]`），不依赖 markitdown 对单页 PDF 的页码判断（初版逐页 `convert_stream` 方案脆弱，已弃）。
- 复用 `app/ingestion/chunking.py` 已有逻辑：`b.page` → `page_nums[0]`/`page_nums`（跨页块取首页）→ `ChunkRecord.page_num`。**无需改 chunking 层**，`page_num` 由"全 0"变为正确的多页集合。

## 3. 实现要点（`app/ingestion/parsers.py`）

1. **整文档转 + `\f` 切页**：`_split_markdown_by_page(whole_md, page_count)` 按 `\f` 切分、空段补齐、超段末段合并。
2. **CJK 标题补回**：markitdown 0.1.7 对「第X章 / 第X节 / 一二三、 / 1.1.」等**不输出 `#`**。新增 `_is_cjk_heading`/`_cjk_heading_level`，PDF 路径传 `recover_cjk_headings=True` 把章节行补成 heading 块（章=1 / 编号1.1=2 / 节条=2 / 其余=3）。MD/DOCX/TXT 路径**默认不补**，避免误把正文当标题。
3. **页眉页脚剔除**：`_strip_repeating_headers_footers` 阈值由「≥3 页」放宽为 **`c>=3 or (c==n_pages and n_pages>=2)`**——整文档 `\f` 切页后页脚在 `\f` 前独立成段，更易切；短文档（每页都有页脚）也能剔。
4. **红页/扫描页降级**：质量门判 `red` 或 markitdown 该页空输出 → 退回 PyMuPDF 原文 `_page_text_to_blocks`（与现状一致）。VLM 扫描页 OCR 属 M2，未接。
5. **死代码清理**：删初版逐页方案残留 `import io`、`from markitdown import ... StreamInfo`、`_page_to_pdf_bytes`。

## 4. 验证结果

### 4.1 结构回归 `scripts/verify_pdf_structure.py`（无外部依赖）— PASS

| 样本 | 块数 | 标题数 | 页码 | 页脚剔除 | chunk page_num |
|------|------|--------|------|----------|----------------|
| policy_after_sales_return.pdf | 20（10 heading） | 10（≥7 ✓） | 1..5 ✓ | ✓ | [1,3,5] ✓ |
| manual_guardian_x1.pdf | 8（4 heading） | 4 ✓ | [1,2] | ✓ | [1] |
| manual_warehouse_ops_2026-02-10.pdf | 8（4 heading） | 4 ✓ | [1,2] | ✓ | [1] |
| sample_scan_page.pdf | 2（红/黄降级） | — | [1] | ✓ | [1] |
| hr_annual_leave_2026.md | 13（7 heading） | 7 ✓ | — | — | MD 路径不补 CJK 标题 ✓ |
| it_account_terminal_management.docx | 22（10 heading） | 10 ✓ | — | — | — |
| misc_canteen_menu.txt | 4 | — | — | — | — |

标题识别样例（policy）：`第一章 适用范围` … `第十章 争议处理` 全部补回。

### 4.2 全量重解析回归（无 LLM）

- **`verify_golden_anchors`：55/55 锚句命中**（38 文档含 5 PDF 全部经 markitdown 重解析，未破坏既有结构）。
- `verify_cli_ingest`：14/0。
- `verify_token_accounting`：5/5。
- `compileall`：RC=0。

### 4.3 未跑项（环境限制，非代码回归）

`m1 / m3 / m4 / m6 / sse` 等需真实 LLM 的回归，**本环境无 `DASHSCOPE_API_KEY` 未执行**。改动面仅限 `parsers.py`（解析层）与 `pyproject.toml`（依赖），不触碰 agent / retrieval / eval 链路，且上述无 LLM 回归全绿，风险可控。

## 5. 关键教训（跨会话复用）

- **markitdown 0.1.7 对中文「第X章」不输出 `#`**——若依赖其原生标题，章节结构会丢。解法：`recover_cjk_headings` 二次识别补回。
- **整文档转 + `\f` 切页 比 逐页 `convert_stream` 稳**：页码权威源交给 PyMuPDF，避免 markitdown 对单页 PDF 的页码判断差异。
- **页脚剔除要整文档切页后做**：markitdown 把页脚接在上段正文里时，按块文本匹配剔不掉；`\f` 切页后页脚独立成段，阈值再放宽到"每页都出现"即可。
- **复用 > 重写**：现有 `_parse_markdown_text` 已支持 heading/table/code/image/paragraph 五类块，PDF 直接复用，零新解析逻辑即拿回页内结构。

## 6. 残留 / 后续（M2）

- 扫描页 OCR：`markitdown-ocr` 插件走 Qwen-VL（现有 `app/ingestion/vlm.py` 桩已就位，未接）。
- 图文混排对齐：PDF 内图片/表格与正文的版面对齐未做（markitdown 表格已结构化，图片仅登记元数据）。
- LLM 回归待在具备 `DASHSCOPE_API_KEY` 的环境补跑。
