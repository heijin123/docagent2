# PDF 内嵌图 OCR 接入报告（图文混排补齐）

- 日期：2026-09-16
- 需求：F1.9（PDF 质量门 / VLM 转录）· M2 残留项收口
- 结论：**非红页内嵌图 OCR 已接入并真实跑通**；`verify_vlm_ocr.py` 18/18 PASS。

## 1. 背景与问题

上一轮 M2 已接「红页整页 OCR」（整页≈0 文本的扫描页 → PyMuPDF 渲染整页 → Qwen-VL 转录）。
但**图文混排页**仍是洞：

- markitdown 对 PDF **内嵌位图不产块**——解析层 `_parse_markdown_text` 只把
  `![alt](http...)` **外链**图片识别为 image 块，本地流式图片（PDF 内嵌 XObject）进不来；
- 后果：页面里**图表 / 带文字的示意图**里的文字**完全丢失**，既不落块也不 OCR；
- 红页整页 OCR 只在「整页几乎无文本」时触发，**混排页有文本层 → 判 green/yellow → 不触发**。

即：**纯文字 ✅、表格 ✅、整页扫描 ✅，唯独混排页的「图内文字」丢失**。

## 2. 方案（与红页路径同源，扩展为双路）

复用已跑通的红页 OCR 机制（PyMuPDF 取图 → base64 data URL → Qwen-VL → paragraph 块），
新增一路处理**非红页内嵌图**：

```
PDF ──PyMuPDF──┬─ 每页文本 → markitdown → 逐页 blocks（heading/paragraph/table）
               │
               ├─ 质量门 grade_page ── red 页 → 渲染整页 PNG → VLM 整页 OCR ─┐
               │                                                          ├─ _merge_vlm_text_blocks
               └─ 非 red 页 → page.get_images() 抽内嵌图 → VLM 图 OCR ──────┘   （按页码插回页末尾）
```

### 关键改动

| 文件 | 改动 |
|---|---|
| `app/ingestion/vlm.py` | `transcribe` 签名泛化：`transcribe(image_bytes, page_no, *, mime="image/png", prompt=None)`；data URL 用传入 MIME（内嵌图可能是 jpeg）；提示词可自定义。ABC / `NoneTranscriber` 同步。 |
| `app/ingestion/pipeline.py` | 新增 `_transcribe_inline_images()`：抽图 → 过滤/去重/限额 → OCR → `source=vlm_ocr_image` paragraph 块；抽出 `_merge_vlm_text_blocks()` 统一「按页码归位」；`run_document` 的 `vlm_note` 合并两类计数。 |

### 成本护栏（token 是一等公民）

- 跳过 **< 64×64** 小图（logo / 图标）；
- 同一 **xref 去重**（同图多页引用只 OCR 一次）；
- 总量上限 **`VLM_MAX_INLINE_IMAGES`（默认 20）**；
- **跳过红页**（已整页 OCR，避免重复计费与噪声）；
- 图 OCR 提示词要求「无文字只回（无文字）」，结果命中该标记则**不产块**（装饰图/纯照片不污染语料）；
- 每次 VLM 调用 emit `log_llm_usage(node="vlm_ocr")`。

### 归位逻辑

`_merge_vlm_text_blocks(parsed, transcripts_by_page, inline_by_page)`：
剔除红页原文块（不可信）→ 把「红页转录 + 内嵌图 OCR」块**插回其所属页的末尾**。
页内块序保持连续，避免图内文字被甩到文末、打散 chunk 上下文。

## 3. 验证

### 3.1 回归脚本 `scripts/verify_vlm_ocr.py`（18/18 PASS）

- **离线（FakeTranscriber）**：红页路径（判 red → `source=vlm_ocr` 块、page 溯源）+ 内嵌图路径
  （混排 fixture 非 red → `source=vlm_ocr_image` 块）+ 合并逻辑（红页原文块移除、块序正确）。
- **真实（Qwen-VL）**：合成扫描页恢复「保养 / 每季度」；**合成混排 fixture 的内嵌图**恢复「保养 / 每季度」。
- ⚠ 修正脚本原有隐患：原 monkeypatch 的是 `vlm_mod.get_transcriber`，而 pipeline 使用的是
  **本模块导入的名字**（`app.ingestion.pipeline.get_transcriber`）→ 离线分支实际未隔离（靠真实 VLM 蒙过）。
  已改为 patch `pl_mod.get_transcriber`，离线真正生效。

### 3.2 合成混排 fixture 构造

`A4 页`：文本层（>100 字符 → 非 red）+ 一张 800×320 带 CJK 文字的 PNG 用 `page.insert_image` 嵌入。
该图 `page.get_images` 可抽出、`extract_image` 返回原始像素 → Qwen-VL OCR 还原文字。

### 3.3 真实语料 + 全量回归

- 5 个 sample PDF `get_images` 均为 **0**（无内嵌位图）→ 内嵌图一路只能在合成 fixture 验证。
- 全量 `app.cli ingest data/samples --rebuild`：**38/38 ok、0 失败**、`provider=dashscope`；
  `sample_scan_page.pdf` 红页转录正常（无内嵌提示，符合语料实际）。
- `verify_pdf_structure` **PASS**（解析层未动）、`golden_anchors` **55/55**、`cli_ingest` **14/0**、`token` **5/5**。
- 单轮真实评测 `app.cli eval`：**recall@5 = 0.873**（≥0.8 → pass）/ MRR ≈0.79 —— **无回归**（检索链路未受影响）。

## 4. 现状总结（PDF 全内容类型）

| 内容 | 状态 | 说明 |
|---|---|---|
| 纯文字 | ✅ | markitdown → 段落/标题，页码正确 |
| 表格 | ✅ | markitdown → md 表格 → table 块（整块不切） |
| 整页扫描/红页 | ✅ | PyMuPDF 渲染整页 → Qwen-VL 整页 OCR |
| **非红页内嵌图** | ✅（本报告） | 抽图 → Qwen-VL OCR → 可检索段落 |
| 图文**版面对齐** | ⏳ | 图与邻近文字的精细位置/切分关系未做（当前按「页内追加」） |
| DOCX 内嵌图 | ⏳ | 仅 PDF 接了内嵌图 OCR |

## 5. 残留 / 后续

1. **版面对齐**：当前内嵌图文字按「该页末尾」追加，未与邻近段落做位置对齐；对严格依赖「图注—正文」顺序的场景可再细化。
2. **DOCX 内嵌图**：`_parse_docx` 已识别 inline image 占位（`[图片]`），但未抽图 OCR。
3. **配额/成本可视化**：内嵌图 OCR 的 token 已记账，但摄取期成本未计入查询账本（与红页一致）。
