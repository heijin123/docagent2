# VLM OCR 真实样本回归（2026-09-16）

用**真实文档**（而非合成 fixture）验证 PDF 两条 VLM 路径：红页整页 OCR、非红页内嵌图 OCR。

## 0. 起因

用户提供 `~/Downloads/2_react_paper_en.pdf`，作为「带内嵌位图的 PDF 样本」，用于补齐内嵌图 OCR 的真实端到端验证。

**核查结论（重要）**：该 PDF **不含任何栅格位图**，无法充当内嵌图样本。详见 §1。

## 1. 核查：2_react_paper_en.pdf 为什么不能作为内嵌图样本

| 项 | 实测 |
|---|---|
| 页数 | 33 |
| 文本量 | ≈107 KB（每页 1.0k–6.5k 字符，有完整文本层） |
| 原始 `/Subtype /Image` 对象数 | **0**（全库 xref 遍历确认） |
| `page.get_images()` / `get_image_info()` | 全页 0 |
| 图形构成 | **矢量**：1330 个 drawing ops；6 个 `Form XObject`，其 `PTEX.FileName(./iclr2023/figure/*.pdf)` 表明是 pdfTeX `\includegraphics` 内联的**矢量 PDF 图** |
| 含图页 | 2 / 5 / 7 / 14 / 15 |

**关键点：矢量图里的文字本就在文本层，未丢内容。** 例：第 5 页 `get_text()` 直接含图内坐标轴标签与图注——

```
... 0 | 5 | 10 | 15 | 20 | #CoT-SC trials | 26 | 28 | 30 | 32 | 34 | HotpotQA EM ...
... Figure 2: PaLM-540B prompting results with respect to number of CoT-SC samples used. ...
```

因此：
- 内嵌图 OCR 路径**正确地不触发**（无栅格图 → 0 次 VLM 调用）；
- 该文档**不存在图文内容丢失**问题。

## 2. 转机：下载夹里有真带位图的文档

全量扫描 `~/Downloads` 下 15 个 PDF，找到真实位图样本：

| 文件 | 页数 | 栅格图 | 图页 | 说明 |
|---|---|---|---|---|
| 环旭电子…法律意见书.pdf | 12 | 2 | p12（text=0，1×1654×2340） | **真扫描页** |
| 天力锂能…2024年年度报告.pdf | 287 | 1 | p138（text=480，1×685×389） | 图文混排，图内含人名/比例 |
| 振华重工…2025年年度报告（英文版）.pdf | 220 | 12 | p10（text=2876，4 张） | 4 张纯照片（图内无文字） |
| 1_hermes_agent_manual_cn.pdf | 83 | 2 | p83（text=286，1×1710×624） | 图内含「微信搜一搜」 |

## 3. 真实样本集（可复现，含溯源）

`scripts/make_real_vlm_samples.py` 从真实 PDF **无损抽取单页**（保留原文本层与位图），生成 4 个精炼样本到 `data/real_samples/`：

| 样本 | 来源 | 特性 | 验证目标 |
|---|---|---|---|
| `scan_signature_page.pdf` | 环旭电子 p12 | 117KB，text=0，imgs=1 | 红页整页 OCR（真实正样本） |
| `mixed_orgchart_page.pdf` | 天力锂能 p138 | 241KB，text=480，imgs=1 | 内嵌图 OCR（图内文字不在文本层 → **真实丢内容**） |
| `mixed_photos_page.pdf` | 振华重工 p10 | 445KB，text=2876，imgs=4 | 「无文字」护栏（纯照片不产块） |
| `react_figure_page.pdf` | 2_react p5 | 138KB，text=3673，imgs=0 | 矢量图 → 内嵌图路径空转 |

其中 `mixed_orgchart_page.pdf` 是最有价值的一项：图内 `王瑞庆 / 蒋轩 / 李雯 / 18.78 / 9.84 / 301152` 经实测**全部不在文本层**（`in t138 == False`），构成真实的图文内容丢失。

## 4. 回归结果：`scripts/verify_vlm_ocr_real.py` — 14 PASS / 0 FAIL

```
VLM_ENABLED=True  model=qwen-vl-max  transcriber=DashScopeTranscriber  has_key=True

[1] scan_signature_page.pdf — 整页扫描 → 红页 VLM OCR
  [PASS] 被判定为红页  — red_pages=[1] text_chars=0
  [PASS] 红页转录产出块  — blocks=1
  [PASS] OCR 还原扫描页文字  — 命中=['方达','律师事务所','季诺','丛大林','盖章']
  [PASS] 块页码溯源正确  — page=1 vs red=1

[2] mixed_orgchart_page.pdf — 文本层 + 位图结构图 → 内嵌图 VLM OCR
  [PASS] 有文本层（非红页）  — red=[] text=607
  [PASS] 图内文字原本不在文本层（构成真实丢内容）  — 文本层含王瑞庆=False
  [PASS] 内嵌图 OCR 产出块  — blocks=1
  [PASS] OCR 还原图内人名/比例  — 命中=['王瑞庆','李雯','9.84','301152']
  [PASS] 块标记 source=vlm_ocr_image
  [PASS] 块带页码  — pages=[1]

[3] mixed_photos_page.pdf — 文本层 + 4 张纯照片 → 「无文字」护栏
  [PASS] 纯照片不产块（无文字被过滤）  — blocks=0

[4] react_figure_page.pdf — 矢量图（栅格图数=0）→ 内嵌图路径空转
  [PASS] 无栅格图  — raster_imgs=0
  [PASS] 矢量图内文字已在文本层（无内容丢失）  — 含 'trials'=True
  [PASS] 内嵌图路径空转（0 块、0 次 VLM 调用）  — pages_with_ocr=[]
```

**红页 OCR 实际还原文本**（节选）：

```
（本页无正文，为《上海市方达律师事务所关于环旭电子股份有限公司2026年
员工持股计划的法律意见书》之签署页）上海市方达律师事务所（盖章）…
负责人：季诺  经办律师：丛大林 / 郗璐璐  2026年9月9日
```

**内嵌图 OCR 实际还原文本**：`王瑞庆`、`李雯`、`9.84`、`301152`（图内股权结构关系）。

## 5. 附带发现（低危，未改）

长中文文档（年报）上 `recover_cjk_headings` 的**标题层级颠倒 + 过度识别**：

- 天力锂能全文档 552 heading / 3862 blocks（14%）；level 分布 L1=289 / L2=119 / L3=144。
- 层级映射反了：`第一节 …`→L2，`一、公司信息`→L3，而 `1、同时按照国际会计准则…`→**L1**（`_cjk_heading_level` 对 `^[0-9]+、` 返回 `dots+1=1`）。
- 长句子被误标 heading（如 `1、同时按照国际会计准则与按照中国会计准则披露的财务报告中净利润和净资产差异情况`）。

**影响评估：仅元数据，不影响召回。** 证据：`heading_level` 除 `verify_m1.py` 的契约断言外**无任何消费方**；`chunking.py` 只按 `block_type in {table, code, image}` 判「原子块」，heading 与 paragraph 同路径（可累积可切），故过度识别不改变 chunk 边界。

**若后续要把 heading 层级用于分层上下文/检索，需先修 `_cjk_heading_level` 的层级映射。** 本次未改（避免动既有 55/55 golden 基线）。

## 6. 结论

- **红页整页 OCR**：在**真实扫描件**上端到端跑通，还原出盖章/签名/日期等扫描内容，页码溯源正确。
- **内嵌图 OCR**：在**真实年报图表**上跑通，救回图内人名/比例（原本不在文本层，属真实内容丢失）；纯照片经「无文字」护栏过滤，不污染语料。
- **矢量图（2_react）**：内嵌图路径正确空转——矢量图文字本就在文本层，OCR 无必要，且无内容丢失。
- 样本不含入主语料（避免改动 38 文档 / 105 chunk 的 eval 基线），仅作独立真实回归。

## 7. 复现

```bash
PY=C:/Users/Administrator/.workbuddy/binaries/python/envs/docagent2/Scripts/python.exe
# 1) 从真实 PDF 生成样本（默认取 ~/Downloads）
PYTHONPATH=D:/workspace/docagent2 $PY scripts/make_real_vlm_samples.py
# 2) 跑真实回归（需 Key + VLM_ENABLED=1）
PYTHONPATH=D:/workspace/docagent2 $PY scripts/verify_vlm_ocr_real.py
```

无 Key / VLM 未启用时，`verify_vlm_ocr_real.py` 只跑离线结构检查并打印 SKIP（不判失败）。
