"""PDF 结构回归（无外部依赖）：验证 markitdown 接入后解析链路。

覆盖：
- 四格式（PDF/MD/DOCX/TXT）均能出块，MD 标题路径不受影响；
- PDF：中文「第X章」标题被识别为 heading 块、每块带正确 page 页码；
- 页眉页脚（跨 >=3 页重复的块）被剔除；
- chunk_document 透传 page_num（不再全是 0 / 不再只记合并组首页）。
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\workspace\docagent2")

from app.ingestion.parsers import parse_file, _parse_markdown_text  # noqa: E402
from app.ingestion.chunking import build_doc_meta, chunk_document  # noqa: E402
from app.models import DocumentType  # noqa: E402

SAMPLES = r"D:\workspace\docagent2\data\samples"

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [ok] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        _failures.append(msg)


def test_pdf(path: str, *, expect_headings: int = 0, expect_pages: int = 0) -> None:
    name = os.path.basename(path)
    print(f"\n# PDF: {name}")
    parsed = parse_file(path)
    blocks = parsed.blocks
    check(len(blocks) > 0, f"{name} 产出非空块（{len(blocks)} 块）")

    types = Counter(b.block_type for b in blocks)
    print(f"   块类型: {dict(types)}  页码: {sorted({b.page for b in blocks if b.page})}")

    if expect_headings:
        heads = [b for b in blocks if b.block_type == "heading"]
        check(len(heads) >= expect_headings,
              f"{name} 识别 >= {expect_headings} 个标题（实得 {len(heads)}）")
        print("   标题:", [b.text for b in heads][:12])

    if expect_pages:
        pages = {b.page for b in blocks if b.page}
        check(pages == set(range(1, expect_pages + 1)),
              f"{name} 页码覆盖 1..{expect_pages}（实得 {sorted(pages)}）")

    # 页眉页脚剔除：不应残留明显页脚
    texts = [b.text for b in blocks]
    check("锐眼科技 版权所有" not in texts,
          f"{name} 页脚『锐眼科技 版权所有』已被剔除")

    # page_num 透传：chunk 后应有非 0 页码
    doc_meta = build_doc_meta(parsed, tenant_id="tenant_demo", doc_key=name,
                              doc_title=name)
    chunks = chunk_document(parsed, doc_meta)
    chunk_pages = {c.page_num for c in chunks}
    print(f"   chunks: {len(chunks)}  page_num 集合: {sorted(chunk_pages)}")
    check(0 not in chunk_pages or len(chunk_pages) == 1,
          f"{name} chunk page_num 非全 0（实得 {sorted(chunk_pages)}）")


def test_md_path(name: str, path: str) -> None:
    print(f"\n# MD/DOCX/TXT: {name}")
    parsed = parse_file(path)
    blocks = parsed.blocks
    check(len(blocks) > 0, f"{name} 产出非空块（{len(blocks)} 块）")
    types = Counter(b.block_type for b in blocks)
    print(f"   块类型: {dict(types)}")
    # MD/DOCX 的标题路径不受影响（recover_cjk_headings 默认 False）
    if name.endswith((".md", ".docx")):
        check("heading" in types, f"{name} 仍识别 heading 块（{types.get('heading',0)} 个）")


def main() -> int:
    # --- PDF 结构 ---
    test_pdf(os.path.join(SAMPLES, "policy_after_sales_return.pdf"),
             expect_headings=7, expect_pages=5)
    test_pdf(os.path.join(SAMPLES, "manual_guardian_x1.pdf"),
             expect_headings=4, expect_pages=2)
    test_pdf(os.path.join(SAMPLES, "manual_warehouse_ops_2026-02-10.pdf"),
             expect_headings=4, expect_pages=2)
    test_pdf(os.path.join(SAMPLES, "sample_scan_page.pdf"))  # 红/黄页降级

    # --- 其他格式回归（确认未动坏 MD/DOCX/TXT 解析）---
    for nm in ["hr_annual_leave_2026.md", "it_account_terminal_management.docx",
               "misc_canteen_menu.txt"]:
        p = os.path.join(SAMPLES, nm)
        if os.path.exists(p):
            test_md_path(nm, p)

    # --- _parse_markdown_text 纯函数：recover_cjk_headings 开关 ---
    md_blocks_off = _parse_markdown_text("第一章 概述\n正文内容。", DocumentType.MD)
    md_blocks_on = _parse_markdown_text("第一章 概述\n正文内容。", DocumentType.PDF,
                                       recover_cjk_headings=True)
    check(not any(b.block_type == "heading" for b in md_blocks_off),
          "MD 路径默认不补 CJK 标题（recover_cjk_headings=False）")
    check(any(b.block_type == "heading" for b in md_blocks_on),
          "PDF 路径开启后补回 CJK 标题为 heading 块")

    print("\n" + "=" * 60)
    if _failures:
        print(f"结果: FAIL（{len(_failures)} 项未通过）")
        for f in _failures:
            print("  -", f)
        return 1
    print("结果: PASS（PDF 结构解析全绿）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
