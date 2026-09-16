"""从真实 PDF 中抽取代表性单页，生成 VLM OCR 真实回归样本（可复现、含溯源）。

样本设计（每页测一种情形）：
  1. scan_signature_page.pdf  ← 环旭电子法律意见书 p12
     整页栅格扫描（get_text=0）→ 红页路径真实正样本。
  2. mixed_orgchart_page.pdf  ← 天力锂能年报 p138
     有文本层 + 股权结构图（人名/比例烤进位图，不在文本层）→ 内嵌图正样本。
  3. mixed_photos_page.pdf    ← 振华重工年报（英文版）p10
     有文本层 + 4 张纯照片（图内无文字）→ 内嵌图「无文字」护栏负样本。
  4. react_figure_page.pdf    ← 2_react_paper_en.pdf p5
     有文本层 + 矢量图（栅格图数=0，图内文字本就在文本层）→ 内嵌图路径空转样本。

用法：
    python scripts/make_real_vlm_samples.py [--src-dir <下载目录>] [--out-dir data/real_samples]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pymupdf

DEFAULT_SRC = Path(os.path.expanduser("~")) / "Downloads"

# (输出名, 源文件名, 页码 1-based, 说明)
SPECS = [
    ("scan_signature_page.pdf",
     "环旭电子：上海市方达律师事务所关于环旭电子股份有限公司2026年员工持股计划的法律意见书.pdf",
     12, "整页栅格扫描（无文本层）→ 红页 VLM 整页 OCR"),
    ("mixed_orgchart_page.pdf",
     "天力锂能：天力锂能集团股份有限公司2024年年度报告（更正后）.pdf",
     138, "文本层 + 股权结构图（图内人名/比例不在文本层）→ 内嵌图 VLM OCR"),
    ("mixed_photos_page.pdf",
     "振华重工：振华重工2025年年度报告（英文版）.pdf",
     10, "文本层 + 4 张纯照片（图内无文字）→ 内嵌图 OCR 应被「无文字」护栏过滤"),
    ("react_figure_page.pdf",
     "2_react_paper_en.pdf",
     5, "文本层 + 矢量图（栅格图数=0，图内文字本就在文本层）→ 内嵌图路径空转"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC))
    ap.add_argument("--out-dir", default="data/real_samples")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for out_name, src_name, page_no, desc in SPECS:
        src = src_dir / src_name
        if not src.exists():
            missing.append(str(src))
            print(f"[MISS] {src_name}（跳过）")
            continue
        doc = pymupdf.open(str(src))
        if page_no > doc.page_count:
            print(f"[SKIP] {src_name} 页数 {doc.page_count} < {page_no}")
            doc.close()
            continue
        # 抽取单页（无损：保留原文本层与位图）
        single = pymupdf.open()
        single.insert_pdf(doc, from_page=page_no - 1, to_page=page_no - 1)
        dst = out_dir / out_name
        single.save(str(dst))
        # 统计（校验抽样后仍保留原特性）
        pg = single[0]
        n_img = len(pg.get_images(full=True))
        n_txt = len(pg.get_text("text"))
        single.close()
        doc.close()
        print(f"[OK] {out_name}  <- {src_name} p{page_no}  "
              f"({dst.stat().st_size // 1024}KB, text={n_txt}, imgs={n_img})  — {desc}")

    if missing:
        print("\n缺失源文件：")
        for m in missing:
            print("  ", m)
        return 1
    print(f"\n生成完成 -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
