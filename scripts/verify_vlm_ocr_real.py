"""VLM OCR **真实样本**回归：用从真实 PDF 抽取的单页样本，跑真实 Qwen-VL 端到端验证。

与 `verify_vlm_ocr.py`（合成 fixture + 离线 FakeTranscriber 为主）互补：
本脚本用**真实文档**（扫描件 / 年报图表 / 照片 / 学术论文）验证两条 VLM 路径：

  1. scan_signature_page.pdf   整页栅格扫描（text=0）→ 红页整页 OCR 应还原页面文字
  2. mixed_orgchart_page.pdf   文本层 + 位图图表（图内人名/比例不在文本层）
                               → 内嵌图 OCR 应还原「王瑞庆/蒋轩/李雯」等图内文字
  3. mixed_photos_page.pdf     文本层 + 4 张纯照片（图内无文字）
                               → 内嵌图 OCR 应被「无文字」护栏过滤，产 0 块
  4. react_figure_page.pdf     文本层 + 矢量图（栅格图数=0）
                               → 内嵌图路径空转（0 块、0 次 VLM 调用）

样本由 `scripts/make_real_vlm_samples.py` 从真实 PDF 抽取（含溯源）。
无 Key / VLM 未启用时只跑离线结构检查并给出 SKIP 提示（不判失败）。

用法：
    python scripts/verify_vlm_ocr_real.py [--dir data/real_samples]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def _norm(s: str) -> str:
    return "".join((s or "").split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/real_samples")
    args = ap.parse_args()

    sample_dir = Path(args.dir)
    if not sample_dir.is_absolute():
        sample_dir = ROOT / sample_dir
    if not sample_dir.is_dir():
        print(f"样本目录不存在：{sample_dir}，请先运行 scripts/make_real_vlm_samples.py")
        return 1

    from app.core.config import settings
    from app.ingestion.parsers import parse_file
    from app.ingestion.pipeline import IngestPipeline
    from app.ingestion.vlm import NoneTranscriber, get_transcriber

    transcriber = get_transcriber()
    real_vlm = not isinstance(transcriber, NoneTranscriber) and settings.has_api_key
    print(f"VLM_ENABLED={settings.vlm_enabled}  model={settings.qwen_vlm_model}  "
          f"transcriber={type(transcriber).__name__}  has_key={settings.has_api_key}")
    print()

    pl = IngestPipeline()

    # ── 1) 红页：整页栅格扫描 ─────────────────────────────
    print("[1] scan_signature_page.pdf — 整页扫描 → 红页 VLM OCR")
    f = sample_dir / "scan_signature_page.pdf"
    parsed = parse_file(str(f))
    red = [q.page_no for q in parsed.page_qualities if q.level == "red"]
    check("被判定为红页", bool(red),
          f"red_pages={red} text_chars={len(parsed.blocks[0].text) if parsed.blocks else 0}")
    if real_vlm and red:
        blks = pl._transcribe_red_pages(f, parsed, red)
        check("红页转录产出块", len(blks) >= 1, f"blocks={len(blks)}")
        if blks:
            txt = _norm(blks[0].text)
            hit = [k for k in ("方达", "律师事务所", "季诺", "丛大林", "盖章") if k in txt]
            check("OCR 还原扫描页文字", bool(hit), f"命中={hit}｜文本={_norm(blks[0].text)[:60]}")
            check("块页码溯源正确", blks[0].page == red[0], f"page={blks[0].page} vs red={red[0]}")
    elif not real_vlm:
        print("     (SKIP 真实 VLM 分支：无 Key 或未启用)")
    print()

    # ── 2) 内嵌图：位图图表（图内含文字）──────────────────
    print("[2] mixed_orgchart_page.pdf — 文本层 + 位图结构图 → 内嵌图 VLM OCR")
    f = sample_dir / "mixed_orgchart_page.pdf"
    parsed = parse_file(str(f))
    red = [q.page_no for q in parsed.page_qualities if q.level == "red"]
    doc_text = _norm(" ".join(b.text for b in parsed.blocks))
    check("有文本层（非红页）", not red and len(doc_text) > 50, f"red={red} text={len(doc_text)}")
    check("图内文字原本不在文本层（构成真实丢内容）",
          "王瑞庆" not in doc_text and "蒋轩" not in doc_text,
          f"文本层含王瑞庆={'王瑞庆' in doc_text}")
    if real_vlm:
        inline = pl._transcribe_inline_images(f, parsed, red)
        blocks = [b for _, bs in inline.items() for _, b in bs]
        check("内嵌图 OCR 产出块", len(blocks) >= 1, f"blocks={len(blocks)}")
        if blocks:
            txt = _norm(" ".join(b.text for b in blocks))
            hit = [k for k in ("王瑞庆", "蒋轩", "李雯", "18.78", "9.84", "301152") if k in txt]
            check("OCR 还原图内人名/比例", bool(hit), f"命中={hit}")
            check("块标记 source=vlm_ocr_image",
                  all(b.metadata.get("source") == "vlm_ocr_image" for b in blocks))
            check("块带页码", all(b.page == 1 for b in blocks),
                  f"pages={[b.page for b in blocks]}")
    else:
        print("     (SKIP 真实 VLM 分支)")
    print()

    # ── 3) 内嵌图：纯照片（图内无文字）────────────────────
    print("[3] mixed_photos_page.pdf — 文本层 + 4 张纯照片 → 「无文字」护栏")
    f = sample_dir / "mixed_photos_page.pdf"
    parsed = parse_file(str(f))
    red = [q.page_no for q in parsed.page_qualities if q.level == "red"]
    if real_vlm:
        inline = pl._transcribe_inline_images(f, parsed, red)
        blocks = [b for _, bs in inline.items() for _, b in bs]
        check("纯照片不产块（无文字被过滤）", len(blocks) == 0,
              f"blocks={len(blocks)}，产出文本={_norm(''.join(b.text for b in blocks))[:40]}")
    else:
        print("     (SKIP 真实 VLM 分支)")
    print()

    # ── 4) 矢量图（无栅格图）→ 内嵌图路径空转 ─────────────
    print("[4] react_figure_page.pdf — 矢量图（栅格图数=0）→ 内嵌图路径空转")
    f = sample_dir / "react_figure_page.pdf"
    parsed = parse_file(str(f))
    red = [q.page_no for q in parsed.page_qualities if q.level == "red"]
    import pymupdf
    _d = pymupdf.open(str(f))
    n_img = len(_d[0].get_images(full=True))
    _d.close()
    check("无栅格图", n_img == 0, f"raster_imgs={n_img}")
    doc_text = _norm(" ".join(b.text for b in parsed.blocks))
    check("矢量图内文字已在文本层（无内容丢失）",
          "Figure" in doc_text or "CoT-SC" in doc_text or "trials" in doc_text,
          f"含 'trials'={'trials' in doc_text}")
    inline = pl._transcribe_inline_images(f, parsed, red)
    check("内嵌图路径空转（0 块、0 次 VLM 调用）", len(inline) == 0,
          f"pages_with_ocr={list(inline)}")
    print()

    print(f"===== 结果：{PASS} 通过 / {FAIL} 失败 =====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
