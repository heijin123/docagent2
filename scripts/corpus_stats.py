"""统计 data/samples 语料规模与切分分布（评估可信度的地基指标）。

为什么需要它
------------
M6 的 recall@5 分母是**整个 chunk 语料**。若 top-5 就能覆盖语料近半，任何检索器
都接近满分，指标失去区分度。本脚本用**项目真实解析+切分链路**（parse_file →
chunk_document，与线上摄取完全一致）统计文档数 / block 数 / chunk 数 / token
分布，并给出 top-5 覆盖率，用来判断"语料规模是否足以支撑 0.8 阈值"。

经验判据
--------
- top-5 覆盖率 = 5 / chunk 总数；
- 覆盖率 > 20%（chunk < 25）时 recall@5 基本恒为 1.0，指标无区分度；
- 覆盖率 < 5%（chunk > 100）时 recall@5 / MRR 才能反映真实检索质量。

用法::

    python scripts/corpus_stats.py
    python scripts/corpus_stats.py --samples data/samples
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="统计评估语料规模与切分分布")
    ap.add_argument("--samples", default=str(BASE / "data" / "samples"),
                    help="语料目录（默认 data/samples）")
    args = ap.parse_args()

    sys.path.insert(0, str(BASE))
    from app.ingestion.chunking import build_doc_meta, chunk_document
    from app.ingestion.parsers import parse_file
    from app.models import SUPPORTED_EXTENSIONS

    root = Path(args.samples)
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    total_chunks = 0
    fmt_counter: Counter[str] = Counter()
    rows: list[tuple[str, int, int, int]] = []

    for p in files:
        parsed = parse_file(p)
        meta = build_doc_meta(parsed, tenant_id="tenant_demo", doc_key=p.name)
        chunks = chunk_document(parsed, meta)
        tokens = sum(c.chunk_size for c in chunks)
        fmt_counter[p.suffix.lower()] += 1
        total_chunks += len(chunks)
        rows.append((p.name, len(parsed.blocks), len(chunks), tokens))

    rows.sort(key=lambda r: -r[2])  # 按 chunk 数降序，长文档排前

    print(f"{'文件':<52}{'blocks':>7}{'chunks':>7}{'tokens':>8}")
    print("-" * 76)
    for name, nb, nc, tok in rows:
        print(f"{name:<52}{nb:>7}{nc:>7}{tok:>8}")
    print("-" * 76)
    print(f"文档数={len(files)}   chunk 总数={total_chunks}")
    print(f"格式分布：{dict(sorted(fmt_counter.items()))}")
    if total_chunks:
        cover = 5 / total_chunks
        verdict = "无区分度（too small）" if cover > 0.20 else (
            "区分度一般" if cover > 0.05 else "区分度良好")
        print(f"top-5 覆盖率 ≈ {cover:.1%}  →  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
