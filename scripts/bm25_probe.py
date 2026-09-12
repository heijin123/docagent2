"""BM25 离线召回探针：无需 API Key，量化 recall@k / MRR，并按题型分组。

为什么需要它
------------
完整 M6 评估走 Hybrid（向量 + BM25），需要 `DASHSCOPE_API_KEY`；未配 Key 时
embedding 降级为 mock，检索层直接 SKIP，指标不可得。本脚本只走 BM25 这条
**不依赖任何外部服务**的路径，给出检索质量的**下界**，用途：

- 配合 `scripts/corpus_stats.py` 判断语料规模是否让 recall@5 / MRR 具备区分度；
- 调整切分策略 / BM25 参数前后做 A/B 对照（改 `--samples` 指向不同语料目录即可）；
- 按 golden 的 `type` 分组观察不同题型差异（semantic / exact / year / paraphrase /
  noisy），定位短板题型（弥补 M6 主评估只报总分的不足）。

指标口径与 `app/eval/metrics.py` 对齐：仅统计锚句**可定位**的用例（located=True），
命中判定为"检索结果 chunk_id ∈ 锚句期望块集合"。MRR 用全库排名（不受 topk 截断）。

注意：这是 BM25 单路下界，线上 Hybrid 经 RRF 融合向量路后通常更高。

用法::

    python scripts/bm25_probe.py
    python scripts/bm25_probe.py --samples data/samples --topk 5
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="BM25 离线召回探针")
    ap.add_argument("--samples", default=str(BASE / "data" / "samples"),
                    help="语料目录（默认 data/samples）")
    ap.add_argument("--golden", default=None, help="golden 集路径（默认 data/golden/qa_golden.json）")
    ap.add_argument("--topk", type=int, default=5, help="recall@k 的 k（默认 5）")
    args = ap.parse_args()

    sys.path.insert(0, str(BASE))
    from rank_bm25 import BM25Okapi

    from app.eval.golden import load_golden, normalize
    from app.ingestion.chunking import build_doc_meta, chunk_document
    from app.ingestion.parsers import parse_file
    from app.models import SUPPORTED_EXTENSIONS
    from app.retrieval.bm25store import tokenize

    # ── 1) 用项目真实解析+切分链路构建内存语料（与线上摄取一致）──
    root = Path(args.samples)
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    chunks = []
    for p in files:
        parsed = parse_file(p)
        meta = build_doc_meta(parsed, tenant_id="tenant_demo", doc_key=p.name)
        chunks.extend(chunk_document(parsed, meta))

    bm25 = BM25Okapi([tokenize(c.content) for c in chunks])

    # ── 2) 加载 golden 并定位锚句期望块 ──
    cases, _meta = load_golden(args.golden)
    norm_corpus = [(c.chunk_id, normalize(c.content)) for c in chunks]
    for case in cases:
        anchor_n = normalize(case.anchor)
        case.expected_chunk_ids = [cid for cid, nc in norm_corpus if anchor_n in nc]
        case.located = bool(case.expected_chunk_ids)

    located = [c for c in cases if c.located]
    cover = args.topk / len(chunks) if chunks else 0.0
    print(f"语料：{len(chunks)} chunks / {len(files)} docs"
          f"   （top-{args.topk} 覆盖率 ≈ {cover:.1%}）")
    print(f"题集：{len(cases)} 条，锚句可定位 {len(located)} 条"
          f"（不可定位者不计入指标分母）\n")
    if not located:
        print("没有可定位用例，无法评估。")
        return 1

    # ── 3) 逐条检索 ──
    per_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "hit1": 0, "hitk": 0, "rr": 0.0})
    misses: list[tuple[str, str, str, int, list[str]]] = []
    n_hit1 = n_hitk = 0
    rr_sum = 0.0

    for case in located:
        scores = bm25.get_scores(tokenize(case.query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        expected = set(case.expected_chunk_ids)

        rank_full = 0
        for pos, i in enumerate(order, start=1):
            if chunks[i].chunk_id in expected:
                rank_full = pos
                break

        hit1 = 1 if rank_full == 1 else 0
        hitk = 1 if 0 < rank_full <= args.topk else 0
        rr = (1.0 / rank_full) if rank_full else 0.0
        n_hit1 += hit1
        n_hitk += hitk
        rr_sum += rr

        g = per_type[case.type]
        g["n"] += 1
        g["hit1"] += hit1
        g["hitk"] += hitk
        g["rr"] += rr

        if not hitk:
            top_docs: list[str] = []
            for i in order[:args.topk]:
                title = chunks[i].doc_title
                if title not in top_docs:
                    top_docs.append(title)
            misses.append((case.id, case.query, case.expected_doc, rank_full, top_docs))

    n = len(located)
    print(f"{'指标':<16}{'值':>10}")
    print("-" * 28)
    print(f"{'recall@' + str(args.topk):<16}{n_hitk / n:>10.3f}")
    print(f"{'recall@1':<16}{n_hit1 / n:>10.3f}")
    print(f"{'MRR':<16}{rr_sum / n:>10.3f}")

    print(f"\n{'type':<12}{'n':>4}{'r@1':>8}{'r@' + str(args.topk):>8}{'MRR':>8}")
    print("-" * 42)
    for t, g in sorted(per_type.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{t:<12}{int(g['n']):>4}{g['hit1'] / g['n']:>8.3f}"
              f"{g['hitk'] / g['n']:>8.3f}{g['rr'] / g['n']:>8.3f}")

    if misses:
        print(f"\n未命中 {len(misses)} 条（正确答案未进 top-{args.topk}）：")
        for cid, query, doc, rank, top_docs in misses:
            where = f"全库第 {rank} 名" if rank else "全库未命中"
            print(f"  {cid}  期望={doc}  ({where})")
            print(f"        query={query}")
            print(f"        top{args.topk} 实际命中文档：{', '.join(top_docs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
