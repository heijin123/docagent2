"""标定第二阶段：词表覆盖度信号（读第一阶段落盘的明细，无需再调 embedding）。

第一阶段结论：向量相似度不可单独判定"无相关资料"（A 组 min 0.406 vs B 组 max 0.542 重叠）。

本阶段测**词面覆盖**：query 的内容词在语料里"有多少是存在的、存在的是否只是烂大街词"。
用 IDF 加权（越罕见的词缺席，越说明整个话题不在库里）：
    coverage = Σ idf(命中语料的词) / Σ idf(所有内容词)
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.bm25store import BM25Store, tokenize  # noqa: E402

# 功能词 / 疑问词：不承载主题，参与计算会把"是什么/怎么样"当成覆盖
_STOP = {
    "的", "了", "吗", "呢", "是", "有", "在", "和", "与", "及", "或", "怎么", "怎样", "如何",
    "什么", "哪些", "哪个", "多少", "为什么", "可以", "需要", "应该", "会不会", "是否",
    "一下", "一般", "通常", "我们", "你们", "公司", "本", "该", "这", "那", "个", "项",
    "请问", "告诉", "介绍", "说明一下", "情况", "方面",
}


def content_terms(query: str) -> list[str]:
    """内容词 = jieba 分词 - 停用词 - 单字 - 纯数字/标点。"""
    out = []
    for t in tokenize(query):
        t = t.strip()
        if not t or t in _STOP:
            continue
        if len(t) < 2:
            continue
        if re.fullmatch(r"[\d\W_]+", t):
            continue
        out.append(t)
    return out


def build_df(store: BM25Store) -> tuple[Counter, int]:
    df: Counter = Counter()
    n = 0
    for row in store.iter_valid_chunks():
        n += 1
        for t in set(tokenize(row["content"])):
            df[t] += 1
    return df, n


def coverage(query: str, df: Counter, n_docs: int) -> dict:
    terms = content_terms(query)
    if not terms:
        return {"terms": [], "cov": 1.0, "absent": [], "min_idf_present": 99.0}
    total = 0.0
    hit = 0.0
    absent: list[str] = []
    present_idfs: list[float] = []
    for t in terms:
        d = df.get(t, 0)
        # 缺席词给 N+1 的 df → 最大 idf（"这词库里一次都没出现"）
        idf = math.log((n_docs + 1) / (d + 1)) + 1.0
        total += idf
        if d == 0:
            absent.append(t)
        else:
            hit += idf
            present_idfs.append(idf)
    return {"terms": terms, "cov": hit / total if total else 1.0,
            "absent": absent,
            "min_idf_present": min(present_idfs) if present_idfs else 0.0,
            "max_idf": max((math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1.0) for t in terms)}


def main() -> int:
    raw = json.loads(Path("data/reports/relevance_calibration.json").read_text(encoding="utf-8"))
    store = BM25Store()
    df, n_docs = build_df(store)
    print(f"语料 {n_docs} chunk / 词表 {len(df)} 词\n")

    for grp, key in (("组 A(应命中)", "in_scope"), ("组 B(应无资料)", "out_scope")):
        rows = []
        for r in raw[key]:
            c = coverage(r["query"], df, n_docs)
            rows.append((c["cov"], r))
        covs = sorted(x[0] for x in rows)
        n = len(covs)
        print(f"── {grp} 覆盖度分布 min={covs[0]:.3f} p25={covs[n // 4]:.3f} "
              f"med={covs[n // 2]:.3f} p75={covs[3 * n // 4]:.3f} max={covs[-1]:.3f}")
        print("   最低 8 条：")
        for cov, r in sorted(rows, key=lambda x: x[0])[:8]:
            c = coverage(r["query"], df, n_docs)
            print(f"     cov={cov:.3f} vec={r['vec_top1']:.3f} 缺词={c['absent'][:4]}  {r['query'][:34]}")
        print()

    a = [coverage(r["query"], df, n_docs)["cov"] for r in raw["in_scope"]]
    b = [coverage(r["query"], df, n_docs)["cov"] for r in raw["out_scope"]]
    print("── 覆盖度阈值扫描 ──")
    print(f"{'C':>6} {'A_survive':>10} {'A_kill':>8} {'B_detect':>9} {'B_miss':>7}  分隔")
    for c in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        surv = sum(1 for v in a if v >= c) / len(a)
        det = sum(1 for v in b if v < c) / len(b)
        sep = "  ← 无重叠" if min(a) >= c > max(b) else ""
        print(f"{c:>6.2f} {surv:>10.3f} {1 - surv:>8.3f} {det:>9.3f} {1 - det:>7.3f}{sep}")

    print("\n── 复合网格：cov < C 且 vec_top1 < T（双证据才判无资料）──")
    print(f"{'C':>5} {'T':>5} {'A_survive':>10} {'A_kill':>8} {'B_detect':>9}")
    grid = []
    for c in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for t in [0.40, 0.45, 0.50, 0.55]:
            a_ok = sum(1 for r in raw["in_scope"]
                       if not (coverage(r["query"], df, n_docs)["cov"] < c
                               and r["vec_top1"] < t)) / len(raw["in_scope"])
            b_ok = sum(1 for r in raw["out_scope"]
                       if coverage(r["query"], df, n_docs)["cov"] < c
                       and r["vec_top1"] < t) / len(raw["out_scope"])
            print(f"{c:>5.2f} {t:>5.2f} {a_ok:>10.3f} {1 - a_ok:>8.3f} {b_ok:>9.3f}")
            grid.append((c, t, a_ok, b_ok))
    safe = [g for g in grid if g[2] >= 1.0]
    if safe:
        best = max(safe, key=lambda g: g[3])
        print(f"\n★ 零误杀工作点：C={best[0]:.2f} T={best[1]:.2f} "
              f"（A 全保留，B 识别 {best[3]:.3f}）")
        for g in safe:
            print(f"   备选 C={g[0]:.2f} T={g[1]:.2f} → B_detect={g[3]:.3f}")
    else:
        print("\n无零误杀工作点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
