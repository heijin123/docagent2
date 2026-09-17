"""检索相关性阈值标定（F2.10 前置实验，2026-09-17）。

问题：向量检索**恒返回 topN**（无距离阈值），库里没有对应资料时也会给回"最像的 N 条"，
导致 retrieve 结果恒非空 → `no_data`（0 LLM 如实告知缺失）只在索引真空时触发；
正常"库里没答案"要烧 answer+verify 2~3 轮 LLM 靠模型自觉。

本脚本用**真实 embedding** 取两组样本的分数分布，找可分阈值：
- 组 A（应命中）：golden 55 条，语料确实覆盖；
- 组 B（应判无资料）：语料明确不覆盖的问题。

指标（在候选阈值 T 上）：
- `survive`：组 A 中 top1 相似度 ≥ T 的比例（**误杀率 = 1 - survive**，最要紧）；
- `detect` ：组 B 中 top1 相似度 < T 的比例（正确判"无资料"）；
- 同时观察向量 top1 与 BM25 top1 的组合信号（词面重叠是独立的第二证据）。

用法：
    PYTHONPATH=D:/workspace/docagent2 python scripts/calibrate_relevance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.golden import load_golden  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402

# 组 B：语料明确不覆盖的问题。
# 前 6 条"远端话题"（显然不在企业制度库），后 6 条"看似企业问题但库中无对应文档"
# ——后者才是真实场景里最需要被识别出来的（用户以为公司有规定）。
OUT_OF_SCOPE = [
    "量子计算机的量子纠错码是怎么实现的",
    "2026年世界杯决赛是哪两支球队",
    "如何训练宠物狗学会握手",
    "咖啡机除垢的正确步骤",
    "钢琴调音一般多久做一次",
    "空气炸锅烤鸡需要多少度烤多久",
    "员工宿舍入住申请的条件和流程是什么",
    "公司班车线路和发车时刻表在哪里查",
    "会议室预订规则和投影设备借用流程",
    "内部推荐人才的奖励标准是多少",
    "员工体检套餐包含哪些项目、怎么预约",
    "办公用品申领流程和领用额度是多少",
]


def _probe(retriever: HybridRetriever, query: str) -> dict:
    """取一次检索的分数信号（向量 top1/top2、BM25 top1、命中数）。"""
    res = retriever.retrieve(query, top_n=5)
    vec = res.detail.get("baseline", {}).get("vector_hits", [])
    bm = res.detail.get("baseline", {}).get("bm25_hits", [])
    v_scores = [float(h.get("score") or 0.0) for h in vec]
    b_scores = [float(h.get("bm25_score") or 0.0) for h in bm]
    return {
        "query": query,
        "n_items": len(res.items),
        "vec_top1": v_scores[0] if v_scores else 0.0,
        "vec_top2": v_scores[1] if len(v_scores) > 1 else 0.0,
        "vec_top5": v_scores[:5],
        "bm25_top1": b_scores[0] if b_scores else 0.0,
        "bm25_top2": b_scores[1] if len(b_scores) > 1 else 0.0,
    }


def main() -> int:
    retriever = HybridRetriever()
    if retriever.embedder.degraded:
        print("embedder 降级（mock）→ 无语义信号，标定无效")
        return 1

    cases, _meta = load_golden()
    print(f"组 A：golden {len(cases)} 条（语料覆盖）")
    in_scope = []
    for i, c in enumerate(cases, 1):
        p = _probe(retriever, c.query)
        p["id"] = c.id
        p["type"] = c.type
        in_scope.append(p)
        if i % 10 == 0:
            print(f"  …{i}/{len(cases)}", flush=True)

    print(f"\n组 B：{len(OUT_OF_SCOPE)} 条（语料不覆盖）")
    out_scope = [_probe(retriever, q) for q in OUT_OF_SCOPE]

    def _stat(vals: list[float]) -> str:
        vals = sorted(vals)
        n = len(vals)
        if not n:
            return "n=0"
        return (f"min={vals[0]:.3f} p10={vals[int(0.10 * n)]:.3f} "
                f"p25={vals[n // 4]:.3f} med={vals[n // 2]:.3f} "
                f"p75={vals[3 * n // 4]:.3f} max={vals[-1]:.3f}")

    a_v1 = [p["vec_top1"] for p in in_scope]
    b_v1 = [p["vec_top1"] for p in out_scope]
    print("\n── 向量 top1 相似度分布 ──")
    print("组 A(应命中):", _stat(a_v1))
    print("组 B(应无资料):", _stat(b_v1))

    print("\n── 组 A 向量 top1 最低的 10 条（决定阈值的下界）──")
    for p in sorted(in_scope, key=lambda x: x["vec_top1"])[:10]:
        print(f"  {p['vec_top1']:.3f}  bm25={p['bm25_top1']:>6.2f}  {p['id']} ({p['type']}) {p['query'][:38]}")

    print("\n── 组 B 向量 top1 最高的 6 条（决定阈值的上界）──")
    for p in sorted(out_scope, key=lambda x: -x["vec_top1"])[:6]:
        print(f"  {p['vec_top1']:.3f}  bm25={p['bm25_top1']:>6.2f}  {p['query'][:38]}")

    print("\n── 阈值扫描 ──")
    print(f"{'T':>6} {'A_survive':>10} {'A_kill':>8} {'B_detect':>9} {'B_miss':>7}  分隔")
    best = None
    for t in [x / 100 for x in range(30, 76, 5)]:
        surv = sum(1 for v in a_v1 if v >= t) / len(a_v1)
        det = sum(1 for v in b_v1 if v < t) / len(b_v1)
        sep = "  ← 无重叠" if min(a_v1) >= t > max(b_v1) else ""
        print(f"{t:>6.2f} {surv:>10.3f} {1 - surv:>8.3f} {det:>9.3f} {1 - det:>7.3f}{sep}")
        if surv >= 0.98:  # 误杀 ≤2% 前提下最大化识别率
            if best is None or det > best[1]:
                best = (t, det, surv)

    if best:
        print(f"\n建议阈值 T={best[0]:.2f}（误杀 {1 - best[2]:.3f}，识别 {best[1]:.3f}）")
    else:
        print("\n结论：无满足「误杀 ≤2%」的阈值 → 单靠向量相似度不可分，需复合信号")

    # 复合信号：向量 top1 低 且 词面也弱（BM25 无实质命中）→ 双重确认
    print("\n── 复合信号（vec_top1<T 且 bm25_top1<1.0）──")
    for t in [0.45, 0.50, 0.55, 0.60]:
        surv = sum(1 for p in in_scope
                   if not (p["vec_top1"] < t and p["bm25_top1"] < 1.0)) / len(in_scope)
        det = sum(1 for p in out_scope
                  if p["vec_top1"] < t and p["bm25_top1"] < 1.0) / len(out_scope)
        print(f"  T={t:.2f}  A_survive={surv:.3f}  B_detect={det:.3f}")

    out = Path("data/reports/relevance_calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"in_scope": in_scope, "out_scope": out_scope},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
