"""评估指标计算（F7.2 / F7.3）。

- 检索层：recall@k（至少一个期望块进 top-k 的用例占比）+ MRR（首个期望块倒数排名的均值）；
  锚句定位失败的用例不计分母（F7.2）；
- 答案层：引用可回查率——citations.chunk_id 必须属于检索命中块（F7.3）。
"""
from __future__ import annotations

from app.eval.golden import GoldenCase


def compute_retrieval_metrics(
    cases: list[GoldenCase],
    topk_results: dict[str, list[dict]],
    k: int = 5,
) -> dict:
    """topk_results: {case_id: [{chunk_id, ...}]}（按分数降序）。

    返回 {recall_at_k, mrr, evaluated, skipped, per_case}。
    """
    evaluated = 0
    skipped = 0
    hits = 0
    rr_sum = 0.0
    per_case: dict[str, dict] = {}
    for case in cases:
        if not case.located:
            skipped += 1
            per_case[case.id] = {"status": "anchor_miss", "hit": False, "mrr": 0.0}
            continue
        ranked = topk_results.get(case.id, [])
        ranked_ids = [r["chunk_id"] for r in ranked[:k]]
        expected = set(case.expected_chunk_ids)
        hit = any(cid in expected for cid in ranked_ids)
        # MRR：首个期望块在结果中的位置（1-based）
        rr = 0.0
        for idx, cid in enumerate(ranked_ids, start=1):
            if cid in expected:
                rr = 1.0 / idx
                break
        evaluated += 1
        hits += 1 if hit else 0
        rr_sum += rr
        per_case[case.id] = {"status": "hit" if hit else "miss",
                             "hit": hit, "mrr": round(rr, 4)}
    recall = hits / evaluated if evaluated else 0.0
    mrr = rr_sum / evaluated if evaluated else 0.0
    return {"recall_at_k": recall, "mrr": mrr, "evaluated": evaluated,
            "skipped": skipped, "per_case": per_case}


def compute_answer_metrics(
    cases: list[GoldenCase],
    answer_results: dict[str, dict],
    retrieval_hits: dict[str, set[str]],
) -> dict:
    """answer_results: {case_id: {citations: [chunk_id], ...}}；
    retrieval_hits: {case_id: {本次检索命中的 chunk_id 全集}}（引用可回查的合法集）。

    返回 {verifiable_rate, verified, skipped, per_case}。
    """
    verified = 0
    skipped = 0
    ok = 0
    per_case: dict[str, dict] = {}
    for case in cases:
        ans = answer_results.get(case.id)
        if ans is None:
            skipped += 1
            per_case[case.id] = {"status": "no_answer", "verifiable": False}
            continue
        cites = set(ans.get("citations", []) or [])
        legal = retrieval_hits.get(case.id, set())
        verified += 1
        # 引用可回查：每个 citation 都在本次检索命中块内
        verifiable = bool(cites) and cites.issubset(legal)
        ok += 1 if verifiable else 0
        per_case[case.id] = {"status": "ok" if verifiable else "unverifiable",
                             "verifiable": verifiable,
                             "citations": sorted(cites)}
    rate = ok / verified if verified else 0.0
    return {"verifiable_rate": rate, "verified": verified, "skipped": skipped,
            "per_case": per_case}
