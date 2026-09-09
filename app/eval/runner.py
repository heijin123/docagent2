"""评估编排（F7.5）：加载 → 锚句定位 → 检索/答案 → 指标 → 门槛判定 → 报告落盘。

- 检索层指标：恒跑（mock 向量时仅链路、指标 SKIP 并标 degraded，见 F7.4）；
- 答案层指标：`--answers` 显式开启（需真 Key，F7.3 可选）；
- 门槛：真实向量 recall@5 ≥ 0.8；mock → 指标 SKIP（degraded 标注，不算失败）。
- 报告：data/reports/eval_report_latest.json（含 provider / degraded / 逐用例明细）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.config import settings
from app.eval.golden import load_golden, locate_expected_chunks
from app.eval.metrics import compute_answer_metrics, compute_retrieval_metrics
from app.retrieval.hybrid import HybridRetriever

RECALL_THRESHOLD = 0.8


class EvalRunner:
    def __init__(self, retriever: HybridRetriever, bm25_store, agent=None):
        self.retriever = retriever
        self.bm25_store = bm25_store
        self.agent = agent  # 可选（--answers 时注入）

    def run(self, golden_path: str | Path | None = None,
            *, with_answers: bool = False, top_k: int = 5,
            tenant_id: str | None = None) -> dict:
        cases, meta = load_golden(golden_path)
        locate_expected_chunks(cases, self.bm25_store, tenant_id=tenant_id)

        degraded = self.retriever.embedder.degraded
        provider = self.retriever.embedder.info().get("provider", "?")
        started = time.time()

        # ── 检索层 ──────────────────────────────────────────
        topk_results: dict[str, list[dict]] = {}
        retrieval_hits: dict[str, set[str]] = {}
        for case in cases:
            res = self.retriever.retrieve(case.query, top_n=top_k)
            topk_results[case.id] = res.items
            retrieval_hits[case.id] = {it["chunk_id"] for it in res.items}

        # mock 向量无语义 → 指标 SKIP（F7.4）
        if degraded:
            ret_metrics = {"recall_at_k": None, "mrr": None, "evaluated": 0,
                           "skipped": len(cases), "per_case": {},
                           "note": "mock 向量无语义，指标 SKIP（仅链路回归）"}
        else:
            ret_metrics = compute_retrieval_metrics(cases, topk_results, k=top_k)

        # ── 答案层（可选）───────────────────────────────────
        ans_metrics: dict | None = None
        if with_answers and self.agent is not None:
            answer_results: dict[str, dict] = {}
            for case in cases:
                try:
                    reply = self.agent.reply(
                        case.query, f"{tenant_id or settings.default_tenant_id}:eval:{case.id}")
                    answer_results[case.id] = {
                        "citations": [c.chunk_id for c in reply.citations],
                        "intent": reply.intent,
                        "confidence": reply.confidence,
                    }
                except Exception as exc:  # noqa: BLE001
                    answer_results[case.id] = {"citations": [], "error": str(exc)}
            ans_metrics = compute_answer_metrics(
                cases, answer_results, retrieval_hits)

        # ── 门槛判定 ────────────────────────────────────────
        recall = ret_metrics["recall_at_k"]
        if degraded:
            verdict = "skipped"   # 仅链路，不算通过也不算失败
        else:
            verdict = "pass" if (recall or 0.0) >= RECALL_THRESHOLD else "fail"

        report = {
            "meta": {
                "generated_at": int(time.time()),
                "schema": meta.get("schema"),
                "golden_file": str(golden_path or settings.base_dir / "data/golden/qa_golden.json"),
                "provider": provider,
                "degraded": degraded,
                "tenant_id": tenant_id or settings.default_tenant_id,
                "top_k": top_k,
                "with_answers": with_answers,
                "elapsed_s": round(time.time() - started, 2),
            },
            "retrieval": ret_metrics,
            "answer": ans_metrics,
            "verdict": verdict,
            "threshold": {"recall_at_k": RECALL_THRESHOLD},
        }
        return report

    @staticmethod
    def write_report(report: dict, path: str | Path | None = None) -> Path:
        p = Path(path or (settings.reports_dir / "eval_report_latest.json"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return p


def run_eval(retriever: HybridRetriever, bm25_store, agent=None, *,
             golden_path: str | Path | None = None, with_answers: bool = False,
             top_k: int = 5, tenant_id: str | None = None) -> tuple[dict, Path]:
    """便捷入口：跑评估并落盘，返回 (report, report_path)。"""
    runner = EvalRunner(retriever, bm25_store, agent=agent)
    report = runner.run(golden_path, with_answers=with_answers,
                        top_k=top_k, tenant_id=tenant_id)
    return report, runner.write_report(report)
