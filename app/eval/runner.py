"""评估编排（F7.5）：加载 → 锚句定位 → 检索/答案 → 指标 → 门槛判定 → 报告落盘。

- 检索层指标：恒跑（mock 向量时仅链路、指标 SKIP 并标 degraded，见 F7.4）；
- 答案层指标：`--answers` 显式开启（需真 Key，F7.3 可选）；
- 子集抽检：`limit=N` 时按 `type` **分层轮询**抽样（每类至少 1 条，再按序补齐），
  用于"只想在有限时间内拿到成本/延迟账本"的快速跑；指标仍按子集计算，
  meta 里同时记 `cases_total / cases_used / case_ids`，避免被误读成全集结论；
- 延迟：对答案层逐条计时，汇总 p50/p95（NFR「单问全链路 P95」的直接证据）；
- 门槛：真实向量 recall@5 ≥ 0.8；mock → 指标 SKIP（degraded 标注，不算失败）。
- 报告：data/reports/eval_report_latest.json（含 provider / degraded / 逐用例明细）。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from app.core.config import settings
from app.core.observability import TokenUsage
from app.eval.golden import GoldenCase, load_golden, locate_expected_chunks
from app.eval.metrics import compute_answer_metrics, compute_retrieval_metrics
from app.retrieval.hybrid import HybridRetriever

RECALL_THRESHOLD = 0.8


def _stratified_subset(cases: list[GoldenCase], limit: int | None) -> list[GoldenCase]:
    """按 `type` 分层轮询抽最多 limit 条，保持原顺序。

    先每类取 1 条（保证 exact / noisy 这类小样本不被整类漏掉），再逐轮按
    各类内部顺序补到 limit。`limit` 为 None/≤0/≥总数 → 返回全集。
    """
    if limit is None or limit <= 0 or limit >= len(cases):
        return cases
    by_type: dict[str, list[GoldenCase]] = {}
    for c in cases:
        by_type.setdefault(c.type, []).append(c)
    types = list(by_type)                      # 首现顺序（确定性，不随 dict 哈希变化）
    picked: list[GoldenCase] = []
    for t in types:
        if len(picked) >= limit:
            break
        picked.append(by_type[t][0])
    i = 1
    while len(picked) < limit:
        added = False
        for t in types:
            if len(picked) >= limit:
                break
            if i < len(by_type[t]):
                picked.append(by_type[t][i])
                added = True
        if not added:
            break
        i += 1
    keep = {c.id for c in picked}
    return [c for c in cases if c.id in keep]


class EvalRunner:
    def __init__(self, retriever: HybridRetriever, bm25_store, agent=None):
        self.retriever = retriever
        self.bm25_store = bm25_store
        self.agent = agent  # 可选（--answers 时注入）

    def run(self, golden_path: str | Path | None = None,
            *, with_answers: bool = False, top_k: int = 5,
            tenant_id: str | None = None, limit: int | None = None) -> dict:
        cases, meta = load_golden(golden_path)
        cases_total = len(cases)
        cases = _stratified_subset(cases, limit)
        locate_expected_chunks(cases, self.bm25_store, tenant_id=tenant_id)
        by_type: dict[str, int] = {}
        for c in cases:
            by_type[c.type] = by_type.get(c.type, 0) + 1

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
        answer_results: dict[str, dict] = {}
        cost_acc: TokenUsage = TokenUsage()  # 整轮评估的 LLM token 总账（价格可见化）
        ans_calls = 0       # 问答轮数（= 题数）
        ans_llm_calls = 0   # 真实 LLM 调用次数（含 verify 重试；>>轮数说明重试在烧钱）
        latencies: list[int] = []
        if with_answers and self.agent is not None:
            # 冷启动口径：thread_id 带"本次运行"盐值。否则 `eval:{case.id}` 固定不变 + Redis
            # checkpointer 会把**上一轮评估**的 messages 残留进来 → 本轮 _history 非空 →
            # rewrite 规则短路失效（实测 2 跳变 3 跳、prompt 混入无关历史），
            # 指标随"第几次跑"漂移、不可横向比较。
            run_salt = time.strftime("%Y%m%d-%H%M%S")
            total = len(cases)
            for idx, case in enumerate(cases, start=1):
                # 逐条打印进度：慢跑时可直接 tail 日志看进行到哪（否则数分钟无输出）
                print(f"  [{idx}/{total}] {case.id} ({case.type}) …", end="", flush=True)
                t0 = time.perf_counter()
                try:
                    reply = self.agent.reply(
                        case.query,
                        f"{tenant_id or settings.default_tenant_id}:eval:{run_salt}:{case.id}")
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    latencies.append(elapsed_ms)
                    u = self.agent.last_usage
                    n_calls = getattr(self.agent, "last_calls", 0)
                    n_retries = getattr(self.agent, "last_retries", 0)
                    n_verified = bool(getattr(self.agent, "last_verified", False))
                    cost_acc = cost_acc + u
                    ans_calls += 1
                    ans_llm_calls += n_calls
                    answer_results[case.id] = {
                        "citations": [c.chunk_id for c in reply.citations],
                        "intent": reply.intent,
                        "confidence": reply.confidence,
                        "latency_ms": elapsed_ms,
                        "llm_calls": n_calls,
                        "retries": n_retries,
                        "verified": n_verified,
                        "usage": u.to_dict(),
                    }
                    print(f" {elapsed_ms}ms intent={reply.intent} "
                          f"cites={len(reply.citations)} tok={u.total_tokens} "
                          f"calls={n_calls} retries={n_retries}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    latencies.append(elapsed_ms)
                    answer_results[case.id] = {"citations": [], "error": str(exc),
                                               "latency_ms": elapsed_ms}
                    print(f" ERROR {exc}", flush=True)
            ans_metrics = compute_answer_metrics(
                cases, answer_results, retrieval_hits)

        # ── 延迟分位（NFR 单问全链路 P95 的直接证据）──────────
        latency_report: dict | None = None
        if latencies:
            s = sorted(latencies)
            idx95 = min(len(s) - 1, max(0, math.ceil(0.95 * len(s)) - 1))
            latency_report = {
                "samples": len(s),
                "mean_ms": int(sum(s) / len(s)),
                "p50_ms": s[len(s) // 2],
                "p95_ms": s[idx95],
                "max_ms": s[-1],
            }

        # ── verify 首过率（"每问被 verify 打回几次" = 重试成本的直接证据）──
        # 口径：分母只算**真正走到 verify 的题**，用 reply 终态留下的 `verified` 标记判定。
        # **不再**按 LLM 调用次数反推：rewrite 规则短路后正常题只有 2 跳，
        # 旧的 `llm_calls >= 3` 判据曾把 20/24 条正常题误判为"未经过 verify"，
        # 使首过率错报成 0 —— 判据必须来自状态痕迹，不能来自次数猜测。
        # chitchat / contact_guidance / no_data 本就不经 verify，自然不计入分母。
        # 判定数 = retries + 1：末次判定必然发生（通过 → finalize，用尽 → disclose）。
        verify_report: dict | None = None
        if ans_metrics is not None:
            reached = [c for c in cases if answer_results.get(c.id, {}).get("verified")]
            retries_of = [answer_results[c.id].get("retries") or 0 for c in reached]
            first_pass = sum(1 for r in retries_of if r == 0)
            verify_report = {
                "reached_verify": len(reached),
                "first_pass": first_pass,
                "first_pass_rate": round(first_pass / len(reached), 4) if reached else 0.0,
                "retried": len(reached) - first_pass,
                "judgements": sum(r + 1 for r in retries_of),
                "avg_judgements_per_query": round(
                    (sum(r + 1 for r in retries_of) / len(reached)), 3) if reached else 0.0,
                "max_retries": max(retries_of) if retries_of else 0,
            }

        # ── 门槛判定 ────────────────────────────────────────
        recall = ret_metrics["recall_at_k"]
        if degraded:
            verdict = "skipped"   # 仅链路，不算通过也不算失败
        else:
            verdict = "pass" if (recall or 0.0) >= RECALL_THRESHOLD else "fail"

        # ── 成本总账（价格可见化，与准确率同列）──────────────
        if ans_metrics is not None and self.agent is not None:
            model = getattr(getattr(self.agent, "llm", None), "model", None)
            ip, op = settings.llm_price(model)
            cost_report = {
                "llm_model": model,
                # llm_calls 此前误填"轮数"（= ans_calls），与 queries 重复，掩盖了重试轮数；
                # 现改为真实 LLM 调用次数，并保留 queries 便于对照。
                "llm_calls": ans_llm_calls,
                "queries": len(answer_results),
                "avg_llm_calls_per_query": round(ans_llm_calls / ans_calls, 2)
                if ans_calls else 0,
                "total_prompt_tokens": cost_acc.prompt_tokens,
                "total_completion_tokens": cost_acc.completion_tokens,
                "total_tokens": cost_acc.total_tokens,
                "est_cost_cny": round(cost_acc.cost(ip, op), 6),
                "avg_tokens_per_query": round(cost_acc.total_tokens / ans_calls, 1)
                if ans_calls else 0,
            }
        else:
            cost_report = None

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
                "limit": limit,
                "cases_total": cases_total,
                "cases_used": len(cases),
                "by_type": by_type,
                "case_ids": [c.id for c in cases],
                "elapsed_s": round(time.time() - started, 2),
            },
            "retrieval": ret_metrics,
            "answer": ans_metrics,
            "cost": cost_report,
            "latency": latency_report,
            "verify": verify_report,
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
             top_k: int = 5, tenant_id: str | None = None,
             limit: int | None = None) -> tuple[dict, Path]:
    """便捷入口：跑评估并落盘，返回 (report, report_path)。"""
    runner = EvalRunner(retriever, bm25_store, agent=agent)
    report = runner.run(golden_path, with_answers=with_answers,
                        top_k=top_k, tenant_id=tenant_id, limit=limit)
    return report, runner.write_report(report)
