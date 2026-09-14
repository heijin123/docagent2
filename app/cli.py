"""CLI 入口：`python -m app.cli ingest <paths...> [--rebuild]` / `python -m app.cli eval [--answers]`。

与 API 同构（M4 接 FastAPI 时复用 IngestPipeline；M6 评估复用 HybridRetriever）；报告表格输出。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.ingestion.pipeline import IngestPipeline, run_ingest
from app.retrieval.embedding import build_embedder


def _fmt_table(reports: list[dict]) -> str:
    """逐文档报告表格。"""
    headers = ["文件", "状态", "格式", "v", "块数", "入库", "跳过", "红", "黄", "耗时(s)", "降级"]
    widths = [len(h) for h in headers]
    rows = []
    for r in reports:
        row = [
            r["filename"], r["status"], r["format"] or "-",
            str(r["version"] or "-"), str(r["chunks_created"]),
            str(r["stored"]), str(r["skipped"]),
            ",".join(map(str, r["red_pages"])) or "-",
            ",".join(map(str, r["yellow_pages"])) or "-",
            str(r["elapsed_s"] or "-"),
            "mock✓" if r["degraded"] else ("-" if r["provider"] == "mock" else "real"),
        ]
        rows.append(row)
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines.append(sep)
    lines.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append(sep)
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def _fmt_notes(reports: list[dict]) -> list[str]:
    """失败原因 / VLM 附注 / 警告 单独列出。"""
    notes: list[str] = []
    for r in reports:
        if r["error"]:
            notes.append(f"✗ {r['filename']}: {r['error']}")
        if r["vlm_note"]:
            notes.append(f"◐ {r['filename']} [VLM] {r['vlm_note']}")
        for w in r.get("warnings", []):
            notes.append(f"△ {r['filename']}: {w}")
    return notes


def cmd_ingest(args: argparse.Namespace) -> int:
    if args.list_models:
        print(build_embedder().info())
        return 0

    reports = run_ingest(args.paths, rebuild=args.rebuild)

    print()
    print(_fmt_table(reports))
    notes = _fmt_notes(reports)
    if notes:
        print()
        print("\n".join(notes))

    ok = sum(1 for r in reports if r["status"] == "ok")
    print(f"\n共 {len(reports)} 个文件，成功 {ok}，失败 {len(reports) - ok}")

    provider = reports[0]["provider"] if reports else "?"
    degraded = any(r["degraded"] for r in reports)
    print(f"Embedding provider={provider}"
          f"{'（degraded: 无 Key 降级 mock，无语义仅链路）' if degraded else ''}")

    _rebuild_anchors()
    return 0 if ok == len(reports) else 1


def _rebuild_anchors() -> None:
    """摄取完成后重建主题锚点词表（供 rewrite 短路判据用）。

    词表是从 BM25 语料派生的话题词，**语料一变就可能过期**；过期只会让判据
    退化成"一律改写"（不劣化正确性），但会白烧一跳，所以这里自动跟上。
    任何失败都不影响 ingest 结果——只提示。
    """
    from app.agent import anchors
    from app.core.config import settings

    try:
        meta = anchors.build_vocab(
            Path(settings.bm25_dir) / "corpus.db",
            Path(__file__).resolve().parents[1] / "data" / "kb_anchors.json",
        )
        print(f"锚点词表已重建：{meta['counts']['union']} 个词 "
              f"（docs={meta['docs']}）")
    except Exception as exc:  # noqa: BLE001 — 不影响摄取结果
        print(f"⚠ 锚点词表重建失败（rewrite 短路会退化为一律改写）：{exc}")


def cmd_eval(args: argparse.Namespace) -> int:
    """M6 评估（F7）：golden 集 + recall@5/MRR + 可选答案层可回查率。"""
    from app.agent.llm import build_llm
    from app.agent.graph import AgentApp
    from app.eval.runner import run_eval
    from app.retrieval.bm25store import BM25Store
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.vectorstore import VectorStore

    embedder = build_embedder()
    vector_store = VectorStore()
    bm25_store = BM25Store()
    retriever = HybridRetriever(
        vector_store=vector_store, bm25_store=bm25_store, embedder=embedder)

    agent = None
    if args.answers:
        # 走 Redis 持久化（与线上一致）；Redis 不可达时 AgentApp 内部自动降级内存，不影响评估正确性
        agent = AgentApp(retriever, build_llm(), memory_checkpoint=False)

    report, path = run_eval(
        retriever, bm25_store, agent=agent,
        golden_path=args.golden, with_answers=args.answers,
        top_k=args.top_k, limit=args.limit)

    m = report["meta"]
    print(f"\n===== 评估报告（provider={m['provider']}"
          f"{'，degraded' if m['degraded'] else ''}）=====")
    if m["cases_used"] != m["cases_total"]:
        print(f"题集: 分层抽样 {m['cases_used']}/{m['cases_total']} 条"
              f"（limit={m['limit']}，按 type 轮询；分组 {m['by_type']}）")
    r = report["retrieval"]
    if m["degraded"]:
        print(f"检索层: recall@5/MRR SKIP（mock 向量无语义，仅链路回归）")
        print(f"  锚句定位: {len(r['per_case'])} 条待定位（mock 下不评估）")
    else:
        print(f"检索层: recall@5 = {r['recall_at_k']:.3f} | MRR = {r['mrr']:.3f} "
              f"| 评估 {r['evaluated']} 条（跳过 {r['skipped']} 条锚句未命中）")
    if report["answer"] is not None:
        a = report["answer"]
        print(f"答案层: 引用可回查率 = {a['verifiable_rate']:.3f} "
              f"| 评估 {a['verified']} 条")
    if report.get("cost"):
        c = report["cost"]
        print(f"成本: 模型={c['llm_model']} | LLM 调用={c['llm_calls']} 次 "
              f"| 总 token={c['total_tokens']} "
              f"(prompt {c['total_prompt_tokens']} / completion {c['total_completion_tokens']}) "
              f"| 预估 ¥{c['est_cost_cny']:.4f} "
              f"| 单问均 {c['avg_tokens_per_query']} token")
    if report.get("latency"):
        lt = report["latency"]
        print(f"延迟: 单问全链路 p50 = {lt['p50_ms']}ms | p95 = {lt['p95_ms']}ms "
              f"| mean = {lt['mean_ms']}ms | max = {lt['max_ms']}ms（{lt['samples']} 条）")
    if report.get("verify"):
        v = report["verify"]
        print(f"校验: verify 首过 {v['first_pass']}/{v['reached_verify']} "
              f"= {v['first_pass_rate']:.3f} | 重试 {v['retried']} 条 "
              f"| 判定 {v['judgements']} 次（单问均 {v['avg_judgements_per_query']}）"
              f" | max 重试 {v['max_retries']}")
    print(f"门槛: recall@5 ≥ {report['threshold']['recall_at_k']} → "
          f"verdict = {report['verdict']}")
    print(f"报告: {path}")

    # mock 降级不算失败（F7.4）；真实向量 recall 不达标 → 退出码 1
    if report["verdict"] == "fail":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="Enterprise-QA-Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="摄取文件/目录（串行，单文档失败不中断）")
    ingest_p.add_argument("paths", nargs="+", help="文件或目录路径")
    ingest_p.add_argument("--rebuild", action="store_true",
                          help="强制版本化重灌（同 hash 也 bump 版本 → 新 chunk_id，旧版本随后软删）")
    ingest_p.add_argument("--list-models", action="store_true", help="显示当前 embedding 配置")
    ingest_p.set_defaults(func=cmd_ingest)

    eval_p = sub.add_parser("eval", help="M6 评估（golden + recall@5/MRR + 可回查率）")
    eval_p.add_argument("--golden", default=None, help="golden 集路径（默认 data/golden/qa_golden.json）")
    eval_p.add_argument("--answers", action="store_true",
                        help="额外跑答案层引用可回查率（需真 Key）")
    eval_p.add_argument("--top-k", type=int, default=5, help="recall@k 的 k（默认 5）")
    eval_p.add_argument("--limit", type=int, default=None,
                        help="只跑分层抽样的前 N 条（按 type 轮询、每类至少 1 条）；默认跑全集 55 条")
    eval_p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
