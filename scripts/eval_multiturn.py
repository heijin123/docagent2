"""多轮评估（step 0）：先复现历史轮（真跑 LLM）→ 再评估末轮 → 量化 history 代价与分类行为。

为什么单独一个脚本（而不是塞进 `app.cli eval`）：
1. 单轮集用**原始 query** 算 recall@5；多轮末轮多是指代/省略句（「那二线城市呢？」），
   原始 query 本身不含可检索内容，混进去会让 recall@5 因口径错误而暴跌——不是真实退化。
   本脚本的检索/可回查口径一律基于**改写后的 effective query**（从 reply.notes 提取）。
2. 单轮集一行 = 一问；多轮集一行 = N 轮，需要逐轮留档（prompt/calls/rewrite 走向）才能归因。

核心产出（对应 step 0 要回答的三个问题）：
- ① history 注入的真实 token 代价：末轮 prompt − 首轮 prompt；`independent` 类因 hop 数
     与首轮相同，是**受控净测量**（无「多一跳 rewrite」混淆）；
- ② `should_skip_rewrite` 分类行为：期望「指代/省略 → rewrite」「独立问题 → 跳过」，
     逐条核对实际走向（notes 里 `rewrite: 原样透传（规则短路）` vs `rewrite: <改写文本>`）；
- ③ 多轮下末轮答案的引用可回查：citations ⊂ effective query 的检索命中集，且是否命中锚句期望块。

用法：PYTHONPATH=D:/workspace/docagent2 python scripts/eval_multiturn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import prompts
from app.agent.graph import AgentApp
from app.agent.llm import build_llm
from app.core.config import settings
from app.eval.golden import load_multiturn_golden, locate_expected_chunks
from app.retrieval.bm25store import BM25Store
from app.retrieval.embedding import build_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vectorstore import VectorStore

TENANT = settings.default_tenant_id


def _rewrite_trace(notes: list) -> tuple[str, str]:
    """从 notes 提取最后一次 rewrite 的走向。

    返回 (mode, effective_query)：mode ∈ {"skip", "llm", "none"}。
    - `rewrite: 原样透传（规则短路）` → skip（effective = 原始 query）
    - `rewrite: <改写文本>`           → llm（effective = 改写文本）
    - 无 rewrite 记录（如 chitchat 直接跳转） → none
    """
    last = None
    for n in notes or []:
        s = str(n)
        if s.startswith("rewrite:"):
            last = s
    if last is None:
        return "none", ""
    body = last[len("rewrite:"):].strip()
    if "规则短路" in body or "原样透传" in body:
        return "skip", ""
    return "llm", body


def main() -> int:
    cases, meta = load_multiturn_golden()
    embedder = build_embedder()
    bm25_store = BM25Store()
    retriever = HybridRetriever(
        vector_store=VectorStore(), bm25_store=bm25_store, embedder=embedder)
    locate_expected_chunks(cases, bm25_store, tenant_id=TENANT)

    degraded = embedder.degraded
    agent = AgentApp(retriever, build_llm(), memory_checkpoint=False)
    ip, op = settings.llm_price(agent.llm.model)
    salt = time.strftime("%Y%m%d-%H%M%S")

    print(f"===== 多轮评估（{len(cases)} 条 / provider={embedder.info().get('provider')}"
          f"{'，degraded' if degraded else ''}）=====")
    print(f"模型={agent.llm.model}  价格=({ip}, {op})/1K  thread 盐值={salt}\n")

    # 锚句定位自检：either 未命中（题作废）或命中过多（锚句不唯一，指标会被稀释）
    print("--- 锚句定位自检 ---")
    for c in cases:
        n = len(c.expected_chunk_ids)
        flag = "OK" if n == 1 else ("未命中→该题跳过" if n == 0 else f"命中 {n} 块（锚句不唯一？）")
        print(f"  {c.id} {c.category:11s} 期望块 {n} → {flag}  | {c.anchor}")
    print()

    per_case: list[dict] = []
    total_cost = 0.0
    total_prompt = total_completion = total_calls = 0

    for case in cases:
        thread = f"{TENANT}:mt:{salt}:{case.id}"
        rows: list[dict] = []
        hist_chars = 0     # 本轮注入的 history 字符数（= 上一轮结束后 render_history 的长度）
        hist_text = ""     # 本轮注入的 history 原文（用于零成本的净代价构造分解）
        final_hist = ""
        for i, q in enumerate(case.turns, start=1):
            if i == len(case.turns):
                final_hist = hist_text
            t0 = time.perf_counter()
            reply = agent.reply(q, thread)
            lat = int((time.perf_counter() - t0) * 1000)
            u = agent.last_usage
            mode, eff = _rewrite_trace(reply.notes)
            eff_q = eff or q
            # 空转改写：走了 LLM rewrite 但改写结果与原文逐字相同 → 这一跳是纯浪费
            # （模型自己也认为无需改写，是规则强制它跑了一次）。
            identity = mode == "llm" and eff.strip() == q.strip()
            rows.append({
                "turn": i,
                "query": q,
                "history_chars": hist_chars,
                "rewrite_mode": mode,
                "rewrite_identity": identity,
                "effective_query": eff,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "llm_calls": agent.last_calls,
                "retries": agent.last_retries,
                "verified": agent.last_verified,
                "latency_ms": lat,
                "intent": reply.intent,
                "citations": [c.chunk_id for c in reply.citations],
                "answer": reply.answer,
                "cost_cny": round(u.cost(ip, op), 6),
            })
            # 下一轮将注入的 history 文本（= 先 ingest 追加本轮 query、_history 再丢掉它）
            msgs = agent.history(thread, limit=999)
            hist_text = prompts.render_history(
                msgs, max_rounds=settings.qa_history_rounds)
            hist_chars = len(hist_text)
            print(f"  {case.id} 第{i}轮 {lat:>6}ms calls={agent.last_calls} "
                  f"rewrite={mode:<4}{'空转!' if identity else '     '} "
                  f"prompt={u.prompt_tokens:<6} "
                  f"cites={len(reply.citations)} hist={hist_chars}字 | {q}")

        last = rows[-1]
        # 引用可回查：以 effective query 的检索命中集为合法集（从原始 query 检索对指代句无意义）
        res = retriever.retrieve(last["effective_query"] or last["query"],
                                 user_permission="internal")
        legal = {it["chunk_id"] for it in res.items}
        cites = set(last["citations"])
        expected = set(case.expected_chunk_ids)
        # 净 history 代价的**构造式分解**（零 LLM）：Δprompt 实测混着「多一跳 rewrite」，
        # 而末轮多数要走 rewrite（见 ②），hop 数与首轮不同的样本无法直接相减。故用
        # 「同题、同证据、只切换 history 段」的方式分离出 history 本身的注入量：
        # answer 一次 + rewrite 一次（independent 类跳过改写时只剩 answer 一次）。
        # verify 不注入 history（刻意设计），增量为 0。
        ev = prompts.render_evidence(res.items)
        _, a_wo = prompts.answer_prompt(last["effective_query"] or last["query"], ev, "")
        _, a_w = prompts.answer_prompt(last["effective_query"] or last["query"], ev, final_hist)
        _, r_wo = prompts.rewrite_prompt(case.query, "")
        _, r_w = prompts.rewrite_prompt(case.query, final_hist)
        row = {
            "id": case.id, "category": case.category, "turns": rows,
            "final_effective_query": last["effective_query"],
            "final_rewrite_mode": last["rewrite_mode"],
            "legal_hits": sorted(legal),
            "expected_chunk_ids": sorted(expected),
            "cites_verifiable": bool(cites) and cites.issubset(legal) if cites else False,
            "anchor_hit": bool(cites & expected) if cites else False,
            "expected_doc": case.expected_doc,
            "note": case.note,
            "final_history_text": final_hist,
            "prompt_ab": {
                # history 注入到 answer prompt 的净增字符（实测值，非估算）
                "answer_history_chars": len(a_w) - len(a_wo),
                # 完整 rewrite prompt 字符（= 多出来那一跳的全部输入）
                "rewrite_prompt_chars": len(r_w),
                # history 注入到 rewrite prompt 的净增字符
                "rewrite_history_chars": len(r_w) - len(r_wo),
            },
        }
        per_case.append(row)
        for r in rows:
            total_cost += r["cost_cny"]
            total_prompt += r["prompt_tokens"]
            total_completion += r["completion_tokens"]
            total_calls += r["llm_calls"]

    # ── ① history 代价：末轮 vs 首轮 ────────────────────────────
    print("\n===== ① prompt 增量（末轮 − 首轮，同题内对照，含真实 history 复现）=====")
    print("id    category    hop1 hop2  prompt1  prompt2   Δprompt  Δ%     末轮hist")
    for r in per_case:
        t1, t2 = r["turns"][0], r["turns"][-1]
        d = t2["prompt_tokens"] - t1["prompt_tokens"]
        pct = 100.0 * d / max(1, t1["prompt_tokens"])
        print(f"{r['id']} {r['category']:11s} {t1['llm_calls']:<4} {t2['llm_calls']:<4}  "
              f"{t1['prompt_tokens']:<8} {t2['prompt_tokens']:<8}  {d:<8} {pct:>5.1f}% "
              f"{t2['history_chars']:>5}字")

    print("\n--- 构造式分解（零 LLM）：Δ 里多少是 history、多少是『多那一跳 rewrite』---")
    print("id    hist字  →answer注入  →rewrite注入  rewrite整跳  history净增")
    c2t = 1.6   # chars/token：本轮 llm_call 日志实测 1.57~1.66（同内容自校准）
    for r in per_case:
        ab = r["prompt_ab"]
        net = ab["answer_history_chars"] + ab["rewrite_history_chars"]
        print(f"{r['id']} {r['turns'][-1]['history_chars']:>6}  "
              f"{ab['answer_history_chars']:>10}  {ab['rewrite_history_chars']:>11}  "
              f"{ab['rewrite_prompt_chars']:>10}  {net:>6}字 ≈{net / c2t:.0f}tok")
    # 外推：上限取两级预算的**生效值**（每条截断 × 轮数×2 条，与总量预算取小者）
    per_msg_cap = settings.qa_history_rounds * 2 * settings.history_per_msg_chars
    binding = "总量预算" if settings.history_total_chars < per_msg_cap else "每条×轮数"
    per_inject = min(per_msg_cap, settings.history_total_chars) / c2t
    print("说明：history 被注入**两次**（rewrite 一跳 + answer 一跳）；verify 不注入（刻意设计）。"
          f"\n      上限 = min(每条 {settings.history_per_msg_chars} 字 × {settings.qa_history_rounds} 轮 × 2 条"
          f" = {per_msg_cap} 字, 总量预算 {settings.history_total_chars} 字) → 绑定的是**{binding}**："
          f"单次注入 ≈{per_inject:.0f} tok，**两次合计 ≈{per_inject * 2:.0f} tok**"
          f"（≈ 单问 prompt 的 {per_inject * 2 / 3000:.1f} 倍）。")
    print("      ⚠ 上表 Δprompt 不能当 history 成本用：它被 evidence 体量波动淹没"
          "（m002 甚至为负）—— 只有『同题同证据、只切 history』的构造值才是净成本。")

    # ── ② rewrite 分类行为 ─────────────────────────────────────
    print("\n===== ② should_skip_rewrite 分类行为 =====")
    print("期望：指代/省略 → 必须 rewrite(llm)｜独立新问题 → 应跳过(skip)")
    for r in per_case:
        last = r["turns"][-1]
        mode = r["final_rewrite_mode"]
        want = "skip" if r["category"] == "independent" else "llm"
        ok = "✓" if mode == want else "✗"
        print(f"  {r['id']} {r['category']:11s} 期望={want:<4} 实际={mode:<4} {ok}"
              f"  calls={last['llm_calls']}"
              f"{'  ⚠空转(改写==原文)' if last['rewrite_identity'] else ''}"
              f"  | {last['query']}")
        if mode == "llm" and last["effective_query"]:
            print(f"        改写 → {last['effective_query'][:100]}")
    skipped = sum(1 for r in per_case if r["final_rewrite_mode"] == "skip")
    identity = sum(1 for r in per_case if r["turns"][-1]["rewrite_identity"])
    print(f"  跳过率 {skipped}/{len(per_case)}；其中空转改写 {identity} 条"
          f"（空转 = 花了 LLM 一跳但产出与原文逐字相同）")

    # ── ③ 末轮答案质量 ─────────────────────────────────────────
    print("\n===== ③ 末轮引用可回查 + 锚句命中 =====")
    for r in per_case:
        last = r["turns"][-1]
        print(f"  {r['id']} {r['category']:11s} 可回查={r['cites_verifiable']} "
              f"锚句命中={r['anchor_hit']} cites={len(last['citations'])} "
              f"intent={last['intent']} conf={0.0} | 期望 {r['expected_doc']}")
        print(f"        A: {last['answer'][:160].replace(chr(10), ' ')}")

    # ── 汇总 ───────────────────────────────────────────────────
    n_turns = sum(len(r["turns"]) for r in per_case)
    n_last = len(per_case)
    ok_ver = sum(1 for r in per_case if r["cites_verifiable"])
    ok_anchor = sum(1 for r in per_case if r["anchor_hit"])
    lats = sorted(r["turns"][-1]["latency_ms"] for r in per_case)
    report = {
        "meta": {
            "generated_at": int(time.time()),
            "golden_file": "data/golden/qa_golden_multiturn.json",
            "llm_model": agent.llm.model,
            "provider": embedder.info().get("provider"),
            "degraded": degraded,
            "cases": n_last,
            "turns_total": n_turns,
            "run_salt": salt,
            "history_per_msg_chars": settings.history_per_msg_chars,
            "history_total_chars": settings.history_total_chars,
            "history_rounds": settings.qa_history_rounds,
        },
        "cost": {
            "llm_calls": total_calls,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "est_cost_cny": round(total_cost, 6),
            "avg_tokens_per_turn": round((total_prompt + total_completion) / n_turns, 1),
        },
        "rewrite_classification": {
            "skip_rate": round(skipped / n_last, 4),
            "identity_rewrites": identity,
            "per_case": [
                {"id": r["id"], "category": r["category"],
                 "expected": "skip" if r["category"] == "independent" else "llm",
                 "actual": r["final_rewrite_mode"],
                 "identity": r["turns"][-1]["rewrite_identity"],
                 "calls_final": r["turns"][-1]["llm_calls"],
                 "effective_query": r["final_effective_query"]}
                for r in per_case
            ],
        },
        "final_turn": {
            "verifiable": ok_ver, "verifiable_rate": round(ok_ver / n_last, 4),
            "anchor_hit": ok_anchor, "anchor_hit_rate": round(ok_anchor / n_last, 4),
            "latency_ms": {"p50": lats[len(lats) // 2], "max": lats[-1]},
        },
        "per_case": per_case,
    }
    out = settings.reports_dir / "eval_multiturn_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== 汇总（{n_last} 条 / {n_turns} 轮真实 LLM）=====")
    print(f"调用 {total_calls} 次 | prompt {total_prompt} tok / completion {total_completion} tok"
          f" | 成本 ¥{total_cost:.4f} | 单轮均 {report['cost']['avg_tokens_per_turn']} tok")
    print(f"末轮：引用可回查 {ok_ver}/{n_last} = {ok_ver / n_last:.3f} | "
          f"锚句命中 {ok_anchor}/{n_last} = {ok_anchor / n_last:.3f}")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
