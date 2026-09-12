"""LangGraph 节点实现（F3.1–F3.9 + F4 记忆接线）。

节点清单与契约 §4.1 一致：ingest → (chitchat→direct_reply | human_handoff→handoff | 其余→rewrite → retrieve → answer[合并意图] → verify)* → finalize；
分支（意图路由 / 置信度 / 重试上限）全部在 graph.py 条件边里（F3.8：确定性逻辑不走 LLM）。
"""
from __future__ import annotations

import logging

from app.agent import prompts, schemas
from app.agent.llm import LLMClient
from app.agent.state import AgentState
from app.models import now_ts

logger = logging.getLogger(__name__)

CONFIRM_TEMPLATE = (
    "知识库中该主题的现行资料不足；找到相关文档《{title}》"
    "已于 {date} 过期，是否仍要查看？（内容可能不适用当前情况）"
)

_GREETING = "你好！我是企业知识助手，可查询制度、流程、政策等资料。请直接描述你的问题。"
_HANDOFF_MSG = ("抱歉，这个问题我无法基于现有知识库可靠回答。已为你转接人工客服，"
               "请描述你的工单信息以便跟进。")


def _fmt_ts(ts: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


class QANodes:
    def __init__(self, retriever, llm: LLMClient, *,
                 user_permission: str = "internal", history_rounds: int = 10,
                 token_sink=None):
        self.retriever = retriever
        self.llm = llm
        self.user_permission = user_permission
        self.history_rounds = history_rounds
        # M4 SSE 流式：非空时 answer 节点走文本流式（LLM stream_answer），
        # 增量块逐个回调（线程安全 sink，由 AgentApp.stream_events 注入）。
        self.token_sink = token_sink

    # ── 记忆：追加本轮 query + 窗口截断 + 本轮输出字段重置（F4.2/F4.3）──
    def ingest(self, state: AgentState) -> dict:
        messages = list(state.get("messages", []))
        messages.append({"role": "user", "content": state["query"], "created_at": now_ts()})
        keep = self.history_rounds * 2  # 10 轮 = 20 条
        # 输出/过程字段为"本轮"语义，checkpointer 会跨轮持久化 → 每轮入口统一重置，
        # 防止上一轮 handoff/degraded 等状态串扰到本轮（如 degraded 泄漏）。
        return {
            "messages": messages[-keep:],
            "intent": "",
            "rewritten_query": state["query"],
            "retrieved": [],
            "expired_candidates": [],
            "confirmation_needed": False,
            "citations": [],
            "answer": "",
            "grounded": False,
            "confidence": 0.0,
            "degraded": False,
            "retry_count": 0,
            "notes": [],
        }

    def _history(self, state: AgentState) -> str:
        """历史文本（不含刚追加的当前轮 query，避免上下文重复）。"""
        msgs = list(state.get("messages", []))
        return prompts.render_history(msgs[:-1], max_rounds=self.history_rounds)

    # ── F3.1 supervisor（已合并进 answer 节点，意图分类见 answer）────
    # 说明：意图分类不再单独走一次 LLM，而是由 answer 节点在生成回答的同一次调用里
    # 一并产出 intent（prompt 要求首行 <intent> 或 JSON 含 intent）；明确寒暄/转人工
    # 仍由 rule_classify_intent 规则短路（零 LLM）。下方 answer 方法即合并实现。

    # ── F3.2 rewrite（含 F2.8 过期确认放行）────────────────────
    def rewrite(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        last_assistant = next((m.get("content") for m in reversed(messages)
                               if m.get("role") == "assistant"), None)
        include_expired = bool(state.get("include_expired"))
        note = ""
        if not include_expired and prompts.should_confirm_expired(last_assistant, state["query"]):
            include_expired = True
            note = "用户已确认查看过期文档（include_expired=true）"

        query = state["query"]
        retry_count = state.get("retry_count", 0)
        retry_hint = state.get("notes", [])
        if retry_count > 0:
            note += f"（第 {retry_count} 次重试改写）"

        # F3.8 确定性短路：无历史 + 无指代 + 非重试 → 原样透传（省一次 LLM）
        history_text = self._history(state)
        if (not include_expired
                and prompts.should_skip_rewrite(query, history_text, retry_count)):
            new_notes = [*state.get("notes", []), "rewrite: 原样透传（规则短路）"]
            return {"rewritten_query": query, "include_expired": False,
                    "notes": new_notes}

        system, user = prompts.rewrite_prompt(query, history_text, note=note)
        out: schemas.RewriteOutput = self.llm.complete_json(
            system, user, schemas.RewriteOutput, temperature=0.1)
        rewritten = (out.rewritten_query or query).strip() or query
        new_notes = [*state.get("notes", []), f"rewrite: {rewritten}"]
        if include_expired:
            new_notes.append("include_expired=true（用户已确认）")
        return {"rewritten_query": rewritten, "include_expired": include_expired,
                "notes": new_notes}

    # ── F3.3 retrieve（F2 混合检索 + F2.8 确认分支标记）────────
    def retrieve(self, state: AgentState) -> dict:
        q = state.get("rewritten_query") or state["query"]
        include_expired = bool(state.get("include_expired"))
        if include_expired:
            res = self.retriever.relaxed_retrieve(q, user_permission=self.user_permission)
            retrieved = [*res.items, *res.expired_candidates]
            confirmation_needed = False
        else:
            res = self.retriever.retrieve(q, user_permission=self.user_permission)
            retrieved = res.items
            # 仅当"现行一无所有且存在过期候选"才进入确认话术（F2.8 提示流程）
            confirmation_needed = (not res.items) and bool(res.expired_candidates)
        new_notes = [*state.get("notes", []), *res.notes]
        if res.degraded:
            new_notes.append("检索降级: " + ", ".join(d["path"] for d in res.degraded))
        return {"retrieved": retrieved,
                "expired_candidates": res.expired_candidates,
                "confirmation_needed": confirmation_needed,
                "notes": new_notes}

    # ── F3.4 answer（合并 supervisor：意图分类 + 生成回答，单次 LLM）──
    def answer(self, state: AgentState) -> dict:
        if state.get("confirmation_needed"):
            cands = state.get("expired_candidates", [])
            first = cands[0]["metadata"] if cands else {}
            ans = CONFIRM_TEMPLATE.format(
                title=first.get("doc_title", "未知名文档"),
                date=_fmt_ts(cands[0]["expired_at"]) if cands and cands[0].get("expired_at")
                else "未知时间")
            if self.token_sink:
                self.token_sink(ans)  # 确认话术也作为 token 放送（前端拼接一致）
            return {"intent": "kb_qa", "answer": ans, "citations": [],
                    "notes": [*state.get("notes", []),
                             "仅命中过期文档 → 已向用户发起确认（F2.8，无 interrupt）"]}

        q = state.get("rewritten_query") or state["query"]
        # 合并 supervisor：明确寒暄/转人工走规则短路（零 LLM）；其余进 kb_qa 合并调用
        rule_intent = prompts.rule_classify_intent(q)
        if rule_intent == "chitchat":
            return {"intent": "chitchat", "answer": _GREETING, "citations": [],
                    "notes": [*state.get("notes", []), "answer: chitchat 规则短路（合并 supervisor）"]}
        if rule_intent == "human_handoff":
            return {"intent": "human_handoff", "answer": _HANDOFF_MSG, "citations": [],
                    "degraded": True,
                    "notes": [*state.get("notes", []), "answer: human_handoff 规则短路（合并 supervisor）"]}

        evidence = prompts.render_evidence(state.get("retrieved", []))
        retry_hint = "请补充引用或修正回答使校验通过" if state.get("retry_count", 0) > 0 else ""
        if self.token_sink:
            # M4 流式路径：首行 <intent> 标签被流式回调解析用于路由，正文照常逐段产出
            system, user = prompts.answer_stream_prompt(
                q, evidence, self._history(state),
                include_expired=bool(state.get("include_expired")), retry_hint=retry_hint)
            out: schemas.AnswerOutput = self.llm.stream_answer(
                system, user, self.token_sink)
        else:
            system, user = prompts.answer_prompt(
                q, evidence, self._history(state),
                include_expired=bool(state.get("include_expired")), retry_hint=retry_hint)
            out = self.llm.complete_json(system, user, schemas.AnswerOutput, temperature=0.2)
        intent = out.intent or "kb_qa"
        if intent not in ("kb_qa", "chitchat", "human_handoff"):
            intent = "kb_qa"
        citations = self._enrich_citations(out.chunk_ids, state.get("retrieved", []))
        return {"intent": intent, "answer": out.answer, "citations": citations,
                "notes": [*state.get("notes", []), f"intent={intent}", f"citations={len(citations)}"]}

    @staticmethod
    def _enrich_citations(chunk_ids: list[str], retrieved: list[dict]) -> list[dict]:
        """LLM 选的 chunk_id → 契约 Citation 形状（防止引用不在证据内的编造）。"""
        by_id = {it["chunk_id"]: it for it in retrieved}
        cites = []
        seen = set()
        for cid in chunk_ids:
            if cid in seen or cid not in by_id:
                continue
            seen.add(cid)
            it = by_id[cid]
            m = it.get("metadata", {})
            cites.append({
                "chunk_id": cid,
                "doc_title": m.get("doc_title", ""),
                "page_num": m.get("page_num", 0),
                "validity": it.get("validity", "valid"),
                "expired_at": it.get("expired_at"),
                "doc_date": m.get("doc_date") or None,
                "image_ids": [x for x in (m.get("image_ids") or "").split(",") if x],
            })
        return cites

    # ── F3.5 verify（独立评估输入；F3.9 置信度规则在此执行）────
    def verify(self, state: AgentState) -> dict:
        q = state.get("rewritten_query") or state["query"]
        evidence = prompts.render_evidence(state.get("retrieved", []))
        system, user = prompts.verify_prompt(q, state.get("answer", ""), evidence)
        # 性能：verify 只需 grounded + confidence 两个标量，收紧 max_tokens 抑制长 reason 拖慢
        out: schemas.VerifyJudgement = self.llm.complete_json(
            system, user, schemas.VerifyJudgement, temperature=0.0, max_tokens=200)
        confidence = out.confidence
        notes = [*state.get("notes", []), f"verify: grounded={out.grounded} conf={confidence:.2f}"]

        # F3.9：过期引用约束（需求明确，不依赖模型自觉）
        cites = state.get("citations", [])
        expired_cites = [c for c in cites if c.get("validity") == "expired"]
        degraded = bool(state.get("degraded"))
        if expired_cites and len(expired_cites) == len(cites) and cites:
            degraded = True
            notes.append("整篇答案仅由过期文档支撑 → degraded=true，请以现行制度为准")
        elif expired_cites:
            confidence = min(confidence, 0.5)  # 下调一档（F3.9）
            notes.append("答案含过期引用 → 置信度下调一档（≤0.5）")
        return {"grounded": bool(out.grounded), "confidence": confidence,
                "degraded": degraded, "notes": notes}

    # ── 分支节点 ─────────────────────────────────────────────
    def retry(self, state: AgentState) -> dict:
        """重试计数 +1（防死循环 F3.6；是否还可重试由条件边判定）。"""
        n = state.get("retry_count", 0) + 1
        return {"retry_count": n, "notes": [*state.get("notes", []), f"retry #{n}"]}

    def direct_reply(self, state: AgentState) -> dict:
        return {"answer": _GREETING, "citations": [], "degraded": False, "intent": "chitchat"}

    def handoff(self, state: AgentState) -> dict:
        return {"answer": _HANDOFF_MSG, "citations": [], "degraded": True,
                "intent": "human_handoff",
                "notes": [*state.get("notes", []), "degraded=true（handoff）"]}

    # ── 收尾：assistant 消息入历史（F4.2 供下一轮 QueryRewrite/Answer）──
    def finalize(self, state: AgentState) -> dict:
        messages = list(state.get("messages", []))
        messages.append({
            "role": "assistant",
            "content": state.get("answer", ""),
            "citations": state.get("citations", []),
            "degraded": bool(state.get("degraded")),
            "created_at": now_ts(),
        })
        keep = self.history_rounds * 2
        return {"messages": messages[-keep:]}
