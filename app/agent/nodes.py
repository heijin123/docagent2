"""LangGraph 节点实现（F3.1–F3.9 + F4 记忆接线）。

节点清单与契约 §4.1 一致：
ingest → (chitchat/contact→direct_reply | 其余→rewrite → retrieve ─┬→ no_data
                                                                  └→ answer[合并意图] → verify)* → finalize
分支（意图路由 / 空检索短路 / 置信度 / 重试上限）全部在 graph.py 条件边里
（F3.8：确定性逻辑不走 LLM）。

职责边界：本模块**不产生任何升级 / 转交动作**——没有转人工节点，不建工单。
所有"该找谁"只以文本形式告知，由客户自行联系（no_data / disclose / contact 话术）。
"""
from __future__ import annotations

import logging

from app.agent import prompts, schemas
from app.agent.llm import LLMClient
from app.agent.state import AgentState
from app.core.config import settings
from app.core.observability import log_kb_gap
from app.models import now_ts

logger = logging.getLogger(__name__)

CONFIRM_TEMPLATE = (
    "知识库中该主题的现行资料不足；找到相关文档《{title}》"
    "已于 {date} 过期，是否仍要查看？（内容可能不适用当前情况）"
)

_GREETING = "你好！我是企业知识助手，可查询制度、流程、政策等资料。请直接描述你的问题。"
# 客户要求"转人工"时只回指引，不承诺转接（系统没有人工座席，也不该由本系统转交）。
_CONTACT_MSG = (
    "本助手只提供知识库资料查询，不具备转接人工的职能。\n"
    "如需人工协助：业务 / 制度类问题请联系你的直属主管或对应主管部门；"
    "需要新增资料入库，请联系知识库管理员。"
)
# 检索为空（现行 + 过期都没有）→ 如实告知缺失，并把"补资料"的动作交回客户自己。
_NO_DATA_MSG = (
    "知识库中未检索到与该问题相关的资料，无法据此回答。\n"
    "如确有需要，请联系知识库管理员将相关资料录入后再来查询。"
)
# verify 用尽仍不达标 → 不转人工，改为确定性披露（保留答案 + 提示局限）
_DISCLOSE_SUFFIX = (
    "\n\n注：以上内容基于现有资料整理，可能不完整或存在偏差，仅供参考；"
    "如需准确信息，请核对原文或向对应主管部门确认。"
)
# 更差的情况：连一条引用都没有（疑似无据生成）→ 显著提示不可采信
_NO_CITE_SUFFIX = (
    "\n\n注：以上内容未能在现有知识库中找到对应出处，请勿直接作为依据；"
    "如需准确信息，请核对原文或向对应主管部门确认。"
)


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
        # 防止上一轮 degraded/answer 等状态串扰到本轮（如 degraded 泄漏）。
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

    # ── F3.1 意图分类（已合并进 answer 节点，实现见 answer）────
    # 说明：意图分类不再单独走一次 LLM，而是由 answer 节点在生成回答的同一次调用里
    # 一并产出 intent（prompt 要求首行 <intent> 或 JSON 含 intent）；明确寒暄 /
    # 要求转人工仍由 rule_classify_intent 规则短路（零 LLM）。下方 answer 即合并实现。

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
        # 规则短路（零 LLM）：寒暄 / 要求转人工都不检索；转人工只产出"该找谁"的指引话术。
        rule_intent = prompts.rule_classify_intent(q)
        if rule_intent == "chitchat":
            return {"intent": "chitchat", "answer": _GREETING, "citations": [],
                    "degraded": False,
                    "notes": [*state.get("notes", []), "answer: chitchat 规则短路"]}
        if rule_intent == "contact_guidance":
            return {"intent": "contact_guidance", "answer": _CONTACT_MSG, "citations": [],
                    "degraded": False,
                    "notes": [*state.get("notes", []),
                              "answer: 要求转人工 → 仅给联系指引（不转交、不建单）"]}

        evidence = prompts.render_evidence(state.get("retrieved", []))
        # 重试 hint 只要求"如实说明、不得编造"，**不再**暗示"补充引用以通过校验"
        # ——旧文案会诱导模型堆砌/伪造引用去骗过 verify。
        retry_hint = ("请如实说明资料的不足之处与信息来源，不要编造或堆砌引用"
                      if state.get("retry_count", 0) > 0 else "")
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
            # 限长：completion 长度是延迟/成本的**第一驱动**（实测单次最长 3184 token / 78.6s）。
            # 注意 max_tokens 需容得下完整 JSON 信封（answer + chunk_ids + intent），
            # 设太小会截断 JSON → 解析失败白重试一次（见 llm._warn_if_truncated 告警）。
            out = self.llm.complete_json(system, user, schemas.AnswerOutput,
                                         temperature=0.2,
                                         max_tokens=settings.answer_max_tokens)
        intent = out.intent or "kb_qa"
        if intent not in ("kb_qa", "chitchat"):
            # 模型若仍吐出旧枚举（如 human_handoff）→ 收敛为 kb_qa，绝不据此转人工
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
        items = state.get("retrieved", [])
        # 证据按引用收窄（性能）：verify 的职责是核「答案的核心论断是否被**它引用的证据**支撑」
        # 与「引用标记是否对应真实证据行」——未被引用的候选对判定毫无贡献，
        # 却占 verify prompt 的 84%（实测 1,845 / 2,187 tok）。故只传被引用的证据行。
        # 兜底：零引用、或引用 id 不在命中集 → 回退全量证据，保持既有判定行为不变
        #（「零引用是否应视为未达标」是独立议题，不在本次改动范围，见待办①）。
        cited_ids = {c.get("chunk_id") for c in state.get("citations", []) if c.get("chunk_id")}
        selected = [it for it in items if it.get("chunk_id") in cited_ids]
        if selected:
            use_items = selected
            scope_note = f"verify: 证据按引用收窄 {len(selected)}/{len(items)} 条"
        else:
            use_items = items
            scope_note = f"verify: 无有效引用 → 回退全量证据 {len(items)} 条"
        evidence = prompts.render_evidence(use_items)
        system, user = prompts.verify_prompt(q, state.get("answer", ""), evidence)
        # 性能：verify 只需 grounded + confidence 两个标量，收紧 max_tokens 抑制长 reason 拖慢
        out: schemas.VerifyJudgement = self.llm.complete_json(
            system, user, schemas.VerifyJudgement, temperature=0.0, max_tokens=200)
        confidence = out.confidence
        notes = [*state.get("notes", []), scope_note,
                 f"verify: grounded={out.grounded} conf={confidence:.2f}"]

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

    def no_data(self, state: AgentState, config=None) -> dict:
        """检索为空（现行 + 过期都无命中）→ 如实告知缺失，并留一条离线线索日志。

        0 次 LLM：旧设计这里要白烧 answer + verify 两轮，最后还转人工；现在直接披露，
        并把"补资料"的动作交回客户自己（联系管理员录入，系统不代办）。

        `event=kb_gap` 只是**离线线索**（供管理员把"查不到的问题"聚类成缺失主题，
        决定补哪几篇文档），**不进任何人工作队列、不触发工单/转交** —— 与本系统
        "只做检索与披露"的职责边界一致。thread_id/request_id 由 LangGraph 注入的
        config 携带（见 graph.thread_config），取不到时留空不影响主流程。

        ⚠ config 参数**刻意不写类型注解**：LangGraph 依据该参数的类型注解判断是否注入，
        而本模块有 `from __future__ import annotations`（注解被字符串化），一旦写成
        `RunnableConfig | None` 反而会被判为"非 RunnableConfig"而跳过注入（并告警）。
        "config" 这个参数名本身就是注入契约。
        """
        conf = (config or {}).get("configurable", {}) or {}
        log_kb_gap(
            str(conf.get("request_id", "") or ""),
            state.get("query", ""),
            rewritten_query=state.get("rewritten_query", ""),
            thread_id=str(conf.get("thread_id", "") or ""),
            expired_candidates=len(state.get("expired_candidates", []) or []),
        )
        if self.token_sink:
            self.token_sink(_NO_DATA_MSG)  # 非 LLM 路径也放 token，保持前端拼接一致
        return {"answer": _NO_DATA_MSG, "citations": [], "degraded": True,
                "intent": "kb_qa",
                "notes": [*state.get("notes", []),
                          "资料不足：检索无命中 → 如实告知缺失（不转人工）"]}

    def disclose(self, state: AgentState) -> dict:
        """verify 用尽仍不达标 → 保留答案 + 确定性披露后缀（不转人工）。

        自保规则：有 citation 才"忠实披露"；一条引用都没有（疑似无据生成）→ 换成
        更重的不可采信提示。后缀经 token_sink 实时补发，保证 SSE 下
        "token 拼接 == done.answer" 仍成立。
        """
        suffix = _DISCLOSE_SUFFIX if state.get("citations") else _NO_CITE_SUFFIX
        ans = state.get("answer", "") or ""
        if self.token_sink:
            self.token_sink(suffix)
        return {"answer": ans + suffix, "degraded": True,
                "notes": [*state.get("notes", []),
                          "校验未达标且重试用尽 → 披露局限（不转人工）"]}

    def direct_reply(self, state: AgentState) -> dict:
        """规则短路的直答：问候 或 转人工请求的"该找谁"指引（都不检索、零 LLM）。"""
        ri = prompts.rule_classify_intent(state.get("query", ""))
        if ri == "contact_guidance":
            return {"answer": _CONTACT_MSG, "citations": [], "degraded": False,
                    "intent": "contact_guidance",
                    "notes": [*state.get("notes", []),
                              "direct_reply: 联系指引（系统不转交、不建单）"]}
        return {"answer": _GREETING, "citations": [], "degraded": False, "intent": "chitchat",
                "notes": [*state.get("notes", []), "direct_reply: 问候（规则短路）"]}

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
