"""LangGraph 图组装（F3 编排 + F4.1 checkpointer）+ 对话运行入口。

职责边界（重要）：本图**只做检索与披露，不发起任何升级 / 转交动作**——没有 handoff 节点，
不建工单、不转接人工、不指定责任人。"找谁办"一律以文本形式告知，由客户自行联系。

图：START → ingest ─┬─ chitchat(rule) → direct_reply ─┐
                    ├─ contact(rule)  → direct_reply ─┤  （要求转人工 → 只给"该找谁"指引话术）
                    └─ 其余 → rewrite → retrieve ─┬─ 检索为空 → no_data ────────┐
                                                  └─ 有结果 → answer(合并意图) ─┬─ chitchat/contact → finalize
                                                                                ├─ confirm(仅过期) → finalize
                                                                                └─ verify ─┬─ ok → finalize
                                                                                           ├─ retry → rewrite（回环）
                                                                                           └─ 超限 → disclose → finalize
分支全部走条件边（F3.8）：入口意图路由、空检索短路、过期确认分支、置信度/重试上限判定都是
确定性代码。意图分类已合并进 answer 节点（同一次 LLM 调用产出 intent + answer），明确寒暄 /
要求转人工仍由 rule_classify_intent 规则短路（零 LLM）。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent.checkpointer import build_checkpointer
from app.agent.llm import LLMClient, build_llm
from app.agent.nodes import QANodes
from app.agent import prompts
from app.agent.state import AgentState
from app.core.config import settings
from app.core.observability import TokenUsage
from app.models.assistant import AssistantReply, Citation

logger = logging.getLogger(__name__)


def build_qa_graph(
    retriever,
    llm: LLMClient | None = None,
    *,
    user_permission: str = "internal",
    confidence_threshold: float | None = None,
    max_retry: int | None = None,
    history_rounds: int | None = None,
    checkpointer=None,
    token_sink=None,
):
    """组装并编译 QA 图。checkpointer 缺省用 build_checkpointer()（Redis→内存降级）。

    token_sink：M4 SSE 流式注入（非空时 answer 节点文本流式回调）；缺省 None = 非流式。
    返回 (compiled_graph, checkpointer_note)。
    """
    llm = llm or build_llm()
    threshold = confidence_threshold if confidence_threshold is not None \
        else settings.qa_confidence_threshold
    retry_max = max_retry if max_retry is not None else settings.qa_max_retry
    rounds = history_rounds if history_rounds is not None else settings.qa_history_rounds

    if checkpointer is None:
        checkpointer, _degraded, note = build_checkpointer()
    else:
        _degraded, note = False, ""

    nodes = QANodes(retriever, llm, user_permission=user_permission,
                    history_rounds=rounds, token_sink=token_sink)

    g = StateGraph(AgentState)
    g.add_node("ingest", nodes.ingest)
    g.add_node("rewrite", nodes.rewrite)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("no_data", nodes.no_data)
    g.add_node("answer", nodes.answer)
    g.add_node("verify", nodes.verify)
    g.add_node("retry", nodes.retry)
    g.add_node("disclose", nodes.disclose)
    g.add_node("direct_reply", nodes.direct_reply)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "ingest")

    def _entry_route(state: AgentState) -> Literal["direct_reply", "rewrite"]:
        # 确定性短路：明确寒暄 / 要求转人工直接路由（零 LLM，且都不检索）。
        # 注意 contact_guidance 只让 direct_reply 输出"该找谁"的指引话术，
        # 系统本身不转交、不建单；其余（含歧义）进 rewrite→retrieve→answer。
        ri = prompts.rule_classify_intent(state.get("query", ""))
        if ri in ("chitchat", "contact_guidance"):
            return "direct_reply"
        return "rewrite"

    g.add_conditional_edges(
        "ingest", _entry_route,
        {"direct_reply": "direct_reply", "rewrite": "rewrite"},
    )

    g.add_edge("rewrite", "retrieve")

    def _route_after_retrieve(state: AgentState) -> Literal["no_data", "answer"]:
        # 确定性短路（成本关键）：现行 + 过期都没命中 → 直接走"无资料"披露，**0 次 LLM**。
        # 旧设计此时会白烧 answer + verify 两轮再转人工；现在既省钱，也不把知识库的
        # 覆盖缺口甩成人工负担。
        # 两种情况都短路：① 检索真无命中（索引真空）；② F2.10 双证据判定「无相关内容」
        # （有候选但既无词表覆盖也无语义相近）——后者是本路由的主要来源，因为向量检索
        # 恒返回 topN，① 几乎不会发生。
        if state.get("confirmation_needed"):
            return "answer"          # F2.8 仅命中过期 → 先走确认话术
        if state.get("no_relevant") or not state.get("retrieved"):
            return "no_data"
        return "answer"

    g.add_conditional_edges("retrieve", _route_after_retrieve,
                            {"no_data": "no_data", "answer": "answer"})

    def _route_after_answer(state: AgentState) -> Literal["direct", "confirm", "verify"]:
        # answer 节点产出 intent；据其路由（F2.8 仅过期确认 → 不 verify 直接 finalize）
        intent = state.get("intent", "kb_qa")
        if intent in ("chitchat", "contact_guidance"):
            return "direct"
        return "confirm" if state.get("confirmation_needed") else "verify"

    g.add_conditional_edges(
        "answer", _route_after_answer,
        {"direct": "finalize", "confirm": "finalize", "verify": "verify"},
    )

    def _route_after_verify(state: AgentState) -> Literal["ok", "retry", "disclose"]:
        # F3.6/F3.8：确定性判定——置信度/grounded 达标即完成；未达标且未超上限→重试；
        # 超限→disclose（保留答案 + 确定性披露后缀），**不再**转人工。
        grounded = bool(state.get("grounded"))
        confidence = float(state.get("confidence", 0.0))
        if grounded and confidence >= threshold:
            return "ok"
        if state.get("retry_count", 0) < retry_max:
            return "retry"
        return "disclose"

    g.add_conditional_edges("verify", _route_after_verify,
                            {"ok": "finalize", "retry": "retry", "disclose": "disclose"})

    g.add_edge("retry", "rewrite")           # 回环：rewrite（带重试 hint）→ retrieve → answer → verify
    g.add_edge("no_data", "finalize")
    g.add_edge("disclose", "finalize")
    g.add_edge("direct_reply", "finalize")
    g.add_edge("finalize", END)

    graph = g.compile(checkpointer=checkpointer)
    return graph, note


def thread_config(thread_id: str, request_id: str = "") -> dict:
    """checkpointer key 维度（F4.1）：thread_id 由调用方按 {tenant}:{user} 传入。

    request_id 一并放进 configurable，供节点落 kb_gap 等离线线索日志时携带
    （可空，不参与 checkpointer 的 thread 定位）。
    """
    return {"configurable": {"thread_id": thread_id, "request_id": request_id}}


class AgentApp:
    """对话运行入口：封装图 + checkpointer 工厂，产出契约 AssistantReply。

    - reply()：非流式（M3 图直接 invoke）；
    - history()：对话历史（F4/契约 §2.5）；
    - stream_events()：M4 SSE 事件生成器（ready/token/citation/done/ping/error）。
    """

    def __init__(self, retriever, llm: LLMClient | None = None, *,
                 user_permission: str = "internal", memory_checkpoint: bool = False):
        self._retriever = retriever
        self._user_permission = user_permission
        self.llm = llm or build_llm()
        ckpt, degraded, note = build_checkpointer(memory=memory_checkpoint)
        self._checkpointer = ckpt  # 共享给每请求重建的流式图（保证同 thread 历史跨流可见）
        self.graph, self.checkpoint_note = build_qa_graph(
            retriever, self.llm, user_permission=user_permission, checkpointer=ckpt)
        if degraded and note:
            logger.warning("checkpointer 降级: %s", note)
        self._ckpt_degraded = degraded
        self.last_usage = TokenUsage()  # 最近一轮 reply 的链路 token 用量
        self.last_calls = 0             # 最近一轮 reply 的真实 LLM 调用次数（-1 = 重试轮数）
        self.last_retries = 0           # 最近一轮 reply 的 verify 重试次数（0 = 一次过）
        self.last_verified = False      # 最近一轮 reply 是否**真正执行了 verify 节点**

    def reply(self, query: str, thread_id: str, *,
              include_expired: bool = False) -> AssistantReply:
        t0 = time.time()
        before = self.llm.meter.snapshot()
        calls_before = self.llm.meter.calls
        state = self.graph.invoke(
            {"query": query, "include_expired": include_expired},
            thread_config(thread_id),
        )
        self.last_usage = self.llm.meter.snapshot() - before
        self.last_calls = self.llm.meter.calls - calls_before
        # verify 重试次数直接读终态：retry_count 由 retry 节点自增（ingest 每轮重置为 0），
        # 因此它精确等于"这一问被 verify 打回几次"，是重试成本的一手证据（不靠日志文本推断）。
        self.last_retries = int(state.get("retry_count", 0))
        # 是否经过 verify：**只认终态留下的痕迹**，不靠"LLM 调用次数"反推。
        # 反例（曾真实发生）：rewrite 规则短路后正常题只有 2 跳（answer+verify），
        # 而按 `llm_calls >= 3` 猜的旧判据会把它们全判成"未经过 verify"。
        self.last_verified = any(str(x).startswith("verify") for x in state.get("notes", []))
        return AssistantReply(
            answer=state.get("answer", ""),
            citations=[Citation(**c) for c in state.get("citations", [])],
            confidence=float(state.get("confidence", 0.0)),
            degraded=bool(state.get("degraded")),
            intent=str(state.get("intent", "kb_qa")),
            # 契约 §2.1 声明 AssistantReply 含 latency_ms；非流式路径此前恒为 0（仅 SSE 路径填），
            # 这里补齐——评估/调用方按 latency_ms 统计单问耗时才对得上流式通道的口径。
            latency_ms=int((time.time() - t0) * 1000),
            notes=list(state.get("notes", [])),
            usage=self.last_usage.to_dict(),
        )

    def stream_events(self, query: str, thread_id: str, *,
                      include_expired: bool = False,
                      request_id: str = "") -> Iterator[dict]:
        """M4 SSE 事件生成器（契约 §2.1 事件序列）。

        yield {"type": "ready"|"token"|"citation"|"done"|"ping"|"error", "data": {...}}
        - 图在 worker 线程执行（token_sink 经 queue 跨线程回传），本生成器即产即转发；
        - 空闲超 sse_ping_interval_s 秒 → yield ping（保活）；
        - citation 事件逐条（data=单条 Citation）；done.data = AssistantReply dict。
        """
        import queue
        import threading

        sink_q: queue.Queue = queue.Queue()  # ("token", str)
        graph, _note = build_qa_graph(
            self._retriever, self.llm, user_permission=self._user_permission,
            checkpointer=self._checkpointer,
            token_sink=lambda text: sink_q.put(("token", text)),
        )

        def _run() -> None:
            try:
                before = self.llm.meter.snapshot()
                calls_before = self.llm.meter.calls
                state = graph.invoke(
                    {"query": query, "include_expired": include_expired},
                    thread_config(thread_id, request_id),
                )
                self.last_usage = self.llm.meter.snapshot() - before
                self.last_calls = self.llm.meter.calls - calls_before
                self.last_retries = int(state.get("retry_count", 0))
                self.last_verified = any(str(x).startswith("verify")
                                         for x in state.get("notes", []))
                sink_q.put(("state", state))
            except Exception as exc:  # noqa: BLE001 — SSE 通道内错误走 error 事件
                logger.exception("stream 图执行异常 thread=%s", thread_id)
                sink_q.put(("error", {
                    "code": "INTERNAL_ERROR", "message": str(exc),
                    "request_id": request_id}))

        t0 = time.time()
        yield {"type": "ready", "data": {"request_id": request_id}}
        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        sent_tokens = 0
        try:
            while True:
                try:
                    kind, payload = sink_q.get(timeout=settings.sse_ping_interval_s)
                except queue.Empty:
                    yield {"type": "ping", "data": {}}
                    continue
                if kind == "token":
                    sent_tokens += 1
                    yield {"type": "token", "data": {"content": payload}}
                elif kind == "state":
                    state = payload
                    citations = [Citation(**c) for c in state.get("citations", [])]
                    reply = AssistantReply(
                        answer=state.get("answer", ""),
                        citations=citations,
                        confidence=float(state.get("confidence", 0.0)),
                        degraded=bool(state.get("degraded")),
                        intent=str(state.get("intent", "kb_qa")),
                        request_id=request_id,
                        latency_ms=int((time.time() - t0) * 1000),
                        notes=list(state.get("notes", [])),
                        usage=self.last_usage.to_dict(),
                    )
                    # 非 LLM 直答路径（chitchat / contact_guidance / no_data / 确认话术）
                    # 无 token 事件 → done 前补发整段，保证"token 拼接 == done.answer"
                    # （前端零特判）。disclose 的披露后缀在节点内经 token_sink 实时补发。
                    if not sent_tokens and reply.answer:
                        yield {"type": "token", "data": {"content": reply.answer}}
                    for c in citations:
                        yield {"type": "citation", "data": c.model_dump()}
                    yield {"type": "done", "data": reply.model_dump()}
                    break
                elif kind == "error":
                    yield {"type": "error", "data": payload}
                    break
        finally:
            worker.join(timeout=10)  # done/error 后回收工作线程

    def history(self, thread_id: str, limit: int = 50) -> list[dict]:
        """最近对话历史（契约 §2.5：limit = 消息条数，默认 50 上限调用方校验）。"""
        try:
            snap = self.graph.get_state(thread_config(thread_id))
            msgs = (snap.values or {}).get("messages", [])
        except Exception:  # noqa: BLE001 内存 saver 无该 thread 时会抛 KeyError
            return []
        return list(msgs)[-limit:]
