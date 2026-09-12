"""LLM 客户端抽象（M3）：DashScope(OpenAI 兼容) / stub 降级双通道。

- DashScopeLLM：chat completions + JSON 输出，temp=0（supervisor/verify 确定性）；
- StubLLM：无 Key 时降级，规则式产出去驱动图结构与测试（链路回归，无语义），degraded 标注；
- build_llm()：有 Key → dashscope；无 Key → stub（degraded=True 不掩盖，与 embedding 同哲学）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel

from app.agent import schemas
from app.core.config import settings
from app.core.observability import (
    TimedSpan, TokenMeter, TokenUsage, log_llm_call, log_llm_usage)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TOKEN_STEP = 12  # stub 分段模拟流式的最小块长

# 合并 supervisor：流式首行 <intent>意图</intent> 标签（意图：kb_qa/chitchat/human_handoff）
_INTENT_TAG_RE = re.compile(r"<intent>\s*([a-z_]+)\s*</intent>", re.IGNORECASE)
_MAX_INTENT_PREFIX = 256  # 标签未闭合前的缓冲上限；超界退化为 kb_qa 原样流式


class LLMClient:
    provider: str = ""
    model: str = ""
    degraded: bool = False

    def __init__(self) -> None:
        # 贯穿整条链路的 token 计量：每次调用累加，单轮 reply 前/后差值=该问成本
        self.meter = TokenMeter()
        self.last_usage = TokenUsage()

    def info(self) -> dict:
        return {"provider": self.provider, "model": self.model, "degraded": self.degraded}

    def complete_json(self, system: str, user: str, schema: type[T],
                      *, temperature: float = 0.0, max_tokens: int = 1500) -> T:
        raise NotImplementedError

    def stream_answer(self, system: str, user: str,
                      on_token) -> schemas.AnswerOutput:
        """M4 流式 answer：on_token(str) 逐段回调产出文本，返回最终结构化结果。

        引用反解约定：文本流结束后从 [来源: ...] 标记映射回证据 chunk_id
        （见 _parse_chunk_ids_from_answer）；映射为空时 chunk_ids=[]，由
        verify/条件边自然兜底（grounded=False → 重试或转人工），不在此处编造。
        """
        raise NotImplementedError


class DashScopeLLM(LLMClient):
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        super().__init__()
        self.provider = "dashscope"
        self.model = model or settings.qwen_llm_model
        self._client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=settings.read_timeout_s,
            max_retries=settings.llm_max_retries,  # 默认 0：关掉 SDK 静默重发，防烧 token
        )

    def complete_json(self, system: str, user: str, schema: type[T],
                      *, temperature: float = 0.0, max_tokens: int = 1500) -> T:
        last_exc: Exception | None = None
        prompt_chars = len(system) + len(user)
        resp = None
        for attempt in range(2):  # 解析失败重试一次（模型偶发返回非合法 JSON）
            span = TimedSpan(name="llm_call").attr(model=self.model)
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            finally:
                # 即使抛错也记录耗时（便于定位超时/慢往返）
                duration_ms = span.stop(log_slow=False)
            # 抓真实 token 用量（贯穿链路成本核算核心；此前从未读取 → 控制台才看得到消耗）
            usage = _usage_from_resp(resp)
            if usage is not None:
                self.meter.add(usage)
                self.last_usage = usage
            log_llm_call("", duration_ms, node="complete_json", model=self.model,
                         prompt_chars=prompt_chars,
                         prompt_tokens=usage.prompt_tokens if usage else 0,
                         completion_tokens=usage.completion_tokens if usage else 0,
                         total_tokens=usage.total_tokens if usage else 0)
            if usage is not None:
                log_llm_usage("", node="complete_json", model=self.model, usage=usage,
                              est_cost_cny=usage.cost(*settings.llm_price(self.model)))
            try:
                return _parse_json_strict(schema, resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("complete_json 解析失败(第%d次)，重试一次: %s", attempt + 1, exc)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("complete_json 重试后仍无结果")

    def stream_answer(self, system: str, user: str,
                      on_token) -> schemas.AnswerOutput:
        """DashScope 文本流式：stream=True 逐块回调（纯文本，非 JSON 模式）。

        合并 supervisor：首行 <intent>...</intent> 标签在流式过程中被解析为意图
        （用于路由），标签之后正文照常逐块推送 → 既合并意图分类又保留 SSE 流式。
        模型未遵循格式时退化为 kb_qa（标签兜底清除，正文不丢）。
        """
        span = TimedSpan(name="llm_call").attr(model=self.model)
        prompt_chars = len(system) + len(user)
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=1500,
            stream=True,
            stream_options={"include_usage": True},  # 让尾块带回真实 usage
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        chunks: list[str] = []
        usage: TokenUsage | None = None
        intent = ""
        intent_resolved = False
        prefix = ""  # 标签闭合前的缓冲（未闭合时不推送，避免把标签推给前端）
        for event in resp:
            if getattr(event, "usage", None) is not None:  # 尾块：真实 token 用量
                usage = TokenUsage.from_openai(event.usage)
                continue
            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = (delta or {}).content
            if not piece:
                continue
            if not intent_resolved:
                prefix += piece
                m = _INTENT_TAG_RE.search(prefix)
                if m:
                    intent = m.group(1).strip()
                    after = prefix[m.end():]
                    intent_resolved = True
                    if after:
                        chunks.append(after)
                        on_token(after)
                elif len(prefix) > _MAX_INTENT_PREFIX:
                    # 模型未遵循首行 <intent> 格式 → 放弃解析，原样流式（默认 kb_qa）
                    intent_resolved = True
                    chunks.append(prefix)
                    on_token(prefix)
                # else: 仍在缓冲标签，暂不推送
            else:
                chunks.append(piece)
                on_token(piece)
        if usage is not None:  # 流式也纳入整条链路计量
            self.meter.add(usage)
            self.last_usage = usage
        duration_ms = span.stop(log_slow=False)
        log_llm_call("", duration_ms, node="stream_answer", model=self.model,
                     prompt_chars=prompt_chars,
                     prompt_tokens=usage.prompt_tokens if usage else 0,
                     completion_tokens=usage.completion_tokens if usage else 0,
                     total_tokens=usage.total_tokens if usage else 0)
        if usage is not None:
            log_llm_usage("", node="stream_answer", model=self.model, usage=usage,
                          est_cost_cny=usage.cost(*settings.llm_price(self.model)))
        text = "".join(chunks).strip()
        text = _INTENT_TAG_RE.sub("", text).strip()  # 兜底清除残留标签
        if not text:
            return schemas.AnswerOutput(answer="", chunk_ids=[], intent=intent or "kb_qa")
        evidence = _extract_tag(user, "evidence")
        chunk_ids = _parse_chunk_ids_from_answer(text, evidence)
        return schemas.AnswerOutput(answer=text, chunk_ids=chunk_ids, intent=intent or "kb_qa")


# ── Stub 降级（无 Key 保链路；规则式，无语义）──────────────────
class StubLLM(LLMClient):
    """规则式桩：按 schema 路由，产出可驱动图跑通的结构化结果。

    子类可覆写 *_rule 注入特定行为（verify_m3 用它构造低置信场景）。
    """

    def __init__(self, model: str = "stub-rule-v1"):
        super().__init__()
        self.provider = "stub"
        self.model = model
        self.degraded = True
        self.calls: list[dict] = []   # 调用记录（测试断言用）

    def complete_json(self, system: str, user: str, schema: type[T],
                      *, temperature: float = 0.0, max_tokens: int = 1500) -> T:
        self.calls.append({"schema": schema.__name__, "system": system, "user": user})
        # 无真实 usage → 估算计入整条链路计量（约 2 字符/token），保证成本可见
        usage = TokenUsage.estimate(len(system) + len(user), 40)
        self.meter.add(usage)
        self.last_usage = usage
        log_llm_usage("", node="stub_complete_json", model=self.model, usage=usage)
        name = schema.__name__
        if name == "RewriteOutput":
            return self.rewrite_rule(user)  # type: ignore[return-value]
        if name == "AnswerOutput":
            return self.answer_rule(user)  # type: ignore[return-value]
        if name == "VerifyJudgement":
            return self.verify_rule(user)  # type: ignore[return-value]
        raise NotImplementedError(f"Stub 不支持 schema: {name}")

    def stream_answer(self, system: str, user: str,
                      on_token) -> schemas.AnswerOutput:
        """stub 流式：复用 answer_rule 产出，按 _TOKEN_STEP 分段回调模拟逐字。"""
        out = self.answer_rule(user)
        usage = TokenUsage.estimate(len(system) + len(user), len(out.answer))
        self.meter.add(usage)
        self.last_usage = usage
        log_llm_usage("", node="stub_stream_answer", model=self.model, usage=usage)
        text = out.answer
        for i in range(0, len(text), _TOKEN_STEP):
            on_token(text[i:i + _TOKEN_STEP])
        return out

    # 各规则返回基础 pydantic 实例
    def rewrite_rule(self, user: str) -> schemas.RewriteOutput:
        q = _extract_tag(user, "query")
        return schemas.RewriteOutput(rewritten_query=q, changed=False)

    def answer_rule(self, user: str) -> schemas.AnswerOutput:
        # 合并 supervisor：自判意图（兜底规则；真实模型由 prompt 产出 intent）
        q = _extract_tag(user, "query")
        if any(k in q for k in ("转人工", "人工客服", "找客服", "human", "投诉")):
            intent = "human_handoff"
        elif len(q.strip()) <= 12 and any(
                k in q.lower() for k in ("你好", "hi", "hello", "在吗", "谢谢", "再见", "拜拜", "早上好", "晚上好", "嗨")):
            intent = "chitchat"
        else:
            intent = "kb_qa"
        # 证据行格式: [c{idx}] {chunk_id} | {title} | 第 N 页 | {doc_date} | {content}
        evidence = _extract_tag(user, "evidence")
        picks: list[str] = []
        for line in evidence.splitlines():
            line = line.strip()
            if line.startswith("[c") and "]" in line:
                rest = line.split("]", 1)[1].strip()
                cid = rest.split("|", 1)[0].strip()
                if cid and cid not in picks:
                    picks.append(cid)
            if len(picks) >= 2:
                break
        if not picks:
            return schemas.AnswerOutput(
                answer="知识库暂无相关现行资料，建议转人工核实。", chunk_ids=[], intent=intent)
        cites = _citations_text(evidence, picks)
        body = "根据检索到的资料：" + cites + "（如需进一步细节请说明）。"
        return schemas.AnswerOutput(answer=body, chunk_ids=picks, intent=intent)

    def verify_rule(self, user: str) -> schemas.VerifyJudgement:
        answer = _extract_tag(user, "answer")
        has_cite = "[来源:" in answer
        conf = 0.9 if has_cite and len(answer) > 20 else 0.3
        return schemas.VerifyJudgement(grounded=has_cite, confidence=conf,
                                       reason="stub: 依据引用标记与长度")


def _extract_tag(text: str, tag: str) -> str:
    """从 prompt 文本中提取 <{tag}>...</{tag}> 片段（无则原文）。"""
    start = f"<{tag}>"
    end = f"</{tag}>"
    if start in text and end in text:
        return text.split(start, 1)[1].split(end, 1)[0]
    return text


def _parse_chunk_ids_from_answer(answer: str, evidence: str) -> list[str]:
    """流式文本结束后，从 [来源: 标题 第 N 页] 标记反解证据 chunk_id（保序去重）。

    反解失败（模型未按格式引用）→ 返回 []，由 verify grounded=False 自然兜底，
    不在此处猜测或编造 chunk_id（防止引用不在证据内）。
    """
    # 证据行: [c{idx}] {chunk_id} | {title} | 第 N 页 | {doc_date} | ...
    title2cid: dict[str, str] = {}
    for line in evidence.splitlines():
        line = line.strip()
        if line.startswith("[c") and "]" in line and "|" in line:
            cols = [c.strip() for c in line.split("]", 1)[1].split("|")]
            if len(cols) >= 2:
                title2cid[cols[1]] = cols[0]
    if not title2cid:
        return []
    # 引用标记: [来源: X] 或 [来源: X 第 N 页]；长标题优先匹配避免前缀歧义
    titles = sorted(title2cid, key=len, reverse=True)
    out: list[str] = []
    for m in re.finditer(r"\[来源:\s*([^\]]+)\]", answer):
        ref = m.group(1).strip()
        for title in titles:
            if title and title in ref:
                cid = title2cid[title]
                if cid not in out:
                    out.append(cid)
                break
    return out


def _citations_text(evidence: str, picks: list[str]) -> str:
    """拼 stub 回答中的引用文本（含 [来源: 标题 页码] 标记，供 verify 判 grounded）。

    证据列序：{chunk_id} | {title} | 第 N 页 | {doc_date} | {content}
    """
    import re as _re

    meta: dict[str, tuple[str, str]] = {}
    for line in evidence.splitlines():
        line = line.strip()
        if line.startswith("[c") and "]" in line and "|" in line:
            rest = line.split("]", 1)[1].strip()
            cols = [c.strip() for c in rest.split("|")]
            cid = cols[0]
            title = cols[1] if len(cols) > 1 else ""
            page_m = _re.search(r"(\d+)", cols[2]) if len(cols) > 2 else None
            meta[cid] = (title, page_m.group(1) if page_m else "")
    out = []
    for cid in picks:
        title, page = meta.get(cid, ("未知文档", ""))
        out.append(f"[来源: {title} 第 {page} 页]" if page else f"[来源: {title}]")
    return "；".join(out)


def _parse_json_strict(schema: type[T], raw: str) -> T:
    """解析模型 JSON 输出：剥离代码围栏 + 抽取首个 {...} 片段 + pydantic 校验。

    修复模型偶发返回非合法 JSON（带 markdown 围栏 / 前后多余文本）导致的 500；
    解析失败由调用方决定是否重试。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    text = text.strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    parsed = json.loads(text)
    return schema.model_validate(parsed)


def _usage_from_resp(resp) -> "TokenUsage | None":
    """从 chat.completions 响应中安全提取 usage（create 抛错/字段缺失 → None）。"""
    if resp is None:
        return None
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    return TokenUsage.from_openai(u)


def build_llm() -> LLMClient:
    """工厂：有 Key → dashscope；无 Key → stub（degraded，保链路回归）。"""
    if settings.has_api_key:
        return DashScopeLLM()
    logger.warning("未配 DASHSCOPE_API_KEY → 对话 LLM 降级 Stub（仅链路回归，无语义）")
    return StubLLM()
