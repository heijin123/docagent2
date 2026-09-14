"""LLM 客户端抽象（M3）：DashScope(OpenAI 兼容) / stub 降级双通道。

- DashScopeLLM：chat completions + JSON 输出，temp=0（verify 确定性）；
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

# 合并意图：流式首行 <intent>意图</intent> 标签（意图只允许 kb_qa/chitchat）
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
        （见 _parse_chunk_ids_from_answer）；映射为空时 chunk_ids=[]，交由
        verify/条件边自然兜底（grounded=False → 重试或披露），不在此处编造。

        流式不变式：on_token 推送的块的拼接**恒等于**返回的 answer，
        前端零特判即可还原答案（见 DashScopeLLM.stream_answer 的 _emit）。
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
                    extra_body=_thinking_extra_body(),
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
            # 限长副作用可见化：被 max_tokens 截断 → JSON 必坏 → 白重试一次（双倍成本）。
            # 不静默：留 WARN 便于按日志调 answer_max_tokens。
            _warn_if_truncated(resp, node="complete_json", max_tokens=max_tokens)
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
            # 与 answer 节点同口径的限长：流式正文同样不许无限长（原硬编码 1500 且模型常无视）
            max_tokens=settings.answer_max_tokens,
            stream=True,
            extra_body=_thinking_extra_body(),
            stream_options={"include_usage": True},  # 让尾块带回真实 usage
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        chunks: list[str] = []
        usage: TokenUsage | None = None
        intent = ""
        finish_reason = ""
        intent_resolved = False
        prefix = ""      # 标签闭合前的缓冲（未闭合时不推送，避免把标签推给前端）
        started = False  # 是否已推送过内容（用于剥掉正文前导空白，保持拼接一致）

        def _emit(piece: str) -> None:
            """推送一块正文：首块剥掉前导空白，其余原样推送。

            这样 "".join(推送块) == 最终 answer 恒成立——SSE 端"token 拼接 ==
            done.answer"的不变式由构造保证，前端零特判。"""
            nonlocal started
            if not started:
                piece = piece.lstrip()
                if not piece:
                    return
                started = True
            chunks.append(piece)
            on_token(piece)

        for event in resp:
            if getattr(event, "usage", None) is not None:  # 尾块：真实 token 用量
                usage = TokenUsage.from_openai(event.usage)
                continue
            if not event.choices:
                continue
            # getattr 兜底：finish_reason 只出现在最后一个 chunk，且测试用的 Fake 流式
            # choice 未必实现该字段——不能假设它一定存在（否则整条 SSE 直接 error 事件）。
            _fr = getattr(event.choices[0], "finish_reason", None)
            if _fr:
                finish_reason = _fr
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
                        _emit(after)
                elif len(prefix) > _MAX_INTENT_PREFIX:
                    # 模型未遵循首行 <intent> 格式 → 放弃解析，原样流式（默认 kb_qa）
                    intent_resolved = True
                    _emit(prefix)
                # else: 仍在缓冲标签，暂不推送
            else:
                _emit(piece)
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
        if finish_reason == "length":
            # 流式被截断 = 前端看到的正文本身就是半截话（无 JSON 解析问题，但观感更差）
            logger.warning(
                "stream_answer 正文被 max_tokens=%s 截断（completion=%s）→ 用户会看到半截回答；"
                "请上调 ANSWER_MAX_TOKENS 或收紧 prompt 长度要求",
                settings.answer_max_tokens, usage.completion_tokens if usage else "?")
        # 刻意**不再**对 text 做 strip / 标签正则替换：任何后处理都会让 answer 与已推送的
        # token 不一致（破坏拼接不变式）。标签已在流式解析阶段被消费（不进 chunks），
        # 正文前导空白由 _emit 剥掉，故此处只需原样 join。
        text = "".join(chunks)
        if not text.strip():
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
        # 自判意图（兜底规则；真实模型由 prompt 产出 intent）。
        # 只产出 kb_qa / chitchat —— "要求转人工"由 prompts.rule_classify_intent 规则
        # 短路处理，不进 LLM 决策（系统不决定找谁，也不代办转交）。
        q = _extract_tag(user, "query")
        if len(q.strip()) <= 12 and any(
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
            # 无证据可引：如实说明缺失，不提"转人工"（是否找他人由用户自己决定）
            return schemas.AnswerOutput(
                answer="知识库现有资料不足以回答该问题。", chunk_ids=[], intent=intent)
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


def _thinking_extra_body() -> dict:
    """思考模式开关（DashScope 扩展参数 `enable_thinking`）。

    qwen3.8-flash **默认开思考**：实测同一道题 reasoning 占 314~877 token，用户看不见
    却按输出价计费，且 token 是串行生成的 → 直接变成延迟。关闭后同题 completion
    1069 → 193 token（≈5.5×）、JSON 仍合法（见 2026-09-14 探测）。
    由 `settings.llm_enable_thinking` 统一控制（默认 False），不提供逐节点参数以保持调用面稳定。
    """
    return {"enable_thinking": bool(settings.llm_enable_thinking)}


def _warn_if_truncated(resp, *, node: str, max_tokens: int) -> None:
    """max_tokens 截断告警：截断会让 JSON 解析失败并触发一次白重试（成本翻倍）。"""
    try:
        finish = resp.choices[0].finish_reason
    except Exception:  # noqa: BLE001 — 结构异常不影响主流程
        return
    if finish == "length":
        usage = _usage_from_resp(resp)
        logger.warning(
            "%s 输出被 max_tokens=%s 截断（completion=%s）→ JSON 极可能不合法，"
            "将白重试一次；请上调 %s",
            node, max_tokens,
            usage.completion_tokens if usage else "?",
            "ANSWER_MAX_TOKENS" if node in ("complete_json", "stream_answer") else "max_tokens")


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
