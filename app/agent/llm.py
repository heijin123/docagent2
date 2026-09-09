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
from app.core.observability import TimedSpan, log_llm_call

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TOKEN_STEP = 12  # stub 分段模拟流式的最小块长


class LLMClient:
    provider: str = ""
    model: str = ""
    degraded: bool = False

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

        self.provider = "dashscope"
        self.model = model or settings.qwen_llm_model
        self._client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=settings.read_timeout_s,
        )

    def complete_json(self, system: str, user: str, schema: type[T],
                      *, temperature: float = 0.0, max_tokens: int = 1500) -> T:
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
            log_llm_call("", span.stop(log_slow=False), node="complete_json",
                         model=self.model, prompt_chars=len(system) + len(user))
        raw = resp.choices[0].message.content or ""
        # 容忍模型偶尔带 markdown 代码围栏
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return schema.model_validate(parsed)

    def stream_answer(self, system: str, user: str,
                      on_token) -> schemas.AnswerOutput:
        """DashScope 文本流式：stream=True 逐块回调（纯文本，非 JSON 模式）。"""
        span = TimedSpan(name="llm_call").attr(model=self.model)
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=1500,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        chunks: list[str] = []
        for event in resp:
            if not event.choices:
                continue  # 可能收到 usage 尾块
            delta = event.choices[0].delta
            piece = (delta or {}).content
            if piece:
                chunks.append(piece)
                on_token(piece)
        log_llm_call("", span.stop(log_slow=False), node="stream_answer",
                     model=self.model, prompt_chars=len(system) + len(user),
                     tokens=len("".join(chunks)))
        text = "".join(chunks).strip()
        if not text:
            return schemas.AnswerOutput(answer="", chunk_ids=[])
        evidence = _extract_tag(user, "evidence")
        chunk_ids = _parse_chunk_ids_from_answer(text, evidence)
        return schemas.AnswerOutput(answer=text, chunk_ids=chunk_ids)


# ── Stub 降级（无 Key 保链路；规则式，无语义）──────────────────
class StubLLM(LLMClient):
    """规则式桩：按 schema 路由，产出可驱动图跑通的结构化结果。

    子类可覆写 *_rule 注入特定行为（verify_m3 用它构造低置信场景）。
    """

    def __init__(self, model: str = "stub-rule-v1"):
        self.provider = "stub"
        self.model = model
        self.degraded = True
        self.calls: list[dict] = []   # 调用记录（测试断言用）

    def complete_json(self, system: str, user: str, schema: type[T],
                      *, temperature: float = 0.0, max_tokens: int = 1500) -> T:
        self.calls.append({"schema": schema.__name__, "system": system, "user": user})
        name = schema.__name__
        if name == "SupervisorIntent":
            return self.classify_rule(user)  # type: ignore[return-value]
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
        text = out.answer
        for i in range(0, len(text), _TOKEN_STEP):
            on_token(text[i:i + _TOKEN_STEP])
        return out

    # 各规则返回基础 pydantic 实例
    def classify_rule(self, user: str) -> schemas.SupervisorIntent:
        # 意图规则：转人工 > 寒暄 > 知识问答（user 文本含系统注入的 query 标签）
        q = _extract_tag(user, "query")
        if any(k in q for k in ("转人工", "人工客服", "找客服", "human")):
            return schemas.SupervisorIntent(intent="human_handoff", reason="命中转人工关键词")
        if len(q.strip()) <= 8 and any(k in q for k in ("你好", "hi", "hello", "在吗", "谢谢", "再见", "?")):
            return schemas.SupervisorIntent(intent="chitchat", reason="短寒暄/问候")
        return schemas.SupervisorIntent(intent="kb_qa", reason="默认知识问答")

    def rewrite_rule(self, user: str) -> schemas.RewriteOutput:
        q = _extract_tag(user, "query")
        return schemas.RewriteOutput(rewritten_query=q, changed=False)

    def answer_rule(self, user: str) -> schemas.AnswerOutput:
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
            return schemas.AnswerOutput(answer="知识库暂无相关现行资料，建议转人工核实。", chunk_ids=[])
        cites = _citations_text(evidence, picks)
        body = "根据检索到的资料：" + cites + "（如需进一步细节请说明）。"
        return schemas.AnswerOutput(answer=body, chunk_ids=picks)

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


def build_llm() -> LLMClient:
    """工厂：有 Key → dashscope；无 Key → stub（degraded，保链路回归）。"""
    if settings.has_api_key:
        return DashScopeLLM()
    logger.warning("未配 DASHSCOPE_API_KEY → 对话 LLM 降级 Stub（仅链路回归，无语义）")
    return StubLLM()
