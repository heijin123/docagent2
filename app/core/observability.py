"""M5 观测模块：耗时统计 + 结构化慢操作日志（JSON 行）。

设计目标（对齐需求 F7 观测 + 部署手册）：
- 零第三方依赖（不引入 prometheus_client/opentelemetry，保持轻量，可被外部采集器吞日志）；
- 统一 `duration_ms` 语义（time.perf_counter 单调钟，单位毫秒，一位小数）；
- `latency()` 上下文管理器 / `TimedSpan` 手动 span 两种用法；
- `log_slow()` 输出单行 JSON，字段固定（ts/level/event/req_id/duration_ms/...），
  便于 Filebeat/Loki/Promtail 按 `event=slow_query` / `event=llm_call` 聚合。

配置（.env）：
- OBS_SLOW_QUERY_MS  检索慢查询阈值，默认 500ms；
- OBS_SLOW_LLM_MS    LLM 往返慢阈值，默认 3000ms；
- OBS_SLOW_INGEST_MS 入库慢阈值，默认 5000ms；
- LOG_FORMAT        json | text（默认 json；text 回退原 console 格式便于本地看）。
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .config import _env_int, _env

logger = logging.getLogger(__name__)

_SLOW_QUERY_MS = _env_int("OBS_SLOW_QUERY_MS", 500)
_SLOW_LLM_MS = _env_int("OBS_SLOW_LLM_MS", 3000)
_SLOW_INGEST_MS = _env_int("OBS_SLOW_INGEST_MS", 5000)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# ─────────────────────────────────────────────────────────────
# Token 用量（贯穿整条链路的成本核算单位，M3+）
# 此前代码从不读取 resp.usage，token 消耗只能去 DashScope 控制台看，
# 导致「为了准确率忽略价格影响」。这里把 usage 做成一等公民：
#   每次 LLM 调用 → TokenMeter 累加 → 单轮 reply 前/后差值 = 该问成本
#   → 结构化 llm_usage 日志（看钱）/ 慢阈值 llm_call 日志（看慢）/ 评估报告（看总账）
# ─────────────────────────────────────────────────────────────
@dataclass
class TokenUsage:
    """单次/聚合 LLM token 用量。total_tokens 由服务端返回，不自行相加以防误差。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __sub__(self, other: "TokenUsage") -> "TokenUsage":
        """单轮差值（reply 前/后 snapshot 相减 = 该轮成本）。"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
        )

    @classmethod
    def from_openai(cls, usage) -> "TokenUsage":
        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    @classmethod
    def estimate(cls, prompt_chars: int, completion_chars: int = 0) -> "TokenUsage":
        """无 usage 时的估算（stub/降级/流式尾块缺失）：中英文混合 ~2 字符/token。"""
        p = prompt_chars // 2
        c = completion_chars // 2
        return cls(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)

    def cost(self, input_per_1k: float, output_per_1k: float) -> float:
        """预估人民币成本（按 1K token 单价）。"""
        return self.prompt_tokens / 1000 * input_per_1k + \
               self.completion_tokens / 1000 * output_per_1k

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class TokenMeter:
    """累计 token 用量：挂在 LLM 实例上，随每次调用累加；reply 前/后差值 = 单轮成本。

    同时计数**真实 LLM 调用次数**：重试/多轮节点会让一次 reply 触发多次调用，
    「调用次数」是比 token 更直观的链路成本信号（重试轮数 = 调用数 - 1）。
    """

    def __init__(self) -> None:
        self._u = TokenUsage()
        self._calls = 0

    def add(self, usage: TokenUsage) -> None:
        if usage is not None:
            self._u = self._u + usage
        self._calls += 1

    def snapshot(self) -> TokenUsage:
        return TokenUsage(**self._u.to_dict())

    @property
    def calls(self) -> int:
        """启动至今累计的真实 LLM 调用次数。"""
        return self._calls

    def reset(self) -> None:
        self._u = TokenUsage()
        self._calls = 0


@dataclass
class TimedSpan:
    """手动 span：start() 打点，stop() 返回耗时并（可选）记录慢日志。"""
    name: str = ""
    start_ms: float = field(default_factory=_now_ms)
    _attrs: dict[str, Any] = field(default_factory=dict)

    def attr(self, **kw: Any) -> "TimedSpan":
        self._attrs.update(kw)
        return self

    def stop(self, *, log_slow: bool = True,
             slow_threshold_ms: int | None = None) -> float:
        duration_ms = _now_ms() - self.start_ms
        if log_slow:
            threshold = slow_threshold_ms if slow_threshold_ms is not None else _SLOW_QUERY_MS
            if duration_ms >= threshold:
                _emit_slow(event=self.name or "slow_op", duration_ms=duration_ms,
                           threshold_ms=threshold, **self._attrs)
        return duration_ms


@contextmanager
def latency(name: str, *, req_id: str = "", slow_threshold_ms: int | None = None,
            **attrs: Any) -> Iterator[TimedSpan]:
    """上下文计时：退出时自动 stop（可选记录慢日志）。"""
    span = TimedSpan(name=name).attr(**attrs)
    try:
        yield span
    finally:
        span.stop(slow_threshold_ms=slow_threshold_ms)


def _emit_slow(*, event: str, duration_ms: float, threshold_ms: int, **attrs: Any) -> None:
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "level": "WARN",
        "event": event,
        "duration_ms": round(duration_ms, 1),
        "threshold_ms": threshold_ms,
        **attrs,
    }
    logger.warning(json.dumps(payload, ensure_ascii=False, default=str))


def log_slow_query(req_id: str, query: str, duration_ms: float,
                   used_roads: list[str] | None = None, hits: int = 0) -> None:
    """检索慢查询（event=slow_query）：仅 duration_ms ≥ OBS_SLOW_QUERY_MS 时落日志。"""
    if duration_ms < _SLOW_QUERY_MS:
        return
    _emit_slow(event="slow_query", duration_ms=duration_ms,
               threshold_ms=_SLOW_QUERY_MS, req_id=req_id,
               query=query[:120], hits=hits, used_roads=used_roads or [])


def log_llm_call(req_id: str, duration_ms: float, *, node: str = "",
                 model: str = "", prompt_chars: int = 0,
                 prompt_tokens: int = 0, completion_tokens: int = 0,
                 total_tokens: int = 0) -> None:
    """LLM 往返慢（event=llm_call）：仅 duration_ms ≥ OBS_SLOW_LLM_MS 时落日志；偏延迟。"""
    if duration_ms < _SLOW_LLM_MS:
        return
    _emit_slow(event="llm_call", duration_ms=duration_ms,
               threshold_ms=_SLOW_LLM_MS, req_id=req_id, node=node,
               model=model, prompt_chars=prompt_chars,
               prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
               total_tokens=total_tokens)


def log_llm_usage(req_id: str, *, node: str = "", model: str = "",
                  usage: "TokenUsage | None" = None,
                  est_cost_cny: float = 0.0) -> None:
    """LLM token 用量（event=llm_usage）：每次调用必记（不受慢阈值限制），供成本聚合。

    与 log_llm_call（慢阈值限流、偏延迟）互补 —— 前者看「钱」，后者看「慢」。
    生产环境可由 Loki/Filebeat 按 event=llm_usage 聚合出按 node/model 的成本曲线。
    """
    if usage is None:
        return
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "level": "INFO",
        "event": "llm_usage",
        "node": node,
        "model": model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "est_cost_cny": round(est_cost_cny, 6),
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def log_ingest(req_id: str, duration_ms: float, *, doc_id: str = "",
               chunks: int = 0, status: str = "") -> None:
    """入库慢（event=ingest_slow）：仅 duration_ms ≥ OBS_SLOW_INGEST_MS 时落日志。"""
    if duration_ms < _SLOW_INGEST_MS:
        return
    _emit_slow(event="ingest_slow", duration_ms=duration_ms,
               threshold_ms=_SLOW_INGEST_MS, req_id=req_id, doc_id=doc_id,
               chunks=chunks, status=status)


def log_kb_gap(req_id: str, query: str, *, rewritten_query: str = "",
               thread_id: str = "", expired_candidates: int = 0,
               reason: str = "") -> None:
    """知识库覆盖缺口线索（event=kb_gap）：检索完全无命中时记录，**每次必记**。

    定位（重要，别当成告警）：这只是一条**离线线索**——供知识库管理员把散落的
    "查不到的问题"聚类成「Top-N 缺失主题」，再决定补哪几篇文档。它
    **不进入任何人工作队列、不触发工单、不做任何转交/升级动作**（对齐系统职责边界：
    本系统只做检索与披露，"补资料"是客户/管理员侧的职能）。

    与 log_llm_usage 同为不设阈值的结构化 JSON 行（level=INFO），字段固定
    （ts/event/req_id/thread_id/query/rewritten_query/expired_candidates），
    生产可由 Loki/Filebeat 按 `event=kb_gap` 聚合出缺口主题排行。
    """
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "level": "INFO",
        "event": "kb_gap",
        "req_id": req_id,
        "thread_id": thread_id,
        "query": (query or "")[:200],
        "rewritten_query": (rewritten_query or "")[:200],
        "expired_candidates": expired_candidates,
        # 触发来源：empty_index（索引真空，真无命中）| relevance_gate（F2.10 双证据判定
        # 不相关——"库里有 A 主题、用户问 B 主题"）。缺口聚类时可据此区分"没入库"与"没覆盖"。
        "reason": reason,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
