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
    """检索慢查询（event=slow_query）。"""
    _emit_slow(event="slow_query", duration_ms=duration_ms,
               threshold_ms=_SLOW_QUERY_MS, req_id=req_id,
               query=query[:120], hits=hits, used_roads=used_roads or [])


def log_llm_call(req_id: str, duration_ms: float, *, node: str = "",
                 model: str = "", prompt_chars: int = 0, tokens: int = 0) -> None:
    """LLM 往返慢（event=llm_call）。"""
    _emit_slow(event="llm_call", duration_ms=duration_ms,
               threshold_ms=_SLOW_LLM_MS, req_id=req_id, node=node,
               model=model, prompt_chars=prompt_chars, tokens=tokens)


def log_ingest(req_id: str, duration_ms: float, *, doc_id: str = "",
               chunks: int = 0, status: str = "") -> None:
    """入库慢（event=ingest_slow）。"""
    _emit_slow(event="ingest_slow", duration_ms=duration_ms,
               threshold_ms=_SLOW_INGEST_MS, req_id=req_id, doc_id=doc_id,
               chunks=chunks, status=status)
