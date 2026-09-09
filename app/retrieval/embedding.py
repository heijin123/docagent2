"""Embedding 双通道（需求 F1.3，契约 4.6 embed）：

- dashscope（默认）：text-embedding-v3，OpenAI 兼容，分批 ≤10 + 指数退避重试；
- mock 降级：本地确定性哈希向量（无语义，仅链路回归），无 Key 时自动切且 degraded=True 不掩盖。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from app.core.config import settings

MOCK_DIMENSIONS = 256


@dataclass
class Embedder:
    provider: str            # dashscope | mock
    model: str
    dimensions: int
    degraded: bool           # True = 期望 dashscope 但降级 mock

    def info(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "degraded": self.degraded,
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "mock":
            return [mock_embedding(t, self.dimensions) for t in texts]
        return self._embed_dashscope(texts)

    # ── dashscope ──────────────────────────────────────────
    def _embed_dashscope(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=settings.read_timeout_s,
        )
        results: list[list[float]] = []
        # 分批 ≤10 + 指数退避重试（doc-agent 实证：大批量 embedding 曾触发断连）
        for i in range(0, len(texts), settings.embedding_batch_size):
            batch = texts[i:i + settings.embedding_batch_size]
            results.extend(self._embed_batch_with_retry(client, batch))
        return results

    def _embed_batch_with_retry(self, client, batch: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(settings.max_retries):
            try:
                resp = client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
                # OpenAI 兼容返回顺序与输入一致
                ordered = sorted(resp.data, key=lambda d: d.index)
                return [d.embedding for d in ordered]
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < settings.max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
        raise RuntimeError(f"Embedding 失败（重试 {settings.max_retries} 次）: {last_err}")


def mock_embedding(text: str, dimensions: int = MOCK_DIMENSIONS) -> list[float]:
    """确定性 bigram 哈希向量：同文本恒同向量（幂等/链路回归用，无语义）。"""
    vec = [0.0] * dimensions
    norm = text.encode("utf-8")
    if not norm:
        return vec
    for i in range(len(norm) - 1):
        bigram = norm[i:i + 2]
        h = int.from_bytes(hashlib.md5(bigram).digest()[:4], "little")
        vec[h % dimensions] += 1.0
    # L2 归一化
    length = sum(v * v for v in vec) ** 0.5
    if length > 0:
        vec = [v / length for v in vec]
    return vec


def build_embedder() -> Embedder:
    """工厂：EMBEDDING_PROVIDER=dashscope（默认）| mock；dashscope 无 Key → 降级 mock + degraded。"""
    provider = settings.embedding_provider
    if provider == "mock":
        return Embedder(provider="mock", model="mock-hash-v1",
                        dimensions=MOCK_DIMENSIONS, degraded=False)
    if provider == "dashscope":
        if settings.has_api_key:
            return Embedder(provider="dashscope", model=settings.qwen_embedding_model,
                            dimensions=settings.embed_dimensions, degraded=False)
        # 无 Key：降级 mock，degraded=True 不掩盖
        return Embedder(provider="mock", model="mock-hash-v1",
                        dimensions=MOCK_DIMENSIONS, degraded=True)
    raise ValueError(f"未知 EMBEDDING_PROVIDER={provider!r}，允许 dashscope | mock")
