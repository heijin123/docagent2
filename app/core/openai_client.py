"""共享的 DashScope OpenAI 客户端单例（2026-09-15 落地）。

embedding / vlm 此前在方法内**每次调用都新建 `OpenAI()`**，长批量 embedding 与
多页 VLM 转录会反复建连、抖动且浪费。本模块把客户端收窄为**进程级单例**
（按 settings 派生，settings 不变则只建一次）。

`llm.py` 的 `DashScopeLLM` 是对象级自持 client、由 `Services` 单例化，故不在此统一；
本单例只服务无状态、被反复调用的 embedding / vlm 两条路径。
"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


@lru_cache(maxsize=1)
def get_dashscope_client() -> OpenAI:
    """返回复用型 DashScope OpenAI 客户端（按 settings 派生，进程内只建一次）。

    超时沿用 `settings.read_timeout_s`（与 llm 一致）；不显式设 `max_retries`，
    保留 OpenAI SDK 默认的短重试——embedding/vlm 自带业务层退避（见
    `Embedder._embed_batch_with_retry` / VLM 失败即降级），避免双重重试烧 token。
    """
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        timeout=settings.read_timeout_s,
    )
