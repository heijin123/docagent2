"""全局配置：路径 / 向量库 / Embedding 模型与 Key。

规则（与 doc-agent 同款）：
- 环境变量优先、.env 兜底；
- EMBEDDING_PROVIDER=dashscope（默认）| mock；
- 配置为 dashscope 但未配 DASHSCOPE_API_KEY → 运行时降级 mock（degraded=True，不掩盖）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根 = 本文件上溯三级（app/core/config.py → app/core → app → 根）
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ── 路径 ──────────────────────────────────────────────
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    samples_dir: Path = BASE_DIR / "data" / "samples"
    chroma_dir: Path = BASE_DIR / "data" / "chroma_db"
    bm25_dir: Path = BASE_DIR / "data" / "bm25"
    registry_db: Path = BASE_DIR / "data" / "registry.db"
    reports_dir: Path = BASE_DIR / "data" / "reports"
    log_dir: Path = BASE_DIR / "data" / "logs"

    # ── Embedding ─────────────────────────────────────────
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "dashscope"))
    dashscope_base_url: str = field(
        default_factory=lambda: _env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    dashscope_api_key: str = field(default_factory=lambda: _env("DASHSCOPE_API_KEY", ""))
    qwen_embedding_model: str = field(default_factory=lambda: _env("QWEN_EMBEDDING_MODEL", "text-embedding-v3"))
    embed_dimensions: int = field(default_factory=lambda: _env_int("EMBED_DIMENSIONS", 1024))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 10))
    max_retries: int = field(default_factory=lambda: _env_int("EMBED_MAX_RETRIES", 3))
    connect_timeout_s: int = field(default_factory=lambda: _env_int("HTTP_CONNECT_TIMEOUT_S", 5))
    read_timeout_s: int = field(default_factory=lambda: _env_int("HTTP_READ_TIMEOUT_S", 60))

    # ── Chunking（需求 F1.2）──────────────────────────────
    chunk_size_tokens: int = field(default_factory=lambda: _env_int("CHUNK_SIZE_TOKENS", 512))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 80))

    # ── PDF 质量门（F1.9）─────────────────────────────────
    quality_min_text_chars: int = field(default_factory=lambda: _env_int("QUALITY_MIN_TEXT_CHARS", 20))
    quality_garble_threshold: float = field(
        default_factory=lambda: float(_env("QUALITY_GARBLE_THRESHOLD", "0.01"))
    )
    quality_yellow_text_chars: int = field(default_factory=lambda: _env_int("QUALITY_YELLOW_TEXT_CHARS", 100))

    # ── 对话 LLM（M3，DashScope OpenAI 兼容）───────────────
    qwen_llm_model: str = field(default_factory=lambda: _env("QWEN_LLM_MODEL", "qwen-plus"))
    qa_confidence_threshold: float = field(
        default_factory=lambda: float(_env("QA_CONFIDENCE_THRESHOLD", "0.6"))
    )
    qa_max_retry: int = field(default_factory=lambda: _env_int("QA_MAX_RETRY", 2))
    qa_history_rounds: int = field(default_factory=lambda: _env_int("QA_HISTORY_ROUNDS", 10))

    # ── Redis checkpointer（F4.1，M3）──────────────────────
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    redis_checkpointer_enabled: bool = field(
        default_factory=lambda: _env_bool("REDIS_CHECKPOINTER_ENABLED", True)
    )

    # ── 观测（M5）─────────────────────────────────────────
    log_format: str = field(default_factory=lambda: _env("LOG_FORMAT", "json"))
    obs_slow_query_ms: int = field(default_factory=lambda: _env_int("OBS_SLOW_QUERY_MS", 500))
    obs_slow_llm_ms: int = field(default_factory=lambda: _env_int("OBS_SLOW_LLM_MS", 3000))
    obs_slow_ingest_ms: int = field(default_factory=lambda: _env_int("OBS_SLOW_INGEST_MS", 5000))

    # ── 服务（M4 API）─────────────────────────────────────
    default_tenant_id: str = field(default_factory=lambda: _env("DEFAULT_TENANT_ID", "tenant_demo"))
    upload_dir: Path = BASE_DIR / "data" / "uploads"     # 上传落盘目录（随文件动态拼路径）
    service_api_key: str = field(default_factory=lambda: _env("SERVICE_API_KEY", ""))
    api_max_inflight: int = field(default_factory=lambda: _env_int("API_MAX_INFLIGHT", 50))
    sse_ping_interval_s: int = field(default_factory=lambda: _env_int("SSE_PING_INTERVAL_S", 15))
    ingest_workers: int = field(default_factory=lambda: _env_int("INGEST_WORKERS", 1))

    @property
    def has_api_key(self) -> bool:
        return bool(self.dashscope_api_key.strip())

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.samples_dir, self.chroma_dir, self.bm25_dir,
                  self.reports_dir, self.log_dir, self.upload_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
