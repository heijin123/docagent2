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
    # LLM 读取超时：qwen 长请求（含完整 evidence 的 answer/verify）常 >60s，
    # 旧默认 60s 触发 SDK 默认 2 次重试 → 同一 prompt 被发 3 次、烧 3 倍 token。
    # 提至 120s 给长请求一次成功机会；配合 llm_max_retries=0 杜绝静默重发。
    read_timeout_s: int = field(default_factory=lambda: _env_int("HTTP_READ_TIMEOUT_S", 120))
    # LLM 客户端重试：默认 0（关掉 OpenAI SDK 默认的 2 次静默重试）。
    # 重试交由业务层可控退避，避免「为成功率无视价格」地偷偷重发整段 prompt。
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 0))

    # ── 输出约束 / 思考模式（2026-09-14 实测后新增）────────────
    # answer 生成长度上限（**只约束可见正文**，不含 reasoning token，见下）。
    # 实证（24 条评估）：answer 不限长时单问 completion 达 2388 token，
    # 单次调用最长 3184 token / 78.6s —— completion 长度就是延迟的第一驱动，
    # 而链路延迟 P95 已超 NFR 17 倍。512 足够容纳"结论 + 3~5 句 + [来源:] 标记"的 JSON 包体。
    # 注意：设得过小会让 JSON 被截断（finish_reason=length → 解析失败 → 白重试一次），
    # 所以不是越小越好；512 是"够装下完整 JSON 信封"的安全值。
    answer_max_tokens: int = field(default_factory=lambda: _env_int("ANSWER_MAX_TOKENS", 512))
    # 是否让模型走「思考（reasoning）」模式。qwen3.8-flash **默认开思考**，实测
    # 单次回答 reasoning 占 314~877 token（用户看不见、却按输出价计费，且串行生成直接
    # 变慢）。实测关闭后同题 completion 由 1069 → 193 token（≈5.5×），JSON 仍合法。
    # 置 1 可改回开思考（若发现答案质量下降，优先只给 verify 开）。
    llm_enable_thinking: bool = field(
        default_factory=lambda: _env_bool("LLM_ENABLE_THINKING", False))

    # ── LLM 定价（估算基线，CNY / 1K tokens，输入/输出）─────────
    # 以 DashScope 官网最新价格为准；此处仅作成本可见化的估算基线。
    # 价格随模型迭代变动频繁，请定期核对 https://help.aliyun.com/zh/model-studio/models
    llm_pricing: dict = field(default_factory=lambda: {
        "qwen-turbo": (0.003, 0.006),
        "qwen-plus": (0.004, 0.012),
        "qwen-max": (0.02, 0.06),
        "qwen-long": (0.0005, 0.002),
        # Qwen3.8-Flash 北京按量（2026-08-27 调价后）：0.8 / 2.7 元每百万 tokens
        "qwen-flash": (0.0008, 0.0027),
        "default": (0.01, 0.03),
    })

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
    # 历史文本的两级预算（防多轮把 prompt 撑爆）。上限是怎么来的：窗口 10 轮 × 2 条
    # × 每条 300 字 ≈ 6,000 字/次注入，而 rewrite 与 answer **各注入一次** → 最坏约
    # 7,500 tok/问（≈ 单问 prompt 的 2.3 倍）；单轮评估恒为 0（无历史），看不见。
    # 改「每条截断 + 总量预算」后最坏约 1,200 字 ≈ 710 tok，且最新一轮必然保留。
    history_per_msg_chars: int = field(
        default_factory=lambda: _env_int("HISTORY_PER_MSG_CHARS", 150))
    history_total_chars: int = field(
        default_factory=lambda: _env_int("HISTORY_TOTAL_CHARS", 1200))

    # ── VLM 红页转录（F1.9，M2）─────────────────────────
    # 红页（扫描/乱码）走 Qwen-VL 转录为文本；VLM_ENABLED=1 且配 Key 才启用，否则降级原文。
    vlm_enabled: bool = field(default_factory=lambda: _env_bool("VLM_ENABLED", False))
    qwen_vlm_model: str = field(default_factory=lambda: _env("QWEN_VLM_MODEL", "qwen-vl-max"))

    # ── 检索相关性判定（F2.10，2026-09-17）──────────────────
    # 向量检索恒返回 topN，无距离阈值 → 库里没资料时也给回"最像的 8 条"，
    # 导致 no_data（0 LLM 如实告知缺失）只在索引真空时触发。这里用
    # 「词表覆盖 + 语义相似」双证据做**高精度短路**（标定见 app/retrieval/relevance.py）。
    # 默认值来自 2026-09-17 实测：golden 55 条全保留（零误杀）、语料不覆盖的 12 条里判出 4 条
    # （"话题完全不在库里"那一类）。**误杀比漏判严重得多**，调参请勿放松这两个条件之一。
    retrieval_min_coverage: float = field(
        default_factory=lambda: float(_env("RETRIEVAL_MIN_COVERAGE", "0.20")))
    retrieval_min_similarity: float = field(
        default_factory=lambda: float(_env("RETRIEVAL_MIN_SIMILARITY", "0.40")))

    # ── 邻近 chunk 上下文扩展（需求 7.1，2026-09-17）─────────
    # 命中的 chunk 可能被切在段落/条款边界（chunk_index 相邻块是同一段落的续），
    # 取邻居块拼入**上下文**（不参与检索指标与命中口径，见 HybridResult.context_items）。
    # 预算刻意收紧：邻居会进 answer 的 evidence（prompt 大头），top-3 命中 × ±1 邻块、
    # 去重后最多 4 条 ≈ +500 tok/问。
    context_expand_enabled: bool = field(
        default_factory=lambda: _env_bool("CONTEXT_EXPAND_ENABLED", True))
    context_expand_top: int = field(default_factory=lambda: _env_int("CONTEXT_EXPAND_TOP", 3))
    context_expand_max: int = field(default_factory=lambda: _env_int("CONTEXT_EXPAND_MAX", 4))

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
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 50))
    allowed_origins: str = field(default_factory=lambda: _env("ALLOWED_ORIGINS", "*"))

    @property
    def has_api_key(self) -> bool:
        return bool(self.dashscope_api_key.strip())

    def llm_price(self, model: str | None = None) -> tuple[float, float]:
        """返回 (输入单价, 输出单价) CNY/1K tokens；按模型名关键字匹配定价表。

        优先级：turbo > plus > long > max > default（qwen3.8-max 命中 'max'）。
        """
        m = (model or self.qwen_llm_model).lower()
        table = self.llm_pricing
        if "turbo" in m:
            return table["qwen-turbo"]
        if "plus" in m:
            return table["qwen-plus"]
        if "long" in m:
            return table["qwen-long"]
        if "flash" in m:
            return table["qwen-flash"]
        if "max" in m:
            return table["qwen-max"]
        return table["default"]

    @property
    def allowed_origins_list(self) -> list[str]:
        raw = (self.allowed_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.samples_dir, self.chroma_dir, self.bm25_dir,
                  self.reports_dir, self.log_dir, self.upload_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
