"""API 服务依赖（M4 全局单例，lifespan 构建）。

共享实例（跨 chat / 上传 / debug / health）：
- VectorStore / BM25Store / embedder / DocRegistry：入库与检索共用同一底层连接对象，
  避免多客户端打开同一 Chroma 目录（Chroma 并发写限制 → 任务单写者串行）；
- HybridRetriever：注入共享 store；
- AgentApp：持共享 checkpointer（Redis→内存降级），流式请求重建图但复用 checkpointer；
- TaskManager：单写者入库。
"""
from __future__ import annotations

import logging

from fastapi import Request

from app.agent.graph import AgentApp
from app.agent.llm import build_llm
from app.api.ingest_tasks import TaskManager
from app.core.config import settings
from app.ingestion.pipeline import IngestPipeline
from app.ingestion.registry import DocRegistry
from app.retrieval.bm25store import BM25Store
from app.retrieval.embedding import build_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Services:
    def __init__(self, *, embedder=None, vector_store=None, bm25_store=None,
                 registry=None, llm=None, memory_checkpoint: bool = False,
                 task_workers: int | None = None):
        """默认全走真实组件；测试可注入 mock embedder / stub llm / 隔离 store。

        memory_checkpoint=True → InMemorySaver（避免依赖 Redis，verify_m4 用）。
        """
        settings.ensure_dirs()
        self.embedder = embedder or build_embedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.registry = registry or DocRegistry()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store, bm25_store=self.bm25_store,
            embedder=self.embedder)
        self.llm = llm or build_llm()
        self.agent = AgentApp(self.retriever, self.llm,
                              memory_checkpoint=memory_checkpoint)
        self.task_manager = TaskManager(
            lambda tenant: IngestPipeline(
                tenant_id=tenant, embedder=self.embedder,
                vector_store=self.vector_store, bm25_store=self.bm25_store,
                registry=self.registry),
            max_workers=task_workers if task_workers is not None
            else settings.ingest_workers,
        )
        logger.info("services 就绪: llm=%s/%s degraded=%s | checkpointer=%s",
                    self.llm.provider, self.llm.model, self.llm.degraded,
                    "redis" if not self.agent._ckpt_degraded else "memory(degraded)")

    def close(self) -> None:
        try:
            self.task_manager.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.vector_store.close()
        except Exception:  # noqa: BLE001
            pass


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_tenant_id(request: Request) -> str:
    """X-Tenant-Id 头；缺省用服务默认租户（demo/单租户部署）。"""
    return request.headers.get("X-Tenant-Id", settings.default_tenant_id)


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def check_thread_tenant(thread_id: str, tenant_id: str) -> None:
    """契约 §1.3：thread_id 建议 {tenant}:{user}，提供时前缀必须与 X-Tenant-Id 一致。"""
    if ":" in thread_id:
        prefix = thread_id.split(":", 1)[0]
        if prefix != tenant_id:
            from app.api.errors import ApiError

            raise ApiError(
                "THREAD_TENANT_MISMATCH",
                f"thread_id 前缀 {prefix!r} 与 X-Tenant-Id {tenant_id!r} 不一致", 422)
