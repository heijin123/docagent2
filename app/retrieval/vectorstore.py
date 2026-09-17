"""VectorStore（契约 4.3）：Chroma 持久化，Append-Only 约束。

- 允许：新增、payload 元数据原位更新（软删除翻 is_valid）、带过滤查询、健康检查；
- 禁止：物理删除 / 替换向量（离线重建专用 _purge 仅 compaction 脚本可调，M1 不暴露）；
- chunk 级幂等：写入前 contains() 查重，命中跳过（断点续跑不重嵌不重计费）。
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.models import ChunkRecord, chunk_to_metadata, now_ts

COLLECTION_NAME = "enterprise_qa"


class VectorStore:
    def __init__(self, persist_dir: str | Path | None = None):
        import chromadb

        self.persist_dir = Path(persist_dir or settings.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        # M1：collection 用无默认 embedding 函数（检索显式 query embedding，防 query_texts 下载默认模型）
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── 写：Append-Only（upsert 语义 = 同 id 覆盖；物理删除仅 compaction）──
    def contains(self, chunk_id: str) -> bool:
        res = self._collection.get(ids=[chunk_id])
        return len(res["ids"]) > 0

    def add(self, records: list[ChunkRecord], embeddings: list[list[float]]) -> int:
        """写入新 chunk（幂等：已存在跳过）。返回实际写入数。"""
        ids, docs, metas, embs = [], [], [], []
        written = 0
        for rec, emb in zip(records, embeddings):
            if self.contains(rec.chunk_id):
                continue  # 断点续跑：已完成 chunk 跳过，不重复计费
            ids.append(rec.chunk_id)
            docs.append(rec.content)
            metas.append(chunk_to_metadata(rec))
            embs.append(emb)
            written += 1
        if ids:
            self._collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        return written

    def set_metadata(self, chunk_ids: list[str], fields: dict) -> None:
        """payload 元数据原位更新（软删除翻 is_valid / update_time），不触碰向量。"""
        if not chunk_ids:
            return
        # 只带要更新的字段（Chroma update 合并其余已有 metadata）
        metas = [{k: fields[k] for k in fields} for _ in chunk_ids]
        self._collection.update(ids=chunk_ids, metadatas=metas)

    def doc_chunk_ids(self, doc_id: str, version: int | None = None) -> list[str]:
        """某文档（可选指定版本）的全部 chunk_id（PATCH 元数据用）。"""
        where: dict = {"doc_id": {"$eq": doc_id}}
        if version is not None:
            where = {"$and": [where, {"version": {"$eq": version}}]}
        res = self._collection.get(where=where, include=[])
        return sorted(res["ids"] or [])

    def soft_delete_doc(self, doc_id: str, version: int) -> int:
        """F1.7：某文档某版本全部 chunk 翻 is_valid=false（先插新后翻旧保证无空窗）。"""
        hits = self._collection.get(
            where={"$and": [{"doc_id": {"$eq": doc_id}}, {"version": {"$eq": version}}]},
            include=["metadatas"],
        )
        ids = hits["ids"]
        if ids:
            self.set_metadata(ids, {"is_valid": False, "update_time": now_ts()})
        return len(ids)

    # ── 查询（检索统一显式 query embedding；where 过滤构造见契约 4.5）──
    def query(self, query_embedding: list[float], *, top_k: int = 8,
              where: dict | None = None) -> list[dict]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if res["ids"] and res["ids"][0]:
            for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                            res["metadatas"][0], res["distances"][0]):
                out.append({"chunk_id": cid, "content": doc, "metadata": meta,
                            "distance": dist, "score": 1.0 - dist})
        return out

    def count(self) -> int:
        return self._collection.count()

    def health(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        """释放底层资源（Windows 文件句柄），测试/优雅退出时调用。"""
        try:
            self._client.clear_system_cache()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def _purge(self) -> None:  # compaction 专用，业务路径禁用
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME)
