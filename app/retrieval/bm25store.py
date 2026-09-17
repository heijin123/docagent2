"""BM25Store（契约 4.4）：rank_bm25 + jieba 中文分词，与向量库双索引同批写入。

M1 以 sqlite 持久化语料；M2 扩展为携带完整过滤元数据（与向量侧 metadata 同键，
见 models.chunk_to_metadata），使 BM25 侧与 Chroma 侧强制过滤规则一致（F2.7/F2.8）。

- `add()`：chunk_id 存在则覆盖（content / 元数据列 + meta_json 同步翻新）；
- `search(query, top_k, where_sql, where_params)`：**每次按过滤条件现建索引**再排序
  （rank_bm25 无原生持久化；语料小，rebuild 成本可接受，M2+ 切 ES，见设计决策）。
- `rebuild()` 保留（兼容 pipeline / health），等价于全量 valid 重建。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import jieba

from app.core.config import settings
from app.models import ChunkRecord, chunk_to_metadata, now_ts


def tokenize(text: str) -> list[str]:
    """jieba 分词（索引与查询两侧必须一致）。"""
    return [t for t in jieba.lcut(text) if t.strip()]


# 过滤所需标量列（元数据列，除主键外均可 ALTER 增量补齐）
_META_COLUMNS: dict[str, str] = {
    "tenant_id": "TEXT NOT NULL DEFAULT 'tenant_demo'",
    "permission": "TEXT NOT NULL DEFAULT 'internal'",
    "category": "TEXT NOT NULL DEFAULT 'general'",
    "department": "TEXT NOT NULL DEFAULT ''",
    "effective_time": "INTEGER NOT NULL DEFAULT 0",
    "doc_year": "INTEGER NOT NULL DEFAULT -1",
    "source": "TEXT NOT NULL DEFAULT ''",
    "doc_date": "TEXT NOT NULL DEFAULT ''",
    "author": "TEXT NOT NULL DEFAULT ''",
    "file_path": "TEXT NOT NULL DEFAULT ''",
    "meta_json": "TEXT NOT NULL DEFAULT '{}'",
}

_BASE_SQL = """
CREATE TABLE IF NOT EXISTS bm25_corpus (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    content     TEXT NOT NULL,
    is_valid    INTEGER NOT NULL DEFAULT 1,
    page_num    INTEGER NOT NULL DEFAULT 0,
    doc_title   TEXT NOT NULL DEFAULT '',
    updated_at  INTEGER NOT NULL
)
"""


class BM25Store:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or (settings.bm25_dir / "corpus.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """建表 + 幂等迁移（存量库缺列时 ALTER 补齐，兼容 M1 库）。"""
        with self._connect() as conn:
            conn.execute(_BASE_SQL)
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(bm25_corpus)")}
            for col, ddl in _META_COLUMNS.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE bm25_corpus ADD COLUMN {col} {ddl}")

    def add(self, records: list[ChunkRecord]) -> int:
        """写入（幂等：chunk_id 存在则整体覆盖）。返回写入数。"""
        now = now_ts()
        with self._connect() as conn:
            for rec in records:
                meta = chunk_to_metadata(rec)
                conn.execute(
                    """
                    INSERT INTO bm25_corpus (
                        chunk_id, doc_id, version, content, is_valid,
                        page_num, doc_title, updated_at,
                        tenant_id, permission, category, department,
                        effective_time, doc_year, source, doc_date,
                        author, file_path, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        doc_id=excluded.doc_id, version=excluded.version,
                        content=excluded.content, is_valid=excluded.is_valid,
                        doc_title=excluded.doc_title, page_num=excluded.page_num,
                        updated_at=excluded.updated_at,
                        tenant_id=excluded.tenant_id, permission=excluded.permission,
                        category=excluded.category, department=excluded.department,
                        effective_time=excluded.effective_time, doc_year=excluded.doc_year,
                        source=excluded.source, doc_date=excluded.doc_date,
                        author=excluded.author, file_path=excluded.file_path,
                        meta_json=excluded.meta_json
                    """,
                    (rec.chunk_id, rec.doc_id, rec.version, rec.content,
                     1 if rec.is_valid else 0, rec.page_num, rec.doc_title, now,
                     meta["tenant_id"], meta["permission"], meta["category"],
                     meta["department"], meta["effective_time"], meta["doc_year"],
                     meta["source"], meta["doc_date"], meta["author"], meta["file_path"],
                     json.dumps(meta, ensure_ascii=False)),
                )
        return len(records)

    def soft_delete_doc(self, doc_id: str, version: int) -> int:
        """F1.7：版本化软删除，翻 is_valid=0（与向量库动作对齐）。"""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE bm25_corpus SET is_valid=0 WHERE doc_id=? AND version=?",
                (doc_id, version),
            )
        return cur.rowcount

    def get_by_chunk_ids(self, chunk_ids: list[str], *,
                         tenant_id: str | None = None,
                         allowed_permissions: list[str] | None = None) -> list[dict]:
        """按 chunk_id 精确取块（邻近上下文扩展用，F2.10/需求 7.1）。

        只返回 `is_valid=1`，并强制与检索路**同一套隔离**（tenant_id + permission 白名单）
        ——邻居块同样不能越权泄漏。返回形状与 `search()` 一致（bm25_score=0.0）。
        """
        if not chunk_ids:
            return []
        sql = "SELECT chunk_id, doc_id, content, meta_json FROM bm25_corpus WHERE is_valid=1"
        params: list = []
        sql += f" AND chunk_id IN ({', '.join('?' * len(chunk_ids))})"
        params.extend(chunk_ids)
        if tenant_id:
            sql += " AND tenant_id=?"
            params.append(tenant_id)
        if allowed_permissions:
            sql += f" AND permission IN ({', '.join('?' * len(allowed_permissions))})"
            params.extend(allowed_permissions)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [{"chunk_id": r["chunk_id"], "content": r["content"],
                 "metadata": json.loads(r["meta_json"] or "{}"), "bm25_score": 0.0}
                for r in rows]

    def doc_chunk_ids(self, doc_id: str, version: int | None = None) -> list[str]:
        """某文档（可选指定版本）的全部 chunk_id（PATCH 元数据用）。"""
        sql = "SELECT chunk_id FROM bm25_corpus WHERE doc_id=?"
        params: list = [doc_id]
        if version is not None:
            sql += " AND version=?"
            params.append(version)
        sql += " ORDER BY chunk_id"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [r["chunk_id"] for r in rows]

    def set_metadata(self, chunk_ids: list[str], fields: dict) -> int:
        """按 chunk_id 批量更新可过滤元数据列（F2.8 有效期设置，与向量侧对齐）。

        只接受白名单列，避免任意列写坏库；未登记列直接忽略（调用方不必知道 schema）。
        """
        allowed = {"effective_time", "category", "department", "permission"}
        cols = [c for c in fields if c in allowed]
        if not chunk_ids or not cols:
            return 0
        # meta_json 与列必须同步（检索过滤走列，citation/证据走 meta_json）
        with self._connect() as conn:
            n = 0
            for cid in chunk_ids:
                row = conn.execute(
                    "SELECT meta_json FROM bm25_corpus WHERE chunk_id=?", (cid,)
                ).fetchone()
                if row is None:
                    continue
                meta = json.loads(row["meta_json"] or "{}")
                meta.update({c: fields[c] for c in cols})
                sets = ", ".join(f"{c}=?" for c in cols)
                conn.execute(
                    f"UPDATE bm25_corpus SET {sets}, meta_json=?, updated_at=? "
                    "WHERE chunk_id=?",
                    (*[fields[c] for c in cols], json.dumps(meta, ensure_ascii=False),
                     now_ts(), cid),
                )
                n += 1
        return n

    def rebuild(self):
        """全量重建索引（兼容入口：等价 search 不带过滤，供 pipeline / health 用）。"""
        self._build_from_sql("", ())

    def _build_from_sql(self, where_sql: str, where_params: tuple):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, content, meta_json FROM bm25_corpus "
                "WHERE is_valid=1" + (f" AND {where_sql}" if where_sql else ""),
                where_params,
            ).fetchall()
        from rank_bm25 import BM25Okapi

        corpus = [row["content"] for row in rows]
        self._tokenized = [tokenize(c) for c in corpus]
        self._bm25 = BM25Okapi(self._tokenized) if corpus else None
        self._rows = [
            {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
             "content": r["content"], "meta": json.loads(r["meta_json"] or "{}")}
            for r in rows
        ]

    def search(self, query: str, top_k: int = 8,
               where_sql: str = "", where_params: tuple = ()) -> list[dict]:
        """BM25 检索（F2.7：与向量侧同一过滤规则，代码层一条规则防漂移）。

        where_sql / where_params 由调用方（HybridRetriever）用与 Chroma where
        等价的谓词构造；每次按过滤语料现建索引再排序。
        """
        self._build_from_sql(where_sql, where_params)
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:top_k]:
            row = self._rows[i]
            out.append({"chunk_id": row["chunk_id"], "content": row["content"],
                        "metadata": row["meta"], "bm25_score": float(scores[i])})
        return out

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) c FROM bm25_corpus WHERE is_valid=1").fetchone()["c"]

    def iter_valid_chunks(self, tenant_id: str | None = None):
        """遍历全部有效块（评估锚句定位用）：yield {chunk_id, doc_id, content, meta}。

        按 chunk_id 升序（doc_id + version + index 自然序）；tenant_id 可选过滤。
        """
        sql = "SELECT chunk_id, doc_id, content, meta_json FROM bm25_corpus WHERE is_valid=1"
        params: tuple = ()
        if tenant_id:
            sql += " AND tenant_id=?"
            params = (tenant_id,)
        sql += " ORDER BY chunk_id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        for r in rows:
            yield {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
                   "content": r["content"], "meta": json.loads(r["meta_json"] or "{}")}

    def health(self) -> bool:
        try:
            self._build_from_sql("", ())
            return True
        except Exception:  # noqa: BLE001
            return False
