"""DocRegistry（需求 F1.8，契约 4.7）：sqlite 幂等登记表。

- 主键 (tenant_id, doc_key)，doc_id 内容无关跨版本稳定；
- content_hash 仅作变更指纹；
- 状态：pending / processing / done / failed。
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


@dataclass
class DocRecord:
    tenant_id: str
    doc_key: str
    doc_id: str
    version: int
    status: str            # pending / processing / done / failed
    content_hash: str
    created_at: int
    updated_at: int
    error: str | None = None
    is_deleted: bool = False   # F5.7 软删除标记（幂等 DELETE / 删后重传判定；非 task_status 枚举）
    effective_time: int = 0    # F2.8 有效期截止（Unix 秒；0 = 永久有效）


_STATUSES = ("pending", "processing", "done", "failed")


class DocRegistry:
    """sqlite 登记表（线程安全：每次操作短连接）。"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or settings.registry_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunk_exists_fn: callable | None = None
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_registry (
                    tenant_id   TEXT NOT NULL,
                    doc_key     TEXT NOT NULL,
                    doc_id      TEXT NOT NULL,
                    version     INTEGER NOT NULL,
                    status      TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at  INTEGER NOT NULL,
                    updated_at  INTEGER NOT NULL,
                    error       TEXT,
                    is_deleted  INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, doc_key)
                )
                """
            )
            # 存量库迁移（M1 无 is_deleted 列）：幂等，重复 ALTER 报错即忽略
            try:
                conn.execute(
                    "ALTER TABLE doc_registry ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            # F2.8 有效期（2026-09-17 补）：此前 effective_time 只存在于 chunk 元数据、
            # 无任何写入入口 → 过期链路（验收标准 8）代码在、验不了。现由 PATCH 端点
            # 同时写登记表（列表展示权威值）与两个索引的 chunk 元数据（检索过滤用）。
            try:
                conn.execute(
                    "ALTER TABLE doc_registry ADD COLUMN effective_time INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass

    def get(self, tenant_id: str, doc_key: str) -> DocRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM doc_registry WHERE tenant_id=? AND doc_key=?",
                (tenant_id, doc_key),
            ).fetchone()
        return _row_to_record(row) if row else None

    def reserve(self, tenant_id: str, doc_key: str, doc_id: str,
                content_hash: str, status: str = "processing") -> DocRecord:
        """CAS 语义：不存在则注册为新文档（version=1）；存在则返回现有记录不覆盖。"""
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM doc_registry WHERE tenant_id=? AND doc_key=?",
                (tenant_id, doc_key),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO doc_registry (tenant_id, doc_key, doc_id, version, status,
                                              content_hash, created_at, updated_at, error)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?, NULL)
                    """,
                    (tenant_id, doc_key, doc_id, status, content_hash, now, now),
                )
                return DocRecord(tenant_id=tenant_id, doc_key=doc_key, doc_id=doc_id,
                                 version=1, status=status, content_hash=content_hash,
                                 created_at=now, updated_at=now)
        return _row_to_record(row)

    def begin_version(self, tenant_id: str, doc_key: str, doc_id: str, content_hash: str) -> DocRecord:
        """内容变更（同 doc_key 异 hash）：version+1 并置 processing（F1.7 版本化软更新）。"""
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE doc_registry SET status='processing', version=version+1,
                                        content_hash=?, updated_at=?
                WHERE tenant_id=? AND doc_key=?
                """,
                (content_hash, now, tenant_id, doc_key),
            )
            row = conn.execute(
                "SELECT * FROM doc_registry WHERE tenant_id=? AND doc_key=?",
                (tenant_id, doc_key),
            ).fetchone()
        return _row_to_record(row)

    def mark(self, tenant_id: str, doc_key: str, status: str, *, error: str | None = None) -> None:
        """标记状态（仅登记表，不影响 chunk 的 is_valid）。"""
        if status not in _STATUSES:
            raise ValueError(f"非法状态 {status!r}，允许: {_STATUSES}")
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE doc_registry SET status=?, updated_at=?, error=?
                WHERE tenant_id=? AND doc_key=?
                """,
                (status, now, error, tenant_id, doc_key),
            )

    def by_doc_id(self, tenant_id: str, doc_id: str) -> DocRecord | None:
        """按 doc_id 反查登记（DELETE 端点用）：doc_id 由 doc_key 派生，理论唯一；
        取 version 最新的一行（唯一保留有效块的行）。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM doc_registry WHERE tenant_id=? AND doc_id=? "
                "ORDER BY version DESC LIMIT 1",
                (tenant_id, doc_id),
            ).fetchone()
        return _row_to_record(row) if row else None

    def mark_deleted(self, tenant_id: str, doc_key: str) -> None:
        """F5.7：软删除标记（业务删除记号；重复 DELETE 幂等）。"""
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "UPDATE doc_registry SET is_deleted=1, updated_at=? "
                "WHERE tenant_id=? AND doc_key=?",
                (now, tenant_id, doc_key),
            )

    def list_docs(self, tenant_id: str, *, include_deleted: bool = False,
                  limit: int = 50, offset: int = 0) -> tuple[list[DocRecord], int]:
        """文档列表（契约 §2.4b GET /v1/documents）。

        返回 (本页记录, 过滤后总数)。默认**不返回已软删**（前端列表口径），
        `include_deleted=true` 可查全部（审计/回收站口径）。按 updated_at 倒序
        （最近操作的文档在前），doc_key 作稳定次序键避免同秒抖动。
        """
        where = "tenant_id=?" + ("" if include_deleted else " AND is_deleted=0")
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) c FROM doc_registry WHERE {where}", (tenant_id,)
            ).fetchone()["c"]
            rows = conn.execute(
                f"SELECT * FROM doc_registry WHERE {where} "
                "ORDER BY updated_at DESC, doc_key ASC LIMIT ? OFFSET ?",
                (tenant_id, max(0, limit), max(0, offset)),
            ).fetchall()
        return [_row_to_record(r) for r in rows], int(total)

    def set_effective_time(self, tenant_id: str, doc_key: str, effective_time: int) -> None:
        """F2.8：登记该文档的有效期截止（0 = 永久有效）。chunk 元数据由调用方同步翻。"""
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "UPDATE doc_registry SET effective_time=?, updated_at=? "
                "WHERE tenant_id=? AND doc_key=?",
                (int(effective_time), now, tenant_id, doc_key),
            )

    def chunk_exists(self, chunk_id: str) -> bool:
        """chunk 级去重（断点续跑）：查询向量库是否已有该 chunk_id。

        契约 4.7 将 chunk_exists 列为 DocRegistry 方法；M1 的实现在 store 侧
        （VectorStore.contains），本方法提供委托入口便于后续任务级续跑统一调用。
        """
        return self._chunk_exists_fn(chunk_id) if self._chunk_exists_fn else False

    def set_chunk_exists_fn(self, fn: callable) -> None:
        """注入 chunk 存在性检查（由 pipeline 绑定 VectorStore.contains）。"""
        self._chunk_exists_fn = fn

    def close(self) -> None:
        pass


def _row_to_record(row: sqlite3.Row) -> DocRecord:
    keys = row.keys()
    return DocRecord(
        tenant_id=row["tenant_id"], doc_key=row["doc_key"], doc_id=row["doc_id"],
        version=row["version"], status=row["status"], content_hash=row["content_hash"],
        created_at=row["created_at"], updated_at=row["updated_at"], error=row["error"],
        is_deleted=bool(row["is_deleted"]),
        # 兼容极早期库（无 effective_time 列时用 .get 语义兜底）
        effective_time=int(row["effective_time"]) if "effective_time" in keys else 0,
    )
