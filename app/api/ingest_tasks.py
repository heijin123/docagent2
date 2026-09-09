"""M4 文档入库后台任务（需求 F6.5，契约 §2.3）。

- ThreadPoolExecutor **单写者**串行入库（Chroma 并发写限制，风险清单第 3 行）；
- 任务状态机 pending → processing(phase: parse/chunk/embed/index + percent) → done/failed；
- in-flight 键 (tenant_id, doc_key) 用于上传端点 409 INGEST_IN_PROGRESS 判定；
- 每任务结束释放 in-flight；进程内保留最近任务供 /v1/tasks/{id} 查询
  （持久登记见 DocRegistry status——pipeline 内部维护）。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from app.core.observability import TimedSpan, log_ingest
from app.models.api import TaskProgress, TaskStatus, TaskWarning

logger = logging.getLogger(__name__)

_MAX_KEEP_TASKS = 200


@dataclass
class TaskRecord:
    task_id: str
    tenant_id: str
    doc_key: str
    doc_id: str
    file_path: Path
    status: str = "pending"
    phase: str = "parse"
    percent: int = 0
    chunks_done: int = 0
    warnings: list[TaskWarning] = field(default_factory=list)
    error: dict | None = None
    duplicated: bool = False
    report: dict | None = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def to_status(self) -> TaskStatus:
        return TaskStatus(
            task_id=self.task_id,
            doc_id=self.doc_id,
            status=self.status,
            progress=TaskProgress(phase=self.phase, percent=self.percent,
                                  chunks_done=self.chunks_done),
            warnings=self.warnings,
            error=self.error,
            duplicated=self.duplicated,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class TaskManager:
    def __init__(self, make_pipeline, max_workers: int = 1):
        self._make_pipeline = make_pipeline
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="ingest")
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskRecord] = {}
        self._inflight: set[tuple[str, str]] = set()
        self._seq = 0

    # ── 查询 ──────────────────────────────────────────────
    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def is_inflight(self, tenant_id: str, doc_key: str) -> bool:
        with self._lock:
            return (tenant_id, doc_key) in self._inflight

    def shutdown(self, wait: bool = False) -> None:
        """应用关闭：停止接收新任务，等待（可选）在跑任务结束。"""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    # ── 提交 ──────────────────────────────────────────────
    def submit(self, *, tenant_id: str, doc_key: str, doc_id: str,
               file_path: Path) -> TaskRecord:
        with self._lock:
            key = (tenant_id, doc_key)
            if key in self._inflight:
                from app.api.errors import ApiError

                raise ApiError(
                    "INGEST_IN_PROGRESS",
                    f"同文档（doc_key={doc_key}）正在处理中，请稍后查询任务状态",
                    409)
            self._seq += 1
            rec = TaskRecord(
                task_id=f"ingest_{int(time.time())}_{self._seq}",
                tenant_id=tenant_id, doc_key=doc_key, doc_id=doc_id,
                file_path=file_path)
            self._tasks[rec.task_id] = rec
            self._inflight.add(key)
            self._prune()
        self._executor.submit(self._run, rec)
        return rec

    def _prune(self) -> None:
        # 进程内只保留最近任务（持久登记在 DocRegistry）
        if len(self._tasks) > _MAX_KEEP_TASKS:
            for tid in list(self._tasks)[: len(self._tasks) - _MAX_KEEP_TASKS]:
                self._tasks.pop(tid, None)

    # ── 执行（executor 线程）───────────────────────────────
    def _run(self, rec: TaskRecord) -> None:
        def _cb(phase: str, percent: int) -> None:
            with self._lock:
                rec.phase = phase
                rec.percent = percent
                rec.updated_at = int(time.time())

        with self._lock:
            rec.status = "processing"
            rec.phase = "parse"
            rec.percent = 5
            rec.updated_at = int(time.time())
        try:
            span = TimedSpan(name="ingest_slow").attr(doc_id=rec.doc_id)
            pl = self._make_pipeline(rec.tenant_id)
            report = pl.run_document(rec.file_path, doc_key=rec.doc_key,
                                     progress_cb=_cb)
            duration_ms = span.stop(log_slow=False)
            log_ingest("", duration_ms, doc_id=rec.doc_id,
                       chunks=int(report.get("stored") or 0),
                       status=report.get("status", ""))
            with self._lock:
                rec.report = report
                rec.duplicated = bool(report.get("duplicated"))
                rec.warnings = _report_warnings(report)
                rec.chunks_done = int(report.get("stored") or 0)
                if report["status"] == "ok":
                    rec.status = "done"
                    rec.phase = "index"
                    rec.percent = 100
                else:
                    rec.status = "failed"
                    err = str(report.get("error") or "入库失败")
                    code = err.split(":", 1)[0].strip() if ":" in err else "INGEST_FAILED"
                    if err.startswith("INGEST_IN_PROGRESS"):
                        code = "INGEST_IN_PROGRESS"
                    rec.error = {"code": code, "message": err}
                    rec.percent = 0
        except Exception as exc:  # noqa: BLE001 — 任务级兜底
            logger.exception("任务异常 %s: %s", rec.task_id, exc)
            with self._lock:
                rec.status = "failed"
                rec.error = {"code": "INTERNAL_ERROR", "message": str(exc)}
        finally:
            with self._lock:
                rec.updated_at = int(time.time())
                self._inflight.discard((rec.tenant_id, rec.doc_key))


def _report_warnings(report: dict) -> list[TaskWarning]:
    """pipeline 报告 warnings（str 列表）+ vlm_note/质量统计 → 契约 TaskWarning 形状。"""
    out: list[TaskWarning] = []
    for w in report.get("warnings", []) or []:
        text = str(w)
        page = None
        # pipeline 告警常见 "第 N 页 ..."
        import re

        m = re.search(r"第\s*(\d+)\s*页", text)
        if m:
            page = int(m.group(1))
        out.append(TaskWarning(type="quality_warning" if page else "parse_warning",
                               page=page, detail=text))
    for p in report.get("red_pages", []) or []:
        out.append(TaskWarning(type="red_page", page=p,
                               detail=f"第 {p} 页红色判级（原文不可信，见 vlm_note）"))
    return out
