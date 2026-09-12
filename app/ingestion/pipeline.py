"""入库编排（M1：摄取端核心流程，需求 F1.1–F1.9 + F1.8 幂等）。

流程：detect → parse（含质量门）→ [vlm 红页转录] → chunk → embed → 双索引写入 → 报告。

幂等分流（DocRegistry）：
- 新 doc_key → 新建（version=1）
- 同 doc_key 同 hash 且 done → 幂等跳过（报告 duplicated）
- 同 doc_key 同 hash processing → 冲突（跳过，不重复启动）
- 同 doc_key 同 hash failed → 允许重试（断点续跑：chunk 级 contains 跳过）
- 同 doc_key 异 hash → 版本化软更新（version+1 新插，旧版本翻 is_valid=false，F1.7）

错误隔离：单文档失败不中断批次（failed → 报告带 error）。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import setup_logging
from app.ingestion.chunking import build_doc_meta, chunk_document
from app.ingestion.parsers import ParsedDocument, UnsupportedFormatError, parse_file
from app.ingestion.registry import DocRegistry
from app.ingestion.vlm import get_transcriber
from app.retrieval.bm25store import BM25Store
from app.retrieval.embedding import Embedder, build_embedder
from app.retrieval.vectorstore import VectorStore

log = setup_logging("pipeline")


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class IngestPipeline:
    def __init__(
        self,
        *,
        tenant_id: str | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        registry: DocRegistry | None = None,
    ):
        settings.ensure_dirs()
        self.tenant_id = tenant_id or settings.default_tenant_id
        self.embedder = embedder or build_embedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.registry = registry or DocRegistry()

    # ── 幂等分流（F1.8）────────────────────────────────────
    def _decide(self, doc_key: str, content_hash: str, doc_id: str,
                *, force: bool = False) -> dict:
        """返回分流决策：{action: new|skip|retry|update|conflict, version, record}。

        force=True（--rebuild）：跳过「同 hash → skip」短路，强制版本化重灌
        （version+1，得到全新 chunk_id）。用于文件未变但需重新解析的场景
        （如 VLM 升级后补红页转录）；否则正常分流会走 skip 直接返回，rebuild 永不生效。
        """
        existing = self.registry.get(self.tenant_id, doc_key)
        if existing is None:
            rec = self.registry.reserve(self.tenant_id, doc_key, doc_id, content_hash)
            return {"action": "new", "version": rec.version, "record": rec}

        if force:
            if existing.status == "processing":
                return {"action": "conflict", "version": existing.version, "record": existing}
            rec = self.registry.begin_version(self.tenant_id, doc_key, doc_id, content_hash)
            return {"action": "update", "version": rec.version, "record": rec}

        if existing.content_hash == content_hash:
            if existing.is_deleted:
                # 软删（F5.7 DELETE）后重传：版本化重入（v+1 全新入库，恢复检索可见）
                rec = self.registry.begin_version(
                    self.tenant_id, doc_key, doc_id, content_hash)
                return {"action": "update", "version": rec.version, "record": rec}
            if existing.status == "done":
                return {"action": "skip", "version": existing.version, "record": existing}
            if existing.status == "processing":
                return {"action": "conflict", "version": existing.version, "record": existing}
            # failed → 重试（同 version 断点续跑）
            self.registry.mark(self.tenant_id, doc_key, "processing")
            return {"action": "retry", "version": existing.version, "record": existing}

        # 同 key 异 hash → 版本化软更新
        rec = self.registry.begin_version(self.tenant_id, doc_key, doc_id, content_hash)
        return {"action": "update", "version": rec.version, "record": rec}

    # ── 单文档执行 ─────────────────────────────────────────
    def run_document(self, file_path: str | Path, *, doc_key: str | None = None,
                     rebuild: bool = False, progress_cb=None,
                     doc_meta_extra: dict | None = None) -> dict:
        """摄取单个文件，返回逐文档报告（业务失败也返回报告 dict，不外抛）。

        progress_cb(phase: str, percent: int)：M4 上传任务进度回写（契约 §2.3）；
        doc_meta_extra：API meta 注入（department/category，契约 §2.2），不进 report。
        """
        path = Path(file_path)
        report = {
            "filename": path.name,
            "status": "ok", "format": None, "doc_id": None, "version": None,
            "chunks_created": 0, "stored": 0, "skipped": 0, "duplicated": False,
            "red_pages": [], "yellow_pages": [], "vlm_note": None,
            "provider": self.embedder.provider, "degraded": self.embedder.degraded,
            "error": None, "elapsed_s": None, "warnings": [],
        }
        t0 = time.time()
        try:
            return self._run_document_inner(
                path, doc_key, rebuild, report, progress_cb, doc_meta_extra)
        except Exception as exc:  # noqa: BLE001 — 单文档失败不拖垮批次
            report["status"] = "failed"
            report["error"] = str(exc)
            report["elapsed_s"] = round(time.time() - t0, 2)
            log.warning("文档摄取失败 %s: %s", path.name, exc)
            return report

    def _run_document_inner(self, path: Path, doc_key: str | None, rebuild: bool,
                            report: dict, progress_cb=None,
                            doc_meta_extra: dict | None = None) -> dict:
        t0 = time.time()

        def _cb(phase: str, percent: int) -> None:
            if progress_cb:
                try:
                    progress_cb(phase, percent)
                except Exception:  # noqa: BLE001 — 进度回写失败不阻断摄取
                    log.debug("progress_cb 回写异常: %s", phase)

        # ── detect + parse ──
        try:
            parsed = parse_file(path)
        except UnsupportedFormatError as exc:
            raise RuntimeError(exc.reason) from exc
        _cb("parse", 30)

        report["format"] = parsed.doc_type.value
        content_hash = _file_hash(path)
        doc_key = doc_key or path.name

        from app.models import make_doc_id
        doc_id = make_doc_id(self.tenant_id, doc_key)
        report["doc_id"] = doc_id

        # ── 幂等分流 ──
        decision = self._decide(doc_key, content_hash, doc_id, force=rebuild)
        version = decision["version"]
        report["version"] = version

        if decision["action"] == "skip":
            report["duplicated"] = True
            report["status"] = "ok"
            report["elapsed_s"] = round(time.time() - t0, 2)
            log.info("幂等跳过 %s（同 hash 已完成，version=%s）", path.name, version)
            return report
        if decision["action"] == "conflict":
            report["status"] = "failed"
            report["error"] = "INGEST_IN_PROGRESS: 同 doc_key 任务处理中"
            report["elapsed_s"] = round(time.time() - t0, 2)
            return report

        # ── 质量门统计 + 红页 VLM（F1.9）──
        red_pages = [p.page_no for p in parsed.page_qualities if p.level == "red"]
        yellow_pages = [p.page_no for p in parsed.page_qualities if p.level == "yellow"]
        report["red_pages"] = red_pages
        report["yellow_pages"] = yellow_pages

        transcriber = get_transcriber()
        if red_pages:
            from app.ingestion.vlm import NoneTranscriber

            if isinstance(transcriber, NoneTranscriber):
                # 无 Key / 未启用 → 红页原文降级入库（R7 不崩），报告明示
                report["vlm_note"] = (
                    f"红页 {red_pages} 无 VLM 转录（未配 Key 或未启用 VLM_ENABLED=1）→ 原文降级入库"
                )
                report["warnings"].extend(f"第 {p} 页红色判级，原文不可信（已降级入库）" for p in red_pages)
            else:
                # VLM 转录路径：红页原文不产块，转录文本追加为 figure_transcript（M2 接真实渲染）
                transcripts = self._transcribe_red_pages(path, parsed, red_pages)
                if transcripts:
                    parsed.blocks = [b for b in parsed.blocks
                                     if b.metadata.get("quality_level") != "red"] + transcripts
                    report["vlm_note"] = f"{len(transcripts)} 个红页已 VLM 转录为 figure_transcript 块"
                else:
                    report["vlm_note"] = "VLM 转录失败/无输出 → 红页保持原文降级入库"
                    report["warnings"].extend(f"第 {p} 页红色判级，原文不可信（转录失败已降级）" for p in red_pages)

        # ── chunk ──
        doc_meta = build_doc_meta(
            parsed,
            tenant_id=self.tenant_id,
            doc_key=doc_key,
            doc_title=path.stem,
            version=version,
            content_hash=content_hash,
            department=(doc_meta_extra or {}).get("department"),
            category=(doc_meta_extra or {}).get("category"),
        )
        records = chunk_document(parsed, doc_meta)
        report["chunks_created"] = len(records)
        if not records:
            self.registry.mark(self.tenant_id, doc_key, "done")
            report["elapsed_s"] = round(time.time() - t0, 2)
            log.warning("%s 解析后无块可入库（可能全为红页且无转录）", path.name)
            return report
        _cb("chunk", 50)

        # ── embed + 双索引写入（先向量后 BM25，F1.4/4.8）──
        texts = [r.content for r in records]
        _cb("embed", 60)
        embeddings = self.embedder.embed_texts(texts)
        _cb("embed", 85)

        # chunk 级幂等：vector 侧 contains 跳过（断点续跑不重嵌）
        stored = self.vector_store.add(records, embeddings)
        self.bm25_store.add(records)

        # 版本化软更新：先插新版本，再翻旧版本 is_valid=false（避免检索空窗，F1.7）。
        # 只在 action=="update" 时翻旧版本；切勿用 `or rebuild` 把 version 本身也翻掉——
        # chunk_id = doc_id_version_index，同版本重灌会连刚写入的新块一起失效（检索全空）。
        if decision["action"] == "update":
            old_version = version - 1
            flipped_v = self.vector_store.soft_delete_doc(doc_id, old_version)
            self.bm25_store.soft_delete_doc(doc_id, old_version)
            if flipped_v:
                report["warnings"].append(f"旧版本 v{old_version} 已软删除 {flipped_v} 块")

        self.bm25_store.rebuild()
        _cb("index", 100)

        # 登记 done（幂等语义：done 后同 hash 重传 → skip）
        self.registry.mark(self.tenant_id, doc_key, "done")

        report["stored"] = stored
        report["skipped"] = len(records) - stored
        report["status"] = "ok"
        report["elapsed_s"] = round(time.time() - t0, 2)
        log.info("入库完成 %s v%s: %s 块（stored=%s skipped=%s）",
                 path.name, version, len(records), stored, len(records) - stored)
        return report

    def _transcribe_red_pages(self, path: Path, parsed: ParsedDocument,
                              red_pages: list[int]) -> list:
        """M1 骨架：真实渲染 + VLM 转录在 M2 接入（需配 Key）。返回转录块列表。"""
        # 占位：M1 不渲染页面，直接返回空 → 走降级原文入库
        return []


def run_ingest(paths: list[str | Path], *, rebuild: bool = False,
               pipeline: IngestPipeline | None = None) -> list[dict]:
    """批量摄取文件/目录（目录递归找支持格式，串行；单文档失败不拖垮批次）。"""
    pl = pipeline or IngestPipeline()
    files = _collect_files(paths)
    reports = [pl.run_document(f, rebuild=rebuild) for f in files]
    return reports


def _collect_files(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    from app.models import SUPPORTED_EXTENSIONS

    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    out.append(f)
        elif path.is_file():
            out.append(path)
        else:
            log.warning("路径不存在，跳过: %s", path)
    return out
