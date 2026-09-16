"""入库编排（M1：摄取端核心流程，需求 F1.1–F1.9 + F1.8 幂等）。

流程：detect → parse（含质量门）→ [vlm：红页整页 OCR + 非红页内嵌图 OCR] → chunk → embed → 双索引写入 → 报告。

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
import os
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import setup_logging
from app.ingestion.chunking import build_doc_meta, chunk_document
from app.ingestion.parsers import ParsedDocument, UnsupportedFormatError, parse_file
from app.ingestion.registry import DocRegistry
from app.ingestion.vlm import get_transcriber
from app.models import Block, DocumentType
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

        # ── 质量门统计 + VLM（F1.9：PDF 红页整页 OCR + PDF/DOCX 内嵌图 OCR）──
        red_pages = [p.page_no for p in parsed.page_qualities if p.level == "red"]
        yellow_pages = [p.page_no for p in parsed.page_qualities if p.level == "yellow"]
        report["red_pages"] = red_pages
        report["yellow_pages"] = yellow_pages

        from app.ingestion.vlm import NoneTranscriber

        transcriber = get_transcriber()
        if isinstance(transcriber, NoneTranscriber):
            if red_pages:
                # 无 Key / 未启用 → 红页原文降级入库（R7 不崩），报告明示
                report["vlm_note"] = (
                    f"红页 {red_pages} 无 VLM 转录（未配 Key 或未启用 VLM_ENABLED=1）→ 原文降级入库"
                )
                report["warnings"].extend(
                    f"第 {p} 页红色判级，原文不可信（已降级入库）" for p in red_pages)
            # 无 VLM 时内嵌图直接跳过（图内文字暂不抽取，符合当前能力边界）
        elif parsed.doc_type == DocumentType.DOCX:
            # DOCX：内嵌图 OCR 原位替换 [图片] 占位块（段落顺序天然对齐，无需页码）
            docx_imgs = self._transcribe_docx_images(path, parsed)
            if docx_imgs:
                report["vlm_note"] = f"{docx_imgs} 个内嵌图已 VLM OCR 为文本块"
        else:
            # PDF：红页整页转录 + 非红页内嵌图 OCR，两类文本块按页码归位
            transcripts_by_page: dict[int, Block] = {}
            if red_pages:
                for b in self._transcribe_red_pages(path, parsed, red_pages):
                    transcripts_by_page[b.page] = b
            inline_by_page = self._transcribe_inline_images(path, parsed, red_pages)

            if transcripts_by_page or inline_by_page:
                self._merge_vlm_text_blocks(parsed, transcripts_by_page, inline_by_page)

                red_n = len(transcripts_by_page)
                img_n = sum(len(v) for v in inline_by_page.values())
                parts = []
                if red_n:
                    parts.append(f"{red_n} 个红页已 VLM 转录")
                if img_n:
                    parts.append(f"{img_n} 个内嵌图已 VLM OCR")
                report["vlm_note"] = "；".join(parts) + " 为文本块"
            elif red_pages:
                report["vlm_note"] = "VLM 转录失败/无输出 → 红页保持原文降级入库"
                report["warnings"].extend(
                    f"第 {p} 页红色判级，原文不可信（转录失败已降级）" for p in red_pages)

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
        """M2：渲染红页为 PNG → VLM 转录为文本块（带正确页码）。返回转录块列表。

        红页（扫描/乱码）PyMuPDF 几乎抽不出可信文本；改用 PyMuPDF 渲染页面为图片，
        交给 Qwen-VL 做 OCR。渲染长边控制在 1280px 以内（控 token/体积，且落在 Qwen-VL
        像素上限内）。转录文本作为 paragraph 块追加，page=原页码，使后续 chunk 的
        page_num 溯源正确（不再指错页）。
        """
        from app.ingestion.vlm import NoneTranscriber

        transcriber = get_transcriber()
        if isinstance(transcriber, NoneTranscriber):
            return []  # 无 Key/未启用 → 编排层降级原文

        import pymupdf
        try:
            doc = pymupdf.open(str(path))
        except Exception:  # noqa: BLE001
            return []

        blocks: list = []
        rendered = 0
        try:
            for page_no in red_pages:
                if rendered >= transcriber.max_pages:
                    break
                try:
                    page = doc[page_no - 1]
                    # 长边 ≤ 1280px：zoom = 1280 / max(w,h)（页尺寸单位 point，72dpi）
                    zoom = 1280.0 / max(page.rect.width, page.rect.height)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                    png = pix.tobytes("png")
                except Exception:  # noqa: BLE001 — 单页渲染失败跳过，不阻断
                    continue
                text = transcriber.transcribe(png, page_no)
                if text:
                    blocks.append(Block(
                        block_type="paragraph",
                        text=text,
                        page=page_no,
                        metadata={"source": "vlm_ocr", "quality_level": "transcribed_red"},
                    ))
                    rendered += 1
        finally:
            doc.close()
        return blocks

    def _transcribe_inline_images(self, path: Path, parsed: ParsedDocument,
                                  red_pages: list[int]) -> dict[int, list[tuple[int, Block]]]:
        """M2：抽取**非红页**内嵌图 → Qwen-VL OCR → 图内文字块（**按图位置锚定插入位**）。

        背景：markitdown 对 PDF 内嵌图不产块（解析层只认 `![...](http...)` 外链），
        图文混排页里的图表/带字示意图文字会丢；本方法用 PyMuPDF 直接抽图并 OCR 补回。
        红页已被整页 OCR 覆盖，故跳过，避免重复计费与噪声。

        版面对齐：用 PyMuPDF `get_image_rects` 取图在页内纵坐标，与「图上方最近的文本块」
        比对，把 OCR 块插到该图在阅读顺序中的位置（而非一律页尾）；锚不到则退回页尾。

        成本护栏：跳过 < 64×64 小图（logo/图标）；同一 xref 去重；总量上限
        VLM_MAX_INLINE_IMAGES（默认 20）。
        返回 {page_no: [(插入位序, Block), ...]}（插入位序=在该页已有块中应插在第几个之前）。
        """
        from app.ingestion.vlm import NoneTranscriber

        transcriber = get_transcriber()
        if isinstance(transcriber, NoneTranscriber):
            return {}  # 无 Key/未启用 → 内嵌图不抽取

        import pymupdf
        try:
            doc = pymupdf.open(str(path))
        except Exception:  # noqa: BLE001
            return {}

        max_inline = int(os.getenv("VLM_MAX_INLINE_IMAGES", "20"))
        result: dict[int, list[tuple[int, Block]]] = {}
        seen_xrefs: set[int] = set()
        total = 0
        inline_prompt = (
            "转录这张图片中的全部文字内容，保留原有结构，用 Markdown 输出。"
            "若图片中没有可读文字（纯图形/照片），只输出「（无文字）」。"
        )
        try:
            for page_no in range(1, doc.page_count + 1):
                if total >= max_inline:
                    break
                if page_no in red_pages:
                    continue  # 红页已整页 OCR
                try:
                    page = doc[page_no - 1]
                    images = page.get_images(full=True)
                except Exception:  # noqa: BLE001 — 单页取图失败跳过
                    continue
                page_blocks = [b for b in parsed.blocks
                               if b.page == page_no
                               and b.metadata.get("quality_level") != "red"]
                spans = self._page_text_spans(page)
                for img in images:
                    if total >= max_inline:
                        break
                    xref = img[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        info = doc.extract_image(xref)
                    except Exception:  # noqa: BLE001
                        continue
                    if (info.get("width") or 0) < 64 or (info.get("height") or 0) < 64:
                        continue  # logo/小图标
                    img_bytes = info.get("image")
                    if not img_bytes:
                        continue
                    ext = (info.get("ext") or "png").lower()
                    mime = f"image/{ext}" if ext in {
                        "png", "jpeg", "jpg", "gif", "webp", "bmp"} else "image/png"
                    text = transcriber.transcribe(
                        img_bytes, page_no, mime=mime, prompt=inline_prompt)
                    if not text:
                        continue
                    clean = text.strip()
                    if not clean or ("无文字" in clean and len(clean) <= 10):
                        continue  # 装饰图/纯图形 → 不产块
                    blk = Block(
                        block_type="paragraph",
                        text=clean,
                        page=page_no,
                        metadata={"source": "vlm_ocr_image", "img_ext": ext},
                    )
                    pos = self._inline_insert_pos(page, xref, page_blocks, spans)
                    result.setdefault(page_no, []).append((pos, blk))
                    total += 1
        finally:
            doc.close()
        return result

    @staticmethod
    def _page_text_spans(page) -> list[tuple[float, float, str]]:
        """PyMuPDF 文本块 (y0, y1, text) 列表，用于把图片锚定到阅读顺序位置。"""
        spans: list[tuple[float, float, str]] = []
        try:
            for b in page.get_text("blocks"):
                # (x0,y0,x1,y1,text,block_no,block_type)；block_type 0=文本
                if len(b) >= 7 and b[6] == 0 and str(b[4]).strip():
                    spans.append((float(b[1]), float(b[3]), str(b[4])))
        except Exception:  # noqa: BLE001
            return []
        return spans

    @staticmethod
    def _inline_insert_pos(page, xref: int, page_blocks: list[Block],
                           spans: list[tuple[float, float, str]]) -> int:
        """内嵌图 OCR 块在该页块序中的插入位：按「图上方的最近文本」锚定。

        - 图在页首（上方无文本）→ 0；
        - 找到图上方最近的文本块 → 取其前 16 字为锚句，在页内块中定位包含它的块，插其后；
        - 锚不到（无文本层/文本不走 markdown 块）→ 页尾（保底等价旧行为）。
        """
        import re as _re

        def _norm(s: str) -> str:
            return _re.sub(r"\s+", "", s or "")

        try:
            rects = page.get_image_rects(xref)
            y_top = min(r.y0 for r in rects) if rects else None
        except Exception:  # noqa: BLE001
            y_top = None
        if y_top is None:
            return len(page_blocks)
        above = [s for s in spans if s[1] <= y_top + 1.0]
        if not above:
            return 0
        anchor_text = max(above, key=lambda s: s[1])[2]
        snippet = _norm(anchor_text)[:16]
        if len(snippet) < 4:
            return len(page_blocks)
        idx = -1
        for i, b in enumerate(page_blocks):
            if snippet in _norm(b.text):
                idx = i
        return idx + 1 if idx >= 0 else len(page_blocks)

    def _merge_vlm_text_blocks(self, parsed: ParsedDocument,
                               transcripts_by_page: dict[int, Block],
                               inline_by_page: dict[int, list[tuple[int, Block]]]) -> None:
        """把红页转录块 + 内嵌图 OCR 块归位：红页原文块剔除，OCR 块按锚定插入位入列。

        红页转录为整页文本，追加到该页末尾；内嵌图 OCR 块按 `_inline_insert_pos` 给出的
        插入位序插到页内对应位置（图文版面对齐）。原地改写 parsed.blocks。
        """
        non_red = [b for b in parsed.blocks
                   if b.metadata.get("quality_level") != "red"]
        merged: list = []
        i, n = 0, len(non_red)
        while i < n:
            cur_page = non_red[i].page
            page_blocks: list = []
            j = i
            while j < n and non_red[j].page == cur_page:
                page_blocks.append(non_red[j])
                j += 1
            # 内嵌图 OCR 块按插入位序分桶（位序=插在该页第几个块之前）
            buckets: dict[int, list] = {}
            for pos, blk in inline_by_page.get(cur_page, []):
                k = max(0, min(pos, len(page_blocks)))
                buckets.setdefault(k, []).append(blk)
            for p in range(len(page_blocks) + 1):
                merged.extend(buckets.get(p, []))
                if p < len(page_blocks):
                    merged.append(page_blocks[p])
            if cur_page in transcripts_by_page:
                merged.append(transcripts_by_page[cur_page])
            i = j
        parsed.blocks = merged

    def _transcribe_docx_images(self, path: Path, parsed: ParsedDocument) -> int:
        """M2：OCR DOCX 内联图并**原位替换** `[图片]` 占位块（段落顺序天然对齐）。

        DOCX 无页概念、但块序=段落阅读顺序，故直接就地改写占位块文本即可对齐（不像 PDF 需
        按纵坐标锚定）。图片字节取自关系部件（rId → related_parts[rid].blob）；OCR 文本非空
        则把占位块改写为 paragraph（source=vlm_ocr_docx）；无文字/失败/取不到字节则丢弃该
        占位块（"[图片]" 无检索价值，留着只是噪声）。返回成功 OCR 的图数。
        """
        from app.ingestion.vlm import NoneTranscriber

        transcriber = get_transcriber()
        if isinstance(transcriber, NoneTranscriber):
            return 0  # 无 Key/未启用 → 保留占位（不改动）

        try:
            import docx
            document = docx.Document(str(path))
        except Exception:  # noqa: BLE001
            return 0

        media: dict[str, tuple[bytes, str]] = {}
        for rid, part in getattr(document.part, "related_parts", {}).items():
            ct = str(getattr(part, "content_type", "") or "").lower()
            if not ct.startswith("image/"):
                continue
            blob = getattr(part, "blob", None)
            if not blob:
                continue
            mime = ct if ct in {
                "image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"} else "image/png"
            media[rid] = (blob, mime)

        inline_prompt = (
            "转录这张图片中的全部文字内容，保留原有结构，用 Markdown 输出。"
            "若图片中没有可读文字（纯图形/照片），只输出「（无文字）」。"
        )
        n = 0
        out: list = []
        for b in parsed.blocks:
            rid = b.metadata.get("img_rid") if b.block_type == "image" else None
            item = media.get(rid) if rid else None
            if item is None:
                if b.block_type == "image" and b.metadata.get("has_inline_image"):
                    continue  # 占位块但取不到图片字节 → 丢弃（避免 "[图片]" 噪声入索引）
                out.append(b)
                continue
            blob, mime = item
            text = transcriber.transcribe(blob, 0, mime=mime, prompt=inline_prompt)
            clean = (text or "").strip()
            if clean and not ("无文字" in clean and len(clean) <= 10):
                out.append(Block(block_type="paragraph", text=clean, page=b.page,
                                 metadata={"source": "vlm_ocr_docx"}))
                n += 1
            # 无文字 → 丢弃占位块（不产空块）
        parsed.blocks = out
        return n


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
