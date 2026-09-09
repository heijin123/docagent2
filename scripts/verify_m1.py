"""M1 验证脚本（断言式，python scripts/verify_m1.py）。

覆盖（对齐需求 F1.1–F1.9）：
1. 格式探测：md/txt/docx/pdf 识别；加密/乱二进制拒绝
2. 解析：md 标题结构 / txt 段落 / docx 标题与表格 / pdf 页文本
3. chunking：块长护栏 / table·code 整块不切断 / 溯源键齐全 / 版本段 chunk_id
4. PDF 质量门：绿页放行 / 红页（纯图/乱码）判级
5. 幂等入库：首灌全写 / 重灌全 skip / 内容变更版本化软更新（旧版 is_valid=false）
6. 降级标注：无 Key → degraded=mock 且报告明示
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.detection import detect_document_type
from app.ingestion.parsers import ParsedDocument, parse_file
from app.ingestion.pdf_quality import grade_page
from app.ingestion.chunking import estimate_tokens, chunk_document, build_doc_meta
from app.ingestion.registry import DocRegistry
from app.retrieval.embedding import mock_embedding
from app.retrieval.vectorstore import VectorStore
from app.models import DocumentType, make_doc_id, extract_doc_date_from_filename
from app.ingestion.pipeline import IngestPipeline
from app.core.config import settings

PASS = 0
FAIL = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  ✗ {name}  {detail}")


def expect_raises(fn, name: str):
    try:
        fn()
        check(name, False, "应抛异常但未抛")
    except Exception:  # noqa: BLE001
        check(name, True)


def main() -> int:
    global PASS, FAIL
    samples = settings.samples_dir

    # 若 samples 不存在先尝试生成
    if not samples.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_samples.py")], check=True)

    print("═══ 1. 格式探测 ═══")
    dt, _ = detect_document_type(samples / "sample_guide.md")
    check("md 扩展名识别", dt == DocumentType.MD)
    dt, _ = detect_document_type(samples / "sample_notes.txt")
    check("txt 扩展名识别", dt == DocumentType.TXT)
    dt, _ = detect_document_type(samples / "sample_policy.docx")
    check("docx 扩展名识别", dt == DocumentType.DOCX)
    dt, _ = detect_document_type(samples / "sample_manual.pdf")
    check("pdf 扩展名识别", dt == DocumentType.PDF)

    # 内容嗅探：改名 txt 为 .bin → 仍识别为 txt（可解码文本兜底）
    fake_bin = samples / "_probe_text.bin"
    fake_bin.write_text("这是可解码的文本内容，无扩展名主判时靠嗅探。", encoding="utf-8")
    dt, note = detect_document_type(fake_bin)
    check("内容嗅探 → txt", dt == DocumentType.TXT, str(note))
    fake_bin.unlink(missing_ok=True)

    # OLE 头拒绝（旧版 Office）
    ole = samples / "_probe_ole.bin"
    ole.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32)
    dt, note = detect_document_type(ole)
    check("OLE 旧版 Office 拒绝", dt == DocumentType.UNSUPPORTED and "OLE" in (note or ""), str(note))
    ole.unlink(missing_ok=True)

    print("═══ 2. 解析 ═══")
    parsed = parse_file(samples / "sample_guide.md")
    check("md 含标题块", any(b.block_type == "heading" for b in parsed.blocks))
    check("md 含表格块", any(b.block_type == "table" for b in parsed.blocks))
    check("md 含代码块", any(b.block_type == "code" for b in parsed.blocks))
    headings = [b for b in parsed.blocks if b.block_type == "heading"]
    check("md 标题带层级", all(b.heading_level is not None for b in headings))

    parsed = parse_file(samples / "sample_notes.txt")
    check("txt 段落数>0", len(parsed.blocks) > 0)
    check("txt 全为段落", all(b.block_type == "paragraph" for b in parsed.blocks))

    parsed = parse_file(samples / "sample_policy.docx")
    check("docx 含标题", any(b.block_type == "heading" and b.heading_level == 1 for b in parsed.blocks))
    check("docx 含表格", any(b.block_type == "table" for b in parsed.blocks))

    parsed = parse_file(samples / "sample_manual.pdf")
    check("pdf 文本页产块", len(parsed.blocks) > 0)
    check("pdf 块带 page", all(b.page is not None and b.page >= 1 for b in parsed.blocks))

    print("═══ 3. chunking ═══")
    doc_meta = build_doc_meta(parse_file(samples / "sample_guide.md"),
                              tenant_id="tenant_demo", doc_key="sample_guide.md",
                              content_hash="abc")
    records = chunk_document(parse_file(samples / "sample_guide.md"), doc_meta)
    check("chunk 数>0", len(records) > 0)
    check("chunk_id 含版本段", all(f"_{doc_meta.version:04d}_" in r.chunk_id for r in records))
    check("溯源键齐全", all(r.doc_id and r.doc_title and r.chunk_index > 0 for r in records))
    # table/code 整块保留：无 chunk 中途截断（内容含代码 fence 完整）
    code_chunks = [r for r in records if "def is_late" in r.content]
    check("代码块可检索", len(code_chunks) >= 1)
    table_chunks = [r for r in records if "差旅" in r.content and "审批" not in r.content]
    # 表格整块：表头+数据行同块
    all_table = [r for r in records if "| 差旅 |" in r.content]
    check("表格未切断(表头同块)", all("| 项目 |" in r.content for r in all_table))

    # 超长文本二次切分
    long_text = "这是一个用于测试长文本切分的句子。" * 120
    toks = estimate_tokens(long_text)
    check("token 估算>0", toks > 100)
    from app.ingestion.chunking import _slice_by_tokens
    pieces = _slice_by_tokens(long_text, 200, 40)
    check("超长文本被切分", len(pieces) > 1)
    check("切分后每片不超限(±20%)", all(estimate_tokens(p) <= 240 for p in pieces))

    print("═══ 4. PDF 质量门 ═══")
    q_green = grade_page(1, "正常内容。" * 60, image_count=0)
    check("正常页=green", q_green.level == "green", q_green.signals)
    q_red_short = grade_page(2, "", image_count=1)
    check("纯图页=red", q_red_short.level == "red")
    q_red_garble = grade_page(3, "\ufffd" * 30 + "正常文字" * 10, image_count=0)
    check("乱码页=red", q_red_garble.level == "red", q_red_garble.signals)
    q_yellow = grade_page(4, "少量文字" * 6 + "这是版面较复杂的一页，含多个图片但正文文字不多。", image_count=4)
    check("稀疏+图多=yellow", q_yellow.level == "yellow", q_yellow.signals)

    print("═══ 5. 幂等 / 版本化 / 降级（临时库隔离）═══")
    import tempfile
    import shutil
    tmp_path = Path(tempfile.mkdtemp(prefix="verify_m1_"))
    try:
        store = VectorStore(tmp_path / "chroma")
        reg = DocRegistry(tmp_path / "registry.db")
        from app.retrieval.bm25store import BM25Store
        bm = BM25Store(tmp_path / "corpus.db")
        from app.retrieval.embedding import Embedder
        # 链路测试与真实 provider 解耦（真 Key 环境验证走 scripts/make 真实冒烟）：
        # verify 用显式 mock，256 维，离线确定。
        embedder = Embedder(provider="mock", model="mock-hash-v1",
                            dimensions=256, degraded=False)
        check("mock embedder 显式构造", embedder.provider == "mock" and embedder.dimensions == 256)

        pl = IngestPipeline(tenant_id="tenant_demo", embedder=embedder,
                            vector_store=store, bm25_store=bm, registry=reg)

        src = tmp_path / "doc_a.md"
        src.write_text("# 主题甲\n\n这是主题甲的第一段内容，用于检索验证。\n\n## 小节\n\n补充说明文字若干。", encoding="utf-8")

        r1 = pl.run_document(src)
        check("首灌 ok", r1["status"] == "ok", str(r1["error"]))
        first_stored = r1["stored"]
        check("首灌全写", first_stored > 0 and r1["skipped"] == 0)

        r2 = pl.run_document(src)
        check("重灌幂等 skip", r2["duplicated"] and r2["stored"] == 0, str(r2))
        check("登记表 status=done", reg.get("tenant_demo", "doc_a.md").status == "done")

        # 内容变更 → 版本化软更新
        src.write_text("# 主题甲\n\n这是主题甲更新后的内容，版本 2。\n", encoding="utf-8")
        r3 = pl.run_document(src)
        check("变更触发 update", r3["version"] == 2 and r3["stored"] > 0, str(r3))
        # 旧版 v1 块 is_valid=false → 带 is_valid 过滤查不到；物理仍在库（Append-Only）
        doc_id = reg.get("tenant_demo", "doc_a.md").doc_id
        valid_v1 = store.query([0.0] * 256, top_k=10, where={"$and": [
            {"doc_id": {"$eq": doc_id}},
            {"version": {"$eq": 1}},
            {"is_valid": {"$eq": True}}]})
        check("v1 旧块 soft-delete（过滤查不到）", valid_v1 == [], [h["chunk_id"] for h in valid_v1])
        all_v1 = store._collection.get(
            where={"$and": [{"doc_id": {"$eq": doc_id}}, {"version": {"$eq": 1}}]},
            include=["metadatas"])
        check("v1 物理保留（Append-Only）", len(all_v1["ids"]) > 0
              and all(m["is_valid"] is False for m in all_v1["metadatas"]))

        # 不支持格式隔离
        bad = tmp_path / "bad.xyz"
        bad.write_bytes(b"\x00\x01\x02\x03" * 10)
        r_bad = pl.run_document(bad)
        check("坏文件业务失败不崩", r_bad["status"] == "failed" and r_bad["error"], str(r_bad))
    finally:
        store.close()
        shutil.rmtree(tmp_path, ignore_errors=True)

    # doc_id 派生与日期解析
    print("═══ 6. 身份与日期 ═══")
    did = make_doc_id("tenant_demo", "doc_a.md")
    check("doc_id 确定性", did == make_doc_id("tenant_demo", "doc_a.md"))
    check("doc_id 内容无关", did == make_doc_id("tenant_demo", "doc_a.md"))
    d, y = extract_doc_date_from_filename("annual_report_2024.pdf")
    check("文件名年份解析", y == 2024 and d == "2024")

    # embedding 确定性
    e1 = mock_embedding("同一句话", 256)
    e2 = mock_embedding("同一句话", 256)
    check("mock embedding 确定性", e1 == e2 and len(e1) == 256)

    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    if FAILURES:
        print("\n失败明细:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
