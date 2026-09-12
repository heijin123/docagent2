"""golden 锚句可定位性校验（M6 配套）。

为什么需要它
------------
锚句定位（`app/eval/golden.py:locate_expected_chunks`）靠的是"归一化后子串匹配"。
一旦锚句与真实 chunk 原文不符——写错字、跨了 chunk 边界、文档改版、切块参数变了——
该用例会**静默**变成 `located=False`，随后被 `metrics.py` 踢出指标分母。

后果很隐蔽：分母变小，`recall@5` 反而可能更好看，题集却在悄悄缩水。
跑 `eval` 只能看到报告里的 `skipped: N`，而 N 不解释"为什么"。

本脚本用**项目真实的解析 + 切块链路**（`parsers.parse_file` + `chunking.chunk_document`）
在本地重建 chunk 语料，再对 golden 集逐条做锚句定位，并把失败原因一并给出：

- 锚句定位失败 → 退出码 1（可挂 CI）；
- 若某个锚句"定位到了"，但期望文档与实际命中文档不一致 → 打印告警（不失败）。
  注意 `expected_doc` 只用于人工核对，不参与打分（`metrics.py` 只用锚句定位出的
  `expected_chunk_ids`），所以这类不一致是"题面信息需修正"，不是指标 bug。

零额外依赖：解析/切块所需依赖（python-docx / pymupdf）本就是项目依赖。

用法::

    python -m scripts.verify_golden_anchors
    python -m scripts.verify_golden_anchors --samples data/samples --golden data/golden/qa_golden.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许 `python scripts/verify_golden_anchors.py` 直接运行
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.eval.golden import load_golden, locate_expected_chunks, normalize  # noqa: E402
from app.ingestion.chunking import build_doc_meta, chunk_document  # noqa: E402
from app.ingestion.parsers import UnsupportedFormatError, parse_file  # noqa: E402
from app.models import SUPPORTED_EXTENSIONS  # noqa: E402


class _InMemoryChunkStore:
    """最小替身：只用 golden.py 需要的 iter_valid_chunks 接口。"""

    def __init__(self, chunks: list):
        self._chunks = chunks

    def iter_valid_chunks(self, tenant_id: str | None = None):
        for c in self._chunks:
            if tenant_id and c.tenant_id != tenant_id:
                continue
            yield {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
                   "content": c.content, "meta": {}}


def build_corpus(samples_dir: Path, tenant_id: str):
    """解析 + 切块，返回 (chunks, doc_files)。与 IngestPipeline 的 M1 阶段同构。"""
    files = sorted(
        p for p in samples_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    chunks: list = []
    doc_files: dict[str, str] = {}
    for path in files:
        try:
            parsed = parse_file(path)
        except UnsupportedFormatError as exc:
            print(f"  [skip] {path.name}: {exc.reason}")
            continue
        meta = build_doc_meta(parsed, tenant_id=tenant_id, doc_key=path.name)
        doc_chunks = chunk_document(parsed, meta)
        chunks.extend(doc_chunks)
        doc_files[meta.doc_id] = path.name
        print(f"  [ok]   {path.name:24} blocks={len(parsed.blocks):3} chunks={len(doc_chunks)}")
    return chunks, doc_files


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 golden 锚句能否在真实 chunk 语料中定位")
    ap.add_argument("--samples", default=str(settings.base_dir / "data" / "samples"))
    ap.add_argument("--golden", default=str(settings.base_dir / "data" / "golden" / "qa_golden.json"))
    ap.add_argument("--tenant-id", default=settings.default_tenant_id)
    args = ap.parse_args()

    samples_dir = Path(args.samples)
    if not samples_dir.is_dir():
        print(f"样例目录不存在：{samples_dir}")
        return 2

    print(f"样例目录：{samples_dir}")
    chunks, doc_files = build_corpus(samples_dir, args.tenant_id)
    print(f"\n语料：{len(chunks)} 个 chunk，覆盖 {len(doc_files)} 个文档")

    cases, meta = load_golden(args.golden)
    print(f"题集：{len(cases)} 条用例（meta.case_count={meta.get('case_count', '—')}）")

    store = _InMemoryChunkStore(chunks)
    located_map = locate_expected_chunks(cases, store, tenant_id=args.tenant_id)

    print("\n" + "=" * 72)
    missed: list = []
    doc_mismatch: list = []
    for case in cases:
        if not case.located:
            missed.append(case)
            continue
        # expected_doc 仅作人工核对：命中块的 doc_id 是否覆盖期望文档
        hit_docs = {cid.rsplit("_", 2)[0] for cid in case.expected_chunk_ids}
        expected_doc_id = next((k for k, v in doc_files.items() if v == case.expected_doc), None)
        if expected_doc_id and expected_doc_id not in hit_docs:
            doc_mismatch.append((case, sorted(doc_files.get(d, d) for d in hit_docs)))

    print(f"锚句定位：{len(cases) - len(missed)}/{len(cases)} 命中")
    if doc_mismatch:
        print(f"\n[告警] {len(doc_mismatch)} 条用例的 expected_doc 与实际命中文档不一致"
              "（不影响打分，但题面信息应修正）：")
        for case, docs in doc_mismatch:
            print(f"  {case.id}  写着 {case.expected_doc}  实际命中 {docs}")

    if missed:
        print(f"\n[失败] {len(missed)} 条锚句无法定位 —— 这些用例会被静默剔出指标分母：\n")
        norm_corpus = [(c.chunk_id, c.doc_id, normalize(c.content)) for c in chunks]
        for case in missed:
            print(f"  {case.id}  anchor={case.anchor!r}")
            print(f"         归一化后={normalize(case.anchor)!r}（{len(normalize(case.anchor))} 字）")
            # 给出最可能的落点：取锚句前 4 字做模糊提示
            probe = normalize(case.anchor)[:4]
            hints = [(cid, doc) for cid, doc, nc in norm_corpus if probe and probe in nc]
            if hints:
                print(f"         可能落点：{[d for _, d in hints]}（前 4 字可匹配，锚句后段不一致）")
            else:
                print("         语料中未找到任何近似片段（锚句可能整体写错，或内容未入库）")
            print()

    print("=" * 72)
    if missed or doc_mismatch:
        print(f"结果：FAIL（定位失败 {len(missed)} / 期望文档不一致 {len(doc_mismatch)}）")
        return 1
    print("结果：PASS（全部锚句可定位，且 expected_doc 与命中文档一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
