"""M6 评估体系验证（F7）。

覆盖：
1. golden 加载 + 校验（缺字段/重复 id/锚句过短 → 抛 ValueError）
2. normalize 归一化（全半角/空白/大小写）
3. 锚句定位：灌入样例语料后，**每条** golden 锚句都能定位到期望块（100% 零容忍，
   否则用例会被静默剔出分母）；写新用例时可先用 `scripts/verify_golden_anchors.py`
   快速体检（免 chromadb，且会打印失败锚句的可能落点）
4. 检索层指标：mock 向量 → 指标 SKIP + degraded 标注（F7.4）
5. 报告落盘 eval_report_latest.json 结构正确
6. CLI `python -m app.cli eval` 子命令存在且可解析
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✔ {name}")
    else:
        FAIL += 1
        print(f"  ✘ {name}  {detail[:400]}")


def _make_services(tmp: Path):
    from app.retrieval.bm25store import BM25Store
    from app.retrieval.embedding import Embedder
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.vectorstore import VectorStore

    emb = Embedder(provider="mock", model="mock-hash-v1", dimensions=256, degraded=True)
    vs = VectorStore(tmp / "chroma")
    bm = BM25Store(tmp / "bm25.db")
    ret = HybridRetriever(vector_store=vs, bm25_store=bm, embedder=emb)
    return emb, vs, bm, ret


def _ingest_samples(tmp: Path, vs, bm) -> None:
    """灌入样例语料（复用 IngestPipeline，隔离临时 store）。"""
    from app.ingestion.pipeline import IngestPipeline
    from app.ingestion.registry import DocRegistry
    from app.retrieval.embedding import Embedder

    emb = Embedder(provider="mock", model="mock-hash-v1", dimensions=256, degraded=True)
    reg = DocRegistry(tmp / "registry.db")
    pl = IngestPipeline(tenant_id="tenant_demo", embedder=emb,
                        vector_store=vs, bm25_store=bm, registry=reg)
    samples = Path(__file__).parent.parent / "data" / "samples"
    for f in sorted(samples.glob("*")):
        if f.suffix.lower() in (".pdf", ".docx", ".md", ".txt"):
            pl.run_document(f, doc_key=f.name)


def test_golden_loading() -> None:
    print("— golden 加载/校验 —")
    from app.eval.golden import load_golden, normalize

    cases, meta = load_golden()
    check("golden 加载 ≥50 条", len(cases) >= 50, f"实际 {len(cases)}")
    check("meta.case_count 与实际条数一致",
          meta.get("case_count", len(cases)) == len(cases),
          f"meta={meta.get('case_count')} 实际={len(cases)}")
    check("golden meta 含 schema", "schema" in meta)

    # normalize
    check("normalize 去空白", normalize("年 假 累 计") == "年假累计")
    check("normalize 全半角", normalize("１２３") == "123")
    check("normalize 大小写", normalize("XB-") == "xb-")

    # 校验：缺字段 → 抛错
    bad = {"meta": {}, "cases": [{"id": "x", "query": "q"}]}
    import tempfile as _tf
    with _tf.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(bad, f)
        bad_path = f.name
    try:
        try:
            load_golden(bad_path)
            check("缺字段 golden 抛 ValueError", False)
        except ValueError:
            check("缺字段 golden 抛 ValueError", True)
    finally:
        Path(bad_path).unlink(missing_ok=True)


def test_anchor_locate(tmp: Path) -> None:
    print("— 锚句定位期望块 —")
    from app.eval.golden import load_golden, locate_expected_chunks

    emb, vs, bm, ret = _make_services(tmp)
    _ingest_samples(tmp, vs, bm)

    cases, _ = load_golden()
    locate_expected_chunks(cases, bm, tenant_id="tenant_demo")
    # 锚句定位失败的用例会被 metrics 静默剔出指标分母（题集缩水，recall 反而可能更好看），
    # 因此这里要求 100% 可定位、零容忍——放宽到 80% 等于给"锚句写错/跨块"留后门。
    missed = [c.id for c in cases if not c.located]
    check("全部锚句可定位（100%，0 容忍）", not missed,
          f"未定位 {len(missed)}/{len(cases)} 条：{missed[:8]}")
    # 每条定位到 ≥1 块
    empty = [c.id for c in cases if c.located and not c.expected_chunk_ids]
    check("所有已定位用例块集非空", not empty, f"{empty[:8]}")
    vs.close()


def test_retrieval_metrics(tmp: Path) -> None:
    print("— 检索层指标（mock → SKIP）—")
    from app.eval.runner import EvalRunner

    emb, vs, bm, ret = _make_services(tmp)
    _ingest_samples(tmp, vs, bm)
    runner = EvalRunner(ret, bm)
    report = runner.run(top_k=5, tenant_id="tenant_demo")

    check("mock 向量 → recall@5 为 None（SKIP）", report["retrieval"]["recall_at_k"] is None)
    check("degraded 标注 True", report["meta"]["degraded"] is True)
    check("verdict = skipped", report["verdict"] == "skipped")
    check("报告含逐用例 per_case", "per_case" in report["retrieval"])
    check("报告含 threshold", "threshold" in report)
    vs.close()


def test_report_write(tmp: Path) -> None:
    print("— 报告落盘 —")
    from app.eval.runner import EvalRunner

    emb, vs, bm, ret = _make_services(tmp)
    _ingest_samples(tmp, vs, bm)
    runner = EvalRunner(ret, bm)
    report = runner.run(top_k=5, tenant_id="tenant_demo")
    p = runner.write_report(report, tmp / "reports" / "eval_report_latest.json")
    check("报告文件存在", p.exists())
    parsed = json.loads(p.read_text(encoding="utf-8"))
    check("报告 JSON 可解析且含 verdict", "verdict" in parsed)
    check("报告含 provider/degraded", "provider" in parsed["meta"] and "degraded" in parsed["meta"])
    vs.close()


def test_cli_eval() -> None:
    print("— CLI eval 子命令 —")
    r = subprocess.run([sys.executable, "-m", "app.cli", "eval", "--help"],
                       capture_output=True, text=True, timeout=60)
    check("app.cli eval --help 退出码 0", r.returncode == 0, r.stderr[:200])
    check("eval 帮助含 --answers/--top-k", "--answers" in r.stdout and "--top-k" in r.stdout)
    check("eval 帮助含 --limit（分层抽样）", "--limit" in r.stdout)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="v6eval_"))
    try:
        test_golden_loading()
        test_anchor_locate(tmp)
        test_retrieval_metrics(tmp)
        test_report_write(tmp)
        test_cli_eval()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n===== M6 验证：{PASS} 通过 / {FAIL} 失败 =====")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
