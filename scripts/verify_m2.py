"""verify_m2.py：M2 混合检索验收断言（隔离临时语料，mock embedding）。

覆盖（对齐需求 F2.1–F2.9 与验收标准 8/11）：
  1. 基础双路入库与 RRF 融合结果完整性（metadata 溯源键、sources、score 单调）
  2. is_valid 软删除过滤（版本化后旧块不召回）
  3. permission 权限过滤（internal 用户不可见 secret；secret 用户可见）
  4. tenant_id 隔离
  5. 现行性：过期文档默认不召回；only_expired 时 expired_candidates 带 validity/expired_at
  6. F2.9 年份感知：a) 命中且语义 top1 同文档 → 采用过滤结果
                       b) 过滤跑题（语义 top1 不在过滤集）→ 回退语义 + note
  7. 单路故障降级（vector / bm25 分别炸 → degraded + 存活路结果）
  8. top_n 截断（默认 ≤ 8）
  9. asyncio.aretrieve 与同步 retrieve 结果一致（同一语料同一排序）
用法: .venv/Scripts/python.exe scripts/verify_m2.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ChunkRecord, now_ts  # noqa: E402
from app.retrieval.bm25store import BM25Store  # noqa: E402
from app.retrieval.embedding import Embedder  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.vectorstore import VectorStore  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✔ {name}")
    else:
        _FAIL += 1
        print(f"  ✘ {name}  {detail[:300]}")


def mk_chunk(doc_id: str, title: str, idx: int, content: str, *,
             page: int = 1, version: int = 1, doc_year: int | None = None,
             eff: int = 0, perm: str = "internal", tenant: str = "tenant_a",
             is_valid: bool = True, category: str = "general") -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id, doc_title=title, source="md", file_path=f"{title}.md",
        version=version, doc_year=doc_year,
        chunk_id=f"{doc_id}_v{version}_{idx:04d}", chunk_index=idx,
        page_num=page, content=content, permission=perm,
        effective_time=eff, is_valid=is_valid, tenant_id=tenant,
        create_time=now_ts(), update_time=now_ts(), embedding_model="mock",
        chunk_size=len(content), category=category, block_type="paragraph",
    )


def ingest(store, bm, embedder, recs: list[ChunkRecord]) -> None:
    embs = embedder.embed_texts([r.content for r in recs])
    store.add(recs, embs)
    bm.add(recs)


def build_corpus(tmp: Path, embedder) -> tuple[VectorStore, BM25Store]:
    """构造多维度隔离语料（doc 前缀区分维度，便于断言）。"""
    vs = VectorStore(tmp / "chroma")
    bm = BM25Store(tmp / "bm25.db")

    # ── 报销主题（现行 v2 / 旧版已软删 v1 / 过期旧制度）────────
    doc_a = "doc_a_bx"
    recs = []
    for i, txt in enumerate([
        "报销单编号规则：XB 开头，后接部门码与流水号，共 12 位数字。",
        "差旅报销流程：先提交申请单，审批后粘贴发票原件，再走财务打款。",
        "发票粘贴要求：按时间顺序平铺粘贴，不得重叠，纸质发票需验真伪。",
    ], start=1):
        recs.append(mk_chunk(doc_a, "报销管理制度", i, txt))
    ingest(vs, bm, embedder, recs)

    # 版本化软更新：v2 新内容，v1 全部翻 is_valid=false
    recs2 = []
    for i, txt in enumerate([
        "报销单编号规则（2026 修订）：XB 开头，后接部门码与流水号，共 12 位。",
        "差旅报销流程（2026 修订）：线上 OA 申请，电子发票上传系统自动核验。",
    ], start=1):
        recs2.append(mk_chunk(doc_a, "报销管理制度", i, txt, version=2, eff=0))
    ingest(vs, bm, embedder, recs2)
    vs.soft_delete_doc(doc_a, 1)
    bm.soft_delete_doc(doc_a, 1)

    # ── 过期旧制度（现行检索不可见，二级候选可见）──────────────
    recs3 = [
        mk_chunk("doc_bx_old", "报销制度（历史版）", 1,
                 "旧版报销制度：纸质单据手工签批，跨部门流程繁琐耗时。",
                 eff=now_ts() - 3600, version=1),
    ]
    ingest(vs, bm, embedder, recs3)

    # ── 考勤主题：internal 可见 / secret 权限隔离 ──────────────
    recs4 = [
        mk_chunk("doc_kq", "考勤管理制度", 1, "员工请假需提前一天在 OA 提交申请，主管审批后生效。"),
        mk_chunk("doc_kq_secret", "考勤薪酬（机密）", 1,
                 "高管薪酬结构与股权激励方案属于保密内容。", perm="secret"),
    ]
    ingest(vs, bm, embedder, recs4)

    # ── 跨租户隔离 ────────────────────────────────────────────
    recs5 = [
        mk_chunk("doc_other_tenant", "另一租户报销规范", 1,
                 "报销单编号规则：OT 开头八位，与 tenant_a 完全不同。", tenant="tenant_b"),
    ]
    ingest(vs, bm, embedder, recs5)

    # ── 年份感知：F=2024 年报；G=2025 年报（主题近似不同期次）──
    recs6 = [
        mk_chunk("doc_ar_2024", "2024 年度经营报告", 1,
                 "2024 年度经营报告：全年营收 1.2 亿，同比增长 15%，毛利率 32%。",
                 doc_year=2024),
        mk_chunk("doc_ar_2025", "2025 年度经营报告", 1,
                 "2025 年度经营报告：全年营收 1.6 亿，同比增长 33%，毛利率 35%。",
                 doc_year=2025),
    ]
    ingest(vs, bm, embedder, recs6)
    return vs, bm


def make_retriever(tmp: Path):
    vs, bm = build_corpus(tmp, _EMB)
    return HybridRetriever(vector_store=vs, bm25_store=bm, embedder=_EMB,
                           tenant_id="tenant_a"), vs, bm


# 链路测试与真实 provider 解耦：显式 mock，256 维，离线确定
_EMB = Embedder(provider="mock", model="mock-hash-v1", dimensions=256, degraded=False)


def run() -> None:
    print("═══ M2 混合检索验证（mock embedding）═══")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tdir:
        tmp = Path(tdir)
        rt, vs, bm = make_retriever(tmp)

        # 1. RRF 融合结果完整性
        print("── 1. RRF 融合 + metadata 完整性 ──")
        r = rt.retrieve("报销单编号规则 XB 开头几位")
        check("现行命中非空", bool(r.items), f"items={len(r.items)}")
        check("无过期候选（现行已足）", r.expired_candidates == [], str(len(r.expired_candidates)))
        need = {"doc_id", "doc_title", "page_num", "permission", "effective_time",
                "version", "tenant_id", "doc_year"}
        meta_ok = all(need.issubset(i["metadata"]) for i in r.items)
        check("metadata 携带完整溯源键", meta_ok,
              str({i["metadata"].get("doc_id") for i in r.items}))
        # 版本化后只应命中 v2 块
        top = r.items[0]["metadata"]
        check("版本化过滤：命中 v2 而非 v1", top["version"] == 2, str(top))
        check("score 单调不增", all(r.items[i]["score"] >= r.items[i + 1]["score"]
                                     for i in range(len(r.items) - 1)))
        check("双路均已参与", set(r.used_roads) == {"vector", "bm25"}, str(r.used_roads))
        check("无降级告警", r.degraded == [], str(r.degraded))
        check("year 未触发（query 无年份）", r.detail["year_filtered"] is False)
        check("top_n ≤ 8", len(r.items) <= 8, str(len(r.items)))

        # 2. 权限过滤
        print("── 2. permission 权限过滤 ──")
        r2 = rt.retrieve("高管薪酬 股权激励 保密")
        ids_secret = {i["metadata"]["doc_id"] for i in r2.items}
        check("internal 用户不可见 secret 文档", "doc_kq_secret" not in ids_secret,
              str(ids_secret))
        r2s = rt.retrieve("高管薪酬 股权激励 保密", user_permission="secret")
        ids_secret_s = {i["metadata"]["doc_id"] for i in r2s.items}
        check("secret 用户可见 secret 文档", "doc_kq_secret" in ids_secret_s,
              str(ids_secret_s))

        # 3. tenant 隔离
        print("── 3. tenant_id 隔离 ──")
        r3 = rt.retrieve("报销单编号规则 OT 开头")
        ids_tenant = {i["metadata"]["doc_id"] for i in r3.items}
        check("跨租户文档不泄漏", "doc_other_tenant" not in ids_tenant, str(ids_tenant))

        # 4. 过期文档（F2.8）
        print("── 4. F2.8 二级候选：仅放宽 effective_time（上层判定现行不足后发起）──")
        r4 = rt.retrieve("纸质单据 手工签批 繁琐 耗时 跨部门")
        check("主检索仍返回现行证据（mock 无语义阈值，判定在上层）",
              bool(r4.items) and r4.expired_candidates == [], str(len(r4.items)))
        r4b = rt.relaxed_retrieve("纸质单据 手工签批 繁琐 耗时 跨部门")
        cands = r4b.expired_candidates
        check("二级候选命中过期文档", any(i["metadata"]["doc_id"] == "doc_bx_old" for i in cands),
              str([i["chunk_id"] for i in cands]))
        if cands:
            e0 = cands[0]
            check("候选标 validity=expired", e0["validity"] == "expired", str(e0["validity"]))
            check("候选带 expired_at(=effective_time)", e0["expired_at"] == e0["metadata"]["effective_time"],
                  str(e0.get("expired_at")))
        check("note 明示需用户确认", any("过期" in n and "确认" in n for n in r4b.notes), str(r4b.notes))
        check("is_valid 未放宽（软删 v1 不出现在候选）",
              all(i["metadata"]["version"] != 1 or i["metadata"]["doc_id"] != "doc_a_bx"
                  for i in cands), "ok")

        # 5. F2.9 年份感知：采用过滤（同文档）
        print("── 5. F2.9a 年份过滤命中且语义 top1 同文档 → 采用 ──")
        r5 = rt.retrieve("2024 年度经营报告 全年营收 同比增长 毛利率 32%")
        docs5 = {i["metadata"]["doc_id"] for i in r5.items}
        check("命中 2024 年报", docs5 == {"doc_ar_2024"}, str(docs5))
        check("note 标注采用年份过滤", any("年份过滤" in n and "命中" in n for n in r5.notes),
              str(r5.notes))

        # 6. F2.9 回退：过滤跑题（doc-agent 实证坑）
        print("── 6. F2.9b 年份过滤跑题 → 回退语义 + note ──")
        # query 主语义指向报销（无年份标签 doc_a），显式年份 2024 只会滤到年报文档 → 跑题
        r6 = rt.retrieve("2024 报销单编号规则 XB 开头")
        top6 = r6.items[0]["metadata"]["doc_id"] if r6.items else ""
        check("回退后采用语义结果（报销文档）", top6 == "doc_a_bx", str(top6))
        check("note 明示回退", any("回退" in n for n in r6.notes), str(r6.notes))

        # 7. 单路故障降级
        print("── 7. 单路故障降级 ──")
        from unittest.mock import patch
        with patch.object(rt.vector_store, "query",
                          side_effect=RuntimeError("vector store down")):
            rv = rt.retrieve("差旅报销流程 线上申请 电子发票")
            paths = {d["path"] for d in rv.degraded}
            check("向量路故障被记录", "vector" in paths, str(rv.degraded))
            check("BM25 单路仍出结果", bool(rv.items), str(len(rv.items)))
            check("used_roads 仅 bm25", set(rv.used_roads) == {"bm25"}, str(rv.used_roads))
        with patch.object(rt.bm25_store, "search",
                          side_effect=RuntimeError("bm25 store down")):
            rb = rt.retrieve("差旅报销流程 线上申请 电子发票")
            paths_b = {d["path"] for d in rb.degraded}
            check("BM25 路故障被记录", "bm25" in paths_b, str(rb.degraded))
            check("向量单路仍出结果", bool(rb.items), str(len(rb.items)))
            check("used_roads 仅 vector", set(rb.used_roads) == {"vector"}, str(rb.used_roads))

        # 8. async 并行检索一致性
        print("── 8. asyncio.aretrieve 并行一致性 ──")
        ra = asyncio.run(rt.aretrieve("报销单编号规则 XB 开头几位"))
        ca = [i["chunk_id"] for i in ra.items]
        cs = [i["chunk_id"] for i in rt.retrieve("报销单编号规则 XB 开头几位").items]
        check("async/sync 排序一致", ca == cs, f"{ca} vs {cs}")
        check("async 结果非空", bool(ca))

        vs.close()

    print(f"\n结果: {_PASS} 通过 / {_FAIL} 失败")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    run()
