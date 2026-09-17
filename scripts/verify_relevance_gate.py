"""检索两条新能力验证（F2.10 相关性判定 + 需求 7.1 邻近上下文扩展）。

离线部分（不需要 Key）：
1. 词表覆盖度语义（命中/缺席/无内容词）；
2. RelevanceGate 四象限：**只有「覆盖低 且 相似低」才判无相关资料**（单低不判）；
3. 邻近上下文扩展：±1 邻块、去重、上限、跳过过期、透传 tenant/permission 隔离；
4. **图级集成**：`no_relevant=True` → 走 no_data 且 **0 次 LLM**（本项的核心价值），
   对照组 `no_relevant=False` → 正常进 answer（LLM 被调用）。

真实部分（需 DASHSCOPE_API_KEY，缺失则 SKIP 不算失败）：
5. golden 55 条**零误杀**（全部 no_relevant=False）+ 语料不覆盖 12 条判出 ≥4 条。

用法：PYTHONPATH=D:/workspace/docagent2 python scripts/verify_relevance_gate.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.checkpointer import build_checkpointer  # noqa: E402
from app.agent.graph import build_qa_graph, thread_config  # noqa: E402
from app.agent.llm import StubLLM  # noqa: E402
from app.agent.nodes import _NO_DATA_MSG  # noqa: E402
from app.retrieval.hybrid import HybridResult, HybridRetriever  # noqa: E402
from app.retrieval.relevance import (CorpusVocabulary, RelevanceGate,  # noqa: E402
                                     content_terms)

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


# ── 假语料库（隔离、确定性）──────────────────────────────────
_FAKE_CHUNKS = [
    "员工年假制度：入职满一年享年假 5 天，可跨年休。",
    "考勤管理：迟到 30 分钟以上按旷工半天处理。",
    "报销制度：差旅费凭电子发票在 OA 系统提交。",
]


class _FakeBM25:
    """最小 BM25Store 替身：只实现 relevance/扩展所需的方法。"""

    def __init__(self, chunks: list[str], records: dict[str, dict] | None = None,
                 *, count_override: int | None = None):
        self.chunks = chunks
        self.records = records or {}
        self.last_query_args: dict | None = None
        self._count_override = count_override

    def count(self) -> int:
        return self._count_override if self._count_override is not None else len(self.chunks)

    def iter_valid_chunks(self, tenant_id: str | None = None):
        for i, c in enumerate(self.chunks):
            yield {"chunk_id": f"c{i}", "doc_id": "d", "content": c, "meta": {}}

    def get_by_chunk_ids(self, chunk_ids, *, tenant_id=None, allowed_permissions=None):
        self.last_query_args = {"ids": list(chunk_ids), "tenant_id": tenant_id,
                                "allowed_permissions": allowed_permissions}
        return [self.records[c] for c in chunk_ids if c in self.records]


def _ctx_item(chunk_id: str, text: str, perm: str = "internal", eff: int = 0) -> dict:
    return {"chunk_id": chunk_id, "content": text, "bm25_score": 0.0,
            "metadata": {"permission": perm, "effective_time": eff, "doc_title": "t"}}


def offline_vocab() -> None:
    print("── 1. 语料词表覆盖度 ──")
    vocab = CorpusVocabulary.build(_FakeBM25(_FAKE_CHUNKS))
    cov_hit, absent_hit = vocab.coverage("年假可以跨年休吗")
    check("内容词命中 → 覆盖度高", cov_hit > 0.9, f"cov={cov_hit} absent={absent_hit}")
    cov_miss, absent_miss = vocab.coverage("宠物狗握手训练教程")
    check("全词缺席 → 覆盖度 0", cov_miss == 0.0 and absent_miss,
          f"cov={cov_miss} absent={absent_miss}")
    cov_none, _ = vocab.coverage("怎么样呢")
    check("无内容词 → 覆盖度 1.0（不判）", cov_none == 1.0, f"cov={cov_none}")
    check("内容词过滤（去停用词/单字）",
          content_terms("员工的话怎么样") == ["员工"], str(content_terms("员工的话怎么样")))


def offline_gate() -> None:
    print("── 2. RelevanceGate 四象限（只有双低才判无资料）──")
    gate = RelevanceGate(_FakeBM25(_FAKE_CHUNKS),
                         min_coverage=0.20, min_similarity=0.40)
    hi = [{"score": 0.80}]
    lo = [{"score": 0.30}]
    r1 = gate.assess("年假可以跨年休吗", hi)              # 覆盖高 + 相似高
    r2 = gate.assess("年假可以跨年休吗", lo)              # 覆盖高 + 相似低
    r3 = gate.assess("宠物狗握手训练教程", hi)            # 覆盖低 + 相似高
    r4 = gate.assess("宠物狗握手训练教程", lo)            # 覆盖低 + 相似低
    check("高覆盖 + 高相似 → 不判", r1["no_relevant"] is False, str(r1))
    check("高覆盖 + 低相似 → 不判（防误杀）", r2["no_relevant"] is False, str(r2))
    check("低覆盖 + 高相似 → 不判（防误杀）", r3["no_relevant"] is False, str(r3))
    check("低覆盖 + 低相似 → 判无相关内容", r4["no_relevant"] is True, str(r4))
    r5 = RelevanceGate(_FakeBM25([], count_override=0)).assess("任意问题", lo)
    check("语料为空 → 不判（交回索引真空路径）", r5["no_relevant"] is False, str(r5))


def offline_context() -> None:
    print("── 3. 邻近上下文扩展 ──")
    doc = "doc_ab12cd34ef56"
    recs = {}
    for i in (1, 2, 3, 4, 5):
        recs[f"{doc}_0001_{i:05d}"] = _ctx_item(f"{doc}_0001_{i:05d}", f"第{i}块内容")
    recs[f"{doc}_0001_00006"] = None  # 占位（不会被取）
    expired = f"{doc}_0001_00002"
    recs[expired] = _ctx_item(expired, "过期邻块", eff=1)  # eff=1 → 远古过期

    fake = _FakeBM25(_FAKE_CHUNKS, recs)
    r = HybridRetriever(vector_store=None, bm25_store=fake, embedder=_NoopEmbedder())
    hit = {"chunk_id": f"{doc}_0001_00003", "content": "第3块", "metadata": {},
           "ranks": {"vector": 1, "bm25": None}, "sources": ["vector"],
           "validity": "valid", "expired_at": None, "score": 0.5}
    ctx = r._expand_context([hit], user_permission="internal", now=10 ** 10)
    ids = [c["chunk_id"] for c in ctx]
    check("取 ±1 邻块（跳过过期邻块）",
          ids == [f"{doc}_0001_00004"], str(ids))
    check("邻居标记 is_context=True 且不计分",
          all(c["is_context"] and c["score"] == 0.0 for c in ctx), str(ctx[:1]))
    check("隔离参数透传（tenant + permission 白名单）",
          fake.last_query_args and fake.last_query_args["tenant_id"]
          and set(fake.last_query_args["allowed_permissions"]) == {"public", "internal"},
          str(fake.last_query_args))

    # 上限：命中 1/3/5 → 邻块 2/4 有 4 个候选，受 context_expand_max 截断
    from app.core.config import settings
    recs2 = {f"{doc}_0001_{i:05d}": _ctx_item(f"{doc}_0001_{i:05d}", f"第{i}块") for i in range(1, 12)}
    fake2 = _FakeBM25(_FAKE_CHUNKS, recs2)
    r2 = HybridRetriever(vector_store=None, bm25_store=fake2, embedder=_NoopEmbedder())
    hits = [{"chunk_id": f"{doc}_0001_{i:05d}", "content": "x", "metadata": {},
             "ranks": {}, "sources": [], "validity": "valid", "expired_at": None,
             "score": 0.5} for i in (2, 5, 8)]
    ctx2 = r2._expand_context(hits, user_permission="internal", now=10 ** 10)
    check(f"总量上限 context_expand_max={settings.context_expand_max}",
          len(ctx2) <= settings.context_expand_max, f"n={len(ctx2)}")
    check("不重复取自命中块", all(c["chunk_id"] not in {h["chunk_id"] for h in hits}
                                  for c in ctx2), str([c["chunk_id"] for c in ctx2]))
    check("非法 chunk_id 安全跳过",
          r2._expand_context([{"chunk_id": "bad-id", "content": "x", "metadata": {},
                              "ranks": {}, "sources": [], "validity": "valid",
                              "expired_at": None, "score": 0.0}],
                             user_permission="internal", now=1) == [], "")


class _NoopEmbedder:
    provider = "noop"
    degraded = True

    def embed_texts(self, texts):
        return [[0.0] * 8 for _ in texts]

    def info(self):
        return {"provider": "noop"}


class _FixedRetriever:
    """按预设返回固定 HybridResult 的检索器替身（测图路由用）。"""

    def __init__(self, result: HybridResult):
        self.result = result
        self.calls = 0

    def retrieve(self, query, **kw):
        self.calls += 1
        return self.result

    def relaxed_retrieve(self, query, **kw):
        self.calls += 1
        return self.result


def offline_graph_routing() -> None:
    print("── 4. 图级集成：no_relevant → no_data（0 次 LLM）──")
    ckpt, _deg, _note = build_checkpointer(memory=True)

    # 4a. 判定无相关内容（items 里仍有 8 条"最像的"，正是问题的来源）
    junk_items = [{"chunk_id": f"c{i}", "content": "弱相关文本", "metadata": {},
                   "ranks": {}, "sources": [], "validity": "valid",
                   "expired_at": None, "score": 0.1} for i in range(8)]
    res_no = HybridResult(query="宠物狗握手训练教程", items=junk_items,
                          relevance={"no_relevant": True, "coverage": 0.0,
                                     "top_vector_score": 0.31,
                                     "reason": "覆盖0.00<0.20 且 相似0.31<0.40"},
                          notes=["相关性判定：无相关内容 → 应如实告知缺失"])
    llm = StubLLM()
    graph, _ = build_qa_graph(_FixedRetriever(res_no), llm, checkpointer=ckpt)
    state = graph.invoke({"query": "怎么训练宠物狗握手？"}, thread_config("t:no"))
    check("answer == 无资料话术", state.get("answer") == _NO_DATA_MSG,
          str(state.get("answer"))[:80])
    check("**0 次 LLM 调用**", len(llm.calls) == 0, f"calls={[c['schema'] for c in llm.calls]}")
    check("无引用", not state.get("citations"), str(state.get("citations")))
    check("degraded=True（如实告知缺失）", state.get("degraded") is True, "")
    check("notes 标明触发来源为相关性判定",
          any("相关性判定无相关内容" in str(n) for n in state.get("notes", [])),
          str(state.get("notes"))[-200:])

    # 4b. 对照组：正常命中 → 进 answer（LLM 被调用），且上下文块进入证据
    hit_items = [{"chunk_id": f"c{i}", "content": "年假制度正文", "metadata": {},
                  "ranks": {}, "sources": [], "validity": "valid",
                  "expired_at": None, "score": 0.7} for i in range(2)]
    ctx = [{"chunk_id": "ctx1", "content": "相邻上下文", "metadata": {},
            "ranks": {}, "sources": ["context"], "validity": "valid",
            "expired_at": None, "score": 0.0, "is_context": True}]
    res_ok = HybridResult(query="年假几天", items=hit_items, context_items=ctx,
                          relevance={"no_relevant": False, "coverage": 1.0,
                                     "top_vector_score": 0.7, "reason": "正常"},
                          notes=["邻近上下文：补入 1 条相邻块"])
    llm2 = StubLLM()
    graph2, _ = build_qa_graph(_FixedRetriever(res_ok), llm2, checkpointer=ckpt)
    st2 = graph2.invoke({"query": "员工年假有几天？"}, thread_config("t:ok"))
    check("对照组进入 LLM 路径", len(llm2.calls) > 0, f"calls={len(llm2.calls)}")
    check("上下文块并入 retrieved（供 answer 使用）",
          any(it["chunk_id"] == "ctx1" for it in st2.get("retrieved", [])),
          str([it["chunk_id"] for it in st2.get("retrieved", [])]))
    check("对照组未走无资料话术", st2.get("answer") != _NO_DATA_MSG, "")


def real_calibration() -> None:
    print("── 5. 真实语料标定（需 Key）──")
    from app.core.config import settings
    if not settings.has_api_key:
        print("  ⚠ 无 DASHSCOPE_API_KEY → SKIP（不计失败）")
        return
    from scripts.calibrate_relevance import OUT_OF_SCOPE
    from app.eval.golden import load_golden

    r = HybridRetriever()
    if r.embedder.degraded:
        print("  ⚠ embedder 降级（mock）→ SKIP")
        return

    cases, _ = load_golden()
    killed = []
    for c in cases:
        res = r.retrieve(c.query)
        if res.no_relevant:
            killed.append(c.id)
    check(f"golden {len(cases)} 条零误杀（no_relevant 全为 False）", not killed,
          f"误杀: {killed}")

    detected = [q for q in OUT_OF_SCOPE if r.retrieve(q).no_relevant]
    check(f"语料不覆盖 {len(OUT_OF_SCOPE)} 条判出 ≥4 条", len(detected) >= 4,
          f"判出 {len(detected)}: {[d[:16] for d in detected]}")
    print(f"    判出明细：{detected}")


def real_agent_e2e() -> None:
    """真实链路（真 embedding + 真 LLM）：库里没有的问题必须 0 次 LLM 出货。

    这是本能力唯一真正值钱的断言——离线桩可以证明路由，只有真实链路能证明
    "原本要烧 answer+verify 2~3 轮的问题，现在 0 次 LLM 就如实告知缺失"。
    """
    print("── 6. 真实 Agent 链路（需 Key）──")
    from app.core.config import settings
    if not settings.has_api_key:
        print("  ⚠ 无 DASHSCOPE_API_KEY → SKIP（不计失败）")
        return
    from app.agent.graph import AgentApp
    from app.agent.llm import build_llm
    from app.retrieval.hybrid import HybridRetriever as _R

    r = _R()
    if r.embedder.degraded:
        print("  ⚠ embedder 降级（mock）→ SKIP")
        return

    app = AgentApp(r, build_llm(), memory_checkpoint=True)
    ans = app.reply("怎么训练宠物狗学会握手？", "verify:no_data")
    check("无相关资料 → 如实告知缺失话术", ans.answer.strip() == _NO_DATA_MSG.strip(),
          ans.answer[:90])
    check(f"**0 次 LLM 调用**（实际 {app.last_calls}）", app.last_calls == 0,
          f"calls={app.last_calls} usage={ans.usage}")
    check("无引用且 degraded", not ans.citations and ans.degraded is True,
          f"cites={len(ans.citations)} degraded={ans.degraded}")

    ans2 = app.reply("员工每年有几天年假？", "verify:has_data")
    check(f"对照组（库里有答案）走 LLM 路径（{app.last_calls} 次）", app.last_calls > 0,
          f"calls={app.last_calls}")
    check("对照组答案非无资料话术", ans2.answer.strip() != _NO_DATA_MSG.strip(),
          ans2.answer[:60])


def main() -> int:
    print("═══ 检索相关性判定 + 邻近上下文扩展 验证 ═══")
    offline_vocab()
    offline_gate()
    offline_context()
    offline_graph_routing()
    real_calibration()
    real_agent_e2e()
    print(f"\n结果：PASS={PASS}  FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
