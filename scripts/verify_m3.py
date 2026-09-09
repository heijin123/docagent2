"""verify_m3.py：M3 LangGraph 多 Agent 验收断言（Stub LLM + 内存 checkpointer 隔离）。

覆盖（需求 F3.1–F3.9 / F4.1–4.3 / 契约 §4.1 与验收标准 4/6/8）：
  1. supervisor 三类意图路由（kb_qa / chitchat / human_handoff）
  2. kb_qa 全链：retrieve → answer（引用格式）→ verify 达标 → done
  3. handoff（supervisor 直达 / verify 重试超限兜底）→ degraded=true
  4. verify 低置信/不 grounded → 重试回环，retry_count ≤ max_retry(2) → 超限强转 handoff（防死循环）
  5. F2.8 过期确认两轮流：仅过期 → 确认话术（无 interrupt）→ 用户"是" → include_expired 放行 →
     引用带 validity=expired；纯过期支撑 → degraded=true
  6. F4.2/F4.3 多轮记忆：同 thread 历史累积、rewrite 可见历史；窗口截断最近 N 轮
用法: .venv/Scripts/python.exe scripts/verify_m3.py
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ChunkRecord, now_ts  # noqa: E402
from app.agent import schemas  # noqa: E402
from app.agent.graph import AgentApp, thread_config  # noqa: E402
from app.agent.llm import StubLLM  # noqa: E402
from app.retrieval.bm25store import BM25Store  # noqa: E402
from app.retrieval.embedding import Embedder  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.vectorstore import VectorStore  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✔ {name}")
    else:
        FAIL += 1
        print(f"  ✘ {name}  {detail[:300]}")


# 链路测试与真实 provider 解耦：显式 mock，256 维，离线确定
_EMB = Embedder(provider="mock", model="mock-hash-v1", dimensions=256, degraded=False)


def mk(doc: str, title: str, i: int, txt: str, *, page_num: int = 1, **kw) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc, doc_title=title, source="md", file_path=f"{title}.md",
        version=1, chunk_id=f"{doc}_v1_{i:04d}", chunk_index=i, page_num=page_num,
        content=txt, create_time=now_ts(), update_time=now_ts(),
        embedding_model="mock", block_type="paragraph", **kw)


def ingest(vs, bm, recs) -> None:
    vs.add(recs, _EMB.embed_texts([r.content for r in recs]))
    bm.add(recs)


def make_retriever(tmp: Path, tenant: str = "tenant_t"):
    vs = VectorStore(tmp / "chroma")
    bm = BM25Store(tmp / "bm25.db")
    recs = [
        mk("doc_leave", "员工请假制度", 1, "员工请假需提前一天在 OA 提交申请，主管审批后生效。病假需附医院证明。"),
        mk("doc_leave", "员工请假制度", 2, "年假按自然年计算，未休年假可顺延至次年 3 月底。"),
        mk("doc_pay", "报销管理制度", 1, "报销单编号规则：XB 开头，后接部门码与流水号，共 12 位。"),
        mk("doc_pay", "报销管理制度", 2, "差旅报销流程：线上 OA 申请，电子发票上传系统自动核验。"),
    ]
    for r in recs:
        r.tenant_id = tenant
    ingest(vs, bm, recs)
    return HybridRetriever(vector_store=vs, bm25_store=bm, embedder=_EMB,
                           tenant_id=tenant), vs, bm


def make_expired_only_retriever(tmp: Path, tenant: str = "tenant_e"):
    """仅含一份过期文档的库（F2.8 验收场景：知识库仅有过期文档）。"""
    vs = VectorStore(tmp / "chroma_e")
    bm = BM25Store(tmp / "bm25_e.db")
    rec = mk("doc_old_kq", "考勤办法（已废止 2025 版）", 1,
             "旧考勤办法：纸质打卡，每月人工统计，2025-06-30 废止。",
             effective_time=now_ts() - 3600, page_num=3)
    rec.tenant_id = tenant
    ingest(vs, bm, [rec])
    return HybridRetriever(vector_store=vs, bm25_store=bm, embedder=_EMB,
                           tenant_id=tenant), vs, bm


class LowConfLLM(StubLLM):
    """强制 verify 永不达标 → 触发重试 → 超限 handoff 路径。"""

    def verify_rule(self, user: str) -> schemas.VerifyJudgement:
        return schemas.VerifyJudgement(grounded=False, confidence=0.1,
                                       reason="stub 注入：永不达标")


def main() -> int:
    print("═══ M3 LangGraph 多 Agent 验证（Stub LLM + 内存检查点）═══")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tdir:
        tmp = Path(tdir)
        rt, vs, bm = make_retriever(tmp)
        stub = StubLLM()

        # ── 1. chitchat 路由 ──
        print("── 1. supervisor 意图路由 ──")
        app = AgentApp(rt, stub, memory_checkpoint=True)
        r1 = app.reply("你好", "t1")
        check("chitchat: intent=chitchat", r1.intent == "chitchat", r1.intent)
        check("chitchat: 不检索不引用", r1.citations == [] and "依据" not in r1.answer)
        check("chitchat: 非 degraded", r1.degraded is False)

        # ── 2. human_handoff 直达 ──
        r2 = app.reply("我要转人工客服投诉", "t1")
        check("handoff: intent=human_handoff", r2.intent == "human_handoff", r2.intent)
        check("handoff: degraded=true", r2.degraded is True)
        check("handoff: 兜底话术", "人工" in r2.answer)

        # ── 3. kb_qa 全链 ──
        print("── 2. kb_qa 全链（检索→生成→verify 达标）──")
        r3 = app.reply("报销单编号规则 XB 开头几位？", "t1")
        check("kb_qa: intent=kb_qa", r3.intent == "kb_qa", r3.intent)
        check("kb_qa: 有引用", len(r3.citations) >= 1, str(r3.citations))
        check("kb_qa: 引用 validity=valid", all(c.validity == "valid" for c in r3.citations))
        check("kb_qa: 答案含 [来源: 标记", "[来源:" in r3.answer)
        check("kb_qa: 置信度达标", r3.confidence >= 0.6, str(r3.confidence))
        check("kb_qa: 非 degraded", r3.degraded is False)

        # ── 4. 低置信 → 重试上限 → handoff（防死循环 F3.6/F3.8）──
        print("── 3. verify 永不达标 → 重试 2 次 → handoff ──")
        app4 = AgentApp(rt, LowConfLLM(), memory_checkpoint=True)
        r4 = app4.reply("报销单编号规则 XB 开头几位？", "t4")
        verify_calls = [c for c in app4.llm.calls if c["schema"] == "VerifyJudgement"]
        check("verify 恰好调用 max_retry+1 次(3)", len(verify_calls) == 3, str(len(verify_calls)))
        check("最终 degraded=true（handoff）", r4.degraded is True)
        check("answer 为兜底话术", "人工" in r4.answer)
        hist4 = app4.history("t4")
        last = hist4[-1]
        check("历史末条 assistant degraded 标注", last.get("degraded") is True)

        # ── 5. F2.8 过期确认两轮流 ──
        print("── 4. F2.8 过期文档：确认话术 → 用户放行 → 带失效标注 ──")
        rte, vse, bme = make_expired_only_retriever(tmp)
        app5 = AgentApp(rte, StubLLM(), memory_checkpoint=True)
        r5 = app5.reply("旧考勤办法 纸质打卡 怎么规定？", "t5")
        check("仅过期: 走确认话术", "是否仍要查看" in r5.answer, r5.answer[:80])
        check("仅过期: 无 citation（未放行不引用）", r5.citations == [])
        r5b = app5.reply("是的，查看", "t5")
        check("放行后: 引用 validity=expired", all(c.validity == "expired" for c in r5b.citations),
              str(r5b.citations))
        check("放行后: citation 带 expired_at", all(c.expired_at for c in r5b.citations))
        check("纯过期支撑: degraded=true", r5b.degraded is True)

        # ── 6. 多轮记忆（F4.2）──
        print("── 5. 多轮记忆 + 历史窗口（F4.2/F4.3）──")
        app6 = AgentApp(rt, StubLLM(), memory_checkpoint=True)
        app6.reply("报销单编号规则是什么？", "t6")
        app6.reply("那它需要谁审批？", "t6")
        hist6 = app6.history("t6", limit=20)
        check("两轮后历史 4 条", len(hist6) == 4, str(len(hist6)))
        rewrite_users = [c["user"] for c in app6.llm.calls if c["schema"] == "RewriteOutput"]
        # M6 性能优化后：首轮无历史无指代走规则短路（不调 rewrite），
        # 故只断言「含指代的第二轮确实走了 rewrite 且携带历史」（核心语义不变）。
        check("含指代轮次 rewrite 携带历史上下文",
              len(rewrite_users) >= 1 and "助手:" in rewrite_users[-1],
              rewrite_users[-1][:80] if rewrite_users else "no rewrite")

        # 窗口截断：12 轮问候 → 消息数封顶 20 条（保留最近 10 轮 = 10 条 user）
        app7 = AgentApp(rt, StubLLM(), memory_checkpoint=True)
        for i in range(12):
            app7.reply("你好", "t7")
        hist7 = app7.history("t7", limit=20)
        user_n = sum(1 for m in hist7 if m.get("role") == "user")
        check("12 轮后历史封顶 20 条", len(hist7) == 20, str(len(hist7)))
        check("窗口保留最近 10 轮（user 恰 10 条，最早 2 轮被截断）", user_n == 10,
              f"user={user_n}")

        vs.close()
        vse.close()

    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
