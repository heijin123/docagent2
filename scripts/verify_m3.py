"""verify_m3.py：M3 LangGraph 多 Agent 验收断言（Stub LLM + 内存 checkpointer 隔离）。

覆盖（需求 F3.1–F3.9 / F4.1–4.3 / 契约 §4.1 与验收标准 4/6/8）：
  1. 意图路由（kb_qa / chitchat / contact_guidance）
  2. kb_qa 全链：retrieve → answer（引用格式）→ verify 达标 → done
  3. 要求转人工 → contact_guidance：只给"该找谁"的指引，**不转交、不建单、不降级**
  4. verify 低置信/不 grounded → 重试回环，retry_count ≤ max_retry(2) → 超限**披露局限**（不转人工）
  5. 检索为空 → no_data（如实告知缺失 + 指向管理员录入，0 次 LLM）+ kb_gap 离线线索日志
  6. F2.8 过期确认两轮流：仅过期 → 确认话术（无 interrupt）→ 用户"是" → include_expired 放行 →
     引用带 validity=expired；纯过期支撑 → degraded=true
  7. F4.2/F4.3 多轮记忆：同 thread 历史累积、rewrite 可见历史；窗口截断最近 N 轮
用法: .venv/Scripts/python.exe scripts/verify_m3.py
"""
from __future__ import annotations

import json
import logging
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
from app.core.config import settings  # noqa: E402
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


class _LogCapture(logging.Handler):
    """捕获结构化 JSON 行日志，用于断言 kb_gap 等业务日志确实落盘。"""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    def events(self, name: str) -> list[dict]:
        out: list[dict] = []
        for line in self.lines:
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and obj.get("event") == name:
                out.append(obj)
        return out


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
    """强制 verify 永不达标 → 触发重试 → 超限走"披露局限"路径（而非转人工）。"""

    def verify_rule(self, user: str) -> schemas.VerifyJudgement:
        return schemas.VerifyJudgement(grounded=False, confidence=0.1,
                                       reason="stub 注入：永不达标")


class _EmptyResult:
    """恒空命中结果（验收"检索为空 → no_data"）。

    `no_relevant` / `context_items` 是 2026-09-17 新增的检索结果字段（F2.10 相关性判定 /
    需求 7.1 邻近上下文），替身必须同步——节点直接按属性访问，缺字段会 AttributeError。
    """

    items: list = []
    expired_candidates: list = []
    context_items: list = []
    relevance: dict = {}
    no_relevant: bool = False
    notes: list = []
    degraded = False


class EmptyRetriever:
    """检索恒为空：验证"无命中 → 如实告知缺失，0 次 LLM，不转人工"。"""

    degraded = False

    def retrieve(self, query, **kwargs):
        return _EmptyResult()

    def relaxed_retrieve(self, query, **kwargs):
        return _EmptyResult()


def main() -> int:
    print("═══ M3 LangGraph 多 Agent 验证（Stub LLM + 内存检查点）═══")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tdir:
        tmp = Path(tdir)
        rt, vs, bm = make_retriever(tmp)
        stub = StubLLM()

        # ── 1. chitchat 路由 ──
        print("── 1. 意图路由 ──")
        app = AgentApp(rt, stub, memory_checkpoint=True)
        r1 = app.reply("你好", "t1")
        check("chitchat: intent=chitchat", r1.intent == "chitchat", r1.intent)
        check("chitchat: 不检索不引用", r1.citations == [] and "依据" not in r1.answer)
        check("chitchat: 非 degraded", r1.degraded is False)

        # ── 2. 要求转人工 → 只给指引（不转交、不建单、不降级）──
        r2 = app.reply("我要转人工客服投诉", "t1")
        check("contact: intent=contact_guidance", r2.intent == "contact_guidance", r2.intent)
        check("contact: 不做降级标记（系统正常履责）", r2.degraded is False)
        check("contact: 明确不转接", "不具备转接" in r2.answer, r2.answer[:80])
        check("contact: 只给联系指引", "联系" in r2.answer)
        check("contact: 不检索不引用", r2.citations == [])

        # ── 3. kb_qa 全链 ──
        print("── 2. kb_qa 全链（检索→生成→verify 达标）──")
        r3 = app.reply("报销单编号规则 XB 开头几位？", "t1")
        check("kb_qa: intent=kb_qa", r3.intent == "kb_qa", r3.intent)
        check("kb_qa: 有引用", len(r3.citations) >= 1, str(r3.citations))
        check("kb_qa: 引用 validity=valid", all(c.validity == "valid" for c in r3.citations))
        check("kb_qa: 答案含 [来源: 标记", "[来源:" in r3.answer)
        check("kb_qa: 置信度达标", r3.confidence >= 0.6, str(r3.confidence))
        check("kb_qa: 非 degraded", r3.degraded is False)

        # ── 4. 低置信 → 重试上限 → 披露局限（防死循环 F3.6/F3.8）──
        print("── 3. verify 永不达标 → 重试 2 次 → 披露局限（不转人工）──")
        app4 = AgentApp(rt, LowConfLLM(), memory_checkpoint=True)
        r4 = app4.reply("报销单编号规则 XB 开头几位？", "t4")
        verify_calls = [c for c in app4.llm.calls if c["schema"] == "VerifyJudgement"]
        check("verify 恰好调用 max_retry+1 次(3)", len(verify_calls) == 3, str(len(verify_calls)))
        check("最终 degraded=true（披露局限）", r4.degraded is True)
        check("保留答案 + 追加披露后缀（未转人工）",
              "仅供参考" in r4.answer and "转接人工" not in r4.answer, r4.answer[-90:])
        check("披露后 intent 仍为 kb_qa", r4.intent == "kb_qa", r4.intent)
        hist4 = app4.history("t4")
        last = hist4[-1]
        check("历史末条 assistant degraded 标注", last.get("degraded") is True)

        # ── 5. 检索为空 → no_data（0 次 LLM，不转人工）+ 落 kb_gap 离线线索 ──
        print("── 4. 检索为空 → 如实告知缺失（0 次 LLM）+ kb_gap 线索日志 ──")
        stub_empty = StubLLM()
        app_e = AgentApp(EmptyRetriever(), stub_empty, memory_checkpoint=True)
        cap = _LogCapture()
        obs_logger = logging.getLogger("app.core.observability")
        _prev_level = obs_logger.level
        obs_logger.setLevel(logging.INFO)     # INFO 级业务日志（默认 root=WARNING 会拦掉）
        obs_logger.addHandler(cap)
        try:
            re_ = app_e.reply("知识库里没有的主题", "t_empty")
        finally:
            obs_logger.removeHandler(cap)
            obs_logger.setLevel(_prev_level)
        check("no_data: 告知未检索到资料", "未检索到" in re_.answer, re_.answer[:80])
        check("no_data: 指向管理员录入（客户自行联系）", "知识库管理员" in re_.answer)
        check("no_data: degraded=true", re_.degraded is True)
        check("no_data: 无引用", re_.citations == [])
        check("no_data: intent=kb_qa", re_.intent == "kb_qa", re_.intent)
        check("no_data: 零 LLM 调用（旧设计要白烧 2~3 轮）", len(stub_empty.calls) == 0,
              str([c["schema"] for c in stub_empty.calls]))
        # kb_gap 离线线索：只落日志供管理员聚类缺失主题，**不进任何人工作队列/不建单**
        gaps = cap.events("kb_gap")
        check("no_data: 落且仅落 1 条 event=kb_gap 线索", len(gaps) == 1, str(gaps))
        if gaps:
            check("kb_gap: 携带原始 query", gaps[0].get("query") == "知识库里没有的主题",
                  str(gaps[0]))
            check("kb_gap: thread_id 经 config 注入成功",
                  gaps[0].get("thread_id") == "t_empty", str(gaps[0]))
            check("kb_gap: 记录 expired_candidates 计数（区分真·零命中）",
                  gaps[0].get("expired_candidates") == 0, str(gaps[0]))

        # ── 6. F2.8 过期确认两轮流 ──
        print("── 5. F2.8 过期文档：确认话术 → 用户放行 → 带失效标注 ──")
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

        # ── 7. 多轮记忆（F4.2）──
        print("── 6. 多轮记忆 + 历史窗口（F4.2/F4.3）──")
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

        # ── render_history 两级预算（2026-09-15 新增）────────────────────
        # 为什么要预算：旧实现是「窗口 10 轮 × 2 条 × 每条 300 字」= 6,000 字/次注入，
        # 而 rewrite 与 answer **各注入一次** → 最坏约 7,500 tok/问（≈ 单问 prompt 的
        # 2.3 倍）。单轮评估无历史、恒为 0，这个上限级风险单轮口径永远看不见。
        # 现改为「每条截断 + 总量预算」，且**从最新往旧**累积：最新一轮必然保留，
        # 优先丢最旧（消解指代依赖最近上文）。窗口退化为安全网。
        from app.agent.prompts import render_history as _render_history
        long_hist: list[dict] = []
        for i in range(10):
            long_hist.append({"role": "user", "content": f"第{i}轮问题" + "问" * 60})
            long_hist.append({"role": "assistant", "content": f"第{i}轮回答" + "答" * 300})
        rendered = _render_history(long_hist, max_rounds=10,
                                   per_msg_chars=150, total_chars=1200)
        check("预算生效：总长不超总量预算", len(rendered) <= 1200, f"len={len(rendered)}")
        _last_line = rendered.split("\n")[-1]
        check("最新一轮完整保留（末行 = 最近一轮回答，且截到 per_msg 上限）",
              _last_line.startswith("助手: 第9轮回答") and len(_last_line) == 4 + 150,
              f"len={len(_last_line)} head={_last_line[:22]!r}")
        check("优先丢最旧（第0轮被丢弃）", "第0轮" not in rendered,
              rendered[:60])
        check("每条都截到 per_msg_chars（无 300 字原文残留）",
              max(len(x) for x in rendered.split("\n")) <= 154,
              str(max(len(x) for x in rendered.split("\n"))))
        check("预算极小时至少保留一条（绝不返回空历史）",
              bool(_render_history(long_hist, per_msg_chars=500, total_chars=10)))
        check("空输入 → 空串（不抛错）", _render_history([], 10) == "")
        check("默认配置下绑定的是总量预算（窗口已退化为安全网）",
              settings.history_total_chars
              < settings.qa_history_rounds * 2 * settings.history_per_msg_chars,
              f"total={settings.history_total_chars} vs "
              f"msg_cap={settings.qa_history_rounds * 2 * settings.history_per_msg_chars}")

        # ── verify 零引用判据（2026-09-15 新增）──────────────────────────
        # 出口纪律：凡「给出答案」的出口都必须带出处，零引用即不可回查的裸答案
        #（q007 曾 0 引用直接出货）。故零引用 → 确定性判未达标，且**不调 LLM**。
        from app.agent.nodes import QANodes
        _stub = StubLLM()
        _nodes = QANodes(rt, _stub)
        _calls_before = _stub.meter.calls
        _res_zero = _nodes.verify({
            "query": "离职当年年假怎么折算", "answer": "按日工资折算。",
            "retrieved": [{"chunk_id": "c1", "metadata": {}, "content": "年假可折算"}],
            "citations": [], "notes": [],
        })
        check("零引用 → grounded=False", _res_zero["grounded"] is False)
        check("零引用 → confidence 归零", _res_zero["confidence"] == 0.0)
        check("零引用 → 不调 LLM（0 成本短路）", _stub.meter.calls == _calls_before,
              f"{_calls_before} → {_stub.meter.calls}")
        check("零引用 → note 以 verify 开头（last_verified 可识别）",
              any(str(n).startswith("verify") for n in _res_zero["notes"]),
              str(_res_zero["notes"]))
        # 有引用时仍走 LLM 判定（不能把正常路径一起短路掉）
        _res_cited = _nodes.verify({
            "query": "离职当年年假怎么折算", "answer": "按日工资折算。[来源: 年假 第1页]",
            "retrieved": [{"chunk_id": "c1", "metadata": {}, "content": "年假可折算"}],
            "citations": [{"chunk_id": "c1", "validity": "valid"}], "notes": [],
        })
        check("有有效引用 → 仍调 LLM 判定（未把正常路径短路）",
              _stub.meter.calls > _calls_before, f"calls={_stub.meter.calls}")
        check("有引用 → 证据按引用收窄的 note 存在",
              any("收窄" in str(n) for n in _res_cited["notes"]), str(_res_cited["notes"]))

        # ── F3.8' should_skip_rewrite 白名单判据（2026-09-14 重构）──────────
        # 语义：默认改写，只在本句被证明「自足」时才跳过；有无历史只作为
        # 「能不能消解」的前提，不再是一票否决（旧实现"有历史就必改写"会把
        # 完全独立的新问题也拖进一次空转改写）。
        from app.agent import anchors as _anchors
        from app.agent import prompts as _prompts
        H = "用户: 差旅住宿标准是多少？\n助手: 一线城市 800 元/天。"
        check("独立新问题 + 有上文 → 跳过（旧实现会改写 = 空转）",
              _prompts.should_skip_rewrite("园区停车月卡多少钱？", H, 0) is True)
        check("含指代 + 有上文 → 改写",
              _prompts.should_skip_rewrite("这个标准是什么时候开始执行的？", H, 0) is False)
        check("无指代词但主题靠上文补全 → 改写（m002 防线）",
              _prompts.should_skip_rewrite("部门经理的标准是多少？", H, 0) is False)
        check("极短句即使含锚点 → 改写（「停车的呢？」）",
              _prompts.should_skip_rewrite("停车的呢？", H, 0) is False)
        check("无上文 → 一律跳过（含指代也无从消解，避免模型编主题）",
              _prompts.should_skip_rewrite("这个标准是什么时候开始执行的？", "", 0) is True)
        check("重试轮 → 必须改写（需注入补充引用 hint）",
              _prompts.should_skip_rewrite("园区停车月卡多少钱？", H, 1) is False)
        check("锚点词表已加载且非空", len(_anchors.load_terms()) > 0,
              f"terms={len(_anchors.load_terms())}")
        check("通用中心词不入词表（标准/规定/流程/管理）",
              not any(t in _anchors.load_terms() for t in ("标准", "规定", "流程", "管理")))
        check("实体不入词表（部门经理）",
              "部门经理" not in _anchors.load_terms())

        vs.close()
        vse.close()

    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
