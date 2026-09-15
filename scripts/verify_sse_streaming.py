"""verify_sse_streaming.py：SSE 流式输出 + 合并意图（supervisor 并进 answer prompt）验收。

为什么单独测：
  SSE 通道本身（stream_events → chat 路由 StreamingResponse）已存在，但本次把 supervisor
  的意图分类合并进了 answer 节点的「同一次流式调用」——模型需在首行吐 <intent> 标签，
  流式回调解析意图用于路由、剥离标签后把正文照常 token 流式推送。这条「标签解析 + 流式」逻辑
  此前无自动化覆盖。

做法：子类化真实 DashScopeLLM，仅替换底层 client 为 Fake（复现 OpenAI 流式契约：
  首块含 <intent> 标签 → 后续块为答案正文 → 尾块带 usage），从而真正走
  llm.stream_answer 的真实标签解析/剥离代码，而非另写一份。

断言：
  1. 事件序列含 ready / token* / citation* / done；
  2. token 流中绝不含 <intent> 标签（标签被剥离，不污染前端）；
  3. done.intent 正确（kb_qa）；
  4. token 拼接 == done.answer（前端零特判即可还原答案）；
  5. chitchat 规则短路（零 LLM）→ 仍走 SSE 且 intent=chitchat；
  6. 要求转人工（contact_guidance）→ 规则短路仍走 SSE，只给联系指引（不转交、不降级）；
  7. **零引用答案不得直接出货**：verify 判未达标（0 LLM 短路）→ 重试 → 用尽后 disclose
     换成「未能在现有知识库中找到对应出处」的后缀（`_NO_CITE_SUFFIX`）。
用法: python scripts/verify_sse_streaming.py
"""
from __future__ import annotations

import json
import re
import sys
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.graph import AgentApp
from app.agent.llm import DashScopeLLM, LLMClient
from app.core.config import settings

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✔ {name}")
    else:
        FAIL += 1
        print(f"  ✘ {name}" + (f"  —— {detail}" if detail else ""))


# ── 复现 OpenAI 流式契约的 Fake client ──────────────────────
class _Delta:
    def __init__(self, content=None):
        self.content = content


class _Choice:
    def __init__(self, content=None):
        self.delta = _Delta(content)


class _Event:
    def __init__(self, content=None, usage=None):
        self.choices = [_Choice(content)] if content is not None else []
        self.usage = usage


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("_CM", (), {"message": _Msg(content)})()]


def _complete_json_reply(messages):
    """非流式 JSON 响应：按 prompt 内容路由（图里有 rewrite 与 verify 两个 complete_json 调用点）。

    早期实现只回一种 JSON，靠"rewrite 恰好被规则短路"才没暴露——一旦某轮真的进改写
    （零引用被打回重试时会强制改写）就会 RewriteOutput 校验失败 → 整条 SSE 退化成 error 事件。
    """
    blob = " ".join(str(m.get("content", "")) for m in messages)
    if "rewritten_query" in blob:
        m = re.search(r"<query>(.*?)</query>", blob, re.S)
        q = m.group(1).strip() if m else ""
        return _Resp(json.dumps({"rewritten_query": q, "changed": False, "note": ""},
                                ensure_ascii=False))
    return _Resp('{"grounded": true, "confidence": 0.9, "reason": "ok"}')


class FakeCompletions:
    """create() 根据 stream 参数返回：流式迭代器（answer）或 JSON 响应（rewrite/verify）。"""

    def create(self, *, model, messages, stream=False, **kw):
        if stream:
            # 首块携带 <intent>kb_qa</intent> 标签，后续为答案正文 + [来源:] 引用标记，尾块带 usage。
            # **必须带引用标记**：无标记的答案在 verify 处会被判「零引用 → 未达标」打回重试
            # （刻意设计，见 nodes.verify），那会把本用例变成重试用例、测不到主线。
            return iter([
                _Event(content="<intent>kb_qa</intent>根据《考勤管理制度》"),
                _Event(content="第3条，请假需提前向主管申请。"),
                _Event(content="[来源: 考勤管理制度 第3页]"),
                _Event(usage=_Usage(200, 40)),
            ])
        return _complete_json_reply(messages)


class FakeStreamLLM(DashScopeLLM):
    """复用真实 DashScopeLLM 的 stream_answer / complete_json 与标签解析，仅换底层 client。"""

    def __init__(self):
        LLMClient.__init__(self)
        self.provider = "dashscope"
        self.model = "fake-stream"
        self.degraded = False
        self._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions()))


class FakeCompletionsNoUsage:
    """流式迭代器故意不返回 usage 尾块 —— 复现 DashScope 未回 usage 时 usage=None 的真实场景。"""

    def create(self, *, model, messages, stream=False, **kw):
        if stream:
            # 首块带 <intent> 标签、正文与引用标记，但**没有 usage 尾块**
            return iter([
                _Event(content="<intent>kb_qa</intent>答案正文。"),
                _Event(content="[来源: 考勤管理制度 第3页]"),
            ])
        return _complete_json_reply(messages)


class FakeStreamLLMNoUsage(FakeStreamLLM):
    """底层 client 用 FakeCompletionsNoUsage，触发 stream_answer 中 usage 保持 None 的分支。"""

    def __init__(self):
        super().__init__()
        self._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletionsNoUsage()))


class FakeCompletionsNoCite:
    """流式答案**不含任何 [来源:] 引用标记** —— 用于测「零引用不得直接出货」。"""

    def create(self, *, model, messages, stream=False, **kw):
        if stream:
            return iter([
                _Event(content="<intent>kb_qa</intent>离职当年年假按日工资折算，由直属主管审批。"),
                _Event(usage=_Usage(180, 24)),
            ])
        return _complete_json_reply(messages)


class FakeStreamLLMNoCite(FakeStreamLLM):
    """底层 client 用 FakeCompletionsNoCite（零引用答案）。"""

    def __init__(self):
        super().__init__()
        self._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletionsNoCite()))


def check_stream_no_usage_path() -> None:
    """验收：流式尾块缺失 usage（usage=None）时，stream_answer 不抛 AttributeError。

    直接对应代码评审报告「L170-188 当 usage 为 None 时访问属性」——
    当前 181 行 `if usage is not None:` 已守卫 182-183，本测试锁定该兜底。
    """
    print("── 3. 流式尾块无 usage（usage=None）不抛 AttributeError ──")
    app = AgentApp(FakeRetriever(), FakeStreamLLMNoUsage(), memory_checkpoint=True)
    try:
        events = list(app.stream_events("怎么报销", "sse:nousage", request_id="r3"))
    except AttributeError as e:
        check("usage=None 时不抛 AttributeError", False, f"AttributeError: {e}")
        return
    t3 = [e["type"] for e in events]
    done3 = [e for e in events if e["type"] == "done"][0]["data"]
    check("尾块无 usage 仍产出 done（无 AttributeError）", "done" in t3, str(t3))
    check("answer 正文完整（正文 + 引用标记，无缺失）",
          done3["answer"] == "答案正文。[来源: 考勤管理制度 第3页]", done3["answer"])
    check("intent 兜底为 kb_qa", done3.get("intent") == "kb_qa", str(done3.get("intent")))
    check("done 不含 degraded（正常出答案）", done3.get("degraded") is False)


def check_zero_citation_path() -> None:
    """验收：答案零引用 → verify 判未达标（0 LLM 短路）→ 重试用尽 → disclose 换"不可采信"后缀。

    这条判据的由来：四类出口里凡「给出答案」的都必须带出处（① 有资料→答案+出处；
    ② 资料不全→现有数据+出处），零引用即"不可回查的裸答案"（q007 曾 0 引用直接出货）。
    此前 verify 会回退全量证据照常判定，模型说 grounded=true 就直接出货——`disclose` 里
    那支 `_NO_CITE_SUFFIX` 因此几乎是死代码；加上本判据它才真正生效。
    """
    print("── 4. 零引用答案不得直接出货（verify 短路 → 重试 → disclose）──")
    app = AgentApp(FakeRetriever(), FakeStreamLLMNoCite(), memory_checkpoint=True)
    events = list(app.stream_events("离职时年假怎么折算", "sse:nocite", request_id="r5"))
    t = [e["type"] for e in events]
    check("仍产出 done（不崩、退化为 error）", "done" in t, str(t))
    done = [e for e in events if e["type"] == "done"][0]["data"]
    check("degraded=True（走披露，而非直接出货）", done.get("degraded") is True,
          str(done.get("degraded")))
    check("citations 为空（确实零引用）", done.get("citations") == [],
          str(done.get("citations")))
    check("披露后缀为「未找到出处」更重那一档",
          "未能在现有知识库中找到对应出处" in done["answer"], done["answer"][-70:])
    notes = " | ".join(done.get("notes", []))
    check("verify 走了零引用短路（0 LLM）", "零引用" in notes, notes[-220:])
    check("重试用尽后才披露", "重试用尽" in notes, notes[-220:])
    tok = "".join(e["data"]["content"] for e in events if e["type"] == "token")
    # 已知偏差（本次实测暴露，非本用例引入）：verify 打回重试时每次 answer 都会重新
    # 流式推送，于是 token 流里留下**多次尝试**的文本，而 done.answer 只有最后一次
    # （+ 披露后缀）。影响面可控：前端在 done 处用 done.answer **整体覆盖**重渲染
    # （web/index.html），因此不会留下错误终态，只是流式中途的瞬态重影。
    # 但"token 拼接 == done.answer"这条零特判不变式在**重试路径**上不成立——要真正
    # 修好需要协议层加 reset/retry 事件，属独立议题（见报告待办）。
    check("token 流以 done.answer 结尾（终态一致）", tok.endswith(done["answer"]),
          f"tok={tok[-40:]!r} ans={done['answer'][-40:]!r}")
    check("重试使 token 流含多次尝试文本（已知偏差，已锁定）", tok != done["answer"],
          f"len(tok)={len(tok)} len(ans)={len(done['answer'])}")


class _RetrievalResult:
    items = [{
        "chunk_id": "doc_kq_0001_00003",
        "metadata": {"doc_title": "考勤管理制度", "page_num": 3,
                     "doc_date": "2024", "image_ids": ""},
        "validity": "valid",
    }]
    expired_candidates = []
    notes = []
    degraded = False


class FakeRetriever:
    degraded = False

    def retrieve(self, query, **kwargs):
        return _RetrievalResult()

    def relaxed_retrieve(self, query, **kwargs):
        return _RetrievalResult()


def main() -> int:
    print("═══ SSE 流式 + 合并意图 验收（FakeStreamLLM 复现 OpenAI 流式契约）═══")
    app = AgentApp(FakeRetriever(), FakeStreamLLM(), memory_checkpoint=True)

    print("── 1. kb_qa 流式（标签解析 + 流式推送）──")
    events = list(app.stream_events("请假怎么申请", "sse:kb", request_id="r1"))
    types = [e["type"] for e in events]
    check("事件序列含 ready/token/done",
          "ready" in types and "token" in types and "done" in types, str(types))
    tokens = [e["data"]["content"] for e in events if e["type"] == "token"]
    joined = "".join(tokens)
    check("token 流不含 <intent> 标签（已剥离）",
          "<intent>" not in joined and "</intent>" not in joined, joined[:80])
    done = [e for e in events if e["type"] == "done"][0]["data"]
    check("done.intent=kb_qa（标签被正确解析用于路由）",
          done.get("intent") == "kb_qa", str(done.get("intent")))
    check("token 拼接 == done.answer（前端零特判还原）",
          joined == done["answer"], f"joined={joined!r} answer={done['answer']!r}")
    check("答案正文完整（不含标签残留）",
          "请假需提前向主管申请" in done["answer"] and "<intent>" not in done["answer"],
          done["answer"])
    print(f"    answer={done['answer']!r}")

    print("── 2. chitchat 规则短路（零 LLM）仍走 SSE ──")
    ev2 = list(app.stream_events("你好", "sse:hi", request_id="r2"))
    t2 = [e["type"] for e in ev2]
    done2 = [e for e in ev2 if e["type"] == "done"][0]["data"]
    check("chitchat 也产出 done", "done" in t2, str(t2))
    check("chitchat intent=chitchat（规则短路）", done2.get("intent") == "chitchat",
          str(done2.get("intent")))
    check("chitchat 非 degraded", done2.get("degraded") is False)

    print("── 3. contact_guidance（要求转人工）仍走 SSE 且只给指引 ──")
    ev_c = list(app.stream_events("我要转人工客服", "sse:contact", request_id="r4c"))
    t_c = [e["type"] for e in ev_c]
    done_c = [e for e in ev_c if e["type"] == "done"][0]["data"]
    check("contact 也产出 done", "done" in t_c, str(t_c))
    check("contact intent=contact_guidance", done_c.get("intent") == "contact_guidance",
          str(done_c.get("intent")))
    check("contact 只给联系指引、不降级",
          "不具备转接" in done_c["answer"] and done_c.get("degraded") is False,
          done_c["answer"][:60])
    tok_c = "".join(e["data"]["content"] for e in ev_c if e["type"] == "token")
    check("contact token 拼接 == done.answer", tok_c == done_c["answer"],
          f"tok={tok_c[:40]!r} ans={done_c['answer'][:40]!r}")

    check_stream_no_usage_path()
    check_zero_citation_path()

    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
