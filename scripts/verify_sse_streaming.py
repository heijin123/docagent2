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
  5. chitchat 规则短路（零 LLM）→ 仍走 SSE 且 intent=chitchat。
用法: python scripts/verify_sse_streaming.py
"""
from __future__ import annotations

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


class FakeCompletions:
    """create() 根据 stream 参数返回：流式迭代器（answer）或 JSON 响应（verify）。"""

    def create(self, *, model, messages, stream=False, **kw):
        if stream:
            # 首块携带 <intent>kb_qa</intent> 标签，后续为答案正文，尾块带 usage
            return iter([
                _Event(content="<intent>kb_qa</intent>根据《考勤管理制度》"),
                _Event(content="第3条，请假需提前向主管申请。"),
                _Event(usage=_Usage(200, 40)),
            ])
        # verify 节点（complete_json，非流式）：返回合法 JSON
        return _Resp('{"grounded": true, "confidence": 0.9, "reason": "ok"}')


class FakeStreamLLM(DashScopeLLM):
    """复用真实 DashScopeLLM 的 stream_answer / complete_json 与标签解析，仅换底层 client。"""

    def __init__(self):
        LLMClient.__init__(self)
        self.provider = "dashscope"
        self.model = "fake-stream"
        self.degraded = False
        self._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions()))


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

    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
