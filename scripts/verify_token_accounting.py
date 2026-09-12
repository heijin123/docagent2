"""整条链路 token 核算接线验收（stub，零真实消耗）。

验证点：
1. LLM 实例挂载 meter / last_usage，complete_json / stream_answer 都累加；
2. AgentApp.reply 前/后差值 = 单轮成本，且写入 AssistantReply.usage；
3. 多次 reply 累加进 llm.meter（total_usage）；
4. config.llm_price 按模型名命中定价；
5. 日志确有 event=llm_usage 结构化行。
"""
import io
import logging
import sys

import app.agent.llm as llm_mod
from app.agent.graph import AgentApp
from app.agent.llm import StubLLM, TokenUsage
from app.core.config import settings
from app.core.observability import TokenUsage as OTokenUsage

# 捕获 INFO 级日志，确认 event=llm_usage 被写出
buf = io.StringIO()
h = logging.StreamHandler(buf)
h.setLevel(logging.INFO)
logging.getLogger("app.core.observability").addHandler(h)
logging.getLogger("app.core.observability").setLevel(logging.INFO)


class _RetrievalResult:
    items = [{
        "chunk_id": "c1",
        "metadata": {"doc_title": "测试制度", "page_num": 3,
                     "doc_date": "2024", "image_ids": ""},
        "validity": "valid",
    }]
    expired_candidates = []
    notes = []
    degraded = False


class FakeRetriever:
    """最小桩：返回 1 条证据，让 kb_qa 全链路（answer[合并意图]→verify）跑通。"""
    degraded = False

    def retrieve(self, query, **kwargs):
        return _RetrievalResult()

    def relaxed_retrieve(self, query, **kwargs):
        return _RetrievalResult()


def main():
    fails = 0

    # ── 1. 纯 LLM 计量 ──
    stub = StubLLM()
    before_total = stub.meter.snapshot().total_tokens
    assert before_total == 0, "meter 初始应为 0"
    # AnswerOutput 已合并 intent（原 supervisor 字段），complete_json 同样计量
    stub.complete_json("sysA", "userA", llm_mod.schemas.AnswerOutput)
    u1 = stub.last_usage
    assert u1.total_tokens > 0, "complete_json 应产出估算 usage"
    assert stub.meter.snapshot().total_tokens == u1.total_tokens, "meter 应累加"

    # stream_answer 也计入
    chunks = []
    stub.stream_answer("sysB", "userB", lambda t: chunks.append(t))
    assert stub.last_usage.total_tokens > 0
    assert stub.meter.snapshot().total_tokens > u1.total_tokens, "stream 后 meter 应再增加"

    print(f"[1] LLM 计量 OK：complete→{u1.total_tokens}tok，累计→{stub.meter.snapshot().total_tokens}tok")

    # ── 2. AgentApp 单轮差值 + AssistantReply.usage（kb_qa 路径才走 LLM 节点）──
    app = AgentApp(FakeRetriever(), StubLLM(), memory_checkpoint=True)
    rep = app.reply("公司报销流程是怎样的", "verify:token:1")
    assert isinstance(rep.usage, dict) and rep.usage.get("total_tokens", 0) > 0, \
        "kb_qa reply 后 AssistantReply.usage 应非空"
    print(f"[2] AgentApp.reply 单轮 usage={rep.usage}  intent={rep.intent}")

    # 再问一次，累计应增加
    total_before = app.llm.meter.snapshot().total_tokens
    rep2 = app.reply("年假怎么申请需要提前多久", "verify:token:2")
    total_after = app.llm.meter.snapshot().total_tokens
    assert total_after > total_before, "多次 reply 应累计进 meter"
    print(f"[3] 两次 reply 累计：{total_before} → {total_after} tok；第二轮 usage={rep2.usage}")

    # ── 4. 定价命中 ──
    assert settings.llm_price("qwen3.8-max") == settings.llm_pricing["qwen-max"], \
        "qwen3.8-max 应命中 max 档"
    assert settings.llm_price("qwen-plus") == settings.llm_pricing["qwen-plus"]
    # 用户已切到 Qwen3.8-Flash：确认命中 flash 档（0.8/2.7 元每百万 tokens → 0.0008/0.0027 每 1K）
    assert settings.llm_price("Qwen3.8-Flash") == settings.llm_pricing["qwen-flash"], \
        "Qwen3.8-Flash 应命中 flash 档"
    inp_f, outp_f = settings.llm_price("Qwen3.8-Flash")
    cost_flash = OTokenUsage(2408, 1141, 3549).cost(inp_f, outp_f)
    print(f"[4] 定价命中 OK：Qwen3.8-Flash={inp_f}/{outp_f} ¥/1K；3549tok 预估 ¥{cost_flash:.4f}")
    assert cost_flash > 0
    inp, outp = settings.llm_price("qwen3.8-max")
    cost = OTokenUsage(2408, 1141, 3549).cost(inp, outp)
    print(f"    （对照 qwen3.8-max={inp}/{outp} ¥/1K；同量 ¥{cost:.4f}，flash 约为 max 的 "
          f"{cost_flash / cost:.1%}）")
    assert cost > 0

    # ── 5. 日志确有 llm_usage ──
    log_txt = buf.getvalue()
    assert 'event": "llm_usage"' in log_txt or "event\":\"llm_usage" in log_txt or \
        '"event": "llm_usage"' in log_txt, "应写出 event=llm_usage 日志"
    n_usage = log_txt.count("llm_usage")
    print(f"[5] 结构化成本日志 OK：捕获 {n_usage} 条 event=llm_usage")

    print("\n全部验收通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
