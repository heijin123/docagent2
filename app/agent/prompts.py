"""Prompt 构建（F3.1/F3.2/F3.4/F3.5/F3.9）。

纪律（风险清单第 1 行，复发 3+ 次的坑）：
- **prompt 内任何 JSON 示例一律 `json.dumps` 生成**，禁止在 f-string/模板里手写 `{`；
- 证据与引用的结构化片段用行格式 `[c{idx}] ...`，由节点/渲染函数产出，不进 f-string 手拼大括号。
"""
from __future__ import annotations

import json
from typing import Iterable


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def render_evidence(items: Iterable[dict]) -> str:
    """把融合命中渲染为证据文本（节点注入 answer/verify 的 user prompt）。

    行格式: [c<序号>] <chunk_id> | <doc_title> | 第 N 页 | <doc_date> | <validity> | <内容截断>
    validity 仅标 expired（现行不标），让模型感知过期引用（F3.9）。
    """
    lines = []
    for idx, it in enumerate(items, start=1):
        m = it.get("metadata", {})
        validity = it.get("validity", "valid")
        flag = "| expired" if validity == "expired" else ""
        content = (it.get("content") or "").replace("\n", " ").strip()
        lines.append(
            f"[c{idx}] {it['chunk_id']} | {m.get('doc_title', '')} "
            f"| 第 {m.get('page_num', 0)} 页 | {m.get('doc_date', '') or ''}{flag} | {content[:500]}"
        )
    return "\n".join(lines)


def render_history(messages: list[dict], max_rounds: int = 10) -> str:
    """最近若干轮历史（F4.3 窗口在 ingest 已截断，这里仅做文本化）。"""
    msgs = [m for m in (messages or []) if m.get("role") in ("user", "assistant")]
    tail = msgs[-(max_rounds * 2):]
    return "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m.get('content', '')[:300]}"
        for m in tail
    )


# ── F3.8 确定性路由短路（性能优化：明确场景不进 LLM）────────────
_HANDOFF_KW = ("转人工", "人工客服", "找客服", "投诉", "人工", "客服")
_CHITCHAT_KW = ("你好", "您好", "hi", "hello", "在吗", "谢谢", "再见", "拜拜", "早上好", "晚上好", "嗨")
# 指代词 / 省略：需结合历史才能理解 → 必须走 rewrite（否则可透传）
_PRONOMINAL_HINTS = ("它", "他", "她", "这个", "那个", "这", "那", "其", "上述", "前面", "刚才", "再", "也")


def rule_classify_intent(query: str) -> str | None:
    """确定性意图分类：明确命中寒暄/转人工关键词时返回 intent，否则 None（走 LLM）。

    只做高置信短路（降低误判风险）；kb_qa 恒返回 None 交 LLM 保证准确。
    """
    q = (query or "").strip()
    if not q:
        return "chitchat"
    if any(k in q for k in _HANDOFF_KW):
        return "human_handoff"
    if len(q) <= 12 and any(k in q.lower() for k in _CHITCHAT_KW):
        return "chitchat"
    return None


def should_skip_rewrite(query: str, history: str, retry_count: int) -> bool:
    """确定性判断：是否可跳过 rewrite 直接透传（省一次 LLM）。

    可跳过条件（全满足）：
    - 非重试轮（重试需注入"补充引用"hint 改写）；
    - 无历史上下文（无指代可消解）；
    - query 不含指代词 / 省略标记（本身已完整可独立检索）。
    """
    if retry_count > 0:
        return False
    if history and history.strip():
        return False
    q = (query or "").strip()
    if not q:
        return False
    return not any(h in q for h in _PRONOMINAL_HINTS)


# ── F3.2 rewrite（含 F2.8 过期确认的自然语言放行）──────────────
_CONFIRM_KW = ("是", "要", "查看", "确认", "看", "可以", "好的", "嗯", "行")


def should_confirm_expired(last_assistant: str | None, user_query: str) -> bool:
    """用户对"过期文档是否仍要查看"的确认（F2.8 自然语言确认，契约 §2.1 include_expired）。"""
    if not last_assistant or "过期" not in last_assistant:
        return False
    if "是否仍要查看" not in last_assistant and "是否" not in last_assistant:
        return False
    head = user_query.strip()
    return bool(head) and head[0] in _CONFIRM_KW or any(
        k in user_query for k in ("查看", "要看", "确认查看"))


def rewrite_prompt(query: str, history: str, note: str = "") -> tuple[str, str]:
    sys_note = ("注意：" + note + "\n") if note else ""
    system = (
        "你是查询改写器。把用户问题改写为可独立检索的完整查询：\n"
        "- 结合历史补全指代（“它”“这个”“那笔报销”等）与省略；\n"
        "- 无历史或问题已完整 → 原样透传（changed=false）；\n"
        "- 只输出 JSON。" + sys_note +
        "输出示例：" + _dumps({"rewritten_query": "2024 年度经营报告的毛利率是多少", "changed": True, "note": ""})
    )
    user = "历史对话：\n" + history + "\n\n<query>" + query + "</query>"
    return system, user


# ── F3.4 answer（含 F3.9 过期约束 / 资料不足直说 / 合并 supervisor 意图）────
def answer_prompt(query: str, evidence: str, history: str,
                  include_expired: bool = False, retry_hint: str = "") -> tuple[str, str]:
    rule = [
        "回答约束：",
        "1. 只依据证据回答，禁止编造；证据不足就明说“知识库现有资料不足以回答”，并建议补充或转人工；",
        "2. 引用证据时必须内嵌引用标记，格式为：[来源: 文档标题 第 N 页]（与证据行的标题/页码一致）；",
        "3. 证据行带 expired 标记的内容属于过期文档：引用时须在该引用位置后紧跟失效提示"
        "“⚠ 该信息来自已于 X 过期的文档，仅供追溯”，不能把过期内容当现行规则回答；",
        "4. 回答现行政策/流程类问题时，若全部证据均 expired → 明确提示请以现行制度为准；",
        "5. chunk_ids 只能填证据行 [cN] 中出现的 chunk_id。",
    ]
    intent_rule = (
        "先判断意图 intent（取 kb_qa / chitchat / human_handoff）：\n"
        "- 若只是寒暄/问候/感谢（如“你好”“谢谢”）→ intent=chitchat，answer 给一句简短友好的问候，chunk_ids=[]；\n"
        "- 若用户明确要求转人工/找客服/投诉 → intent=human_handoff，answer 给转接话术，chunk_ids=[]；\n"
        "- 其余依赖知识库的问题 → intent=kb_qa，按上述约束从证据回答。"
    )
    if include_expired:
        rule.append("6. 用户已确认可查看过期文档，但过期引用仍须遵守第 3/4 条失效提示。")
    system = ("你是企业知识库问答助手。输出 JSON：intent（意图）、answer（最终回答文本）、"
              "chunk_ids（本次回答引用的证据 chunk_id 列表）。\n"
              + intent_rule + "\n" + "\n".join(rule) + "\n输出示例：" +
              _dumps({"intent": "kb_qa",
                      "answer": "根据《考勤管理制度》……（含 [来源: ...] 标记）",
                      "chunk_ids": ["doc_xxx_0001_00002"]}))
    user = (
        "历史对话（语境参考）：\n" + history + "\n\n"
        "<query>" + query + "</query>\n\n"
        + ("（注：本回答属重试轮，上次验证未达标。" + retry_hint + "）\n" if retry_hint else "")
        + "检索到的证据（可能含过期，行内已标注）：\n<evidence>\n" + evidence + "\n</evidence>"
    )
    return system, user


def answer_stream_prompt(query: str, evidence: str, history: str,
                         include_expired: bool = False,
                         retry_hint: str = "") -> tuple[str, str]:
    """M4 SSE 流式 answer：纯文本输出（非 JSON），首行内嵌 <intent> 标签。

    与 answer_prompt 共享引用/过期约束，仅输出形态不同——流式通道无法先拿
    chunk_ids 再逐字产出，改为"文本流式产出 → 结束后从 [来源: ...] 反解 chunk_ids"
    （_parse_chunk_ids_from_answer），token 事件即最终答案本身。

    合并 supervisor：首行必须以 <intent>意图</intent> 开头（意图取 kb_qa / chitchat /
    human_handoff），紧接着输出最终回答正文；流式回调在首行命中即解析意图用于路由、
    剥离标签后把正文照常 token 流式推送，既合并意图分类又保留 SSE 流式。
    """
    rule = [
        "输出要求：",
        "1. 第一行必须以 <intent>意图</intent> 开头（意图取 kb_qa / chitchat / human_handoff），"
        "紧接着输出最终回答正文，不要输出 JSON 或任何包裹格式；",
        "2. 只依据证据回答，禁止编造；证据不足就明说“知识库现有资料不足以回答”，并建议补充或转人工；",
        "3. 引用证据时必须内嵌引用标记，格式为：[来源: 文档标题 第 N 页]（与证据行的标题/页码一致），"
        "可多次引用不同文档；",
        "4. 证据行带 expired 标记的内容属于过期文档：引用时须在该引用位置后紧跟失效提示"
        "“⚠ 该信息来自已于 X 过期的文档，仅供追溯”，不能把过期内容当现行规则回答；",
        "5. 回答现行政策/流程类问题时，若全部证据均 expired → 明确提示请以现行制度为准；",
        "6. 意图判断：寒暄/问候/感谢 → intent=chitchat（正文给简短问候）；明确要求转人工/找客服/投诉 → "
        "intent=human_handoff（正文给转接话术）；其余依赖知识库的问题 → intent=kb_qa（从证据回答并内嵌 [来源:]）。",
    ]
    if include_expired:
        rule.append("7. 用户已确认可查看过期文档，但过期引用仍须遵守第 4/5 条失效提示。")
    system = "你是企业知识库问答助手。\n" + "\n".join(rule)
    user = (
        "历史对话（语境参考）：\n" + history + "\n\n"
        "<query>" + query + "</query>\n\n"
        + ("（注：本回答属重试轮，上次验证未达标。" + retry_hint + "）\n" if retry_hint else "")
        + "检索到的证据（可能含过期，行内已标注）：\n<evidence>\n" + evidence + "\n</evidence>"
    )
    return system, user


# ── F3.5 verify（独立评估，不给思维链；F3.9 置信度规则在节点代码）──
def verify_prompt(query: str, answer: str, evidence: str) -> tuple[str, str]:
    system = (
        "你是回答质量校验器。只做判定，不修改答案。判定两项：\n"
        "1. grounded：答案核心论断是否被给定证据支撑（含引用标记是否对应真实证据行）；\n"
        "2. confidence：0-1 综合置信度（覆盖度、一致性、引用真实性）。\n"
        "reason 最多一句话（≤20 字），不要展开分析。\n"
        "输出 JSON。输出示例："
        + _dumps({"grounded": True, "confidence": 0.85, "reason": "要点均有引用"})
    )
    user = (
        "用户问题：<query>" + query + "</query>\n\n"
        "待校验答案（不做修改）：<answer>" + answer + "</answer>\n\n"
        "可引用证据：\n<evidence>\n" + evidence + "\n</evidence>"
    )
    return system, user
