"""Prompt 构建（F3.1/F3.2/F3.4/F3.5/F3.9）。

纪律（风险清单第 1 行，复发 3+ 次的坑）：
- **prompt 内任何 JSON 示例一律 `json.dumps` 生成**，禁止在 f-string/模板里手写 `{`；
- 证据与引用的结构化片段用行格式 `[c{idx}] ...`，由节点/渲染函数产出，不进 f-string 手拼大括号。
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable

from app.agent import anchors
from app.core.config import settings


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


def render_history(messages: list[dict], max_rounds: int = 10, *,
                   per_msg_chars: int | None = None,
                   total_chars: int | None = None) -> str:
    """最近若干轮历史的文本化（**带硬预算**，防多轮把 prompt 撑爆）。

    为什么需要预算：历史在链路里被注入**两次**（rewrite 一次、answer 一次），
    原先"窗口 10 轮 × 每条 300 字"的写法上限是 6,000 字/次 → 最坏约 7,500 tok/问，
    而单轮评估没有历史、恒为 0，这个上限级风险单轮口径永远测不到。

    两级预算都**从最新往旧**累积：
    - 每条消息先截到 `per_msg_chars`；
    - 全量再截到 `total_chars`，超出即停 → **最新一轮必然完整保留**，优先丢最旧
      （消解指代依赖的是最近上文，丢旧的代价最小）。
    `max_rounds` 退化为安全网：预算通常先于轮数窗口生效（6,000 字 > 1,200 字）。
    """
    per_msg = per_msg_chars if per_msg_chars is not None else settings.history_per_msg_chars
    total = total_chars if total_chars is not None else settings.history_total_chars
    msgs = [m for m in (messages or []) if m.get("role") in ("user", "assistant")]
    tail = msgs[-(max_rounds * 2):]

    picked: list[str] = []
    used = 0
    for m in reversed(tail):
        line = (f"{'用户' if m['role'] == 'user' else '助手'}: "
                f"{(m.get('content') or '')[:per_msg]}")
        # `picked` 非空才允许因预算 break → 单条超预算时至少留一条，绝不返回空历史
        if picked and used + len(line) > total:
            break
        picked.append(line)
        used += len(line) + 1  # +1 计入换行
    return "\n".join(reversed(picked))


# ── F3.8 确定性路由短路（性能优化：明确场景不进 LLM）────────────
# 客户要求"转人工 / 找客服 / 投诉"→ 只产出"该找谁"的指引话术（不代办、不承诺转接）。
_CONTACT_KW = ("转人工", "人工客服", "找客服", "投诉", "人工", "客服")
_CHITCHAT_KW = ("你好", "您好", "hi", "hello", "在吗", "谢谢", "再见", "拜拜", "早上好", "晚上好", "嗨")
# 回指（指代）判定：需结合历史才能消解 → 必须改写。
# ⚠ 单字代词必须**按分词整词**匹配，不能按子串：否则「这周」「那次」「其他」这类复合词
# 会被误判成回指——2026-09-15 由 m007 实测抓出（「员工食堂这周的菜单是什么？」是完整独立
# 问题，却因含「这」被判需改写，且改写结果与原文逐字相同 = 纯空转一跳）。
_PRONOUNS = ("它", "他", "她", "这", "那", "此", "其", "该")
# 多字指代短语用子串匹配：本身无歧义，且不依赖分词是否把它们切成一个词
_PRONOUN_PHRASES = ("这个", "那个", "这些", "那些", "上述", "前面", "刚才",
                    "该文档", "该标准", "该制度")


@lru_cache(maxsize=1024)
def _tokens(query: str) -> tuple[str, ...]:
    """分词（与检索层同一套 jieba）。缺依赖 → 空元组，退化为只靠多字短语判回指。"""
    try:
        import jieba  # noqa: PLC0415 — 延迟导入，避免拖慢链路冷启动
        return tuple(t for t in jieba.lcut(query or "") if t.strip())
    except Exception:  # noqa: BLE001
        return ()


def _has_anaphora(query: str) -> bool:
    """本句是否含回指：整词判单字代词 + 子串判多字短语。"""
    q = (query or "").strip()
    if any(p in q for p in _PRONOUN_PHRASES):
        return True
    return any(t in _PRONOUNS for t in _tokens(q))


def rule_classify_intent(query: str) -> str | None:
    """确定性意图分类：明确命中寒暄 / 要求转人工关键词时返回 intent，否则 None（走 LLM）。

    只做高置信短路（降低误判风险）；kb_qa 恒返回 None 交 LLM 保证准确。
    返回 contact_guidance 的含义是"用户要人 → 由系统给'该找谁'的指引话术"，
    **不是**系统去转交人工（系统不发起任何转交动作）。
    """
    q = (query or "").strip()
    if not q:
        return "chitchat"
    if any(k in q for k in _CONTACT_KW):
        return "contact_guidance"
    if len(q) <= 12 and any(k in q.lower() for k in _CHITCHAT_KW):
        return "chitchat"
    return None


_MIN_SELF_CONTAINED_LEN = 8


def _has_generic_head_after_de(q: str) -> bool:
    """是否为「…的 + 通用中心词」结构（如"部门经理的标准"）。

    这类句子的主题由「的」前的修饰语决定（"部门经理"只是角色，不是知识库主题），
    单独拎出来无法判断是哪个标准 → 不算自足。
    """
    return any("的" + h in q for h in anchors.GENERIC_HEADS)


def _is_self_contained(query: str) -> bool:
    """本句是否**明确不需要改写**：能独自读懂、且自带检索抓手。

    四条同时成立才算自足（任一不满足 → 视为需要上下文的消解）：
    1. 无回指（它/这个/那… 没有前文无从消解）；
    2. 非极短句（≤ 8 字信息量不足）；
    3. 非「…的 + 通用中心词」结构（主题落在修饰语上，而修饰语常是角色/实体）；
    4. 含主题锚点（词表由语料自动生成，见 app/agent/anchors.py）。
    """
    q = (query or "").strip()
    if not q:
        return False
    if _has_anaphora(q):
        return False
    if len(q) <= _MIN_SELF_CONTAINED_LEN:
        return False
    if _has_generic_head_after_de(q):
        return False
    return anchors.has_topic_anchor(q)


def should_skip_rewrite(query: str, history: str, retry_count: int) -> bool:
    """确定性判断：是否可跳过 rewrite 直接透传（省一次 LLM 往返）。

    **白名单哲学**（2026-09-14 定案）：rewrite 的价值是消解本句对上下文的依赖，
    而实际场景里大多数问题都需要改写；判错方向也不对称——多改只浪费一跳，
    漏消解却会直接让检索拿到一句读不懂的话。所以**默认改写**，只在本句被
    证明「自足」（`_is_self_contained`）时才跳过：

    - 非重试轮（重试必须注入"补充引用"hint 改写）→ 改写；
    - 本句自足 → **跳过**（与有没有历史无关；旧实现是"有历史就必改写"，会把
      完全独立的新问题也拖进一次空转改写）；
    - 本句不自足 + 有历史 → 改写（有东西可消解）；
    - 本句不自足 + 无历史 → **跳过**（没有可消解的对象，硬改只会让模型编个主题）。
    """
    if retry_count > 0:
        return False
    q = (query or "").strip()
    if not q:
        return False
    if _is_self_contained(q):
        return True
    return not (history and history.strip())


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


# ── F3.4 answer（含 F3.9 过期约束 / 资料不足直说 / 合并意图）────
def answer_prompt(query: str, evidence: str, history: str,
                  include_expired: bool = False, retry_hint: str = "") -> tuple[str, str]:
    rule = [
        "约束：",
        "1. 只依据证据作答，禁止编造。证据不足即明说“知识库现有资料不足以回答”并指出缺什么；"
        "不要建议转人工或联系客服（要不要找他人由用户自己决定）。",
        "2. 先给结论，正文 ≤3 句（合计 ≤200 字）。不复述问题、不成段抄录证据原文、"
        "不写“综上所述/总的来说”等套话；有例外情形只在结论后补 1 句。",
        "3. 每个论断后内嵌引用：[来源: 文档标题 第 N 页]，须与证据行的标题/页码一致。",
        "4. 证据行带 expired 标记者属过期文档：引用处须紧跟“⚠ 该信息来自已于 X 过期的文档，"
        "仅供追溯”，不得当现行规则回答；若全部证据均 expired → 提示请以现行制度为准。",
        "5. chunk_ids 只能填证据行 [cN] 中出现的 chunk_id。",
    ]
    intent_rule = (
        "先判 intent（只允许 kb_qa / chitchat）：寒暄/问候/感谢（如“你好”“谢谢”）→ chitchat，"
        "answer 给一句简短问候、chunk_ids=[]；其余依赖知识库的问题 → kb_qa，按上述约束作答。"
    )
    if include_expired:
        rule.append("6. 用户已确认可查看过期文档；过期引用仍须遵守第 4 条。")
    system = ("你是企业知识库问答助手。输出 JSON：intent、answer（回答文本）、"
              "chunk_ids（引用的证据 chunk_id 列表）。\n"
              + intent_rule + "\n" + "\n".join(rule) + "\n示例：" +
              _dumps({"intent": "kb_qa",
                      "answer": "……[来源: 文档标题 第 1 页]",
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

    合并意图：首行必须以 <intent>意图</intent> 开头（意图只允许 kb_qa / chitchat），
    紧接着输出最终回答正文；流式回调在首行命中即解析意图用于路由、剥离标签后把正文
    照常 token 流式推送，既合并意图分类又保留 SSE 流式。
    """
    rule = [
        "输出要求：",
        "1. 第一行必须是 <intent>意图</intent>（只允许 kb_qa / chitchat），紧接着输出回答正文；"
        "不要输出 JSON 或任何包裹格式。",
        "2. 只依据证据作答，禁止编造。证据不足即明说“知识库现有资料不足以回答”并指出缺什么；"
        "不要建议转人工或联系客服（要不要找他人由用户自己决定）。",
        "3. 先给结论，正文 ≤3 句（合计 ≤200 字）。不复述问题、不成段抄录证据原文、"
        "不写“综上所述/总的来说”等套话；有例外情形只在结论后补 1 句。",
        "4. 每个论断后内嵌引用：[来源: 文档标题 第 N 页]，须与证据行的标题/页码一致；可引用多篇。",
        "5. 证据行带 expired 标记者属过期文档：引用处须紧跟“⚠ 该信息来自已于 X 过期的文档，"
        "仅供追溯”，不得当现行规则回答；若全部证据均 expired → 提示请以现行制度为准。",
        "6. 寒暄/问候/感谢 → intent=chitchat（正文一句问候）；其余依赖知识库的问题 → intent=kb_qa。",
    ]
    if include_expired:
        rule.append("7. 用户已确认可查看过期文档；过期引用仍须遵守第 5 条。")
    system = "你是企业知识库问答助手。\n" + "\n".join(rule)
    user = (
        "历史对话（语境参考）：\n" + history + "\n\n"
        "<query>" + query + "</query>\n\n"
        + ("（注：本回答属重试轮，上次验证未达标。" + retry_hint + "）\n" if retry_hint else "")
        + "检索到的证据（可能含过期，行内已标注）：\n<evidence>\n" + evidence + "\n</evidence>"
    )
    return system, user


# ── F3.5 verify（独立评估，不给思维链；F3.9 置信度规则在节点代码）──
# 注意：evidence 由 verify 节点按「答案实际引用的 chunk」收窄后传入（未被引用的候选
# 对判定无贡献，曾占本跳 prompt 的 84%）；零引用时才回退全量。
def verify_prompt(query: str, answer: str, evidence: str) -> tuple[str, str]:
    system = (
        "你是回答质量校验器。只判定，不改答案：\n"
        "1. grounded：答案核心论断是否被给定证据支撑，引用标记是否对应真实证据行；\n"
        "2. confidence：0-1 置信度（覆盖度、一致性、引用真实性）。\n"
        "reason ≤20 字，不展开。输出 JSON，示例："
        + _dumps({"grounded": True, "confidence": 0.85, "reason": "要点均有引用"})
    )
    user = (
        "问题：<query>" + query + "</query>\n\n"
        "答案：<answer>" + answer + "</answer>\n\n"
        "相关证据：\n<evidence>\n" + evidence + "\n</evidence>"
    )
    return system, user
