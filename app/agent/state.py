"""QAState（契约 §4.1 QAState 的权威实现）。

约定：
- `messages`：对话历史（list[dict]，role ∈ user/assistant），由 ingest 节点负责追加与窗口截断
  （F4.3：最多保留最近 10 轮 = 20 条消息），不用 langchain BaseMessage + add_messages，
  避免引入额外运行时；checkpointer 每轮结束自动持久化整份 state（F4.1）。
- 字段冗余是有意的：answer/confidence 等保留在 state 里供条件边读取（F3.8 分支在代码里）。
"""
from __future__ import annotations

from typing import Literal, TypedDict

# 全量状态意图：kb_qa / chitchat / contact_guidance。
# contact_guidance = 用户要求转人工 / 问该找谁 → 只产出"该联系谁"的指引话术
# （规则短路，零 LLM）；系统不代为转交、不建工单、不指定责任人。
Intent = Literal["kb_qa", "chitchat", "contact_guidance"]


class AgentState(TypedDict, total=False):
    # 对话与记忆（F4）
    messages: list[dict]               # [{"role","content","created_at","citations"?,"degraded"?}]
    query: str                         # 本轮用户输入
    include_expired: bool              # F2.8：用户确认后放行过期文档
    # 意图与改写
    intent: Intent
    rewritten_query: str
    # 检索结果（F2）
    retrieved: list[dict]              # 现行融合命中（含 metadata / validity / expired_at）
    expired_candidates: list[dict]     # F2.8 二级候选（仅过期）
    confirmation_needed: bool          # 仅命中过期 → 先向用户确认再查看
    # 生成与校验
    citations: list[dict]              # [{chunk_id, doc_title, page_num, validity, expired_at?, doc_date?}]
    answer: str
    grounded: bool                     # verify 判定（供条件边）
    confidence: float
    # 兜底与防死循环（F3.6/F3.7）
    degraded: bool
    retry_count: int                   # 已重试次数，max_retry=2，条件边判定
    notes: list[str]                   # 过程说明（年份回退/降级/过期提示等，report/debug 用）
