"""节点结构化输出 schema（F3.2 rewrite / F3.4 answer / F3.5 verify）。

意图分类（原 F3.1 supervisor）已合并进 AnswerOutput.intent，单一 LLM 调用产出。
字段与需求逐条对应；枚举值取契约 §5（intent 只允许追加）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntentLiteral = Literal["kb_qa", "chitchat", "human_handoff"]


class RewriteOutput(BaseModel):
    """rewrite 输出（F3.2）：把指代/省略改写为可独立检索的 query。"""

    rewritten_query: str
    changed: bool = Field(default=False, description="相对原始 query 是否改写")
    note: str = Field(default="", description="改写说明；无历史透传时为空")


class AnswerOutput(BaseModel):
    """answer 输出（F3.4/F3.9）：意图 + 生成回答 + 从给定证据中选引用的 chunk_id。

    intent 由 answer 节点（合并了原 supervisor 的意图分类）一并产出，
    供条件边路由（chitchat/human_handoff 不再单独走 LLM）。
    """

    intent: IntentLiteral = Field(default="kb_qa", description="意图：kb_qa/chitchat/human_handoff")
    answer: str
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="只允许引用给定证据中出现的 chunk_id，禁止编造")


class VerifyJudgement(BaseModel):
    """verify 输出（F3.5/F3.9）：自校验结果（独立评估，输入不含生成思维链）。"""

    grounded: bool = Field(description="答案是否被引用内容支撑")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 综合置信度")
    reason: str = Field(default="", description="判定依据（简短）")
