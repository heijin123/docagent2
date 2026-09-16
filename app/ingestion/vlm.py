"""VLM 转录器（需求 F1.9，M1 提供抽象与无 Key 降级路径）。

- 红页 → Transcriber.transcribe(page_render) → 文本；
- 无 Key / 失败 / 超 max_pages → 返回 None，编排层降级为原文入库 + vlm_note（R7 不崩）。
M1 不强制接入真实 VLM：构造器用 get_transcriber() 工厂，配 Key 且 VLM_ENABLED=1 时
返回 DashScopeTranscriber，否则返回 NoneTranscriber（恒 None）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import settings


class Transcriber(ABC):
    """红页整页转录接口。"""

    @abstractmethod
    def transcribe(self, page_image_path: str, page_no: int) -> str | None:
        """转录单页图片为文本；失败返回 None。"""


class NoneTranscriber(Transcriber):
    """无 Key / 未启用：恒返回 None，编排降级原文入库。"""

    def transcribe(self, page_image_path: str, page_no: int) -> str | None:
        return None


class DashScopeTranscriber(Transcriber):
    """Qwen-VL 整页转录（OpenAI 兼容 /chat/completions，image_url 传 base64 或本地文件）。

    用法示例（M2 接红页渲染后启用）：
        transcriber = DashScopeTranscriber(model="qwen-vl-max")
        text = transcriber.transcribe(page_image_path, page_no)
    """

    def __init__(self, model: str = "qwen-vl-max", max_pages: int = 20):
        import os

        self.model = model
        self.max_pages = max_pages or int(os.getenv("VLM_MAX_PAGES_PER_DOC", "20"))

    def transcribe(self, page_image_path: str, page_no: int) -> str | None:
        # 2026-09-15：复用进程级单例，避免每次调用新建连接（见 app.core.openai_client）
        from app.core.openai_client import get_dashscope_client

        client = get_dashscope_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"file://{page_image_path}"}},
                            {"type": "text", "text": "请完整转录本页内容，保留版式信息。仅输出页面文字。"},
                        ],
                    }
                ],
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or None
        except Exception:  # noqa: BLE001 — 转录失败由编排降级
            return None


def get_transcriber() -> Transcriber:
    """工厂：VLM_ENABLED=1 且配 Key → DashScope；否则 NoneTranscriber。"""
    import os

    if os.getenv("VLM_ENABLED", "0") == "1" and settings.has_api_key:
        return DashScopeTranscriber()
    return NoneTranscriber()
