"""VLM 转录器（需求 F1.9，M2 真实接入 Qwen-VL：红页整页 OCR + 非红页内嵌图 OCR）。

- 红页（扫描/乱码）→ 渲染整页 PNG → Transcriber.transcribe(png_bytes) → 文本；
- 非红页内嵌图（图表/带字示意图）→ 抽取图片字节 → Transcribe.transcribe(img_bytes, mime=…) → 文本；
- 无 Key / 未启用 / 失败 / 超 max_pages → 返回 None，编排层降级为原文入库 + vlm_note（R7 不崩）。
- 远程 DashScope（OpenAI 兼容）**不接受 file:// 本地路径**，图片必须以 base64 data URL 传入；
  故 transcribe 接收内存中的图片字节 + MIME，由调用方（pipeline）渲染页面/抽取图片后传入。

token 记账：每次 VLM 调用 emit log_llm_usage（node="vlm_ocr"），对齐「token 是一等公民」。
"""
from __future__ import annotations

import base64
import time
import os

from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.observability import TokenUsage, log_llm_usage


class Transcriber(ABC):
    """图片（红页整页 / 内嵌图）转录接口。"""

    @abstractmethod
    def transcribe(self, image_bytes: bytes, page_no: int, *,
                   mime: str = "image/png", prompt: str | None = None) -> str | None:
        """转录单张图片（内存字节）为文本；失败返回 None。

        mime: 图片 MIME（image/png|image/jpeg…），用于构造 data URL；
        prompt: 可选自定义提示词（红页整页 vs 内嵌图可不同），None 用默认 OCR 提示。
        """


class NoneTranscriber(Transcriber):
    """无 Key / 未启用：恒返回 None，编排降级原文入库。"""

    def transcribe(self, image_bytes: bytes, page_no: int, *,
                   mime: str = "image/png", prompt: str | None = None) -> str | None:
        return None


class DashScopeTranscriber(Transcriber):
    """Qwen-VL 图片转录（OpenAI 兼容 /chat/completions，image_url 传 base64 data URL）。

    既用于红页整页 OCR，也用于非红页内嵌图 OCR（图表/带字示意图）。
    远程 DashScope 不接受 file:// 本地路径，故 transcribe 接收内存图片字节 + MIME，
    由调用方（pipeline）渲染页面/抽取图片后传入。

    用法示例：
        transcriber = DashScopeTranscriber(model="qwen-vl-max")
        text = transcriber.transcribe(page_png_bytes, page_no)              # 红页整页
        text = transcriber.transcribe(img_bytes, page_no, mime="image/jpeg")  # 内嵌图
    """

    def __init__(self, model: str = "qwen-vl-max", max_pages: int = 20):
        import os

        self.model = model
        self.max_pages = max_pages or int(os.getenv("VLM_MAX_PAGES_PER_DOC", "20"))

    def transcribe(self, image_bytes: bytes, page_no: int, *,
                   mime: str = "image/png", prompt: str | None = None) -> str | None:
        # 2026-09-15：复用进程级单例，避免每次调用新建连接（见 app.core.openai_client）
        from app.core.openai_client import get_dashscope_client

        client = get_dashscope_client()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        ocr_prompt = prompt or (
            "请完整转录本页全部文字内容，保留原有的标题、段落与表格结构，"
            "用 Markdown 输出；仅输出页面文字，不要解释或补充。"
        )
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": ocr_prompt},
                        ],
                    }
                ],
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
            # token 记账：VLM 也是 LLM 调用，prompt_tokens 含图片 token 成本
            usage = getattr(resp, "usage", None)
            if usage is not None:
                log_llm_usage(
                    "", node="vlm_ocr", model=self.model,
                    usage=TokenUsage(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    ),
                    est_cost_cny=0.0,  # 摄取期成本未计入按查询账本，token 数已记录
                )
            return text or None
        except Exception:  # noqa: BLE001 — 转录失败由编排降级
            _dt = int((time.time() - t0) * 1000)
            from app.core.logging import setup_logging
            setup_logging("vlm").warning(
                "VLM 转录失败 page=%s model=%s %dms", page_no, self.model, _dt)
            return None


def get_transcriber() -> Transcriber:
    """工厂：VLM_ENABLED=1 且配 Key → DashScope；否则 NoneTranscriber。"""
    if settings.vlm_enabled and settings.has_api_key:
        return DashScopeTranscriber(model=settings.qwen_vlm_model)
    return NoneTranscriber()
