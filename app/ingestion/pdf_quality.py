"""PDF 质量门：逐页红/黄/绿三档判级（需求 F1.9，吸收自 doc-agent）。

全部信号零 LLM 成本、阈值可调（config.quality_*）。判级对象为 PyMuPDF 文本抽取结果。

| 档 | 含义 | 处置 |
|---|---|---|
| red   | 文本不可信：乱码率高 / 纯图扫描页（几乎无文本） | 不产块；转 VLM 转录（无 Key 降级原文 + note） |
| yellow| 文本可用但版面复杂：文本稀疏 / 含大量图片 | 正常入库 + parse_warning |
| green | 正常放行 | 直接入库 |
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings

# 乱码信号：U+FFFD 替换符 / C0 控制字符（\t\n\r 除外）
_GARBLE_CHARS = str.maketrans({"\ufffd": "\ufffd"})
_CONTROL_EXCLUDE = {"\t", "\n", "\r"}


@dataclass
class PageQuality:
    """单页质量判级结果。"""

    page_no: int          # 1-based，对应 PDF 页码
    level: str            # red / yellow / green
    text_chars: int = 0   # 有效文本字符数
    garble_ratio: float = 0.0
    image_count: int = 0
    signals: list[str] = field(default_factory=list)


def _clean_text(text: str) -> str:
    """剔除控制字符后返回，供计数/乱码率计算。"""
    return "".join(ch for ch in text if ch not in _CONTROL_EXCLUDE)


def _count_garble(text: str) -> int:
    """乱码字符数：U+FFFD + 遗留控制字符。"""
    return sum(1 for ch in text if ch == "\ufffd" or (ord(ch) < 32 and ch not in _CONTROL_EXCLUDE))


def grade_page(page_no: int, text: str, image_count: int = 0) -> PageQuality:
    """对单页文本判级。text 为 PyMuPDF page.get_text() 原始输出。"""
    cleaned = _clean_text(text)
    text_chars = len(cleaned.strip())
    garble_count = _count_garble(cleaned)
    garble_ratio = garble_count / text_chars if text_chars > 0 else 0.0

    signals: list[str] = []
    level = "green"

    # 红：乱码率超阈值（坏字体）
    if text_chars > 0 and garble_ratio > settings.quality_garble_threshold:
        level = "red"
        signals.append(f"乱码率 {garble_ratio:.1%} > {settings.quality_garble_threshold:.0%}")

    # 红：纯图/扫描页（几乎无文本）
    if text_chars < settings.quality_min_text_chars:
        level = "red"
        signals.append(f"文本过短 {text_chars} 字符 < {settings.quality_min_text_chars}（疑似扫描/纯图页）")

    # 黄：文本稀疏或版面复杂
    if level == "green" and (
        text_chars < settings.quality_yellow_text_chars or image_count >= 3
    ):
        level = "yellow"
        signals.append(f"文本稀疏 {text_chars} 字符或含 {image_count} 张图（版面复杂）")

    return PageQuality(
        page_no=page_no,
        level=level,
        text_chars=text_chars,
        garble_ratio=round(garble_ratio, 4),
        image_count=image_count,
        signals=signals,
    )


def summarize(pages: list[PageQuality]) -> dict:
    """整文档统计：{red: [页码], yellow: [页码], green: N}。"""
    return {
        "red_pages": [p.page_no for p in pages if p.level == "red"],
        "yellow_pages": [p.page_no for p in pages if p.level == "yellow"],
        "green_count": sum(1 for p in pages if p.level == "green"),
    }
