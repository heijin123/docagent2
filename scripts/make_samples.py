"""生成 M1 演示语料到 data/samples/。

- sample_guide.md：md 标题结构 + 表格 + 代码块（测试标题结构切分）
- sample_notes.txt：txt 段落（测试段落切）
- sample_policy.docx：docx 标题 + 段落 + 表格（测试 docx 解析）
- sample_manual.pdf：纯文本 PDF（测试 pdf 解析 + 质量门绿页）
- sample_scan_page.pdf：含一页纯图/少文本的 PDF（测试质量门红页降级路径）
"""
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # scripts/.. = 项目根? 本文件放 scripts/
SAMPLES = BASE / "data" / "samples"

MD_CONTENT = """# 员工手册 v2

## 1. 考勤制度

员工每天上班需打卡，考勤周期为自然月。

### 1.1 打卡规则

- 上班时间不晚于 09:30
- 下班时间不早于 18:00
- 一天至少打卡两次

```python
def is_late(clock_in):
    return clock_in > datetime(9, 30)
```

## 2. 报销规范

报销单编号规则：XB- 开头，后接 6 位数字。

| 项目 | 上限 | 说明 |
|---|---|---|
| 差旅 | 500/天 | 需发票 |
| 餐饮 | 100/天 | 无需发票 |

### 2.1 审批链

报销 5000 元以上需总监审批。
"""

TXT_CONTENT = """企业知识库使用说明

本系统支持 PDF、Word、Markdown 与纯文本格式文档的入库检索。

上传文档后系统会进行解析、切分与向量化，回答问题时附带引用来源。

首次使用请先阅读《管理员手册》。若检索不到结果，请检查文档是否已成功入库。
"""

DOCX_REQ = "2026-09-09 员工休假制度 试运行稿"


def gen_docx(path: Path, title: str = "员工休假制度") -> None:
    import docx
    from docx.shared import Pt

    document = docx.Document()
    document.add_heading(title, level=1)
    document.add_heading("一、适用范围", level=2)
    document.add_paragraph("本制度适用于全体员工。年假按自然年计算，可跨年累计不超过 5 天。")
    document.add_heading("二、请假流程", level=2)
    document.add_paragraph("请假需提前一天在系统提交申请，由直属主管审批。")
    table = document.add_table(rows=3, cols=2)
    table.style = "Table Grid"
    for r, (k, v) in enumerate([
        ("请假类型", "审批人"),
        ("年假/事假", "直属主管"),
        ("病假(>3天)", "部门总监"),
    ]):
        table.cell(r, 0).text = k
        table.cell(r, 1).text = v
    document.add_heading("三、附则", level=2)
    document.add_paragraph("本制度自发布之日起试行，解释权归人力资源部。")
    document.save(str(path))


def _cjk_font():
    """PyMuPDF 默认字体不支持中文：优先用 simhei（Windows 常见），否则 None（仅 ASCII）。"""
    import os

    for candidate in (
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def gen_pdf(path: Path, with_scan_page: bool = False) -> None:
    import pymupdf

    font_file = _cjk_font()
    doc = pymupdf.open()
    page1 = doc.new_page()
    if font_file:
        page1.insert_font(fontname="cjk", fontfile=font_file)
        fontname = "cjk"
    else:
        fontname = "helv"  # 无中文字体则降级（演示语料 ASCII）
    page1.insert_textbox(
        pymupdf.Rect(50, 60, 545, 200),
        "年度经营分析报告\n\n本报告覆盖上半年经营数据，含营收、成本与毛利分析。\n"
        "营收同比增长 12%，成本控制在预算内，毛利提升至 28%。",
        fontsize=12,
        fontname=fontname,
    )
    if with_scan_page:
        # 第二页：纯图片页（无文本）→ 质量门红页
        page2 = doc.new_page()
        page2.draw_rect(pymupdf.Rect(60, 120, 500, 500), color=(0.2, 0.2, 0.2), width=1.5)
        page2.insert_textbox(pymupdf.Rect(80, 130, 480, 480), "", fontsize=12)
    doc.save(str(path))
    doc.close()


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (SAMPLES / "sample_guide.md").write_text(MD_CONTENT, encoding="utf-8")
    (SAMPLES / "sample_notes.txt").write_text(TXT_CONTENT, encoding="utf-8")
    gen_docx(SAMPLES / "sample_policy.docx")
    gen_pdf(SAMPLES / "sample_manual.pdf", with_scan_page=False)
    gen_pdf(SAMPLES / "sample_scan_page.pdf", with_scan_page=True)
    print(f"演示语料已生成到 {SAMPLES}")


if __name__ == "__main__":
    main()
