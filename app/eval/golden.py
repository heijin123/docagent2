"""golden 集加载 + 校验 + 锚句定位（F7.1）。

设计（吸收 doc-agent，零人工标注块 id）：
- golden 每条 = query + expected_doc + anchor（锚句）；
- 锚句经 `normalize` 归一化（去空白/全半角/大小写）后，与全库有效 chunk content
  做子串匹配 → 定位期望块集合；
- 校验规则（失败即不跑，见 F7.1）：缺字段 / 重复 id / 锚句过短（<6 字）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings

MIN_ANCHOR_LEN = 6

_WS_RE = re.compile(r"\s+")
_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９（）【】，。；：？！％．ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789()[],.;:?!%." + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 2,
)


def normalize(text: str) -> str:
    """归一化：全半角统一 + 去所有空白 + 小写（子串匹配的稳健基础）。"""
    t = text.translate(_FULLWIDTH)
    t = _WS_RE.sub("", t)
    return t.lower()


@dataclass
class GoldenCase:
    id: str
    query: str
    expected_doc: str
    anchor: str
    type: str = "semantic"
    # 运行时填充
    expected_chunk_ids: list[str] = field(default_factory=list)
    located: bool = False


def load_golden(path: str | Path | None = None) -> tuple[list[GoldenCase], dict]:
    """加载 + 校验 golden 集。返回 (cases, meta)。

    校验失败抛 ValueError（缺字段 / 重复 id / 锚句过短），符合 F7.1「失败不跑」。
    """
    p = Path(path or (settings.base_dir / "data" / "golden" / "qa_golden.json"))
    raw = json.loads(p.read_text(encoding="utf-8"))
    meta = raw.get("meta", {})
    cases_raw = raw.get("cases", [])
    if not cases_raw:
        raise ValueError("golden 集为空（cases=[]）")

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for i, c in enumerate(cases_raw):
        cid = c.get("id", "")
        query = c.get("query", "")
        expected_doc = c.get("expected_doc", "")
        anchor = c.get("anchor", "")
        if not (cid and query and expected_doc and anchor):
            raise ValueError(f"golden #{i} 缺字段（id/query/expected_doc/anchor 必填）")
        if cid in seen:
            raise ValueError(f"golden id 重复: {cid}")
        if len(normalize(anchor)) < MIN_ANCHOR_LEN:
            raise ValueError(f"golden {cid} 锚句过短（<{MIN_ANCHOR_LEN} 字）: {anchor!r}")
        seen.add(cid)
        cases.append(GoldenCase(
            id=cid, query=query, expected_doc=expected_doc,
            anchor=anchor, type=c.get("type", "semantic")))
    return cases, meta


def locate_expected_chunks(cases: list[GoldenCase], bm25_store,
                           tenant_id: str | None = None) -> dict[str, GoldenCase]:
    """锚句定位期望块：遍历全库有效 chunk，归一化子串匹配。

    返回 {case_id: case}（原地填充 expected_chunk_ids / located）。
    锚句定位失败（内容已改 / 未入库）→ located=False，该用例不计指标分母（F7.2）。
    """
    # 预归一化全库 chunk content（一次扫描）
    corpus: list[tuple[str, str, str]] = []  # (chunk_id, norm_content, raw_content)
    for row in bm25_store.iter_valid_chunks(tenant_id=tenant_id):
        corpus.append((row["chunk_id"], normalize(row["content"]), row["content"]))

    for case in cases:
        anchor_n = normalize(case.anchor)
        hits = [cid for cid, nc, _ in corpus if anchor_n in nc]
        case.expected_chunk_ids = hits
        case.located = bool(hits)
    return {c.id: c for c in cases}
