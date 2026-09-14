"""手动重建主题锚点词表（自动重建已挂在 `app.cli ingest` 之后）。

用法：
  PYTHONPATH=D:/workspace/docagent2 python scripts/build_anchor_vocab.py --show
  PYTHONPATH=D:/workspace/docagent2 python scripts/build_anchor_vocab.py --max-df 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent import anchors  # noqa: E402

DB = ROOT / "data" / "bm25" / "corpus.db"
OUT = ROOT / "data" / "kb_anchors.json"

PROBES = ["报销", "住宿", "停车", "年假", "月卡", "打卡", "营收", "毛利", "入库",
          "加班", "差旅", "发票", "补贴", "请假", "考勤", "事假",
          "部门经理", "标准", "规定", "流程", "上限", "费用", "公司", "员工",
          "管理", "一定", "万元", "七天", "一人", "工手", "业发"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-df", type=int, default=8, help="正文词最大文档频率")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    meta = anchors.build_vocab(DB, OUT, max_df=args.max_df)
    print(f"✓ {OUT.relative_to(ROOT)}  {meta['counts']}  docs={meta['docs']}")

    if args.show:
        terms = sorted(anchors.load_terms())
        print(f"\n[词表] 共 {len(terms)}  长度分布="
              + str({n: sum(1 for t in terms if len(t) == n) for n in (2, 3, 4)}))
        for n in (2, 3, 4):
            grp = [t for t in terms if len(t) == n]
            print(f"  {n} 字: {' '.join(grp[:45])}")
        print("\n[探针]")
        ts = set(terms)
        for p in PROBES:
            print(f"  {p:8}{'✓ 锚点' if p in ts else '✗ 不是'}")


if __name__ == "__main__":
    main()
