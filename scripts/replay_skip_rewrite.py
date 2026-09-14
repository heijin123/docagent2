"""校验 `should_skip_rewrite` 白名单判据（纯规则，零 LLM）。

对「旧实现（有历史就改写）」与新实现（白名单：自足才跳过）做离线回放，并
统计新判据在语料上的真实行为：golden 多轮 5 条 / 单轮 55 条 / 人工探针。

用法：PYTHONPATH=D:/workspace/docagent2 python scripts/replay_skip_rewrite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent import anchors, prompts  # noqa: E402

SINGLE = ROOT / "data" / "golden" / "qa_golden.json"
MULTI = ROOT / "data" / "golden" / "qa_golden_multiturn.json"
RESULT = ROOT / "data" / "reports" / "eval_multiturn_latest.json"


def judge_old(query: str, history: str, retry_count: int) -> bool:
    """旧实现（原样复刻，用于对照）。"""
    if retry_count > 0:
        return False
    if history and history.strip():
        return False
    q = (query or "").strip()
    if not q:
        return False
    return not any(h in q for h in prompts._PRONOMINAL_HINTS)


def judge_new(query: str, history: str, retry_count: int) -> bool:
    return prompts.should_skip_rewrite(query, history, retry_count)


def judge_no_veto(query: str, history: str, retry_count: int) -> bool:
    """变体：去掉「…的 + 通用中心词」否决，只留回指 / 长度 / 锚点三条。"""
    if retry_count > 0:
        return False
    q = (query or "").strip()
    if not q:
        return False
    ok = (not any(h in q for h in prompts._PRONOMINAL_HINTS)
          and len(q) > prompts._MIN_SELF_CONTAINED_LEN
          and anchors.has_topic_anchor(q))
    if ok:
        return True
    return not (history and history.strip())


JUDGES = {"旧": judge_old, "新(白名单)": judge_new, "新-去否决": judge_no_veto}

# 人工探针：(query, 有上文时理想判定, 备注)。理想值填 "任意" 表示不比对（已知保守偏差）。
PROBES = [
    ("园区停车月卡多少钱？", "跳过", ""),
    ("差旅住宿标准是多少？", "跳过", ""),
    ("年假可以跨年累计多少天？", "跳过", "jieba 词典缺「年假」，靠标题片段补回"),
    ("差旅报销的住宿上限是多少？", "跳过", "的紧邻「住宿」而非通用词，否决不触发"),
    ("企业发票怎么开？", "任意", "7 字 ≤ 8，被判需改写——短句下限的保守代价"),
    ("员工手册在哪里下载？", "跳过", "靠标题片段「工手」命中（结论正确、机制偶然）"),
    ("停车的呢？", "改写", "有锚点但 4 字 ≤ 8 → 短句下限拦住"),
    ("那二线城市呢？", "改写", "回指"),
    ("部门经理的标准是多少？", "改写", "「部门经理」实体不入词表，且无锚点"),
    ("那市内交通补贴呢？", "改写", "回指"),
    ("这个制度什么时候开始执行？", "改写", "回指"),
    ("标准是多少？", "改写", "无锚点 + 通用中心词"),
    ("差旅报销每天的上限是多少？", "改写", "的紧邻通用词「上限」→ 结构否决"),
]


def main() -> None:
    meta = anchors.vocab_meta()
    print(f"锚点词表：{len(anchors.load_terms())} 个词  {meta.get('counts', {})}")
    print(f"          built_at={meta.get('built_at')}  max_df={meta.get('max_df')}")
    print()

    print("=" * 100)
    print("一、golden 多轮 5 条末轮（末轮有上文，判据真正起作用的地方）")
    print("=" * 100)
    golden = json.loads(MULTI.read_text(encoding="utf-8"))
    res = json.loads(RESULT.read_text(encoding="utf-8"))
    hist = {c["id"]: c["final_history_text"] for c in res["per_case"]}
    ident = {c["id"]: c["turns"][-1]["rewrite_identity"] for c in res["per_case"]}
    print(f"{'case':6}{'cat':12}{'末轮 query':26}{'应有':8}"
          + "".join(f"{k:10}" for k in JUDGES) + " 实际")
    print("-" * 100)
    rows = []
    for case in golden["cases"]:
        cid, cat = case["id"], case["category"]
        q = case["turns"][-1]
        expect = "skip" if cat == "independent" else "llm"
        v = {k: ("skip" if fn(q, hist.get(cid, ""), 0) else "llm") for k, fn in JUDGES.items()}
        print(f"{cid:6}{cat:12}{q:26}{expect:8}"
              + "".join(f"{v[k]:10}" for k in JUDGES)
              + ("  逐字相同(空转)" if ident.get(cid) else ""))
        rows.append((cid, expect, v))
    print()
    for k in JUDGES:
        bad = [cid for cid, e, v in rows if v[k] != e]
        print(f"  {k:10} 一致 {len(rows)-len(bad)}/{len(rows)}"
              + (f"   ✗ {','.join(bad)}" if bad else ""))

    print()
    print("=" * 100)
    print("二、单轮 55 条：首轮无上文（新判据恒跳过；此处测「若在多轮里会不会自足」）")
    print("=" * 100)
    single = json.loads(SINGLE.read_text(encoding="utf-8"))["cases"]
    sc = [c for c in single if prompts._is_self_contained(c["query"])]
    print(f"自足 {len(sc)}/{len(single)}（{len(sc)/len(single):.1%}）→ 多轮里这些题能省掉改写跳")
    by_type: dict[str, list[int]] = {}
    for c in single:
        t = c.get("type", "?")
        by_type.setdefault(t, [0, 0])
        by_type[t][0] += 1
        if prompts._is_self_contained(c["query"]):
            by_type[t][1] += 1
    for t, (tot, ok) in sorted(by_type.items()):
        print(f"    {t:12} 自足 {ok}/{tot}")
    print("\n  非自足样例（含原因，前 10 条）：")
    shown = 0
    for c in single:
        q = c["query"]
        if prompts._is_self_contained(q):
            continue
        why = []
        if any(h in q for h in prompts._PRONOMINAL_HINTS):
            why.append("回指")
        if len(q) <= prompts._MIN_SELF_CONTAINED_LEN:
            why.append(f"≤{prompts._MIN_SELF_CONTAINED_LEN}字")
        if prompts._has_generic_head_after_de(q):
            why.append("的+通用中心词")
        if not anchors.has_topic_anchor(q):
            why.append("无锚点")
        print(f"    [{c['type']:10}] {q:26} {'/'.join(why)}")
        shown += 1
        if shown >= 10:
            break

    print()
    print("=" * 100)
    print("三、人工探针（含假锚点对抗样例）")
    print("=" * 100)
    print(f"{'query':26}{'理想':7}" + "".join(f"{k:10}" for k in JUDGES) + " 备注")
    print("-" * 100)
    mismatch = []
    for q, want, note in PROBES:
        v = {k: ("skip" if fn(q, "存在上文", 0) else "llm") for k, fn in JUDGES.items()}
        got = "跳过" if v["新(白名单)"] == "skip" else "改写"
        if want != "任意" and want != got:
            mismatch.append((q, want, got))
        print(f"{q:26}{want:7}" + "".join(f"{v[k]:10}" for k in JUDGES) + f" {note}")
    print(f"\n  与理想不符：{len(mismatch)}" +
          ("".join(f"\n    ✗ {q} 理想={w} 实得={g}" for q, w, g in mismatch) if mismatch else ""))

    print()
    print("=" * 100)
    print("四、短句下限敏感性（_MIN_SELF_CONTAINED_LEN）")
    print("=" * 100)
    origin = prompts._MIN_SELF_CONTAINED_LEN
    golden_cases = [(c["id"], c["turns"][-1], "skip" if c["category"] == "independent" else "llm")
                    for c in golden["cases"]]
    guard = [(q, want) for q, want, _ in PROBES if want != "任意"]
    print(f"{'阈值':6}{'单轮自足':10}{'golden一致':12}{'探针一致':10}")
    for th in (4, 5, 6, 7, 8, 9, 10):
        prompts._MIN_SELF_CONTAINED_LEN = th
        sc_n = sum(1 for c in single if prompts._is_self_contained(c["query"]))
        g_ok = sum(1 for _cid, q, want in golden_cases
                   if ("skip" if prompts.should_skip_rewrite(q, "x", 0) else "llm") == want)
        p_ok = sum(1 for q, want in guard
                   if ("跳过" if prompts.should_skip_rewrite(q, "x", 0) else "改写") == want)
        mark = " ← 当前" if th == origin else ""
        print(f"{th:<6}{f'{sc_n}/{len(single)}':10}{f'{g_ok}/{len(golden_cases)}':12}"
              f"{f'{p_ok}/{len(guard)}':10}{mark}")
    prompts._MIN_SELF_CONTAINED_LEN = origin


if __name__ == "__main__":
    main()
