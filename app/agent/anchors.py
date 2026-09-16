"""主题锚点词表：判断一个 query 是否含「能独立定位知识库主题」的词。

**词表不手写**，由 `build_vocab()`（本模块）从语料（`data/bm25/corpus.db`）自动
生成到 `data/kb_anchors.json`；`ingest` 完成后会自动重建（见 `app/cli.py`），
手动重建用 `scripts/build_anchor_vocab.py`。本模块其余部分只负责加载与查询。

设计动机（2026-09-14）：`should_skip_rewrite` 采用**白名单**哲学——
默认改写，只在本句被证明「自足」时才跳过。自足的判据之一是「含主题锚点」。

为什么不能手写：手工列的话题词覆盖不住真实语料（实测 55 条单轮题里 52.7%
不含手写表里的词，而"打卡/营收/毛利/文档入库"都是真实话题）→ 词表必须来自语料。

为什么要有 `GENERIC_HEADS`：语料里「标准/规定/流程/管理/公司/员工/经理」这类
**通用中心词**高频出现，若当成锚点，则「部门经理的标准是多少？」会被误判为自足
（它其实必须靠上文才知道是哪个标准）。故生成时**含任一通用词的候选一律剔除**——
这同时把「部门经理」这类实体排除掉（含「经理」）。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

# 通用中心词 / 机构通用语（封闭类）：既用于生成时剔除候选，也用于运行时的
# 「…的 + 通用中心词」结构否决（见 prompts._is_self_contained）。
GENERIC_HEADS: tuple[str, ...] = (
    # 文档类型名词
    "标准", "规定", "制度", "办法", "流程", "规范", "要求", "细则", "指南", "手册",
    "方案", "通知", "说明", "指引", "协议", "承诺", "报告", "记录", "清单", "条例",
    "政策", "模板", "目录", "索引",
    # 抽象 / 度量通用语
    "管理", "金额", "费用", "时间", "情况", "部分", "条件", "范围", "内容", "事项",
    "原则", "目的", "依据", "结果", "数据", "信息", "系统", "服务", "工作", "方式",
    "上限", "额度", "比例", "类别", "等级", "状态", "期限", "天数",
    # 机构 / 角色通用语
    "人员", "公司", "部门", "员工", "主管", "经理", "负责人", "领导", "客户",
    "供应商", "产品", "业务", "项目", "计划", "账号", "权限",
    # 动词 / 程序通用语
    "执行", "审批", "申请", "提交", "完成", "应当", "有权", "责任", "义务", "适用",
)

# 功能词 / 疑问词 / 数量时间语：候选直接丢弃（生成期使用）
FUNCTION_STOP: tuple[str, ...] = (
    "什么", "怎么", "多少", "哪些", "哪个", "哪几", "可以", "需要", "是否", "如何",
    "以上", "以下", "之内", "之后", "之前", "之内", "一个", "两个", "三个", "每次",
    "每人", "每天", "每年", "每月", "每周", "一次", "两次", "三次", "其他", "我们",
    "你们", "他们", "这个", "那个", "这些", "那些", "其中", "根据", "按照", "由于",
    "因为", "所以", "但是", "如果", "并且", "以及", "或者", "不能", "不得", "进行",
    "有关", "相关", "本次", "一天", "一年", "一周", "一月", "半天", "当天", "年内",
    "不少于", "不超过", "不低于", "不得超", "视情况", "原则上", "特殊情况",
)

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "kb_anchors.json"
_NGRAM_LENS = (2, 3, 4)


def _vocab_path() -> Path:
    override = os.getenv("KB_ANCHORS_PATH")
    return Path(override) if override else _DEFAULT_PATH


@lru_cache(maxsize=1)
def load_terms() -> frozenset[str]:
    """加载锚点词表。文件缺失 / 损坏 → 返回空集（退化为「一律改写」，行为安全）。"""
    p = _vocab_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return frozenset(t for t in data.get("terms", []) if isinstance(t, str))
    except Exception:  # noqa: BLE001 — 缺词表不应让问答链路失败
        return frozenset()


def vocab_meta() -> dict:
    try:
        return json.loads(_vocab_path().read_text(encoding="utf-8")).get("meta", {})
    except Exception:  # noqa: BLE001
        return {}


def has_topic_anchor(query: str) -> bool:
    """query 是否含词表中的锚点。

    用**子串匹配**（按 query 的 2~4 字片段反查词表，O(len(q))）而非分词匹配：
    jieba 词典缺词（如「年假」被切成 年 + 假）会把真话题词切碎，分词匹配就漏了。
    词表本身已是「词级、去通用词」的干净集合，故子串匹配不会引入跨词边界的假命中。
    """
    terms = load_terms()
    if not terms:
        return False
    q = (query or "").strip()
    for n in _NGRAM_LENS:
        for i in range(len(q) - n + 1):
            if q[i:i + n] in terms:
                return True
    return False


def anchors_in(query: str) -> list[str]:
    """命中详情（调试 / 可观测用）。"""
    terms = load_terms()
    q = (query or "").strip()
    hits: list[str] = []
    for n in _NGRAM_LENS:
        for i in range(len(q) - n + 1):
            g = q[i:i + n]
            if g in terms and g not in hits:
                hits.append(g)
    return hits


# ── 词表生成（语料变了要重建；ingest 后自动调用，见 app/cli.py）──────

_EDGE_FUNCTION_CHARS = set(
    "的了与和及上不在是有为以由从对向并其该本这那我你他们一二三四五六七八九十"
    "百千万个之所可能要应须将被把让使则即如若且或但而等各每第条款章节项次种"
    "无未非很更最又也还就都只再另须经由此故"
)
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_CN_TAIL_RE = re.compile(r"[（(].*?[）)]|\s+")
_CHAPTER_RE = re.compile(r"第[一二三四五六七八九十]+[章条节]")
_KEEP_POS = ("n", "v")  # 名词类 / 动词类（含 vn 名动词）


def _blocked(term: str) -> bool:
    if not (2 <= len(term) <= 4) or not _CJK_RE.match(term):
        return True
    if any(g in term for g in GENERIC_HEADS):
        return True
    if any(s in term for s in FUNCTION_STOP):
        return True
    return term[0] in _EDGE_FUNCTION_CHARS or term[-1] in _EDGE_FUNCTION_CHARS


def _title_of(content: str, max_len: int = 24) -> str:
    head = (content or "").strip().split("\n")[0]
    head = _CN_TAIL_RE.sub("", head)
    return _CHAPTER_RE.split(head)[0][:max_len].strip()


def _composites_from(tokens) -> set[str]:
    """相邻 CJK token 拼接成复合词（2~4 字，过 `_blocked`）。

    jieba 常把「加班费」切成「加班/费」、「餐补」切成「餐/补」——单字/双字被拆后，
    真话题词反而进不了词表（且「加班」跨文档 DF 超限被过滤）。把相邻 token 重新拼回，
    复合词只在少数文档出现 → DF 带内 → 被保留为锚点（如「加班费」「餐补」）。
    按相邻成对拼接（不跨标点、不跨句），长度限 2~4 字避免噪音。
    """
    out: set[str] = set()
    run: list[str] = []
    for w, _f in tokens:
        if _CJK_RE.match(w):
            run.append(w)
            if len(run) >= 2:
                g = run[-2] + run[-1]
                if 2 <= len(g) <= 4 and not _blocked(g):
                    out.add(g)
        else:
            run = []
    return out


def build_vocab(db_path: str | Path, out_path: str | Path,
                max_df: int = 8) -> dict:
    """从 BM25 语料生成锚点词表并落盘，返回 meta（语料变更后需重建）。

    三层来源（详见模块 docstring）：标题词 + 标题 2 字片段 + 正文词（DF 带内），
    再按词性（n*/v*）与通用词/功能词/边界规则过滤。
    """
    import jieba.posseg as pseg  # noqa: PLC0415 — 延迟导入，避免拖慢链路冷启动

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = list(con.execute(
        "select doc_id, chunk_id, content from bm25_corpus where is_valid = 1"))
    con.close()

    by_doc: dict[str, list[tuple[str, str]]] = {}
    for doc_id, chunk_id, content in rows:
        by_doc.setdefault(doc_id, []).append((chunk_id, content or ""))

    term_doc: dict[str, set[str]] = {}
    composite_doc: dict[str, set[str]] = {}
    title_tokens: set[str] = set()
    title_ngrams: set[str] = set()
    titles: dict[str, str] = {}

    for doc_id, chunks in by_doc.items():
        chunks.sort()
        title = _title_of(chunks[0][1])
        titles[doc_id] = title
        title_toks = list(pseg.lcut(title))
        for w, f in title_toks:
            if f[0] in _KEEP_POS and not _blocked(w):
                title_tokens.add(w)
        for i in range(len(title) - 1):
            g = title[i:i + 2]
            if not _blocked(g):
                title_ngrams.add(g)
        for g in _composites_from(title_toks):
            composite_doc.setdefault(g, set()).add(doc_id)
        for _cid, content in chunks:
            content_toks = list(pseg.lcut(content))
            for w, f in content_toks:
                if f[0] in _KEEP_POS and not _blocked(w):
                    term_doc.setdefault(w, set()).add(doc_id)
            for g in _composites_from(content_toks):
                composite_doc.setdefault(g, set()).add(doc_id)

    body_terms = {t for t, docs in term_doc.items() if 1 <= len(docs) <= max_df}
    body_composites = {t for t, docs in composite_doc.items() if 1 <= len(docs) <= max_df}
    terms = sorted(title_tokens | title_ngrams | body_terms | body_composites)
    data = {
        "meta": {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": str(db_path),
            "tokenizer": "jieba.posseg + title n-gram",
            "docs": len(by_doc), "chunks": len(rows), "max_df": max_df,
            "counts": {"title_tokens": len(title_tokens),
                       "title_ngrams": len(title_ngrams - title_tokens),
                       "body_tokens": len(body_terms - title_tokens - title_ngrams),
                       "body_composites": len(body_composites - title_tokens - title_ngrams - body_terms),
                       "union": len(terms)},
        },
        "titles": titles,
        "terms": terms,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    load_terms.cache_clear()  # 重建后立刻生效
    return data["meta"]
