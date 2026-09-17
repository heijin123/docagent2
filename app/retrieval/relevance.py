"""相关性判定（F2.10）：让系统知道「自己不知道」。

**问题**：向量检索恒返回 topN（Chroma 无距离阈值），库里没有对应资料时也会给回
「最像的 8 条」→ `retrieve` 结果恒非空 → `no_data`（0 LLM 如实告知缺失）只在**索引真空**
时才触发。正常"库里没这个主题"的场景要烧 answer + verify 2~3 轮 LLM，靠模型读 prompt
自觉说"资料不足"——验收标准 3（无相关内容时明确降级不编造）此前靠自觉，不靠机制。

**标定实验（2026-09-17，scripts/calibrate_relevance.py + calibrate_coverage.py）**
取 golden 55 条（语料覆盖）vs 12 条（语料不覆盖）实测两组分数分布：

| 信号 | 组 A（应命中） | 组 B（应无资料） | 可分性 |
|---|---|---|---|
| 向量 top1 余弦 | min **0.406** / med 0.639 | max **0.542** / med 0.422 | ❌ 大面积重叠 |
| 词表覆盖度（IDF 加权） | min 0.110 / med 0.725 | p75 0.505 / 4 条为 0 | ❌ 口语改写会掉覆盖 |
| **复合**（覆盖 < C 且余弦 < T） | C=0.20/T=0.40 → **55/55 保留** | 4/12 判出 | ✅ 零误杀 |

**设计纪律（本模块只做高精度短路）**：
- 两个条件**必须同时**成立才判"无相关资料"——单看任一项都会误杀可答题；
- 宁可**漏判**（继续走 LLM 路径，最坏是原来那 2~3 轮）也**不误杀**（把库里有答案的
  问题回答成"没资料"是实质性回归）；
- 命中即走 `no_data` 出口（0 次 LLM），与出口③「查不到 → 告知缺失」语义一致。

**词表覆盖度**用 IDF 加权：越罕见的词缺席，越说明整个话题不在库里（"是什么 / 流程 /
标准"这类烂大街词缺席与否没有信息量）。用 `Σ idf(命中的词) / Σ idf(全部内容词)`。
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 功能词 / 疑问词：不承载主题。纳入计算会把"是什么/怎么样"当成覆盖，稀释信号。
STOP_TERMS = {
    "的", "了", "吗", "呢", "是", "有", "在", "和", "与", "及", "或", "怎么", "怎样", "如何",
    "什么", "哪些", "哪个", "多少", "为什么", "可以", "需要", "应该", "会不会", "是否",
    "一下", "一般", "通常", "我们", "你们", "公司", "本", "该", "这", "那", "个", "项",
    "请问", "告诉", "介绍", "情况", "方面",
    # 2026-09-17 补：纯疑问/虚词，留着会把"怎么样呢"这类无主题句判成"覆盖 0"而误触发
    "的话", "怎么样", "怎么办", "干嘛", "多久", "多长时间", "哪里", "哪儿", "多少度",
}

_NON_WORD_RE = re.compile(r"[\d\W_]+")

# 字符级兜底词表规模上限：超过则不建（避免大语料下内存/构建成本失控，
# 此时退化为"只按分词精确匹配"，宁可少判也不误判）
_MAX_FALLBACK_CHARS = 4_000_000


def content_terms(query: str) -> list[str]:
    """内容词 = jieba 分词 − 停用词 − 单字 − 纯数字/标点（与 BM25 同一套分词）。"""
    from app.retrieval.bm25store import tokenize

    out: list[str] = []
    for t in tokenize(query or ""):
        t = t.strip()
        if not t or t in STOP_TERMS or len(t) < 2 or _NON_WORD_RE.fullmatch(t):
            continue
        out.append(t)
    return out


@dataclass
class CorpusVocabulary:
    """语料词表 + 文档频次（覆盖度计算的基础）。

    **为什么还要字符级兜底**（2026-09-17 实测踩到）：jieba 分词是**上下文相关**的——
    同一串字在不同语境会被切得不同（语料里"员工年假制度…可跨年休"切成 `年/假/跨/年/休`，
    而查询"年假可以跨年休吗"切成 `年/假/跨年/休`）。纯词级精确匹配会因此把**真实存在
    的主题**判成缺席，压低覆盖度 → 有误杀风险。故 df 查不到时，再看该词是否以
    **字符二元组子集**形式存在于语料（近似子串判定）。方向刻意偏保守：
    判定"存在"更容易 → 覆盖度偏高 → 更少触发"无相关资料"（漏判可忍，误杀不可忍）。
    """

    df: Counter
    n_chunks: int
    bigrams: set[str] | None = None

    @classmethod
    def build(cls, bm25_store) -> CorpusVocabulary:
        from app.retrieval.bm25store import tokenize

        df: Counter = Counter()
        bigrams: set[str] = set()
        total_chars = 0
        n = 0
        for row in bm25_store.iter_valid_chunks():
            n += 1
            content = row["content"] or ""
            for t in set(tokenize(content)):
                df[t] += 1
            if total_chars <= _MAX_FALLBACK_CHARS:
                flat = re.sub(r"\s+", "", content)
                total_chars += len(flat)
                bigrams.update(flat[i:i + 2] for i in range(len(flat) - 1))
        return cls(df=df, n_chunks=n,
                   bigrams=bigrams if total_chars <= _MAX_FALLBACK_CHARS else None)

    def _contains(self, term: str) -> bool:
        """字符级兜底：term 的每个字符二元组都出现在语料里（近似子串判定）。"""
        if self.bigrams is None or len(term) < 2:
            return False
        return all(term[i:i + 2] in self.bigrams for i in range(len(term) - 1))

    def df_of(self, term: str) -> int:
        """词频：词级精确命中优先；否则字符级兜底命中记 1（罕见但确实存在）。"""
        d = self.df.get(term, 0)
        if d:
            return d
        return 1 if self._contains(term) else 0

    def idf(self, term: str) -> float:
        """IDF：缺席词给 df=0 → 最大权重（"这个词库里一次都没出现"）。"""
        return math.log((self.n_chunks + 1) / (self.df_of(term) + 1)) + 1.0

    def coverage(self, query: str) -> tuple[float, list[str]]:
        """返回 (覆盖度, 缺席词表)。无内容词 → (1.0, [])（判不了，按"有覆盖"放行）。"""
        terms = content_terms(query)
        if not terms:
            return 1.0, []
        total = sum(self.idf(t) for t in terms)
        absent = [t for t in terms if self.df_of(t) == 0]
        hit = sum(self.idf(t) for t in terms if self.df_of(t) > 0)
        return (hit / total if total else 1.0), absent


class RelevanceGate:
    """按「语料覆盖 + 语义相似」双证据判定"库里到底有没有相关内容"。

    词表懒加载并按语料 chunk 数变化失效（ingest 后可自动跟上，无需重启）；
    小语料下重建一次是毫秒级，大语料请看 `_vocab_ttl_s` 兜底节流。
    """

    def __init__(self, bm25_store, *, min_coverage: float | None = None,
                 min_similarity: float | None = None, vocab_ttl_s: float = 300.0):
        from app.core.config import settings

        self.bm25_store = bm25_store
        self.min_coverage = (settings.retrieval_min_coverage
                             if min_coverage is None else min_coverage)
        self.min_similarity = (settings.retrieval_min_similarity
                               if min_similarity is None else min_similarity)
        self._vocab_ttl_s = vocab_ttl_s
        self._vocab: CorpusVocabulary | None = None
        self._vocab_n: int = -1
        self._vocab_built_at: float = 0.0

    def _get_vocab(self) -> CorpusVocabulary | None:
        try:
            n = self.bm25_store.count()
        except Exception:  # noqa: BLE001 — 取不到语料规模 → 不判（放行走 LLM）
            return None
        if n <= 0:
            return None
        fresh = (self._vocab is not None and n == self._vocab_n
                 and (time.time() - self._vocab_built_at) < self._vocab_ttl_s)
        if not fresh:
            self._vocab = CorpusVocabulary.build(self.bm25_store)
            self._vocab_n = n
            self._vocab_built_at = time.time()
        return self._vocab

    def assess(self, query: str, vector_hits: list[dict]) -> dict:
        """双证据判定。返回 dict（恒含 `no_relevant`，供节点/日志消费）。

        判定为真需**同时**满足：
        1. `coverage < min_coverage`：话题内容词基本不在语料词表里（考虑 IDF 权重）；
        2. `top_vector_score < min_similarity`：语义上也没有相近内容。
        """
        top = max((float(h.get("score") or 0.0) for h in vector_hits), default=0.0)
        vocab = self._get_vocab()
        if vocab is None:
            return {"no_relevant": False, "coverage": None, "top_vector_score": top,
                    "absent_terms": [], "reason": "语料为空/不可读 → 不判"}
        cov, absent = vocab.coverage(query)
        low_cov = cov < self.min_coverage
        low_sim = top < self.min_similarity
        no_relevant = low_cov and low_sim
        if no_relevant:
            reason = (f"覆盖{cov:.2f}<{self.min_coverage:.2f} 且 "
                      f"相似{top:.2f}<{self.min_similarity:.2f}（缺词 {absent[:4]}）")
        else:
            reason = (f"覆盖{cov:.2f} 相似{top:.2f} → 不判无资料"
                      + ("（覆盖低但语义相近）" if low_cov else ""))
        return {"no_relevant": no_relevant, "coverage": round(cov, 4),
                "top_vector_score": round(top, 4), "absent_terms": absent[:8],
                "reason": reason}


_gate_cache: dict[int, RelevanceGate] = {}


def get_relevance_gate(bm25_store) -> RelevanceGate:
    """进程内每 store 一个 gate（避免每问重建语料词表）。"""
    key = id(bm25_store)
    gate = _gate_cache.get(key)
    if gate is None:
        gate = RelevanceGate(bm25_store)
        _gate_cache[key] = gate
    return gate
