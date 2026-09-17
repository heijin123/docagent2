"""HybridRetriever（需求 F2，契约 4.5）：向量 + BM25 并行召回 + RRF 融合。

职责边界（对齐需求）：
- F2.1/F2.2  向量 top_k=20（cosine，由 store 排序）+ BM25 top_k=20（jieba 分词）；
- F2.3/F6.4  两路并行（async 入口 asyncio.gather + to_thread；单路失败降级为单路结果 + degraded）；
- F2.4       RRF 融合 score=Σ 1/(k+rank)，k=60，输出 top_n=8；
- F2.6       结果携带完整 metadata（citation 溯源用）；
- F2.7       两路强制同一过滤：is_valid=true + tenant_id 隔离 + permission 标签；
              category/department 为可配过滤（默认关闭）；
- F2.8       现行性过滤（effective_time==0 永久有效 | now<=effective_time 现行）；
              现行不足 → 顺序发起二级候选（仅放宽 effective_time，不动 is_valid/permission）；
- F2.9       年份感知两段式裁决（query 显式年份 → doc_year 过滤检索；
              过滤命中与语义 top1 同 doc_id 才采用，否则回退语义结果 + note）。
- F2.10      相关性判定（`relevance`）：「库里到底有没有相关内容」——向量恒返回 topN，
              需显式判无相关资料才能走 no_data（0 LLM），见 app/retrieval/relevance.py。
- 需求 7.1   邻近 chunk 上下文扩展（`context_items`）：命中块被切在段落/条款边界时补齐
              相邻块。**刻意与 `items` 分开放**——命中口径与评估指标（recall@5 读 items）
              不能被上下文掺水。

同步入口 `retrieve()` 供 M3 LangGraph 节点直接调用；
异步入口 `aretrieve()` 供 M4 SSE 层调用（两路真并行）。
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.observability import TimedSpan, log_slow_query
from app.models import now_ts
from app.retrieval.bm25store import BM25Store
from app.retrieval.embedding import Embedder, build_embedder
from app.retrieval.relevance import get_relevance_gate
from app.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60
ROAD_TOPK = 20
FINAL_TOPN = 8

# 权限分级：public < internal < secret；用户可见 = 不高于其级别的标签
PERMISSION_LEVEL = {"public": 0, "internal": 1, "secret": 2}

YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_CHUNK_ID_RE = re.compile(r"^(?P<doc>.+)_(?P<ver>\d{4})_(?P<idx>\d{5})$")


def parse_chunk_id(chunk_id: str) -> tuple[str, int, int] | None:
    """chunk_id = f"{doc_id}_{version:04d}_{index:05d}" → (doc_id, version, index)。

    doc_id 形如 `doc_<12hex>`（自身不含下划线），故从右往左切两段即可，无需额外索引。
    """
    m = _CHUNK_ID_RE.match(chunk_id or "")
    if not m:
        return None
    return m.group("doc"), int(m.group("ver")), int(m.group("idx"))


@dataclass
class HybridResult:
    """单次检索的完整结果（F2.6 元数据 + F2.8 过期分支 + F2.9 note + F5.8 调试明细）。"""

    query: str
    items: list[dict] = field(default_factory=list)          # 现行结果（最终采用）
    expired_candidates: list[dict] = field(default_factory=list)  # 二级候选：仅过期命中
    context_items: list[dict] = field(default_factory=list)  # 需求 7.1 邻近上下文（非命中）
    relevance: dict = field(default_factory=dict)            # F2.10 相关性判定结果
    notes: list[str] = field(default_factory=list)           # F2.9 回退 / F2.8 降级说明
    degraded: list[dict] = field(default_factory=list)       # [{path, error}]
    used_roads: list[str] = field(default_factory=list)      # 实际参与融合的路
    detail: dict = field(default_factory=dict)               # 双路原始明细（debug 端点用）

    @property
    def only_expired(self) -> bool:
        """主检索现行不足且仅命中过期文档 → Agent 走 F2.8 提示话术。"""
        return not self.items and bool(self.expired_candidates)

    @property
    def no_relevant(self) -> bool:
        """F2.10：库里没有与问题相关的内容（双证据判定，高精度）。"""
        return bool(self.relevance.get("no_relevant"))


def allowed_permissions(user_permission: str) -> list[str]:
    """按用户权限标签计算可见的 permission 集合（public<=internal<=secret）。"""
    level = PERMISSION_LEVEL.get(user_permission, 1)
    return [p for p, lv in PERMISSION_LEVEL.items() if lv <= level]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        embedder: Embedder | None = None,
        *,
        tenant_id: str | None = None,
        road_top_k: int = ROAD_TOPK,
        rrf_k: int = RRF_K,
        final_top_n: int = FINAL_TOPN,
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.embedder = embedder or build_embedder()
        self.tenant_id = tenant_id or settings.default_tenant_id
        self.road_top_k = road_top_k
        self.rrf_k = rrf_k
        self.final_top_n = final_top_n

    # ── 过滤谓词构造（两路共用一条规则，防漂移）─────────────────
    def _filters(
        self,
        *,
        include_expired: bool,
        doc_years: list[int] | None,
        category: str | None,
        department: str | None,
        user_permission: str,
        now: int,
    ) -> tuple[list[dict], list[str], list]:
        """返回 (chroma_where_parts, bm25_sql_parts, bm25_params)。

        chroma parts 由调用方拼 $and；bm25 parts 拼 AND 串。两条链从同一组谓词出发。
        """
        allowed = allowed_permissions(user_permission)
        perm_ph = ", ".join("?" * len(allowed))
        chroma_parts: list[dict] = [
            {"is_valid": {"$eq": True}},
            {"tenant_id": {"$eq": self.tenant_id}},
            {"permission": {"$in": allowed}},
        ]
        sql_parts = ["is_valid=1", "tenant_id=?", f"permission IN ({perm_ph})"]
        params: list = [self.tenant_id, *allowed]

        if not include_expired:
            # F2.8：默认只召回现行（effective_time==0 永久有效 | now<=effective_time）
            chroma_parts.append(
                {"$or": [
                    {"effective_time": {"$eq": 0}},
                    {"effective_time": {"$gte": now}},
                ]}
            )
            sql_parts.append("(effective_time=0 OR effective_time>=?)")
            params.append(now)

        if doc_years:
            chroma_parts.append({"doc_year": {"$in": doc_years}})
            sql_parts.append(f"doc_year IN ({', '.join('?' * len(doc_years))})")
            params.extend(doc_years)
        if category:
            chroma_parts.append({"category": {"$eq": category}})
            sql_parts.append("category=?")
            params.append(category)
        if department is not None:
            chroma_parts.append({"department": {"$eq": department}})
            sql_parts.append("department=?")
            params.append(department)

        return chroma_parts, sql_parts, params

    # ── 单次检索（双路原始命中 + 融合）──────────────────────────
    def _roads_sync(
        self, query_emb: list[float], query: str, where: dict | None,
        bm25_where: tuple[str, tuple],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """顺序执行双路（同步入口）。返回 (vector_hits, bm25_hits, degraded)。"""
        degraded: list[dict] = []
        try:
            vector_hits = self.vector_store.query(query_emb, top_k=self.road_top_k, where=where)
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量路检索失败，降级为 BM25 单路: %s", exc)
            vector_hits = []
            degraded.append({"path": "vector", "error": str(exc)})
        try:
            bm25_hits = self.bm25_store.search(
                query, top_k=self.road_top_k,
                where_sql=bm25_where[0], where_params=bm25_where[1],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 路检索失败，降级为向量单路: %s", exc)
            bm25_hits = []
            degraded.append({"path": "bm25", "error": str(exc)})
        return vector_hits, bm25_hits, degraded

    async def _roads_async(
        self, query_emb: list[float], query: str, where: dict | None,
        bm25_where: tuple[str, tuple],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """两路真并行（F2.3/F6.4：asyncio.gather + to_thread 包同步 SDK）。"""
        async def _vector() -> tuple[list[dict], dict | None]:
            try:
                hits = await asyncio.to_thread(
                    lambda: self.vector_store.query(
                        query_emb, top_k=self.road_top_k, where=where))
                return hits, None
            except Exception as exc:  # noqa: BLE001
                logger.warning("向量路检索失败，降级为 BM25 单路: %s", exc)
                return [], {"path": "vector", "error": str(exc)}

        async def _bm25() -> tuple[list[dict], dict | None]:
            try:
                hits = await asyncio.to_thread(
                    lambda: self.bm25_store.search(
                        query, top_k=self.road_top_k,
                        where_sql=bm25_where[0], where_params=bm25_where[1]))
                return hits, None
            except Exception as exc:  # noqa: BLE001
                logger.warning("BM25 路检索失败，降级为向量单路: %s", exc)
                return [], {"path": "bm25", "error": str(exc)}

        (vh, vd), (bh, bd) = await asyncio.gather(_vector(), _bm25())
        degraded = [d for d in (vd, bd) if d]
        return vh, bh, degraded

    # ── RRF 融合（F2.4）────────────────────────────────────────
    def _fuse(self, vector_hits: list[dict], bm25_hits: list[dict],
              now: int, top_n: int | None = None) -> list[dict]:
        merged: dict[str, dict] = {}
        for road, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, h in enumerate(hits, start=1):
                cid = h["chunk_id"]
                item = merged.get(cid)
                if item is None:
                    meta = dict(h["metadata"])
                    eff = meta.get("effective_time") or 0
                    item = {
                        "chunk_id": cid,
                        "content": h["content"],
                        "metadata": meta,
                        "ranks": {"vector": None, "bm25": None},
                        "sources": [],
                        "validity": "expired" if (eff and eff < now) else "valid",
                        "expired_at": eff if (eff and eff < now) else None,
                    }
                    merged[cid] = item
                item["ranks"][road] = rank
                if road not in item["sources"]:
                    item["sources"].append(road)
        # RRF：score = Σ 1/(k + rank)；只在某一路的项另一路 rank=None → 该路贡献 0
        for item in merged.values():
            item["score"] = sum(
                1.0 / (self.rrf_k + r) for r in item["ranks"].values() if r is not None)
        fused = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return fused[: (top_n or self.final_top_n)]

    # ── 一次"过滤窗口"内的完整召回（向量 + BM25 + 融合）─────────
    def _build_query_spec(
        self, *, include_expired: bool, doc_years: list[int] | None,
        category: str | None, department: str | None,
        user_permission: str, now: int,
    ) -> tuple[dict | None, tuple[str, tuple]]:
        chroma_parts, sql_parts, params = self._filters(
            include_expired=include_expired, doc_years=doc_years,
            category=category, department=department,
            user_permission=user_permission, now=now,
        )
        where = {"$and": chroma_parts} if chroma_parts else None
        bm25_where = (" AND ".join(sql_parts), tuple(params))
        return where, bm25_where

    def _search_once(
        self, query: str, *, include_expired: bool,
        doc_years: list[int] | None = None,
        category: str | None = None, department: str | None = None,
        user_permission: str = "internal", now: int | None = None,
    ) -> dict:
        """同步窗口检索（F2.3 单路失败降级为单路）。"""
        now = now or now_ts()
        where, bm25_where = self._build_query_spec(
            include_expired=include_expired, doc_years=doc_years,
            category=category, department=department,
            user_permission=user_permission, now=now)
        query_emb = self.embedder.embed_texts([query])[0]
        vh, bh, degraded = self._roads_sync(query_emb, query, where, bm25_where)
        return self._pack(query, vh, bh, degraded, now)

    async def _search_once_async(
        self, query: str, *, include_expired: bool,
        doc_years: list[int] | None = None,
        category: str | None = None, department: str | None = None,
        user_permission: str = "internal", now: int | None = None,
    ) -> dict:
        """异步窗口检索（两路 gather 并行，asyncio 事件循环内可用）。"""
        now = now or now_ts()
        where, bm25_where = self._build_query_spec(
            include_expired=include_expired, doc_years=doc_years,
            category=category, department=department,
            user_permission=user_permission, now=now)
        query_emb = self.embedder.embed_texts([query])[0]
        vh, bh, degraded = await self._roads_async(query_emb, query, where, bm25_where)
        return self._pack(query, vh, bh, degraded, now)

    def _pack(self, query: str, vh: list[dict], bh: list[dict],
              degraded: list[dict], now: int) -> dict:
        fused = self._fuse(vh, bh, now)
        used = [r for r, h in (("vector", vh), ("bm25", bh)) if h]
        return {"vector_hits": vh, "bm25_hits": bh, "fused": fused,
                "degraded": degraded, "used_roads": used, "now": now}

    # ── F2.10 相关性判定：库里到底有没有相关内容 ────────────────
    def _assess_relevance(self, query: str, base: dict) -> dict:
        """基于**语义基准路**判定（年份过滤/二级候选都不该改变"有没有相关内容"）。

        判定逻辑与标定见 app/retrieval/relevance.py；此处只负责取信号与落 note。

        护栏：`provider=mock` 或 `degraded` 时**不判**——mock 是哈希向量，余弦分数没有
        语义含义（合约测试/无 Key 降级环境都走这条路）。此时若照常判定，会因"相似度
        恒低"把**库里有答案的问题**判成无资料（真实误杀）。宁可让本能力在 mock 下失效。
        """
        prov = (getattr(self.embedder, "provider", "") or "").lower()
        if getattr(self.embedder, "degraded", False) or prov == "mock":
            return {"no_relevant": False, "coverage": None, "top_vector_score": None,
                    "absent_terms": [],
                    "reason": f"向量 provider={prov or '?'}/degraded（相似度无语义）→ 不判"}
        try:
            gate = get_relevance_gate(self.bm25_store)
            return gate.assess(query, base.get("vector_hits") or [])
        except Exception as exc:  # noqa: BLE001 — 判定失败绝不阻断检索
            logger.warning("相关性判定失败，跳过（不影响检索）: %s", exc)
            return {"no_relevant": False, "coverage": None, "top_vector_score": None,
                    "absent_terms": [], "reason": f"判定异常跳过: {exc}"}

    # ── 需求 7.1 邻近 chunk 上下文扩展 ─────────────────────────
    def _expand_context(self, items: list[dict], *, user_permission: str,
                        now: int) -> list[dict]:
        """取前 N 条命中在同文档同版本内的 chunk_index ±1 邻块，作为**上下文**返回。

        与 `items` 分开返回的理由（重要）：
        - 评估指标（recall@5 / MRR）读的是 `items`，掺入邻居会让指标虚高、与历史基线不可比；
        - 邻居是"补全语义"而非"命中证据"，口径要能区分。

        护栏：预算 top-3×±1、总量 `context_expand_max`；跳过过期邻块（避免把过期内容
        当上下文引进来）；强制 tenant + permission 隔离（BM25Store.get_by_chunk_ids）。
        """
        if not settings.context_expand_enabled or not items:
            return []
        wanted: dict[tuple[str, int], set[int]] = {}
        for it in items[: max(1, settings.context_expand_top)]:
            parsed = parse_chunk_id(it.get("chunk_id", ""))
            if not parsed:
                continue
            doc_id, ver, idx = parsed
            for nb in (idx - 1, idx + 1):
                if nb >= 1:
                    wanted.setdefault((doc_id, ver), set()).add(nb)
        if not wanted:
            return []
        want_ids = [f"{d}_{v:04d}_{i:05d}" for (d, v), idxs in wanted.items() for i in idxs]
        try:
            rows = self.bm25_store.get_by_chunk_ids(
                want_ids, tenant_id=self.tenant_id,
                allowed_permissions=allowed_permissions(user_permission))
        except Exception as exc:  # noqa: BLE001 — 扩展失败不影响主检索
            logger.warning("邻近 chunk 扩展失败（忽略）: %s", exc)
            return []

        have = {it.get("chunk_id") for it in items}
        out: list[dict] = []
        for row in rows:
            cid = row["chunk_id"]
            if cid in have:
                continue
            meta = dict(row["metadata"])
            eff = meta.get("effective_time") or 0
            if eff and eff < now:
                continue  # 过期邻块不进上下文（避免混淆现行结论）
            have.add(cid)
            out.append({
                "chunk_id": cid, "content": row["content"], "metadata": meta,
                "ranks": {"vector": None, "bm25": None}, "sources": ["context"],
                "validity": "valid", "expired_at": None, "score": 0.0, "is_context": True,
            })
            if len(out) >= max(0, settings.context_expand_max):
                break
        return out

    # ── 公开入口 ───────────────────────────────────────────────
    def retrieve(
        self, query: str, *, top_n: int | None = None,
        user_permission: str = "internal",
        category: str | None = None, department: str | None = None,
        now: int | None = None,
    ) -> HybridResult:
        """同步检索入口（M3 节点用）。完整 F2.8 两级 + F2.9 两段式裁决。"""
        span = TimedSpan(name="slow_query").attr(query=query[:120])
        now = now or now_ts()
        # 第一步：恒做不限年份的现行检索（F2.9 基准 + F2.8 主检索共用一次）
        base = self._search_once(
            query, include_expired=False, category=category,
            department=department, user_permission=user_permission, now=now,
        )
        result = HybridResult(query=query, degraded=list(base["degraded"]),
                              used_roads=list(base["used_roads"]))
        result.items = self._trim(base["fused"], top_n)
        result.detail["baseline"] = {
            "vector_hits": base["vector_hits"], "bm25_hits": base["bm25_hits"],
        }
        # F2.10：先判「有没有相关内容」——它为真时下方二级候选无需再发（话题都不在库里，
        # 过期候选同样不相关），也由它决定是否走 no_data 出口（0 LLM）。
        result.relevance = self._assess_relevance(query, base)
        if result.no_relevant:
            result.notes.append(
                f"相关性判定：无相关内容（{result.relevance['reason']}）→ 应如实告知缺失")

        # F2.9 两段式裁决：显式年份 → 过滤检索；命中与语义 top1 同 doc 才采用
        years = sorted({int(y) for y in YEAR_RE.findall(query)})
        if years and not result.no_relevant:
            filtered = self._search_once(
                query, include_expired=False, doc_years=years,
                category=category, department=department,
                user_permission=user_permission, now=now,
            )
            result.detail["year_filtered"] = True
            result.detail["years"] = years
            top_doc = base["fused"][0]["metadata"]["doc_id"] if base["fused"] else None
            f_docs = {i["metadata"]["doc_id"] for i in filtered["fused"]}
            if filtered["fused"] and top_doc and top_doc in f_docs:
                result.items = self._trim(filtered["fused"], top_n)
                result.notes.append(
                    f"年份过滤 {years} 命中且与语义 top1 同文档 → 采用年份过滤结果")
            else:
                result.notes.append(
                    f"年份过滤 {years} 为空或跑题（语义 top1 doc_id 不在过滤结果）"
                    "→ 回退不限年份语义结果")
            result.detail["filtered_fused"] = filtered["fused"]
            result.degraded.extend(filtered["degraded"])
        else:
            result.detail["year_filtered"] = False

        # F2.8 自动兜底：当前过滤窗口下现行列表完全为空（库空/全被过滤）时，
        # 顺序放宽 effective_time 取过期候选（无现行可答，直接给出确认分支所需信息）。
        # 注意：语义检索恒返回 topN（无距离阈值），"现行不足"的常规判定发生在调用方
        # （Answer/Generate 节点看到证据不足）→ 由其显式调 relaxed_retrieve() 发起二级候选。
        if not result.items and not result.no_relevant:
            relaxed = self._search_once(
                query, include_expired=True, category=category,
                department=department, user_permission=user_permission, now=now,
            )
            result.expired_candidates = self._trim(
                [i for i in relaxed["fused"] if i["validity"] == "expired"], top_n)
            if result.expired_candidates:
                result.notes.append(
                    "现行资料不足，命中过期文档（详见 expired_candidates）→ 需用户确认后查看")
            result.detail["relaxed"] = {
                "vector_hits": relaxed["vector_hits"], "bm25_hits": relaxed["bm25_hits"]}
            result.degraded.extend(relaxed["degraded"])
            result.used_roads = relaxed["used_roads"] or result.used_roads

        # 需求 7.1：命中块可能被切在段落/条款边界 → 补同文档相邻块作上下文
        result.context_items = self._expand_context(
            result.items, user_permission=user_permission, now=now)
        if result.context_items:
            result.notes.append(
                f"邻近上下文：补入 {len(result.context_items)} 条相邻块（非命中，仅供补全语义）")
        duration_ms = span.stop(log_slow=False)
        log_slow_query("", query, duration_ms,
                       used_roads=result.used_roads, hits=len(result.items))
        return result

    def relaxed_retrieve(
        self, query: str, *, top_n: int | None = None,
        user_permission: str = "internal",
        category: str | None = None, department: str | None = None,
        now: int | None = None,
    ) -> HybridResult:
        """F2.8 二级候选（由调用方在"现行不足"判定后顺序发起，见 F2.8 流程图）。

        仅放宽 effective_time（含过期），**不**放宽 is_valid / permission / doc_year；
        复用与主检索同一过滤链与融合器，避免两套逻辑漂移。

        - `expired_candidates`：需要用户确认才能应用的过期命中（validity=expired）；
        - `items`：放宽窗口内仍属现行的命中的子集（尽力而为，通常为空）。
        """
        now = now or now_ts()
        res = self._search_once(
            query, include_expired=True, category=category,
            department=department, user_permission=user_permission, now=now,
        )
        result = HybridResult(query=query, degraded=list(res["degraded"]),
                              used_roads=list(res["used_roads"]))
        expired = self._trim([i for i in res["fused"] if i["validity"] == "expired"], top_n)
        result.expired_candidates = expired
        result.items = self._trim(
            [i for i in res["fused"] if i["validity"] != "expired"], top_n)
        if expired:
            result.notes.append(
                f"命中 {len(expired)} 条过期文档（validity=expired）→ 需用户确认后查看，"
                "答案须带失效提示")
        result.detail["baseline"] = {
            "vector_hits": res["vector_hits"], "bm25_hits": res["bm25_hits"]}
        result.relevance = self._assess_relevance(query, res)
        result.context_items = self._expand_context(
            result.items, user_permission=user_permission, now=now)
        return result

    async def arelaxed_retrieve(
        self, query: str, *, top_n: int | None = None,
        user_permission: str = "internal",
        category: str | None = None, department: str | None = None,
        now: int | None = None,
    ) -> HybridResult:
        """relaxed_retrieve 的异步版（M4 SSE 用，双路并行）。"""
        now = now or now_ts()
        res = await self._search_once_async(
            query, include_expired=True, category=category,
            department=department, user_permission=user_permission, now=now,
        )
        result = HybridResult(query=query, degraded=list(res["degraded"]),
                              used_roads=list(res["used_roads"]))
        expired = self._trim([i for i in res["fused"] if i["validity"] == "expired"], top_n)
        result.expired_candidates = expired
        result.items = self._trim(
            [i for i in res["fused"] if i["validity"] != "expired"], top_n)
        if expired:
            result.notes.append("命中过期文档（validity=expired）→ 需用户确认后查看")
        result.relevance = self._assess_relevance(query, res)
        result.context_items = self._expand_context(
            result.items, user_permission=user_permission, now=now)
        return result

    async def aretrieve(
        self, query: str, *, top_n: int | None = None,
        user_permission: str = "internal",
        category: str | None = None, department: str | None = None,
        now: int | None = None,
    ) -> HybridResult:
        """异步入口（M4 SSE 用）：双路 gather 并行召回，逻辑与 retrieve 一致。"""
        span = TimedSpan(name="slow_query").attr(query=query[:120])
        now = now or now_ts()
        base = await self._search_once_async(
            query, include_expired=False, category=category,
            department=department, user_permission=user_permission, now=now,
        )
        result = HybridResult(query=query, degraded=list(base["degraded"]),
                              used_roads=list(base["used_roads"]))
        result.items = self._trim(base["fused"], top_n)
        result.detail["baseline"] = {
            "vector_hits": base["vector_hits"], "bm25_hits": base["bm25_hits"]}
        result.relevance = self._assess_relevance(query, base)
        if result.no_relevant:
            result.notes.append(
                f"相关性判定：无相关内容（{result.relevance['reason']}）→ 应如实告知缺失")

        years = sorted({int(y) for y in YEAR_RE.findall(query)})
        if years and not result.no_relevant:
            filtered = await self._search_once_async(
                query, include_expired=False, doc_years=years,
                category=category, department=department,
                user_permission=user_permission, now=now,
            )
            top_doc = base["fused"][0]["metadata"]["doc_id"] if base["fused"] else None
            f_docs = {i["metadata"]["doc_id"] for i in filtered["fused"]}
            if filtered["fused"] and top_doc and top_doc in f_docs:
                result.items = self._trim(filtered["fused"], top_n)
                result.notes.append(f"年份过滤 {years} 命中且与语义 top1 同文档 → 采用年份过滤结果")
            else:
                result.notes.append(
                    f"年份过滤 {years} 为空或跑题 → 回退不限年份语义结果")
            result.detail["year_filtered"] = True
            result.degraded.extend(filtered["degraded"])
        else:
            result.detail["year_filtered"] = False

        if not result.items and not result.no_relevant:
            relaxed = await self._search_once_async(
                query, include_expired=True, category=category,
                department=department, user_permission=user_permission, now=now,
            )
            result.expired_candidates = [
                i for i in relaxed["fused"] if i["validity"] == "expired"]
            if result.expired_candidates:
                result.notes.append("现行资料不足，命中过期文档 → 需用户确认后查看")
            result.degraded.extend(relaxed["degraded"])

        result.context_items = self._expand_context(
            result.items, user_permission=user_permission, now=now)
        if result.context_items:
            result.notes.append(
                f"邻近上下文：补入 {len(result.context_items)} 条相邻块（非命中，仅供补全语义）")
        duration_ms = span.stop(log_slow=False)
        log_slow_query("", query, duration_ms,
                       used_roads=result.used_roads, hits=len(result.items))
        return result

    @staticmethod
    def _trim(items: list[dict], top_n: int | None) -> list[dict]:
        return items[:top_n] if top_n else items


def build_retriever(**kw) -> HybridRetriever:
    return HybridRetriever(**kw)
