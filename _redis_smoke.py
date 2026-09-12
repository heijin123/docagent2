"""临时冒烟：验证 Redis checkpointer 跨进程持久化（Windows 侧 → WSL2 Redis）。

用法：
    python _redis_smoke.py write   # 进程 A：跑一轮对话，落到 Redis
    python _redis_smoke.py read    # 进程 B：全新进程，仅凭 thread_id 读回历史
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

THREAD = "tenant_demo:smoke:redis"


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "write"

    import redis as redis_py

    from app.core.config import settings
    from app.agent.checkpointer import build_checkpointer

    # ── 0. 工厂行为 ────────────────────────────────────────
    ckpt, degraded, note = build_checkpointer(memory=False)
    print(f"[factory] type={type(ckpt).__name__} degraded={degraded} note={note!r}")
    print(f"[redis_url] {settings.redis_url}")

    from app.retrieval.bm25store import BM25Store
    from app.retrieval.embedding import build_embedder
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.vectorstore import VectorStore
    from app.agent.graph import AgentApp
    from app.agent.llm import build_llm

    r = redis_py.Redis.from_url(settings.redis_url, decode_responses=False)
    before = r.dbsize()

    if mode == "write":
        embedder = build_embedder()
        retriever = HybridRetriever(
            vector_store=VectorStore(), bm25_store=BM25Store(), embedder=embedder)
        app = AgentApp(retriever, build_llm(), memory_checkpoint=False)
        print(f"[agent] ckpt_degraded={app._ckpt_degraded} note={app.checkpoint_note!r}")

        reply = app.reply("差旅住宿费的标准上限是多少？", THREAD)
        print(f"[reply] intent={reply.intent} conf={reply.confidence:.3f} "
              f"cites={len(reply.citations)} degraded={reply.degraded}")
        print(f"[reply] answer={reply.answer[:160]!r}")
        print(f"[redis] dbsize {before} -> {r.dbsize()}")
        return 0

    # ── read：全新进程，不碰 agent 图，仅用 checkpointer 读回 ──
    cfg = {"configurable": {"thread_id": THREAD}}
    try:
        tup = ckpt.get_tuple(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[read] get_tuple FAIL {type(exc).__name__}: {exc}")
        return 2
    if tup is None:
        print("[read] NO CHECKPOINT（跨进程读不到 → 未真正落到 Redis）")
        return 3
    msgs = (tup.checkpoint.get("channel_values") or {}).get("messages", [])
    print(f"[read] checkpoint_id={tup.checkpoint.get('id')}")
    print(f"[read] messages={len(msgs)}  types={[type(m).__name__ for m in msgs]}")
    for m in msgs[-4:]:
        print(f"   - raw: {repr(m)[:150]}")

    # 项目自身入口：AgentApp.history()（API /threads/{id}/history 走这条）
    embedder = build_embedder()
    retriever = HybridRetriever(
        vector_store=VectorStore(), bm25_store=BM25Store(), embedder=embedder)
    app = AgentApp(retriever, build_llm(), memory_checkpoint=False)
    hist = app.history(THREAD, limit=10)
    print(f"[history] AgentApp.history() -> {len(hist)} 条")
    for h in hist[-4:]:
        print(f"   - {repr(h)[:150]}")

    print(f"[redis] dbsize={r.dbsize()}  (进程 A 之前={before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
