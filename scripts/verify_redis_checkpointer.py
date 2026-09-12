"""verify_redis_checkpointer.py：Redis 检查点器连通 + 跨进程持久化验收（F4.1）。

为什么单独有这个脚本：
  verify_m3 全程 `memory_checkpoint=True`（InMemorySaver，故意不碰网络），
  于是 **Redis 路径从未被任何自检覆盖**。2026-09-12 实测发现
  `langgraph-checkpoint-redis` 0.5.x 已移除 `RedisSaver(conn=...)` 签名，
  工厂每次启动都静默降级内存版（跨进程记忆实际失效），而日志只留一行 WARN。
  本脚本把这条链路补上验收。

断言：
  1. Redis 可达时工厂返回 RedisSaver 且 degraded=False（不静默降级）
  2. 构造签名与 0.5.x 对齐（redis_client= 存在、旧 conn= 已移除）
  3. setup() 建出 RediSearch 索引（checkpoint / checkpoint_write）
  4. **跨进程持久化**：本进程写一轮 → 子进程凭 thread_id 读回（内存版必然读不到）
  5. Redis 不可达时降级 InMemorySaver + degraded=True，且不抛异常

若无 Redis：打印 SKIP 并 0 退出（与 verify_m3「不依赖外部服务」同精神）。
用法: python scripts/verify_redis_checkpointer.py
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
THREAD = "verify_redis:checkpointer"
# StubLLM 的 chitchat 规则：长度 ≤8 且含问候词 → 走 direct_reply，不触发检索、无网络
SENTINEL = "你好探针Z9"


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✔ {name}")
    else:
        FAIL += 1
        print(f"  ✘ {name}" + (f"  —— {detail}" if detail else ""))


class _NoRetrieveRetriever:
    """chitchat 路径不检索；若被调用即为路由错误，直接炸出来。"""

    def retrieve(self, *a, **k):
        raise AssertionError("本用例走 chitchat，不应触发检索")

    relaxed_retrieve = retrieve


def _read_from_child(thread_id: str) -> dict:
    """子进程（全新解释器）仅凭 thread_id 读回 checkpoint。"""
    code = (
        "import json,sys;"
        f"sys.path.insert(0, r'{ROOT}');"
        "import redis;"
        "from langgraph.checkpoint.redis import RedisSaver;"
        "from app.core.config import settings;"
        "c = redis.Redis.from_url(settings.redis_url, decode_responses=False);"
        "s = RedisSaver(redis_client=c);"
        f"t = s.get_tuple({{'configurable': {{'thread_id': '{thread_id}'}}}});"
        "msgs = ((t.checkpoint.get('channel_values') or {}).get('messages', [])) if t else [];"
        "print(json.dumps({'found': t is not None,"
        " 'cid': t.checkpoint.get('id') if t else None, 'messages': msgs},"
        " ensure_ascii=False))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, encoding="utf-8", cwd=str(ROOT))
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "")[-400:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    import redis as redis_py

    from app.agent.checkpointer import build_checkpointer
    from app.core.config import settings

    print("── 0. 工厂连通（memory=False → 期望 Redis）──")
    ckpt, degraded, note = build_checkpointer(memory=False)
    kind = type(ckpt).__name__
    if degraded or kind != "RedisSaver":
        print(f"  ⊘ SKIP：Redis 不可用（{note or kind}）——本环境跳过 Redis 验收")
        print(f"\n结果: 0 通过 / 0 失败（SKIP）")
        return 0
    check("工厂返回 RedisSaver 且未降级", kind == "RedisSaver" and not degraded,
          f"{kind} degraded={degraded} note={note!r}")

    print("── 1. 构造签名与 0.5.x 对齐（回归：旧 conn= 已移除）──")
    params = list(inspect.signature(type(ckpt).__init__).parameters)
    check("支持 redis_client=（0.5.x 写法）", "redis_client" in params, str(params))
    check("旧 conn= 已不存在（若回归此断言先炸）", "conn" not in params, str(params))

    print("── 2. setup() 建出 RediSearch 索引 ──")
    client = redis_py.Redis.from_url(settings.redis_url, decode_responses=False)
    idx = [x.decode() if isinstance(x, bytes) else str(x)
           for x in client.execute_command("FT._LIST")]
    check("索引 checkpoint 存在", "checkpoint" in idx, str(idx))
    check("索引 checkpoint_write 存在", "checkpoint_write" in idx, str(idx))

    print("── 3. 写入一轮对话（StubLLM，走 chitchat，无网络）──")
    from app.agent.graph import AgentApp
    from app.agent.llm import StubLLM

    before = client.dbsize()
    app = AgentApp(_NoRetrieveRetriever(), StubLLM(), memory_checkpoint=False)
    check("AgentApp 未降级（真走 Redis）",
          not getattr(app, "_ckpt_degraded", True), repr(app.checkpoint_note))
    reply = app.reply(SENTINEL, THREAD)
    after = client.dbsize()
    check("写入产生新 Redis 键", after > before, f"dbsize {before} -> {after}")
    hist_local = app.history(THREAD, limit=10)
    check("同进程内 history() 可见 2 条", len(hist_local) == 2, str(len(hist_local)))

    print("── 4. 跨进程持久化（核心：内存版必然失败）──")
    child = _read_from_child(THREAD)
    if "error" in child:
        check("子进程读取成功", False, child["error"])
    else:
        check("子进程读到 checkpoint", child.get("found") is True, json.dumps(child)[:200])
        msgs = child.get("messages") or []
        check("子进程读回 2 条消息", len(msgs) == 2, str(len(msgs)))
        joined = json.dumps(msgs, ensure_ascii=False)
        check("消息内容完整（含探针 query 原文）", SENTINEL in joined, joined[:160])
        check("助手回复非空",
              any(m.get("role") == "assistant" and m.get("content") for m in msgs if isinstance(m, dict)),
              joined[:160])
        print(f"     checkpoint_id={child.get('cid')}")

    print("── 5. 不可达降级（不抛异常、不假装成功）──")
    import dataclasses

    import app.core.config as cfg_mod

    orig = cfg_mod.settings
    cfg_mod.settings = dataclasses.replace(orig, redis_url="redis://127.0.0.1:1/0")
    try:
        ck2, deg2, note2 = build_checkpointer(memory=False)
        check("不可达 → InMemorySaver", type(ck2).__name__ == "InMemorySaver", type(ck2).__name__)
        check("不可达 → degraded=True", deg2 is True, repr(deg2))
        check("降级原因可见（note 非空）", bool(note2), repr(note2))
    except Exception as exc:  # noqa: BLE001
        check("不可达时不应抛异常", False, f"{type(exc).__name__}: {exc}")
    finally:
        cfg_mod.settings = orig

    print("── 6. 清理探针 thread ──")
    victims = [k for k in client.scan_iter(match=f"*{THREAD}*", count=200)]
    if victims:
        client.delete(*victims)
    left = [k for k in client.scan_iter(match=f"*{THREAD}*", count=200)]
    check("探针键已清理", not left, f"残留 {len(left)}")

    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
