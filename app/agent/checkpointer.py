"""Checkpointer 工厂（F4.1）：RedisSaver 持久化 → 不可达时降级 InMemorySaver。

thread_id = `{tenant_id}:{user_id}`（契约 §2.1 建议格式），由调用方（graph invoke config）传入。

- 降级不掩盖：连接失败打 degraded 日志，调用方可读 `last_degraded` 标注；
- 探测：短超时 PING，避免启动阻塞；
- 测试环境（verify_m3）直接传 memory=True 强制内存检查点，跳过网络探测。

注意 langgraph-checkpoint-redis 0.5.x 的 API 变更：`RedisSaver(conn=...)` 已移除，
改为 `redis_client=` / `redis_url=` / `connection_args=`。

`setup()` **必须逐个实例调用**（不能只做一次）——它除了建 RediSearch 索引
（`create(overwrite=False)`，已存在则直接返回，故幂等且廉价），还会初始化**实例级**
`self._key_registry`；漏调的话该实例读 pending writes 时会 AttributeError
（`'NoneType' object has no attribute 'make_write_keys_zset_key'`）。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_checkpointer(memory: bool = False):
    """返回 (checkpointer, degraded, note)。

    degraded=True 说明期望 Redis 但已降级内存（进程内，重启丢失）——日志与调用方可见。
    """
    if memory:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), True, "测试模式：强制内存检查点（无跨进程持久化）"

    from app.core.config import settings

    if not settings.redis_checkpointer_enabled:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), True, "REDIS_CHECKPOINTER_ENABLED=false → 内存检查点"

    # 尝试 Redis（同步 RedisSaver + 短超时 PING 探测）
    try:
        import redis as redis_py
        from langgraph.checkpoint.redis import RedisSaver

        client = redis_py.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
            decode_responses=False,
        )
        client.ping()  # 失败即抛 → 走降级

        # 0.5.x 签名：redis_client=（旧的 conn= 已移除）
        saver = RedisSaver(redis_client=client)
        # setup() 幂等且廉价（create(overwrite=False) 已存在直接返回），但会初始化实例级
        # _key_registry，必须每实例调用——进程级只调一次会让第二个实例读 pending 时崩溃。
        saver.setup()  # 建 RediSearch 索引，幂等；失败即抛 → 走降级
        return saver, False, ""
    except Exception as exc:  # noqa: BLE001
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning("Redis 不可达(%s)，降级 InMemorySaver（本进程内持久化，重启丢失）", exc)
        return InMemorySaver(), True, f"Redis 不可达({exc}) → 内存检查点"
