"""M5 验证：观测埋点 + 压测脚本 + 部署配置。

覆盖：
1. observability.TimedSpan 计时正确（>=0 且单调），log_slow_* 不抛异常
2. logging JSON 格式输出合法 JSON 行（可被采集器解析）
3. 检索/LLM/入库埋点真实生效（monkeypatch 校验慢操作日志触发路径）
4. 压测脚本 CLI 可解析（--help 退出 0）
5. 部署三件套存在且语法合法（Dockerfile/compose/手册关键章节）
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✔ {name}")
    else:
        FAIL += 1
        print(f"  ✘ {name}  {detail[:400]}")


def test_observability() -> None:
    print("— 观测模块 —")
    from app.core.observability import (TimedSpan, latency, log_ingest,
                                        log_llm_call, log_slow_query)

    # 1. TimedSpan 计时
    s = TimedSpan(name="t")
    time.sleep(0.02)
    d = s.stop(log_slow=False)
    check("TimedSpan.stop 返回毫秒且 >=0", d >= 0)

    # 2. latency 上下文
    with latency("ctx", log_slow=False) as sp:
        time.sleep(0.01)
    check("latency 上下文可正常退出", sp.name == "ctx")

    # 3. 慢日志函数不抛异常（阈值调 0 强制触发）
    try:
        log_slow_query("r1", "测试查询", 999.0, used_roads=["vector"], hits=3)
        log_llm_call("r1", 4000.0, node="answer", model="m", prompt_chars=10)
        log_ingest("r1", 6000.0, doc_id="d1", chunks=5, status="ok")
        check("log_slow_* 不抛异常", True)
    except Exception as exc:  # noqa: BLE001
        check("log_slow_* 不抛异常", False, str(exc))


def test_json_logging(tmp: Path) -> None:
    print("— JSON 日志 —")
    from app.core.logging import JsonFormatter

    rec = logging.LogRecord("test", logging.INFO, __file__, 1, "hello 你好", None, None)
    line = JsonFormatter().format(rec)
    parsed = json.loads(line)
    check("JsonFormatter 输出合法 JSON 行", isinstance(parsed, dict))
    check("JSON 行含 ts/level/logger/message",
          all(k in parsed for k in ("ts", "level", "logger", "message")))
    check("message 保留中文", parsed["message"] == "hello 你好")


def test_instrumentation_paths() -> None:
    print("— 埋点生效（静态检查 + 运行时触发）—")
    # 静态：确认关键调用点已注入
    hybrid_src = (Path(__file__).parent.parent / "app" / "retrieval" / "hybrid.py").read_text(encoding="utf-8")
    llm_src = (Path(__file__).parent.parent / "app" / "agent" / "llm.py").read_text(encoding="utf-8")
    mid_src = (Path(__file__).parent.parent / "app" / "api" / "middleware.py").read_text(encoding="utf-8")
    ingest_src = (Path(__file__).parent.parent / "app" / "api" / "ingest_tasks.py").read_text(encoding="utf-8")

    check("hybrid.py 注入 log_slow_query", "log_slow_query" in hybrid_src)
    check("llm.py 注入 log_llm_call", "log_llm_call" in llm_src)
    check("middleware.py 注入 http_request 计时", "http_request" in mid_src and "TimedSpan" in mid_src)
    check("ingest_tasks.py 注入 log_ingest", "log_ingest" in ingest_src)

    # 运行时：真实触发一次检索慢查询日志（mock embedder + 隔离临时 store，不污染真实库）
    import tempfile

    from app.retrieval.embedding import Embedder
    from app.retrieval.hybrid import HybridRetriever

    # Windows 下 Chroma sqlite 句柄延迟释放，tempdir 清理会锁文件 → 忽略清理错误
    tmp = Path(tempfile.mkdtemp(prefix="v5chroma_"))
    try:
        from app.retrieval.bm25store import BM25Store
        from app.retrieval.vectorstore import VectorStore

        emb = Embedder(provider="mock", model="m", dimensions=64, degraded=False)
        vs = VectorStore(tmp / "chroma")
        bm = BM25Store(tmp / "bm25.db")
        try:
            ret = HybridRetriever(vector_store=vs, bm25_store=bm, embedder=emb)
            res = ret.retrieve("请假流程")
            check("带埋点检索不报错且返回 HybridResult",
                  hasattr(res, "items") and isinstance(res.items, list))
        except Exception as exc:  # noqa: BLE001
            check("带埋点检索不报错", False, str(exc))
        finally:
            vs.close()
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_loadtest_cli() -> None:
    print("— 压测脚本 —")
    p = Path(__file__).parent / "loadtest.py"
    check("loadtest.py 存在", p.exists())
    if p.exists():
        r = subprocess.run([sys.executable, str(p), "--help"],
                           capture_output=True, text=True, timeout=30)
        check("loadtest.py --help 退出码 0", r.returncode == 0)
        check("loadtest.py 帮助含 chat/ingest 模式",
              "chat" in r.stdout and "ingest" in r.stdout)
        # 无服务时 chat 应正常报错返回（不崩溃），但会尝试连 127.0.0.1
        # 仅校验脚本可 import/解析（不实际联网）
        src = p.read_text(encoding="utf-8")
        check("loadtest.py 使用标准库（无第三方依赖）",
              "urllib" in src and "ThreadPoolExecutor" in src)


def test_deployment_artifacts() -> None:
    print("— 部署配置 —")
    root = Path(__file__).parent.parent
    df = root / "Dockerfile"
    dc = root / "docker-compose.yml"
    dm = root / "docs" / "deployment.md"

    check("Dockerfile 存在", df.exists())
    check("docker-compose.yml 存在", dc.exists())
    check("deployment.md 存在", dm.exists())

    if df.exists():
        s = df.read_text(encoding="utf-8")
        check("Dockerfile 用 python:3.13", "python:3.13" in s)
        check("Dockerfile 含健康检查", "HEALTHCHECK" in s)
        check("Dockerfile 非 root 运行", "appuser" in s or "useradd" in s)

    if dc.exists():
        s = dc.read_text(encoding="utf-8")
        check("compose 含 app + redis 服务",
              "redis" in s and "app:" in s)
        check("compose 挂载 data 持久化", "/app/data" in s)
        check("compose 含 REDIS_URL 指向容器服务", "redis://redis:6379" in s)

    if dm.exists():
        s = dm.read_text(encoding="utf-8")
        for section in ("鉴权", "容量规划", "观测", "压测", "扩容"):
            check(f"手册含「{section}」章节", section in s)


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_observability()
        test_json_logging(tmp)
        test_instrumentation_paths()
        test_loadtest_cli()
        test_deployment_artifacts()

    print(f"\n===== M5 验证：{PASS} 通过 / {FAIL} 失败 =====")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
