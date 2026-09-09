"""M5 压测脚本（零第三方依赖）：并发 SSE 问答 + 文档入库。

用法（先 `python -m app.cli ingest --sample` 或启动服务并灌入语料）：

    # 纯 SSE 问答压测：50 并发 x 20 轮
    python scripts/loadtest.py --mode chat --concurrency 50 --rounds 20

    # 入库压测：上传 data/samples 下所有文件，10 并发（受 INGEST_WORKERS 单写者限流）
    python scripts/loadtest.py --mode ingest --concurrency 10

    # 混合 + 自定义地址
    python scripts/loadtest.py --base http://127.0.0.1:8000 --mode chat \
        --concurrency 30 --rounds 10 --query "请假流程是什么"

设计：
- 用标准库 urllib + concurrent.futures.ThreadPoolExecutor（不引入 httpx/aiohttp/locust，
  可直接 `uv run scripts/loadtest.py` 运行）；
- SSE 解析：读 text/event-stream 累积 data 行，遇到 event: done/error 结束；
- 指标：吞吐(QPS)、延迟分位(p50/p90/p99/max)、错误率、SSE 事件分布；
- 两种模式共用一套统计器，输出对齐 md 表格便于写入部署手册。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "http://127.0.0.1:8000"
API = "/api/v1"

QUERIES = [
    "请假流程是什么？",
    "公司年假有多少天？",
    "报销需要哪些材料？",
    "转人工客服",
    "你好",
    "考勤制度 2024 年的规定",
    "入职需要提交什么资料？",
    "加班补贴怎么算？",
]


def _post(url: str, data: dict, headers: dict | None = None,
          timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _chat_once(base: str, question: str, idx: int, thread: str) -> dict:
    """单次 SSE 问答，返回 (status, duration_ms, events, err)。"""
    url = f"{base}{API}/chat"
    body = json.dumps({
        "question": question, "thread_id": thread, "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    events: list[str] = []
    err = ""
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            status = resp.status
            buf = ""
            for raw in resp:
                buf += raw.decode("utf-8", "replace")
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    ev, data = "", ""
                    for line in frame.splitlines():
                        if line.startswith("event: "):
                            ev = line[7:].strip()
                        elif line.startswith("data: "):
                            data = line[6:].strip()
                    if ev:
                        events.append(ev)
                        if ev in ("done", "error"):
                            # 提前结束（若服务端仍继续发送也断不开，此处仅标记）
                            break
    except Exception as exc:  # noqa: BLE001
        status = -1
        err = str(exc)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    return {"status": status, "duration_ms": duration_ms,
            "events": events, "err": err}


def _ingest_once(base: str, path: Path, idx: int) -> dict:
    """单次文件上传（multipart），返回 (status, duration_ms, err)。"""
    url = f"{base}{API}/documents"
    boundary = f"----loadtest{idx}{int(time.time()*1000)}"
    fname = path.name
    raw = path.read_bytes()
    # 手工构造 multipart 体（标准库无便捷 multipart 编码）
    parts = [
        f"--{boundary}",
        f'Content-Disposition: form-data; name="file"; filename="{fname}"',
        "Content-Type: application/octet-stream",
        "",
    ]
    head = ("\r\n".join(parts) + "\r\n").encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    payload = head + raw + tail
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    t0 = time.perf_counter()
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
        err = f"HTTP {e.code}"
    except Exception as exc:  # noqa: BLE001
        status = -1
        err = str(exc)
    return {"status": status, "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "err": err}


def _report(title: str, results: list[dict]) -> None:
    ok = [r for r in results if r["status"] in (200, 202)]
    bad = [r for r in results if r["status"] not in (200, 202)]
    durs = sorted(r["duration_ms"] for r in ok)
    total = sum(r["duration_ms"] for r in results)
    n = len(results)
    okn = len(ok)

    def pct(p: float) -> float:
        if not durs:
            return 0.0
        return durs[min(len(durs) - 1, int(p * len(durs)))]

    wall = (max(r["duration_ms"] for r in results) -
            min(r["duration_ms"] for r in results)) or 1.0
    qps = okn / (wall / 1000.0) if wall else 0.0

    print(f"\n===== {title} =====")
    print(f"总请求 {n} | 成功 {okn} | 失败 {len(bad)} | 错误率 {len(bad)/max(n,1)*100:.1f}%")
    if okn:
        print(f"延迟(ms)  均值 {statistics.mean(durs):.0f} | "
              f"p50 {pct(0.50):.0f} | p90 {pct(0.90):.0f} | "
              f"p99 {pct(0.99):.0f} | max {durs[-1]:.0f}")
    print(f"吞吐      约 {qps:.1f} req/s（并发窗口法，仅量级参考）")
    if bad:
        from collections import Counter
        print("错误分布", dict(Counter(r["err"] or f"status={r['status']}"
                                  for r in bad).most_common(5)))
    if results and "events" in results[0]:
        from collections import Counter
        evc = Counter()
        for r in results:
            evc.update(r["events"])
        print("SSE 事件", dict(evc.most_common()))


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 压测脚本")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--mode", choices=["chat", "ingest"], default="chat")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=10, help="chat 每线程轮数")
    ap.add_argument("--query", default="", help="固定问题（默认轮换内置语料）")
    ap.add_argument("--samples-dir", default="data/samples",
                    help="ingest 模式读取的文件目录")
    args = ap.parse_args()

    results: list[dict] = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    def run_chat():
        out = []
        for i in range(args.rounds):
            q = args.query or QUERIES[(i + idx) % len(QUERIES)]
            out.append(_chat_once(args.base, q, idx, f"tenant_demo:loadtest{idx}"))
        return out

    def run_ingest():
        files = sorted(Path(args.samples_dir).glob("*"))
        files = [f for f in files if f.suffix.lower() in
                 (".pdf", ".docx", ".md", ".txt")]
        if not files:
            return [{"status": -1, "duration_ms": 0, "err": "无样例文件"}]
        return [_ingest_once(args.base, files[idx % len(files)], idx)]

    total = args.concurrency if args.mode == "ingest" else args.concurrency
    worker = run_chat if args.mode == "chat" else run_ingest
    with ThreadPoolExecutor(max_workers=total) as ex:
        futs = [ex.submit(worker) for idx in range(total)]
        for fut in as_completed(futs):
            with lock:
                results.extend(fut.result())

    wall = time.perf_counter() - t0
    _report("SSE 问答压测" if args.mode == "chat" else "入库压测", results)
    print(f"总墙钟 {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
