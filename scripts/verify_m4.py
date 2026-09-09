"""M4 端到端验证（FastAPI TestClient + 隔离 mock/stub 服务）。

覆盖（契约 §2）：
1. /health 健康检查（vector/bm25 up）
2. /v1/chat stream=false 返回 AssistantReply（intent/citations/confidence）
3. /v1/chat stream=true SSE 事件序 ready→token*→citation*→done，token 拼接 == done.answer
4. 校验：空 question → 422 VALIDATION_INVALID_ARGUMENT；thread 租户前缀不匹配 → 422 THREAD_TENANT_MISMATCH
5. /v1/threads/{id}/history 多轮后消息可查
6. 上传 202 → 轮询任务 done → 再传同内容 200 duplicated；异 doc_key 第二个任务
7. DELETE /documents/{doc_id} 204 + 重复删除幂等 204 → debug/retrieve 不再命中
8. /v1/debug/retrieve 结构（fused/filters/degraded）
9. 不支持类型上传 → 415
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.agent.llm import StubLLM
from app.api.deps import Services
from app.api.main import app
from app.ingestion.registry import DocRegistry
from app.retrieval.bm25store import BM25Store
from app.retrieval.embedding import Embedder
from app.retrieval.vectorstore import VectorStore

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


def make_test_services(tmp: Path) -> Services:
    emb = Embedder(provider="mock", model="mock-hash-v1", dimensions=256, degraded=False)
    vs = VectorStore(tmp / "chroma")
    bm = BM25Store(tmp / "bm25.db")
    reg = DocRegistry(tmp / "registry.db")
    return Services(embedder=emb, vector_store=vs, bm25_store=bm, registry=reg,
                    llm=StubLLM(), memory_checkpoint=True)


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    """解析 text/event-stream → [(event, data_dict)]。"""
    out = []
    for block in raw.decode("utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[len("event: "):]
            elif line.startswith("data: "):
                data += line[len("data: "):]
        if ev:
            out.append((ev, json.loads(data)))
    return out


def wait_task(client: TestClient, task_id: str, timeout_s: float = 30) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/api/v1/tasks/{task_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise TimeoutError(f"任务 {task_id} 超时")


def run() -> int:
    print("═══ M4 API 端到端验证（隔离 mock/stub）═══")
    tmp = Path(tempfile.mkdtemp(prefix="verify_m4_"))
    test_svcs: Services | None = None
    try:
        with TestClient(app) as client:
            # 用隔离测试服务替换启动期真实服务（stub LLM + mock embedder + 临时库）
            test_svcs = make_test_services(tmp)
            app.state.services = test_svcs
            app.state.inflight = asyncio.Semaphore(50)
            svc = test_svcs

            # ── 1. health ─────────────────────────────────
            print("── 1. /health ──")
            r = client.get("/api/v1/health")
            h = r.json()
            check("health 200", r.status_code == 200, r.text)
            check("vector/bm25 up", h["checks"].get("vector_store") == "up"
                  and h["checks"].get("bm25") == "up", str(h["checks"]))
            check("redis down 但 degraded（内存降级设计内）", h["checks"].get("redis") == "down"
                  and h["status"] == "degraded", str(h))

            # ── 2. 上传两份语料（202 → done）───────────────
            print("── 2. 上传入库 ──")
            kb1 = ("# 报销制度\n\n差旅报销流程：线上 OA 申请，电子发票上传系统自动核验。\n\n"
                   "报销单编号规则：XB 开头共 12 位数字。\n").encode("utf-8")
            r = client.post("/api/v1/documents",
                            files={"file": ("reimburse.md", kb1,
                                            "text/markdown")})
            check("上传 202", r.status_code == 202, f"{r.status_code} {r.text}")
            up1 = r.json()
            t1 = wait_task(client, up1["task_id"])
            check("任务 done", t1["status"] == "done", str(t1))
            check("任务进度 100", t1["progress"]["percent"] == 100, str(t1["progress"]))
            doc_id1 = up1["doc_id"]

            kb2 = ("# 考勤管理\n\n员工请假需提前一天在 OA 提交，由直属主管审批。\n\n"
                   "迟到 30 分钟以上按旷工半天处理。\n").encode("utf-8")
            r = client.post("/api/v1/documents",
                            files={"file": ("attendance.md", kb2, "text/markdown")})
            up2 = r.json()
            t2 = wait_task(client, up2["task_id"])
            check("第二份 done", t2["status"] == "done", str(t2))

            # ── 3. chat 非流式 ────────────────────────────
            print("── 3. /v1/chat 非流式 ──")
            r = client.post("/api/v1/chat", json={
                "thread_id": "tenant_demo:u1", "question": "报销单编号规则 XB 开头几位？",
                "stream": False})
            check("非流式 200", r.status_code == 200, r.text)
            reply = r.json()
            check("answer 非空", bool(reply.get("answer")), reply.get("answer", "")[:80])
            check("intent=kb_qa", reply.get("intent") == "kb_qa", str(reply.get("intent")))
            check("citations 非空", len(reply.get("citations", [])) > 0,
                  str(reply.get("citations"))[:150])
            check("request_id 透传", reply.get("request_id", "").startswith("req_"),
                  reply.get("request_id", ""))
            check("latency_ms>0", reply.get("latency_ms", 0) > 0, str(reply.get("latency_ms")))

            # ── 4. SSE 流式事件序 ──────────────────────────
            print("── 4. SSE 事件流 ──")
            with client.stream("POST", "/api/v1/chat", json={
                    "thread_id": "tenant_demo:u2", "question": "员工请假流程 主管审批 几天？",
                    "stream": True}) as r:
                check("SSE HTTP 200", r.status_code == 200, str(r.status_code))
                events = parse_sse(b"".join(r.iter_bytes()))
            types = [e for e, _ in events]
            check("含 ready", types and types[0] == "ready", str(types[:3]))
            has_done = "done" in types
            check("含 done", has_done, str(types))
            if has_done:
                di = types.index("done")
                pre = types[:di]
                # ready 必须最先；token 与 citation 顺序可交错但都在 done 前
                order_ok = (pre[0] == "ready"
                            and set(pre[1:]).issubset({"token", "citation", "ping"}))
                check("done 前事件合法", order_ok, str(pre))
                tok = "".join(d["content"] for t, d in events if t == "token")
                done = next(d for t, d in events if t == "done")
                check("token 拼接 == done.answer", tok == done.get("answer", ""),
                      f"token={tok[:40]!r} answer={done.get('answer','')[:40]!r}")
                check("done.citations 与 citation 事件一致",
                      len(done.get("citations", [])) == sum(1 for t, _ in events
                                                            if t == "citation"),
                      str(len(done.get("citations", []))))

            # 指代多轮（F4.3）：上一轮提报销 → 这轮"它有几位数字"应改写检索
            r = client.post("/api/v1/chat", json={
                "thread_id": "tenant_demo:u1",
                "question": "它一共几位数字？", "stream": False})
            check("多轮指代可答", r.status_code == 200 and bool(r.json().get("answer")),
                  r.text[:120])

            # ── 5. history ────────────────────────────────
            print("── 5. 对话历史 ──")
            r = client.get("/api/v1/threads/tenant_demo:u1/history?limit=50")
            hist = r.json()
            roles = [m["role"] for m in hist["messages"]]
            check("历史含 user+assistant", "user" in roles and "assistant" in roles,
                  str(roles))
            check("历史条数上限截断", len(hist["messages"]) <= 50,
                  str(len(hist["messages"])))

            # ── 6. 校验错误路径 ────────────────────────────
            print("── 6. 错误路径 ──")
            r = client.post("/api/v1/chat", json={
                "thread_id": "tenant_demo:u3", "question": "   ", "stream": False})
            check("空 question → 422", r.status_code == 422
                  and r.json()["error"]["code"] == "VALIDATION_INVALID_ARGUMENT", r.text)
            r = client.post("/api/v1/chat", json={
                "thread_id": "other_tenant:u3", "question": "你好", "stream": False})
            check("thread 租户不匹配 → 422", r.status_code == 422
                  and r.json()["error"]["code"] == "THREAD_TENANT_MISMATCH", r.text)
            r = client.post("/api/v1/documents",
                            files={"file": ("evil.exe", b"MZ...", "application/octet-stream")})
            check("不支持类型 → 415", r.status_code == 415
                  and r.json()["error"]["code"] == "DOC_UNSUPPORTED_TYPE", r.text)
            r = client.get("/api/v1/tasks/ingest_nonexistent")
            check("任务不存在 → 404", r.status_code == 404
                  and r.json()["error"]["code"] == "INGEST_NOT_FOUND", r.text)

            # ── 7. 幂等 / 删除 ─────────────────────────────
            print("── 7. 幂等重传与软删除 ──")
            r = client.post("/api/v1/documents",
                            files={"file": ("reimburse.md", kb1, "text/markdown")})
            body = r.json()
            check("同内容重传 → 200 duplicated", r.status_code == 200
                  and body["duplicated"] is True, f"{r.status_code} {body}")

            r = client.delete(f"/api/v1/documents/{doc_id1}")
            check("DELETE → 204", r.status_code == 204, str(r.status_code))
            r = client.delete(f"/api/v1/documents/{doc_id1}")
            check("重复 DELETE 幂等 → 204", r.status_code == 204, str(r.status_code))
            r = client.delete("/api/v1/documents/doc_not_exist_000")
            check("删除不存在 → 404", r.status_code == 404
                  and r.json()["error"]["code"] == "DOC_NOT_FOUND", r.text)

            # ── 8. debug/retrieve ──────────────────────────
            print("── 8. /v1/debug/retrieve ──")
            r = client.post("/api/v1/debug/retrieve",
                            json={"query": "考勤 迟到 旷工 处理"})
            check("debug 200", r.status_code == 200, r.text)
            dbg = r.json()
            check("debug fused 非空", len(dbg["fused"]) > 0, str(len(dbg["fused"])))
            first = dbg["fused"][0]
            check("fused 字段完整", {"chunk_id", "fused_rank", "validity", "doc_title"}
                  .issubset(first.keys()), str(first))
            check("filters 含 is_valid", dbg["filters"].get("is_valid") is True,
                  str(dbg["filters"]))

            # ── 9. 删除后检索不再命中 ───────────────────────
            print("── 9. 删除后检索隔离 ──")
            r = client.post("/api/v1/debug/retrieve",
                            json={"query": "报销单编号规则 XB 开头 12 位"})
            cids = {i["chunk_id"] for i in r.json()["fused"]}
            del_cids = [c for c in cids if c.startswith(doc_id1)]
            check("被删文档块不出现在现行检索", del_cids == [], str(sorted(cids)))

            print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
            return 1 if FAIL else 0
    finally:
        if test_svcs is not None:
            try:
                test_svcs.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(run())
