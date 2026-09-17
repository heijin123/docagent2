"""文档列表 + 有效期（契约 §2.4b / §2.4c）端到端验证（隔离 mock 服务，无需 Key）。

背景（2026-09-17 补齐）：
- `/v1/documents` 此前只有 POST / DELETE —— **没有列表**，前端刷新即丢、DELETE 需要的
  doc_id 也拿不到 → 验收标准 12「软删除后前端列表同步」无法成立；
- `effective_time`（F2.8 有效期）只有读取方、**没有任何写入入口** → 过期链路
  （验收标准 8）代码在、验不了。现由 `PATCH /v1/documents/{doc_id}` 单一写入口补上。

覆盖：
1. GET 列表：空 → 上传后可见 / 字段完整 / 分页（limit+offset）/ total 是过滤后总数；
2. PATCH：写有效期 → 登记表 + 向量 + BM25 三处同步；
3. **F2.8 端到端**：设成过去时间 → 主检索不再召回该文档，`relaxed_retrieve` 转为其过期候选；
4. 404：不存在的 doc_id；
5. DELETE 后列表默认不含，`include_deleted=true` 可查（验收标准 12）。

用法：PYTHONPATH=D:/workspace/docagent2 python scripts/verify_documents_api.py
"""
from __future__ import annotations

import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.agent.llm import StubLLM  # noqa: E402
from app.api.deps import Services  # noqa: E402
from app.api.main import app  # noqa: E402
from app.ingestion.registry import DocRegistry  # noqa: E402
from app.models import now_ts  # noqa: E402
from app.retrieval.bm25store import BM25Store  # noqa: E402
from app.retrieval.embedding import Embedder  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.vectorstore import VectorStore  # noqa: E402

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


def wait_task(client: TestClient, task_id: str, timeout_s: float = 30) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/api/v1/tasks/{task_id}")
        body = r.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise TimeoutError(f"任务 {task_id} 超时")


def run() -> int:
    print("═══ 文档列表 + 有效期 API 验证（隔离 mock）═══")
    tmp = Path(tempfile.mkdtemp(prefix="verify_docs_api_"))
    svc = Services(
        embedder=Embedder(provider="mock", model="mock-hash-v1", dimensions=256,
                          degraded=False),
        vector_store=VectorStore(tmp / "chroma"),
        bm25_store=BM25Store(tmp / "bm25.db"),
        registry=DocRegistry(tmp / "registry.db"),
        llm=StubLLM(), memory_checkpoint=True,
    )
    with TestClient(app) as client:
        app.state.services = svc

        print("── 1. 空列表 ──")
        r = client.get("/api/v1/documents")
        check("GET 200", r.status_code == 200, r.text)
        body = r.json()
        check("空列表 total=0 items=[]", body["total"] == 0 and body["items"] == [],
              str(body))

        print("── 2. 上传两篇 → 列表可见 ──")
        kb1 = "# 年假制度\n\n员工入职满一年享年假 5 天，可跨年休。\n".encode()
        kb2 = "# 考勤制度\n\n迟到 30 分钟以上按旷工半天处理。\n".encode()
        ups = []
        for name, data in (("leave.md", kb1), ("attendance.md", kb2)):
            rr = client.post("/api/v1/documents",
                             files={"file": (name, data, "text/markdown")})
            check(f"上传 {name} 202", rr.status_code == 202, f"{rr.status_code} {rr.text}")
            up = rr.json()
            t = wait_task(client, up["task_id"])
            check(f"{name} 任务 done", t["status"] == "done", str(t))
            ups.append(up)

        r = client.get("/api/v1/documents")
        body = r.json()
        check("列表 total=2", body["total"] == 2, str(body["total"]))
        check("列表 items=2", len(body["items"]) == 2, str(len(body["items"])))
        first = body["items"][0]
        check("列表字段完整",
              {"doc_id", "doc_key", "version", "status", "is_deleted",
               "effective_time", "updated_at"} <= set(first),
              str(sorted(first)))
        check("status=done", all(i["status"] == "done" for i in body["items"]),
              str([i["status"] for i in body["items"]]))
        check("默认 effective_time=0（永久有效）",
              all(i["effective_time"] == 0 for i in body["items"]),
              str([i["effective_time"] for i in body["items"]]))

        print("── 3. 分页 ──")
        p1 = client.get("/api/v1/documents?limit=1&offset=0").json()
        p2 = client.get("/api/v1/documents?limit=1&offset=1").json()
        check("limit=1 → 1 条但 total=2", len(p1["items"]) == 1 and p1["total"] == 2,
              str(p1))
        check("offset 生效（两页 doc_id 不同）",
              p1["items"][0]["doc_id"] != p2["items"][0]["doc_id"],
              f"{p1['items'][0]['doc_id']} vs {p2['items'][0]['doc_id']}")
        check("limit 上限被夹到 200",
              client.get("/api/v1/documents?limit=9999").json()["limit"] == 200, "")

        print("── 4. PATCH 设置有效期（未来 → 仍现行）──")
        target = ups[0]
        doc_id = target["doc_id"]
        future = now_ts() + 86400
        r = client.patch(f"/api/v1/documents/{doc_id}",
                         json={"effective_time": future})
        check("PATCH 200", r.status_code == 200, r.text)
        pr = r.json()
        check("chunks_updated > 0", pr["chunks_updated"] > 0, str(pr))
        check("返回新版有效期", pr["effective_time"] == future, str(pr))

        listed = {i["doc_id"]: i for i in client.get("/api/v1/documents").json()["items"]}
        check("登记表已同步（列表可见）", listed[doc_id]["effective_time"] == future,
              str(listed[doc_id]))

        # 三处同步：Chroma / BM25 chunk 元数据
        cids = svc.vector_store.doc_chunk_ids(doc_id, target["version"])
        check("向量侧 chunk_id 非空", len(cids) > 0, str(cids))
        rows = svc.bm25_store.get_by_chunk_ids(cids)
        check("BM25 侧 chunk 元数据同步（meta_json）",
              all(r["metadata"].get("effective_time") == future for r in rows),
              str([r["metadata"].get("effective_time") for r in rows]))

        retr = HybridRetriever(vector_store=svc.vector_store,
                               bm25_store=svc.bm25_store, embedder=svc.embedder)
        res = retr.retrieve("年假可以跨年休吗")
        check("未来有效期 → 主检索仍召回该文档",
              any(it["metadata"]["doc_id"] == doc_id for it in res.items),
              str([it["metadata"]["doc_id"] for it in res.items]))

        print("── 5. PATCH 设成过去 → F2.8 过期链路生效 ──")
        past = now_ts() - 86400
        r = client.patch(f"/api/v1/documents/{doc_id}", json={"effective_time": past})
        check("PATCH 200", r.status_code == 200, r.text)
        res2 = retr.retrieve("年假可以跨年休吗")
        check("主检索不再召回过期文档",
              all(it["metadata"]["doc_id"] != doc_id for it in res2.items),
              str([(it["metadata"]["doc_id"], it["validity"]) for it in res2.items]))
        rel = retr.relaxed_retrieve("年假可以跨年休吗")
        check("放宽窗口后进入过期候选（validity=expired）",
              any(it["metadata"]["doc_id"] == doc_id
                  and it["validity"] == "expired" for it in rel.expired_candidates),
              str([(it["metadata"]["doc_id"], it["validity"])
                   for it in rel.expired_candidates]))
        check("放宽结果的 note 提示需用户确认",
              any("过期" in n for n in rel.notes), str(rel.notes))

        print("── 6. 恢复永久有效 + 错误路径 ──")
        r = client.patch(f"/api/v1/documents/{doc_id}", json={"effective_time": 0})
        check("可恢复为永久有效（0）", r.json()["effective_time"] == 0, r.text)
        res3 = retr.retrieve("年假可以跨年休吗")
        check("恢复后主检索重新召回",
              any(it["metadata"]["doc_id"] == doc_id for it in res3.items), "")
        r = client.patch("/api/v1/documents/doc_nonexistent",
                         json={"effective_time": 0})
        check("不存在 doc_id → 404 DOC_NOT_FOUND",
              r.status_code == 404 and r.json()["error"]["code"] == "DOC_NOT_FOUND",
              r.text)
        r = client.patch(f"/api/v1/documents/{doc_id}", json={"effective_time": -1})
        check("负数有效期 → 422", r.status_code == 422, r.text)

        print("── 7. 软删除与列表同步（验收标准 12）──")
        victim = ups[1]["doc_id"]
        r = client.delete(f"/api/v1/documents/{victim}")
        check("DELETE 204", r.status_code == 204, r.text)
        after = client.get("/api/v1/documents").json()
        check("默认列表不再含已删除文档",
              all(i["doc_id"] != victim for i in after["items"]),
              str([i["doc_id"] for i in after["items"]]))
        check("total 同步减少", after["total"] == 1, str(after["total"]))
        withdel = client.get("/api/v1/documents?include_deleted=true").json()
        check("include_deleted=true 可查到（审计口径）",
              any(i["doc_id"] == victim and i["is_deleted"] for i in withdel["items"]),
              str([(i["doc_id"], i["is_deleted"]) for i in withdel["items"]]))

    print(f"\n结果：PASS={PASS}  FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
