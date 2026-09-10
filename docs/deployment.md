# 部署手册（M5）

本手册覆盖 Enterprise-QA-Agent 的容器化部署、鉴权、容量规划、观测与压测基线。
系统定位：混合检索（Chroma + BM25 + RRF）+ LangGraph 多 Agent 企业问答服务。

---

## 1. 快速部署（Docker Compose）

前置：Docker 20.10+、已配置 `.env`（至少 `DASHSCOPE_API_KEY`）。

```bash
cp .env.example .env
vim .env                    # 填 DASHSCOPE_API_KEY、SERVICE_API_KEY
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

验证健康：

```json
{"status":"ok","version":"0.1.0","checks":{"vector_store":"up","bm25":"up","redis":"up","llm":"up"}}
```

- `status=ok`：核心链路（向量库/BM25）正常且 Redis 正常。
- `status=degraded` + HTTP 200：Redis 不可达，已降级内存 checkpointer（跨会话记忆失效，但问答可用）。
- `status=degraded` + HTTP 503：向量库或 BM25 不可用（**核心链路不可用，必须告警**）。

---

## 2. 配置项（关键）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 空 | 必填；空则 embedding/LLM 双降级 mock（仅链路回归，无语义） |
| `QWEN_EMBEDDING_MODEL` | text-embedding-v3 | 向量模型（需与 `EMBED_DIMENSIONS` 一致） |
| `QWEN_LLM_MODEL` | qwen-plus | 对话模型 |
| `SERVICE_API_KEY` | 空 | **生产必填**；启用 `Authorization: Bearer <key>` 鉴权 |
| `REDIS_URL` | redis://localhost:6379/0 | checkpointer 地址 |
| `REDIS_CHECKPOINTER_ENABLED` | 1 | 1=用 Redis 跨会话；0=强制内存 |
| `INGEST_WORKERS` | 1 | 入库线程数；**保持 1**（Chroma 并发写限制） |
| `API_MAX_INFLIGHT` | 50 | 并发在途问答上限，超限 429 |
| `MAX_UPLOAD_MB` | 50 | 单文件上传大小上限，超限 413 |
| `ALLOWED_ORIGINS` | `*` | CORS 白名单（逗号分隔域名）；生产收紧，本地默认 `*` |
| `LOG_FORMAT` | json | json=生产日志采集；text=本地调试 |

---

## 3. 鉴权与安全

1. **生产必须设置 `SERVICE_API_KEY`**（非空即启用 Bearer 鉴权，中间件校验）。
2. 客户端请求带 `Authorization: Bearer <key>` 与 `X-Tenant-Id`（多租户隔离）。
3. `X-Tenant-Id` 决定向量/BM25 检索的 tenant 过滤；`thread_id` 前缀须与之一致（`{tenant}:{user}`），否则 422。
4. CORS 由 `ALLOWED_ORIGINS`（逗号分隔域名）控制，默认 `*` 仅限内网 demo；**生产按域名收紧**（如 `ALLOWED_ORIGINS=https://a.example.com,https://b.example.com`）。
5. 上传 `meta.permission` 不接受客户端传入（模型未定义该字段，防提权）；权限默认 `internal`。

---

## 4. 扩容与容量规划

| 维度 | 建议 | 原因 |
|---|---|---|
| 副本数 | 水平扩容（多副本 + 共享 Redis） | 单进程多 worker 会各自打开 Chroma 连接，存在并发写风险 |
| 数据目录 | `./data` 挂卷持久化 | 向量库/BM25/registry.db/uploads/日志 |
| Redis | AOF 持久化 + 定期备份 | checkpoint 是跨会话记忆唯一持久层 |
| 入库并发 | `INGEST_WORKERS=1` | 单写者串行，避免 Chroma 写冲突 |
| 问答并发 | `API_MAX_INFLIGHT` 对齐 LLM 配额 | 每请求多轮 LLM 往返，需控在途量防雪崩 |

**水平扩容注意**：Chroma 的 `data/chroma_db` 是本地目录，多副本共享同一挂卷会冲突。
若需多副本，二选一：
- (a) 多副本**只读**共享（入库走独立单写者实例）；
- (b) 迁移到 Chroma 服务端 / 其他分布式向量库（超出当前范围，见 M6 评估阶段再议）。

---

## 5. 观测

- **日志**：`data/logs/agent.log`（`LOG_FORMAT=json` 时单行 JSON，含 `ts/level/logger/message`）。
- **慢操作事件**（WARN 级，JSON 行，可按 `event` 聚合告警）：
  - `slow_query`：检索 >500ms（含 query、hits、used_roads）
  - `llm_call`：LLM 往返 >3000ms（含 node、model、prompt_chars、tokens）
  - `ingest_slow`：入库 >5000ms（含 doc_id、chunks、status）
  - `http_request`：每请求耗时 + 状态码（INFO 级，req_id 贯穿）
- **阈值可调**：`OBS_SLOW_QUERY_MS` / `OBS_SLOW_LLM_MS` / `OBS_SLOW_INGEST_MS`。

接入采集（任选）：
- Loki + Promtail：直接吞 JSON 行日志；
- Filebeat → ELK：同目录采集；
- 自建告警：`jq 'select(.event=="slow_query")' data/logs/agent.log`。

---

## 6. 压测基线

内置零依赖压测脚本（标准库，无需 locust/httpx）：

```bash
# 先灌入语料
.venv/Scripts/python.exe -m app.cli ingest data/samples --rebuild
# 启动服务
.venv/Scripts/python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 并发 SSE 问答：20 并发 x 10 轮
.venv/Scripts/python.exe scripts/loadtest.py --mode chat --concurrency 20 --rounds 10
# 入库压测
.venv/Scripts/python.exe scripts/loadtest.py --mode ingest --concurrency 10
```

输出指标：吞吐（QPS）、延迟分位（p50/p90/p99/max）、错误率、SSE 事件分布。

**参考基线**（本机 mock/stub，无网络）：
- 单请求端到端 SSE（真实 DashScope）约 16.8s（4 次 LLM 往返，偏慢，见 §7 优化）。
- mock 模式下主要瓶颈在 Chroma 查询与 BM25 分词，QPS 取决于本机 CPU。

> 压测前确认服务已灌入语料，否则 `chat` 会大量命中"资料不足→转人工"分支，延迟不代表真实负载。

---

## 7. 已知性能项与优化方向

| 问题 | 现状 | 优化方向 |
|---|---|---|
| 全链路 16.8s | 4 次 LLM 往返（supervisor/rewrite/answer/verify）串行 | ① rewrite 仅在有上下文歧义时触发；② verify 可并行/采样；③ 流式 answer 已即时吐 token，感知延迟更低 |
| 向量路 latency | Chroma 本地查询 | 已并行 BM25；大数据量评估 hnsw 参数 |
| 入库串行 | 单写者 | 多文档 batch embed（embedding_batch_size） |

---

## 8. 回滚与运维

- 回滚：`docker compose down && git checkout <tag> && docker compose up -d --build`。
- 数据备份：`data/`（含 chroma_db、bm25、registry.db）+ Redis AOF。
- 蓝绿/灰度：多副本 + 独立 Redis DB 分片（`REDIS_URL` 指向不同 db index）。
