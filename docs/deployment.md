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

> **容器配置的两点注意**
> 1. `docker-compose.yml` 的 `environment:` 是**白名单**——只有列出的变量会注入容器。
>    宿主机 `.env` 仅用于 compose 的 `${VAR}` 变量替换，**未在白名单登记的变量改了不起作用**。
>    新增可调配置时请同步登记（`scripts/verify_m5.py` 会校验安全/限额关键项）。
> 2. 镜像构建**不依赖 `uv.lock`**（本项目 `.gitignore` 忽略锁文件）：Dockerfile 用
>    `COPY pyproject.toml` + `uv sync`（未加 `--frozen`），构建时在镜像内解析依赖并生成锁文件。
>    若追求可复现构建，可提交 `uv.lock` 并改回 `COPY pyproject.toml uv.lock ./` + `--frozen`。
> 3. 健康检查语义两边一致：**HTTP 503（degraded）也算存活**，只有进程不可达才判 unhealthy。

---

## 2. 配置项（关键）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 空 | 必填；空则 embedding/LLM 双降级 mock（仅链路回归，无语义） |
| `QWEN_EMBEDDING_MODEL` | text-embedding-v3 | 向量模型（需与 `EMBED_DIMENSIONS` 一致） |
| `QWEN_LLM_MODEL` | qwen-plus | 对话模型（**id 全小写**，大小写错会 404 model_not_found） |
| `ANSWER_MAX_TOKENS` | 512 | answer 正文长度上限（只约束可见正文，不含 reasoning）。调小可省 token/延迟，但**太大会白花钱、太小小会截断 JSON**（截断→解析失败→白重试一次，日志有 `输出被 max_tokens 截断` 告警） |
| `LLM_ENABLE_THINKING` | 0 | 是否开思考模式。qwen3.8-flash **默认开思考**，实测单题 reasoning 占 314~877 token（用户看不见但按输出价计费、还串行拖慢）；关闭后同题 completion 1069→193 token |
| `HTTP_READ_TIMEOUT_S` | 120 | LLM 读取超时；配合 `LLM_MAX_RETRIES=0` 杜绝 SDK 静默重发整段 prompt |
| `LLM_MAX_RETRIES` | 0 | LLM 客户端重试次数；**保持 0**，重试由业务层可控退避接管 |
| `SERVICE_API_KEY` | 空 | **生产必填**；启用 `Authorization: Bearer <key>` 鉴权 |
| `REDIS_URL` | redis://localhost:6379/0 | checkpointer 地址 |
| `REDIS_CHECKPOINTER_ENABLED` | 1 | 1=用 Redis 跨会话；0=强制内存 |
| `INGEST_WORKERS` | 1 | 入库线程数；**保持 1**（Chroma 并发写限制） |
| `API_MAX_INFLIGHT` | 50 | 并发在途问答上限，超限 429 |
| `MAX_UPLOAD_MB` | 50 | 单文件上传大小上限，超限 413（读前 `file.size` 预筛 + 流式计数兜底） |
| `ALLOWED_ORIGINS` | `*` | CORS 白名单（逗号分隔域名）；生产收紧，本地默认 `*` |
| `LOG_FORMAT` | json | json=生产日志采集；text=本地调试 |

---

## 3. 鉴权与安全

1. **生产必须设置 `SERVICE_API_KEY`**（非空即启用 Bearer 鉴权，中间件校验）。
   ⚠️ **不要写成 `SERVICE_API_KEY=   # 留空 = 关闭鉴权`** —— python-dotenv 只剥离「值非空」时后随的注释；`=` 后没有实值直接跟 `#` 时，整段 `# ...` 会被当作密钥值，导致**意外开启鉴权**、所有未带 key 的请求 401。注释必须独立成行，值留空即写成 `SERVICE_API_KEY=`。
2. 客户端请求带 `Authorization: Bearer <key>` 与 `X-Tenant-Id`（多租户隔离）。
3. `X-Tenant-Id` 决定向量/BM25 检索的 tenant 过滤；`thread_id` 前缀须与之一致（`{tenant}:{user}`），否则 422。
4. CORS 由 `ALLOWED_ORIGINS`（逗号分隔域名）控制，默认 `*` 仅限内网 demo；**生产按域名收紧**（如 `ALLOWED_ORIGINS=https://a.example.com,https://b.example.com`）。
5. 上传 `meta.permission` 不接受客户端传入（模型未定义该字段，防提权）；权限默认 `internal`。
6. **上传大小两层防护**：应用层由 `MAX_UPLOAD_MB` 控制——先按 `file.size`（Starlette 解析 multipart 时统计的真实字节数，非客户端声明）预筛，再在分块读取中累加计数兜底（覆盖 chunked 传输与谎报长度），全程不把整份文件读进内存，超限返回 `413 DOC_TOO_LARGE`。
   建议**网关层同步限制**，让超大请求在收包阶段就被掐断、根本进不了 Python 进程：

   ```nginx
   client_max_body_size 50m;   # 与 MAX_UPLOAD_MB 保持一致
   ```

   注意：nginx 值应 **≥** `MAX_UPLOAD_MB`，否则超限请求会先被 nginx 拦掉并返回 HTML 错误页，前端拿不到统一的 JSON 错误体。

---

## 4. 扩容与容量规划

| 维度 | 建议 | 原因 |
|---|---|---|
| 副本数 | 水平扩容（多副本 + 共享 Redis） | 单进程多 worker 会各自打开 Chroma 连接，存在并发写风险 |
| 数据目录 | `./data` 挂卷持久化 | 向量库/BM25/registry.db/uploads/日志/锚点词表 |
| Redis | AOF 持久化 + 定期备份 | checkpoint 是跨会话记忆唯一持久层 |
| 入库并发 | `INGEST_WORKERS=1` | 单写者串行，避免 Chroma 写冲突 |
| 问答并发 | `API_MAX_INFLIGHT` 对齐 LLM 配额 | 每请求多轮 LLM 往返，需控在途量防雪崩 |

**锚点词表（`data/kb_anchors.json`）**：`rewrite` 短路判据用的主题词，由 `ingest`
完成后自动重建（`app/cli.py::_rebuild_anchors`），也可手动跑
`scripts/build_anchor_vocab.py`。它是从 BM25 语料派生的，**语料变更后不重建就会过期**；
过期只会让判据退化成"一律改写"（不影响正确性，只多一次 LLM 往返），且词表缺失/损坏
时自动退化为同一行为，不会中断问答。路径可用 `KB_ANCHORS_PATH` 覆盖。

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
- **业务事件**（INFO 级，不受慢阈值限制，每次必记，供成本/内容运营聚合）：
  - `llm_usage`：每次 LLM 调用的 token 与预估成本（node/model/prompt_tokens/completion_tokens/est_cost_cny）
  - `kb_gap`：**检索完全无命中**时记录的覆盖缺口线索（req_id/thread_id/query/rewritten_query/expired_candidates）；**仅供离线聚类"缺失主题"**，不进任何人工作队列、不触发工单
- **阈值可调**：`OBS_SLOW_QUERY_MS` / `OBS_SLOW_LLM_MS` / `OBS_SLOW_INGEST_MS`。

接入采集（任选）：
- Loki + Promtail：直接吞 JSON 行日志；
- Filebeat → ELK：同目录采集；
- 自建告警：`jq 'select(.event=="slow_query")' data/logs/agent.log`；
- 知识库缺口盘点（离线、非告警）：`jq -r 'select(.event=="kb_gap") | .query' data/logs/agent.log | sort | uniq -c | sort -rn` —— 把"查不到的问题"聚成缺失主题清单，交管理员决定补哪几篇。

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
- 单请求端到端 SSE（真实 DashScope，合并前 qwen3.8-max）：约 16.8s（3 次 LLM 往返 supervisor+answer+verify，NFR P95≤8s 未达标）；现合并意图进 answer + 换 Qwen3.8-Flash 后 kb_qa 仅 2 次往返，延迟预期显著下降，待重新压测确认（见 §7）。
- mock 模式下主要瓶颈在 Chroma 查询与 BM25 分词，QPS 取决于本机 CPU。

> 压测前确认服务已灌入语料，否则 `chat` 会大量命中"检索为空 → no_data"分支（0 次 LLM，延迟虚低），不代表真实负载。

---

## 7. 已知性能项与优化方向

| 问题 | 现状 | 优化方向 |
|---|---|---|
| 全链路延迟 | kb_qa 现 2 次 LLM 往返（answer[合并意图] / verify）串行；含改写歧义时 +1（rewrite） | ① supervisor 已合并进 answer 同一次调用（少一次往返）；② rewrite 仅在有上下文歧义时触发；③ verify 可并行/采样；④ 流式 answer 即时吐 token，感知延迟更低 |
| 向量路 latency | Chroma 本地查询 | 已并行 BM25；大数据量评估 hnsw 参数 |
| 入库串行 | 单写者 | 多文档 batch embed（embedding_batch_size） |

---

## 8. 回滚与运维

- 回滚：`docker compose down && git checkout <tag> && docker compose up -d --build`。
- 数据备份：`data/`（含 chroma_db、bm25、registry.db）+ Redis AOF。
- 蓝绿/灰度：多副本 + 独立 Redis DB 分片（`REDIS_URL` 指向不同 db index）。
