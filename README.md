# Enterprise-QA-Agent — 企业级智能问答 Agent

> RAG + LangGraph 多 Agent 企业问答系统。混合检索（向量 + BM25 + RRF）、引用溯源、SSE 流式问答 API。
> 当前进度：**M1–M7 全部完成**。M1 摄取 / M2 检索 / M3 Agent 编排 / M4 API / M5 观测部署 / M6 评估体系（golden + recall@5/MRR + 引用可回查率）/ M7 Web 前端（零构建静态调试客户端）均已闭环。

需求与契约：见 [`docs/`](docs/)（requirements v1.2 / api-contract v1.3 / doc-agent 吸收记录）。

## M1 已交付能力

| 能力 | 说明 |
|---|---|
| 7→4 格式解析 | PDF（PyMuPDF）/ DOCX / MD / TXT，扩展名主判 + 内容嗅探兜底（含 OLE 旧版拒绝） |
| 统一 Block | 解析器可插拔，`parse(file) -> list[Block]`（heading/paragraph/table/code/image/figure_transcript） |
| 定制 chunking | 512 token 上限 + 80 overlap；**table/code 整块不切断**；超长按行/句递归二次切 |
| PDF 质量门 | 逐页红/黄/绿零成本判级：乱码率 / 纯图页 / 文本过短 |
| VLM 转录钩子 | 红页 → `figure_transcript`（配 Key + VLM_ENABLED=1）；无 Key 降级原文入库 + 报告明示（R7 不崩） |
| 幂等入库 | `(tenant, doc_key)` 登记表 + `content_hash` 指纹：同 hash 重传全 skip；异 hash → **版本化软更新**（旧版 is_valid=false） |
| 双索引写入 | Chroma（向量，Append-Only）+ BM25（rank_bm25 + jieba），写入先向量后 BM25，先插新版再翻旧版（无空窗） |
| Embedding 双通道 | dashscope text-embedding-v3（分批 10 + 指数退避）/ 无 Key 自动降级 mock（degraded 标注不掩盖） |
| chunk 断点续跑 | 写入前 contains() 查重，跳过已完成 chunk（不重复计费） |
| CLI 报告 | 逐文档表格（状态/格式/v/块数/红黄绿/耗时/降级）+ 幂等跳过日志 |

## M2 已交付能力

| 能力 | 说明 |
|---|---|
| 统一过滤谓词 | 单源 `_filters`：`is_valid=true` + tenant 隔离 + permission 分级（public≤internal≤secret）+ 现行性窗口 + 可配 category/department；向量（Chroma where）与 BM25（SQL）从同一谓词生成，**防双路规则漂移**（F2.7） |
| 并行召回 | `HybridRetriever.retrieve()` 同步（M3 图节点用）/ `aretrieve()` 异步双路 gather 并行（F2.3/F6.4）；单路失败 → degraded 标注 + 存活路结果，不阻断 |
| RRF 融合 | score = Σ 1/(60+rank)，输出 top_n=8；命中带 `sources`（vector/bm25）、双路 rank、融合分 |
| 结果溯源完整 | 命中携带 chunk_id / doc_title / page_num / version / permission / effective_time / doc_year 等全 metadata（citation 直接可用） |
| 时效两级（F2.8） | 现行不足由上层判定后调 `relaxed_retrieve()` 顺序发起二级候选（**仅**放宽 effective_time）；过期命中标 `validity=expired` + `expired_at`，走"需用户确认"分支；列表为空时自动兜底同语义 |
| 年份感知（F2.9） | 显式年份 → `doc_year` 过滤检索；与语义 top1 同 doc_id 才采用，否则**回退语义结果 + note 明示**（doc-agent 实证坑已内置） |
| BM25 元数据对齐 | bm25_corpus 增列（tenant/permission/category/department/effective_time/doc_year + meta_json），`ALTER` 幂等迁移兼容 M1 存量库 |
| 验证 | `verify_m2.py`：30 断言（过滤/权限/隔离/过期/年份双分支/单路降级/async 一致性） |

## M3 已交付能力

| 能力 | 说明 |
|---|---|
| LangGraph 图 | `ingest → (chitchat/contact→direct_reply | 其余→rewrite→retrieve ─┬→ no_data └→ answer(合并意图)→verify) → finalize` + retry 回环（契约 §4.1 节点清单） |
| 职责边界 | **只做检索 + 披露，不发起升级 / 转交**：无转人工节点、不建工单、不指定责任人；"该找谁"只以文本告知，由客户自行联系 |
| 意图路由（F3.1，已合并） | 意图分类合并进 answer 节点（同一次 LLM 调用产出 intent+answer）；明确寒暄 / 要求转人工由 `rule_classify_intent` 规则短路（零 LLM），枚举 kb_qa/chitchat/contact_guidance |
| 查询改写（F3.2） | 结合对话历史消歧；**短路判据为白名单**——默认改写，只在本句被证明"自足"（无回指 / 非极短 / 非「…的+通用中心词」/ 含主题锚点）时才跳过；主题锚点词表从语料自动生成（`app/agent/anchors.py` + `data/kb_anchors.json`，ingest 后自动重建）。无历史时不改写（无消解对象，避免模型编主题）。过期确认的自然语言放行（"是/查看"→ include_expired） |
| 检索节点（F3.3） | 挂 F2 HybridRetriever；仅命中过期 → 发确认话术（无 interrupt，F2.8）；**检索为空 → no_data（0 LLM）** |
| Answer（F3.4/F3.9） | 强制 `[来源: 文档 页码]` 引用格式 + 引用只允许来自证据（防编造）；资料不足直说且**不建议转人工**；过期引用失效提示规则进 prompt |
| Verify（F3.5/F3.9） | 独立校验（输入只含引用内容+答案，不给思维链）；证据按引用收窄（未被引用的候选对判定无贡献，曾占本跳 prompt 84%）；**零引用 → 确定性判未达标（0 LLM 短路）** → 重试 → 用尽则 disclose；含过期引用 → confidence ≤0.5；纯过期支撑 → degraded=true |
| 防死循环（F3.6/F3.8） | 置信度/重试判定全在条件边：verify 不达标且 retry<2 → rewrite 回环；超限 → **disclose（保留答案 + 披露局限后缀）** |
| 资料不足 / 转人工兜底 | `no_data`：如实告知缺失 + 指向管理员录入；`contact_guidance`：只给"该找谁"指引——两者均**不转交** |
| 对话记忆（F4） | checkpointer 持久化（thread_id=`{tenant}:{user}`）；消息窗口截断最近 10 轮=20 条；**历史文本两级预算**（每条 `HISTORY_PER_MSG_CHARS` + 总量 `HISTORY_TOTAL_CHARS`，从最新往旧累积、优先丢最旧）；每轮入口重置本轮输出字段防跨轮串扰 |
| 双通道降级 | LLM：DashScope（QWEN_LLM_MODEL 可配，JSON/流式双模式，现用 Qwen3.8-Flash）/ 无 Key→Stub 规则式（保链路回归）；checkpointer：RedisSaver / 不可达→InMemorySaver（降级不掩盖） |
| 验证 | `verify_m3.py`：60 断言（意图路由 / 引用达标链 / contact 指引 / 重试 2 次披露局限 / 空检索 no_data 0-LLM / 过期两轮流 / 记忆与窗口 + **历史预算** / rewrite 短路判据 / **零引用未达标**）；`verify_sse_streaming.py`：24 断言（含零引用不得直接出货）；`verify_cli_ingest.py`：14 断言（cmd_ingest 全路径） |

## M4 已交付能力

| 能力 | 说明 |
|---|---|
| FastAPI 服务 | `app/api/main.py`（create_app）：CORS + X-Request-Id 中间件 + 统一错误体 + 可选鉴权（配置 SERVICE_API_KEY 后启用） |
| SSE 流式问答 | `POST /api/v1/chat`（F5.1）：事件序 ready → token* → citation* → done / error，空闲 >15s ping 保活；**token 事件即 LLM 真实增量**（answer 节点文本流式 + 引用反解）；非 LLM 直答（chitchat/contact_guidance/no_data/确认话术）done 前整段补发 token，`disclose` 的披露后缀经 token_sink 实时补发，前端拼接零特判 |
| 非流式回复 | `stream=false` 返回单个 AssistantReply（含 request_id/latency_ms），经并发闸（默认 50）超限 429 |
| 文档异步入库 | `POST /api/v1/documents`（F5.2/F6.5）：三态 202（新/变更/删后重传）/ 200 duplicated / 409 INGEST_IN_PROGRESS；单写者线程池串行入库（Chroma 并发写限制）+ 阶段进度回写（parse/chunk/embed/index） |
| 任务状态 | `GET /api/v1/tasks/{id}`（F5.3）：progress.phase/percent、warnings（红页/质量告警）、error{code,message} |
| 软删除 | `DELETE /api/v1/documents/{doc_id}`（F5.7）：翻 is_valid=false + registry is_deleted 标记，重复删除幂等 204；删后重传 → 版本化重入 |
| 对话历史 | `GET /api/v1/threads/{thread_id}/history?limit=`（F5.4） |
| 检索调试 | `POST /api/v1/debug/retrieve`（F5.8）：双路 rank/score + RRF fused 明细 + filters/degraded/notes |
| 健康检查 | `GET /api/v1/health`（F5.5）：vector/bm25 核心 down → 503；仅 Redis 降级 → 200 degraded（设计内降级，见契约 v1.3） |
| 流式引用 | 文本流结束后从 `[来源: 标题 页码]` 反解 chunk_id（`_parse_chunk_ids_from_answer`）；反解失败 verify grounded=False → 重试/披露自然兜底，不编造引用 |
| 验证 | `verify_m4.py`：35 断言（SSE 事件序/校验错误路径/上传三态轮询/删除幂等/debug 结构/隔离） |

## M5 已交付能力

| 能力 | 说明 |
|---|---|
| 结构化观测 | `app/core/observability.py`：零第三方依赖计时 + JSON 行日志——慢操作（`slow_query`/`llm_call`/`ingest_slow`/`http_request`）+ 业务事件（`llm_usage` 成本账本、`kb_gap` 检索缺口线索），`req_id` 贯穿；`LOG_FORMAT=json`（默认）输出单行 JSON 供 Loki/ELK 采集 |
| 耗时埋点 | middleware（每请求耗时+状态码）/ hybrid 检索（慢查询，含 hits/used_roads）/ LLM 往返（node/model/tokens）/ 入库（doc_id/chunks/status）；阈值 `OBS_SLOW_*_MS` 可调 |
| 压测脚本 | `scripts/loadtest.py`：纯标准库（urllib + ThreadPoolExecutor），并发 SSE 问答 + 文档入库两模式，输出吞吐/延迟分位 p50/p90/p99/错误率/SSE 事件分布 |
| 容器化部署 | `Dockerfile`（多阶段 + 非 root + 健康检查）+ `docker-compose.yml`（app + Redis AOF）+ `docs/deployment.md`（鉴权/扩容/容量规划/观测/压测基线/优化方向） |
| 验证 | `verify_m5.py`：29 断言（观测模块/JSON 日志/埋点生效/压测 CLI/部署三件套） |

## M6 已交付能力

| 能力 | 说明 |
|---|---|
| golden 评估集 | `data/golden/qa_golden.json`：**55 条**（语义 / 精确编号 / 跨年份 / 改写稳健 / 噪声 五类），每条 `query → 期望文档 + 锚句`；锚句归一化（全半角/空白/大小写）子串匹配全库定位期望块，**零人工标注块 id**。锚句可定位性由 `scripts/verify_golden_anchors.py` 用真实解析+切块链路体检（55/55 命中） |
| 检索层指标 | `app/eval/metrics.py`：recall@5（期望块进 top-k 的用例占比）+ MRR；锚句定位失败用例不计分母（F7.2） |
| 答案层指标 | 引用可回查率——`citations.chunk_id` 必须属于检索命中块（F7.3，`--answers` 开启，需真 Key） |
| 门槛判定 | 真实向量 recall@5 ≥ 0.8 → pass；mock 向量无语义 → 指标 SKIP 并标 degraded（F7.4） |
| 报告 + CLI | `data/reports/eval_report_latest.json`（provider/degraded/逐用例明细 + `cost` / `latency` / `verify` 三块聚合；逐题含 `llm_calls` / `retries` / `verified`）；`python -m app.cli eval [--answers] [--top-k K] [--limit N]`。评估 thread 带**本次运行盐值**（保证冷启动，指标可跨轮比较） |
| 实测 | 真实 DashScope 向量（`qwen3.7-text-embedding-flash`，1024 维；语料 38 文档 / 105 chunk）：**recall@5 = 0.855 / MRR = 0.791**，55 条全评估（verdict=pass）；题型分组 semantic 0.875 / paraphrase 1.000 / exact 0.750 / year 1.000 / noisy 0.000。同语料 **BM25 单路下界**（`scripts/bm25_probe.py`，无需 Key）recall@5=0.836 / MRR=0.673 → RRF 融合向量路 +0.019 / +0.118 |
| 验证 | `verify_m6.py`：19 断言（golden 校验/normalize/锚句定位 100%/mock SKIP/报告落盘/CLI） |
| 已知局限 | 语料 38 文档 / 105 chunk（top-5 覆盖率 4.8%）。**exact（数字/编号）0.750、noisy（口语改写）0.000 是两块短板**：干扰文档含同款关键词把 BM25 分数摊薄，口语问法又天然缺字面重叠——两者都靠向量路补偿，也正是 Hybrid 存在的理由。规模体检与题型分组见 `scripts/corpus_stats.py` / `scripts/bm25_probe.py`。答案层引用可回查率需 `--answers`（走真实 LLM，本轮未跑） |

## M7 已交付能力（Web 前端）

| 能力 | 说明 |
|---|---|
| 提问页 | `web/index.html`：SSE 流式问答（token 增量渲染 + 引用卡片 + 意图/置信度/耗时徽章）+ 对话历史侧栏 + 服务健康状态 + 新会话/流式开关/过期资料开关 |
| 文档管理页 | `web/upload.html`：多文件拖拽/选择**并发上传**，上传即 202 入队（先入库），后台清洗切片（parse→chunk→embed→index 四阶段进度条 + warnings + 失败重试），任务实时轮询 |
| 静态挂载 | FastAPI `StaticFiles` 挂载 `web/`，根路径 `/` 重定向到提问页；零构建零依赖（纯 HTML/CSS/JS，直接浏览器访问 `http://127.0.0.1:8000/`） |
| 验证 | 端到端实测：SSE 事件序 ready→token*→citation→done、上传三态（202/200/409）、任务轮询 done、软删除 204 全通过 |

## 快速开始

```bash
uv sync --dev                 # 清华镜像，Python ≥3.13（.python-version 锁定 3.13）
.venv/Scripts/python.exe scripts/make_samples.py      # 基础演示语料（5 份）
.venv/Scripts/python.exe scripts/make_corpus.py       # 扩语料批次一：干扰项 + 格式覆盖（13 份）
.venv/Scripts/python.exe scripts/make_corpus_long.py  # 扩语料批次二：长文档（20 份，撑大 chunk 数）
.venv/Scripts/python.exe -m app.cli ingest data/samples   # 摄取（无 Key 自动 mock）
.venv/Scripts/python.exe scripts/verify_m1.py       # M1 验证：41 断言
.venv/Scripts/python.exe scripts/verify_m2.py       # M2 验证：30 断言
.venv/Scripts/python.exe scripts/verify_m3.py       # M3 验证：60 断言（stub LLM + 内存检查点）
.venv/Scripts/python.exe scripts/verify_cli_ingest.py  # CLI ingest 路径验证：14 断言（桩，覆盖 cmd_ingest 全路径）
.venv/Scripts/python.exe scripts/verify_m4.py       # M4 验证：35 断言（TestClient + 隔离服务）
.venv/Scripts/python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000   # 启动 API + Web 前端
# 浏览器打开 http://127.0.0.1:8000/ （提问页），/upload.html 为文档管理页
```

配真 Key / Redis（可选）：
- 项目根建 `.env`：`DASHSCOPE_API_KEY=sk-xxx`（对话 LLM + embedding 自动切真模型）
- Redis（跨会话记忆）：`REDIS_URL=redis://localhost:6379/0`（本机或 WSL2 内起 redis-server 即可；不可达自动降级内存并明示）
- **真实环境冒烟清单**（stub/mock 只保链路）：① 真 Key 下跑一轮 kb_qa 看真实检索→引用→verify；② Redis 起来后同一 thread 两轮对话，重开进程历史仍在；③ 过期文档确认流（验收标准 8）。

API 一览（契约 `docs/api-contract.md`，OpenAPI：`docs/openapi.yaml`）：

```bash
# SSE 流式问答
curl -N -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' -H 'X-Tenant-Id: tenant_demo' \
  -d '{"thread_id":"tenant_demo:user_demo","question":"报销单编号规则 XB 开头几位？"}'
# 文档上传 → 202 task_id，轮询 /api/v1/tasks/{id} 至 done
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H 'X-Tenant-Id: tenant_demo' -F file=@data/samples/sample_guide.md
# 检索调试
curl -X POST http://127.0.0.1:8000/api/v1/debug/retrieve \
  -H 'Content-Type: application/json' -d '{"query":"年假 跨年 审批"}'
```

## 目录结构

```
app/
├── core/            # config（环境变量+.env）/ 结构化日志
├── models/          # 数据契约：Block / DocMeta / ChunkRecord / Citation / doc_id 派生
├── ingestion/
│   ├── detection.py # 格式探测
│   ├── parsers.py   # 可插拔解析器（pdf/docx/md/txt）
│   ├── pdf_quality.py # 质量门红黄绿判级（F1.9）
│   ├── chunking.py  # 分格式定制切片 + 二次切分
│   ├── registry.py  # DocRegistry 幂等登记（sqlite）
│   ├── vlm.py       # VLM 转录器抽象（dashscope / none 降级）
│   └── pipeline.py  # 摄取编排：detect→parse→chunk→embed→双索引
├── retrieval/
│   ├── embedding.py # dashscope/mock 双通道
│   ├── vectorstore.py # Chroma（Append-Only + soft-delete）
│   ├── bm25store.py # rank_bm25 + jieba（扩展元数据列 + 过滤式检索）
│   └── hybrid.py    # HybridRetriever：并行召回 + RRF + F2.8/F2.9
├── agent/
│   ├── state.py     # AgentState（契约 §4.1）
│   ├── schemas.py   # 节点结构化输出（intent Literal / verify）
│   ├── prompts.py   # prompt 构建（JSON 示例一律 json.dumps；M4 加 answer_stream_prompt 纯文本模板）
│   ├── llm.py       # DashScope / Stub 双通道（M4 加 stream_answer 流式）
│   ├── nodes.py     # ingest/rewrite/retrieve/no_data/answer(合并意图)/verify/retry/disclose/direct_reply/finalize
│   ├── checkpointer.py # RedisSaver → InMemorySaver 降级工厂
│   └── graph.py     # StateGraph 组装 + AgentApp（reply/history/stream_events）
├── api/             # M4 FastAPI 服务层
│   ├── main.py      # create_app：lifespan / 中间件 / 错误处理 / 路由挂载
│   ├── deps.py      # Services 全局单例（共享 store/retriever/agent/task_manager）
│   ├── middleware.py # X-Request-Id + 可选鉴权（SERVICE_API_KEY）+ tenant
│   ├── errors.py    # ApiError + 统一错误体（契约 §1.5）
│   ├── ingest_tasks.py # 入库后台任务（单写者 + 进度 + in-flight 409）
│   └── routes/      # chat(SSE)/documents/tasks/threads/debug/health
├── models/          # 数据契约：Block / DocMeta / ChunkRecord / Citation / api.py（请求/响应模型）
├── eval/            # M6 评估体系
│   ├── golden.py    # golden 加载/校验 + 锚句归一化子串定位期望块
│   ├── metrics.py   # recall@5 / MRR / 引用可回查率
│   └── runner.py    # 编排 + 门槛判定 + 报告落盘
└── cli.py           # python -m app.cli ingest / eval
web/                       # M7 前端（零构建纯静态，FastAPI StaticFiles 挂载）
├── index.html             # 提问页（SSE 流式 + 引用 + 历史 + 健康）
├── upload.html            # 文档管理页（并发上传 + 清洗切片进度轮询）
├── common.css             # 共享暗色样式
└── common.js              # 共享 API 客户端 + SSE 解析
scripts/make_samples.py  # 演示语料生成
scripts/verify_m1.py     # M1 验证断言
scripts/verify_m2.py     # M2 检索验证断言
scripts/verify_m3.py     # M3 图编排验证断言
scripts/verify_m4.py     # M4 API 端到端验证断言
scripts/verify_m5.py     # M5 观测/部署验证断言
scripts/verify_m6.py     # M6 评估验证断言
scripts/verify_golden_anchors.py  # golden 锚句可定位性体检（免 chromadb，写用例时用）
scripts/loadtest.py      # M5 压测脚本
scripts/eval_multiturn.py  # 多轮评估（先真跑历史轮，再评估末轮；history 成本 + 跳改写行为）
scripts/build_anchor_vocab.py  # 手动重建主题锚点词表（ingest 后会自动重建）
scripts/replay_skip_rewrite.py # rewrite 短路判据离线回放（纯规则、零 LLM，改判据前先跑）
data/samples/            # 演示语料（md/txt/docx/pdf/含红页 pdf）
data/golden/qa_golden.json  # 评估 golden 集（55 条）
data/golden/qa_golden_multiturn.json  # 多轮 golden 集（5 条，schema 1.1-multiturn）
data/kb_anchors.json     # 主题锚点词表（由语料派生，勿手改；ingest 后自动重建）
data/chroma_db|bm25|registry.db|uploads  # 运行时数据（gitignore）
```

## Roadmap

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 解析 + chunking + 质量门 + 双索引 + 幂等版本化 | ✅ 45/45 |
| M2 | 混合检索：向量 + BM25 并行 + RRF + 年份/时效过滤 | ✅ 30/30 |
| M3 | LangGraph 多 Agent：rewrite/retrieve/no_data/answer(合并意图)/verify/disclose + Redis checkpointer | ✅ 60/60（stub） |
| M4 | FastAPI：SSE 流式 + 文档上传 + 软删除 + debug/health | ✅ 35/35 |
| M5 | 观测（JSON 日志+耗时埋点+慢操作）+ 压测脚本 + Docker/部署手册 | ✅ 42/42 |
| M6 | 评估体系：golden recall@5/MRR + 引用可回查率（题集 55 条 / 语料 38 文档 105 chunk） | ✅ 19/19 |
| M7 | Web 前端：提问页（SSE 流式）+ 文档管理页（并发上传+进度） | ✅ 端到端实测 |

> 已知工程坑（实现期实测）：Chroma `query_texts` 会触发默认模型下载 → 检索统一走显式 embedding；jieba 在 Py3.14 无 wheel 需锁 3.13（`.python-version`）；PyMuPDF 默认字体不含中文，生成语料需 `insert_font(fontfile=simhei.ttf)`；`VectorStore.query` 的 `top_k/where` 是 keyword-only，`asyncio.to_thread` 传参须用 lambda；BM25 SQL 占位符数必须与参数数动态匹配（permission 白名单长度可变）；**langgraph 须 ≥1.2.11**（旧版 langgraph 0.5.0 与 langchain-core 1.6 冲突报 MRO 错误，故须升到 1.2.11+）；langgraph-checkpoint-redis 取 0.5.x（0.5.2 验证可用）；pydantic v2 静默忽略 extra 字段——构造 ChunkRecord 时元数据键名必须精确（`effective_time` 写成 `eff` 会被丢弃且不报错）；checkpoint 跨轮持久化 → 每轮入口必须重置"本轮输出"字段（degraded/intent/citations 等），否则上一轮 degraded/intent 状态串扰下一轮；**Py3.12+ `StopIteration` 不能经 `asyncio.to_thread` 的 Future 传播**（转 RuntimeError）→ SSE 迭代器用哨兵对象收尾；FastAPI 路由 prefix 若自带 `/v1` 再 include `prefix=/api/v1` 会双前缀 → 各 router 去掉版本段统一由 include 加；新版 FastAPI include_router 为 `_IncludedRouter` 惰性挂载（openapi 才可查完整路径）；**Windows 下 Chroma 的 sqlite 句柄延迟释放** → 测试用 `TemporaryDirectory` 清理会 `PermissionError [WinError 32]`，须 `mkdtemp + shutil.rmtree(ignore_errors=True)` 并在 `close()` 后手工清理。**`.env` 内联注释陷阱**：python-dotenv 只剥离「值非空」时后随的注释——`SERVICE_API_KEY=` 后直接跟 `#`（中间无实值）会把整段注释当成密钥 → 意外开启鉴权、接口全 401；该注释必须独立成行。**`ingest --rebuild` 曾静默清空索引**：旧实现 `old_version = version if rebuild else version-1`，当 `action=="new"` 时 `old_version` 恰等于刚写入的版本 → 新块被自己翻成 `is_valid=false`，检索返回空且无任何报错；现改为「同 hash 也强制 bump 版本重灌，只失效上一版本」，`verify_m1.py` 已加 4 条回归断言。
>
> **`qwen3.8-flash` 默认开思考（reasoning），`max_tokens` 压不住它**（2026-09-14 实测）：同一道题 reasoning 占 314~877 completion token——用户看不见，却按**输出价**计费，且 token 是串行生成的、直接变成延迟（这是当时 P95 135.9s 的头号成因）。更坑的是 `max_tokens` **只约束可见正文**：设 64 时 `finish_reason=length`、正文被截断、JSON 不合法 → `complete_json` 白重试一次（成本翻倍）。所以"给 answer 限长"必须**关思考 + 限可见正文**一起做（`LLM_ENABLE_THINKING=0` + `ANSWER_MAX_TOKENS=512`）；只调 `max_tokens` 不但无效，还可能更亏。实测同 24 题：completion 57,320 → 5,638 token（-90%）、p50 41.8s → 6.2s、p95 135.9s → 11.1s，而 recall/可回查率**完全不变**。
>
> **CLI 子命令只跑 `--help` 等于没测**（2026-09-14 实测）：`cmd_ingest` 曾出现 `_rebuild_anchors` 函数体内混入 `cmd_ingest` 尾部代码（引用 `reports`/`ok`）→ 一旦真跑 ingest 就 `NameError`，而 `--help` 走的是 argparse、根本不进函数体，冒烟测试全绿也发现不了。凡新增 CLI 子命令行为，验证必须**真正调用该函数**（`scripts/verify_cli_ingest.py` 用桩驱动 `cmd_ingest`，覆盖返回码 / provider 行 / 词表重建 / 软失败 / 空输入 5 条路径）。
>
> **历史文本必须限长，而"单轮评估"结构上看不见这件事**（2026-09-15）：`render_history` 旧写法是「窗口 10 轮 × 2 条 × 每条 300 字」= 6,000 字/次注入，而历史在链路里被注入**两次**（rewrite 一跳 + answer 一跳；verify 刻意不注入）→ 最坏约 7,500 tok/问 ≈ 单问 prompt 的 2.3 倍。为什么九轮评估都没抓到：**单轮集没有历史，这一项恒为 0**；而当时的 5 条多轮样本**每条恰好 2 轮**，正好落在窗口阈值之下——既测不到 10 轮窗口，也几乎碰不到 300 字截断。现改为两级预算（`HISTORY_PER_MSG_CHARS=150` + `HISTORY_TOTAL_CHARS=1200`），且**从最新往旧累积**：最新一轮必然保留、优先丢最旧（消解指代依赖最近上文）；轮数窗口退化为安全网（预算先于窗口生效）。教训：**评估口径本身会决定你能看见哪些问题**——只在单轮场景测，多轮的成本结构永远是盲区。
>
> **零引用 = 不可回查的裸答案，必须判未达标**（2026-09-15）：四类出口里凡「给出答案」的（① 有资料→答案+出处；② 资料不全→现有数据+出处）都必须带出处，但 verify 此前的兜底是"零引用 → 回退全量证据照常判定"——模型只要说 grounded=true 就直接出货（q007 有前科），`disclose` 里那支 `_NO_CITE_SUFFIX`（"未能在现有知识库中找到对应出处"）因此**几乎成了死代码**。现在零引用一律确定性判 `grounded=False / confidence=0` 并**短路不调 LLM**（答案按构造即不可校验，没有可"判"的东西），落回既有的 retry→disclose 出口。副作用要认：模型如实写"资料里没有这条"时也会被打回重试（hint 正是"如实说明资料不足之处与信息来源"），最多多花两次往返后进披露。
>
> **⚠ 已知偏差：重试路径破坏「token 拼接 == done.answer」**（2026-09-15 实测暴露，**未修**）：verify 打回重试时，每次 answer 都会重新流式推送，于是 token 流里留下**多次尝试**的文本，而 `done.answer` 只有最后一次（+ 披露后缀）。影响可控——前端在 `done` 处用 `done.answer` **整体覆盖**重渲染（`web/index.html`），不会留下错误终态，只是流式中途的瞬态重影。要真正修好需要协议层加 `reset`/`retry` 事件（契约 §2.1 变更），已锁定在 `verify_sse_streaming.py` 的第 4 组断言里，避免它被当成"已修复"。
>
> **「单字代词必须按分词整词匹配，不能按子串」**（2026-09-15，多轮评估实测抓出）：`should_skip_rewrite` 的回指判定原先对 `("它","这","那",…)` 做**子串**匹配，于是「员工食堂**这**周的菜单是什么？」被判成指代句 → 强制执行一次改写，而改写结果与原文**逐字相同** = 纯空转一跳（长历史下每问白烧一次 rewrite）。改为 **jieba 整词判单字代词 + 子串判多字短语**（`_has_anaphora`）：`这周/那次/其他` 被 jieba 切成整词 → 不再误命中；`这个/上述/该文档` 这类多字指代仍靠子串兜住（自身无歧义，也不依赖分词边界）。代价是判据现在**依赖 jieba 是否把复合词切为整词**（缺依赖时退化为只判多字短语，偏保守）。修完单轮集自足率 81.8% → **89.1%**（此前约 4 道题被误判需改写），golden 多轮 **7/7**、人工探针 **13/13** 全对。教训与「词表干净才敢用子串匹配」同源：**任何"含某字即命中"的判据，先问分词边界**。
>
> **⚠ 修正：长历史样本并未真正触发总量预算**（2026-09-15 复测）：先前称"8 轮历史必然越过 `HISTORY_TOTAL_CHARS=1200`"**是错的**——真实答案平均约 90 字，8 轮只累计到 636~1,030 字。真实被触发的只有 **per-msg 150 字上限**（m007 出现多行恰好 154 字 = 前缀 4 + 150，答案被硬截在「[来源:」处）。故把 m007 由 8 轮补到 **10 轮历史**（恰为窗口上限，避免与轮数窗口混淆）使其真正越过 1,200；m006 保持 8 轮，作为「长历史但未截断」的对照下界。总量预算本身是确定性纯函数，四条行为（不超上限 / 优先丢最旧 / 极小预算保底非空 / 默认绑定总量）已由 `verify_m3` 零 LLM 断言覆盖——真跑样本的作用只是证明**真实历史确实能长到这个量级**。
