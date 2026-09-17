# 接口契约 — Enterprise-QA-Agent

| 项 | 内容 |
|---|---|
| 文档版本 | v1.3 |
| 关联文档 | `docs/enterprise-qa-agent-requirements.md`（需求与机制设计） |
| 变更记录 | v1.3（2026-09-09，M4 API 落地）：§2.1 `AssistantReply` 增加可选 `request_id`/`latency_ms`/`notes`；§2.2 明确 `meta.department/category` 落库、`meta.permission` 一律忽略（防提权）；§2.4 DELETE 语义补 registry is_deleted 标记与删后重传（版本化重入）；§2.7 健康语义放宽（Redis 不可达属设计内降级 → HTTP 200 + status=degraded，核心依赖 down 才 503）；错误码登记 `INGEST_FAILED`；§2.6 debug/retrieve 请求补 `department`；§6.1 生成 `docs/openapi.yaml` 机器可读副本。<br>v1.2（2026-09-09）：吸收 doc-agent——citation 事件与 AssistantReply.citations 增加 `doc_date`/`image_ids`；新增 §2.8 图片回填端点 + 错误码 `IMG_NOT_FOUND`；§4.5 过滤构造增加可选 `doc_year`；block_type 枚举追加 `figure_transcript` |
| 权威声明 | **本文档是接口字段、枚举、错误码、校验规则的唯一权威。** 代码（app/models）必须与本文档一一对应；两处不一致时以本文档为准并修正代码。 |

---

## 1. 总则与通用约定

### 1.1 编码与命名

- HTTP 协议，请求/响应体一律 `application/json; charset=utf-8`（上传除外）
- 字段命名统一 **snake_case**；枚举值统一 **小写下划线**（如 `kb_qa`、`contact_guidance`）
- 时间戳统一 **Unix 秒（int）**，禁止毫秒/字符串日期混用
- 空值语义：可选字段用 `null`，不传或传 null 等价；禁止用 `""` 表示空
- 数值限制：所有数值为 JSON number；禁止 NaN / Infinity；浮点分数仅允许出现在内部调试接口

### 1.2 路径与版本

- 所有 API 前缀 `/api/v1`（如 `/api/v1/chat`），本文档正文省略前缀
- 兼容性规则（见第 6 节）：向后兼容的变更（加可选字段、枚举追加）版本号不变；**破坏性变更必须 bump 到 `/v2`**，并在需求文档同步

### 1.3 鉴权与租户隔离

| Header | 必填 | 说明 |
|---|---|---|
| `Authorization: Bearer <service_key>` | 条件必填 | 服务密钥。**仅当服务端配置 `SERVICE_API_KEY` 非空时强制**；未配置则鉴权关闭、放行（默认租户兜底），缺失/无效→401（仅鉴权启用时生效） |
| `X-Tenant-Id` | 推荐必填 | 租户标识。服务端据此注入 `permission` 过滤条件；缺省时回落 `DEFAULT_TENANT_ID`（默认 `tenant_demo`），多租户场景必须显式传 |
| `X-Request-Id` | 否 | 客户端透传的追踪 ID（uuid），服务端回显；缺省由服务端生成 |

- **客户端不可直接传 `permission` / `department` 等过滤字段用于提权**——检索过滤条件全部由服务端依据租户-用户权限注入
- `thread_id` 建议格式 `{tenant}:{user}`（如 `tenant_a:user_001`）；若提供，服务端校验其前缀与 `X-Tenant-Id` 一致，不一致返回 422 `THREAD_TENANT_MISMATCH`

### 1.4 通用响应头

所有响应携带 `X-Request-Id`。限流响应（429）额外携带 `Retry-After`（秒）。

### 1.5 统一错误体（非 SSE）

```json
{
  "error": {
    "code": "RETRIEVAL_UNAVAILABLE",
    "message": "检索服务暂时不可用，请稍后重试",
    "request_id": "req_xxxx",
    "details": {}
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| code | string | 错误码，见第 3 节字典，**禁止自造 code** |
| message | string | 面向调用方的可读信息，不得包含堆栈/内部路径 |
| request_id | string | 追踪 ID，贯穿日志 |
| details | object | 可选，结构化补充（如字段校验明细） |

### 1.6 SSE 内的错误

SSE 通道内发生错误时 **HTTP 状态保持 200**，通过 `error` 事件下发同一错误体，随后服务端关闭流（见 2.8）。

---

## 2. 外部接口契约

### 2.1 POST /v1/chat — 企业问答（流式）

**请求体**

```json
{
  "thread_id": "tenant_a:user_001",
  "question": "年假可以跨年休吗？",
  "stream": true,
  "include_expired": false
}
```

| 字段 | 类型 | 必填 | 默认 | 约束与说明 |
|---|---|---|---|---|
| thread_id | string | 是 | — | 1~128 字符；Redis checkpointer 的 key 维度 |
| question | string | 是 | — | strip 后 1~4000 字符，否则 422 |
| stream | bool | 否 | true | false 时返回单个 AssistantReply（非 SSE） |
| include_expired | bool | 否 | false | true = 主检索即允许过期文档参与；**仍受 F2.8 回答约束**（引用必带失效标注） |

**SSE 事件序列契约**

```
必选顺序：ready → (token)* → (citation)* → done  |  任意时刻可中断为 error
可选：空闲 > 15s 时服务端发 ping（保活）
```

| 事件 | 触发 | data 字段 |
|---|---|---|
| `ready` | 图启动 | `{request_id}` |
| `token` | 生成节点逐字产出 | `{content: string}` |
| `citation` | 生成节点引用一个 chunk | `{chunk_id, doc_title, page_num, validity, expired_at?, doc_date?, image_ids?}` |
| `done` | 正常结束 | 完整 AssistantReply（见下） |
| `error` | 任意异常 | 统一错误体（见 1.5） |
| `ping` | 保活 | `{}` |

**done 事件 data（= AssistantReply 模型，stream=false 时即为响应体）**

```json
{
  "answer": "可以。根据《考勤管理制度》……",
  "citations": [
    {"chunk_id": "doc_a1_0001_0012", "doc_title": "考勤管理制度.pdf", "page_num": 12, "validity": "valid", "doc_date": "2025-01-01", "image_ids": []}
  ],
  "confidence": 0.86,
  "degraded": false,
  "intent": "kb_qa",
  "latency_ms": 3200,
  "request_id": "req_xxxx"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| answer | string | 最终答案；含过期引用时**必须**内嵌"⚠ 已于 {时间} 失效"提示 |
| citations | array | 引用列表；`validity ∈ valid / expired`，expired 必有 `expired_at`；`doc_date` 为文档日期（年份错位自查，F2.9）；`image_ids` 非空时前端按 §2.8 回填原图（F1.10） |
| confidence | float | 0~1，Verify 节点输出 |
| degraded | bool | true = 走降级路径（资料不足 / 仅过期文档支撑 / 校验未达标的披露） |
| intent | string | `kb_qa / chitchat / contact_guidance`（后两类不检索、零 LLM） |
| latency_ms | int | 全链路耗时（v1.3 补入模型） |
| request_id | string | 追踪 ID（v1.3 补入模型） |
| notes | array | 可选过程说明（年份回退/降级/过期提示等，调试用，v1.3 补入模型） |

**时序约束**：客户端收到 `done`/`error` 后应主动关闭；服务端在 `done` 后立即关闭流。客户端断开（TCP 断开）→ 服务端取消生成、释放并发额度，不允许后台继续消耗 token。

### 2.2 POST /v1/documents — 文档入库

`multipart/form-data`；租户上下文以 `X-Tenant-Id` 头为准（兼容字段 `tenant_id`，同时存在时以 header 为准）。

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | binary | 是 | 支持 pdf / docx / md / txt；扩展名 + MIME 双重校验，否则 **415 DOC_UNSUPPORTED_TYPE** |
| meta | string(JSON) | 否 | `{"doc_key": "/docs/api.md", "department": "backend", "category": "技术文档"}`；`doc_key` 缺省 = 规范化文件名；`permission` 不接受客户端传入 |

**响应**

| 场景 | 状态码 | body |
|---|---|---|
| 新文档 / 内容变更（doc_key 不存在，或同 doc_key 不同 content_hash） | 202 | `{"task_id": "ingest_9f3a", "doc_id": "doc_xxx", "version": 1, "status": "pending", "duplicated": false}`（内容变更时 `doc_id` 不变、`version` 递增，走 F1.7 版本化软更新） |
| 幂等命中（同 doc_key 且同 content_hash 已完成） | 200 | 同上，`"duplicated": true`（不重复处理，F1.8） |
| 幂等冲突（同 doc_key 处理中） | 409 | 错误码 `INGEST_IN_PROGRESS` |

> 身份规则（Registry 主键 = `(tenant_id, doc_key)`）：`doc_id = "doc_" + sha256(tenant_id + ":" + doc_key)[:12]`，**与内容无关、跨版本稳定**——这是版本化软更新（F1.7）成立的前提，doc_id 禁止按 content_hash 派生。`doc_key` 默认 = 规范化文件名，可经 meta.doc_key 显式指定（如目录相对路径）；`content_hash` 仅作变更指纹与幂等检测。

### 2.3 GET /v1/tasks/{task_id} — 任务状态

```json
{
  "task_id": "ingest_9f3a",
  "doc_id": "doc_xxx",
  "status": "processing",
  "progress": {"phase": "embed", "percent": 62, "chunks_done": 31},
  "warnings": [{"type": "parse_warning", "page": 7, "detail": "乱码占比超阈值"}],
  "error": null,
  "created_at": 1751241600,
  "updated_at": 1751241700
}
```

| 字段 | 说明 |
|---|---|
| status | `pending / processing / done / failed`（唯一权威枚举，与 DocRegistry 状态一致，见 4.7） |
| progress.phase | `parse / chunk / embed / index` |
| error | failed 时给出 `{code, message}`，如 `DOC_ENCRYPTED` |
| warnings | 部分失败/质量告警不阻断任务，在此上报 |

任务不存在 → 404 `INGEST_NOT_FOUND`。

### 2.4 DELETE /v1/documents/{doc_id} — 软删除

- 语义：仅置该 doc 全部 chunk `is_valid=false`（F1.7），不物理删除
- 成功 / 重复删除（幂等）→ **204**；doc_id 不存在 → 404 `DOC_NOT_FOUND`
- 异步翻状态，返回 204 不代表立即对检索生效；对"删除后立刻查询"的场景允许短暂延迟（最终一致）

### 2.4b GET /v1/documents — 文档列表（2026-09-17 补）

```
?include_deleted=false&limit=50&offset=0   (limit 上限 200)
```

```json
{
  "total": 38,
  "limit": 50,
  "offset": 0,
  "items": [
    {"doc_id": "doc_9f3a1b2c3d4e", "doc_key": "policy_after_sales_return.pdf",
     "version": 3, "status": "done", "is_deleted": false, "effective_time": 0,
     "content_hash": "ab12…", "created_at": 1751241600, "updated_at": 1751241700,
     "error": null}
  ]
}
```

| 参数 | 说明 |
|---|---|
| include_deleted | 默认 `false` = 前端列表口径（不含已软删）；`true` = 审计/回收站口径 |
| total | **过滤后总数**（分页前的分母），前端据此渲染分页器 |
| status | `pending / processing / done / failed`（与 §2.3、4.7 同一枚举） |
| effective_time | 0 = 永久有效；`now > effective_time` → 已过期（F2.8） |

**为什么必须有这个接口**：此前 `/v1/documents` 只有 POST / DELETE——前端**刷新即丢**
（只看得到本次会话的任务态），而 DELETE 需要 `doc_id`，前端根本拿不到它。验收标准 12
「软删除后前端列表同步」因此无法成立。排序：`updated_at DESC, doc_key ASC`（最近操作在前、
同秒稳定）。

### 2.4c PATCH /v1/documents/{doc_id} — 设置有效期（2026-09-17 补）

```json
请求：{"effective_time": 1751241600}    // 0 = 永久有效
响应：{"doc_id": "doc_9f3a…", "version": 3, "effective_time": 1751241600,
       "chunks_updated": 3}
```

- 404 `DOC_NOT_FOUND`（doc_id 不存在）；`effective_time < 0` → 422
- 这是 `effective_time` 的**唯一写入口**：此前该字段只有读取方（检索 where 过滤、citation
  的 `validity`），**没有任何写入路径** → F2.8 过期链路（验收标准 8）代码在、验不了
- 一次写三处，避免"登记表说有期、索引里还是永久"的漂移：① Registry（列表权威值）、
  ② Chroma chunk 元数据（向量路过滤）、③ BM25 列 + `meta_json`（词法路过滤与证据渲染）
- 只翻**当前版本**的 chunk（历史版本已 `is_valid=false`，翻它们无意义）
- 见效时机：`now > effective_time` 后主检索不再召回该文档，仅在 F2.8 二级候选
  （`relaxed_retrieve`）中出现为 `validity=expired`，需用户确认后才可见

### 2.5 GET /v1/threads/{thread_id}/history — 对话历史

```
?limit=50  (默认 50，上限 200)
```

```json
{
  "thread_id": "tenant_a:user_001",
  "messages": [
    {"role": "user", "content": "年假可以跨年休吗？", "created_at": 1751241600},
    {"role": "assistant", "content": "可以……", "citations": [], "degraded": false, "created_at": 1751241650}
  ]
}
```

> 注意：对话上下文（Rewrite / Answer 用）只取最近 10 轮（F4.3），与 history 接口返回条数无关。

### 2.6 POST /v1/debug/retrieve — 检索调试验证（P1，仅供开发调试）

```json
{
  "query": "年假跨年规则",
  "include_expired": false,
  "department": null,
  "top_k": 8
}
```

响应返回两路明细与融合结果（调试用，允许浮点分数）：

```json
{
  "fused": [
    {"chunk_id": "doc_a1_0001_0012", "doc_title": "考勤管理制度.pdf", "fused_rank": 1, "fused_score": 0.0158,
     "vector_rank": 2, "vector_score": 0.81, "bm25_rank": 1, "bm25_score": 12.4, "validity": "valid"}
  ],
  "filters": {"is_valid": true, "tenant_id": "tenant_a", "expired_allowed": false}
}
```

### 2.7 GET /v1/health — 健康检查

```json
{
  "status": "ok",
  "version": "0.1.0",
  "checks": {"redis": "up", "vector_store": "up", "bm25": "up", "llm": "up"}
}
```

核心依赖（vector_store / bm25）不可用 → HTTP 503，`status: "degraded"`，`checks` 中对应项为 `"down"`。`llm` 恒为 `"up"`：无 Key 时的 stub 降级仍属"功能可用"，不参与 503 判定。（Redis checkpointer 不可达属设计内降级，见下方 v1.3 注。）

> v1.3 语义放宽（实现期实测）：Redis checkpointer 不可达时系统按设计降级内存继续服务（§4.7 降级哲学）——此时 `checks.redis="down"` 但 **HTTP 保持 200**，`status="degraded"`；仅核心依赖（vector_store / bm25）不可用才返回 503。本地无 Redis 的开发环境因此可正常调用 /health。

### 2.8 GET /api/v1/images/{image_id} — 图片资源回填（F1.10）

- 命中块文本含 `[IMAGE:{image_id}]` 占位符时，前端按此地址渲染原图
- 200：图片 binary（按实际内容返回 `Content-Type`）；404：`IMG_NOT_FOUND`
- 图片文件由入库管线落盘 + 登记表管理（幂等，永不推断删除）；本端点只读，不做任何写入

---

## 3. 错误码字典

code 统一 `PREFIX_REASON`。**新增错误码必须先在本表登记**，禁止使用本文档之外的 code。

| code | HTTP | 触发条件 | 调用方处理 |
|---|---|---|---|
| AUTH_MISSING_KEY | 401 | 无 Authorization | 携带密钥重试 |
| AUTH_INVALID_KEY | 401 | 密钥无效 | 检查密钥 |
| AUTH_FORBIDDEN | 403 | 密钥无权限访问目标租户 | 联系管理员 |
| THREAD_TENANT_MISMATCH | 422 | thread_id 前缀与 X-Tenant-Id 不一致 | 修正参数 |
| VALIDATION_INVALID_ARGUMENT | 422 | 字段校验失败（长度/类型/枚举） | 按 details 修正 |
| DOC_NOT_FOUND | 404 | doc_id 不存在 | 检查 doc_id |
| DOC_UNSUPPORTED_TYPE | 415 | 上传格式不支持 | 检查文件格式 |
| INGEST_IN_PROGRESS | 409 | 同 doc_key 任务处理中 | 稍后查询任务状态 |
| INGEST_NOT_FOUND | 404 | task_id 不存在 | 检查 task_id |
| DOC_ENCRYPTED | 409/任务failed | 解析阶段发现加密 PDF | 上传解密后的文件（任务级错误码） |
| INGEST_FAILED | 任务failed | 入库任务失败（解析/格式/无块等，message 带原因） | 查看任务 error.message 修正后重传 |
| IMG_NOT_FOUND | 404 | image_id 不存在（未入库或已被业务层清理） | 检查 image_id / 重新上传文档 |
| RETRIEVAL_UNAVAILABLE | 503 | 向量/BM25 双路均不可用 | 按 Retry-After 重试 |
| LLM_UNAVAILABLE | 503 | LLM 重试 3 次仍失败 | 稍后重试 |
| LLM_TIMEOUT | 504 | LLM 单次响应超时 | 稍后重试 / 减小问题长度 |
| RATE_LIMITED | 429 | 触发并发额度（单实例在途请求数 > 默认 50） | 按 Retry-After 退避 |
| INTERNAL_ERROR | 500 | 未分类异常 | 携带 request_id 反馈 |

---

## 4. 内部模块契约（Python 接口）

模块边界与字段定义以需求文档第 3.2 / 7 节为准。以下签名为**强制契约**，实现不得改变出入参语义。

### 4.1 Parser（可插拔）

```python
# 所有解析器实现统一接口（新增格式只加实现）
def parse(file_path: Path) -> list[Block]: ...

class Block(BaseModel):
    block_type: Literal["heading", "paragraph", "table", "image", "code"]
    text: str                       # table 为 Markdown；image 为占位描述
    page: int | None = None
    heading_level: int | None = None
    metadata: dict = {}             # 图片路径、表格行列数等（内部使用，不进入 chunk payload）
```

### 4.2 Chunker

```python
# 输入 Block 序列 + 文档级元数据，输出带偏移的 chunk 记录
def chunk_document(blocks: list[Block], doc_meta: DocMeta) -> list[ChunkRecord]: ...

class ChunkRecord(BaseModel):       # 与 7.1 Chunk 同构 + 正文
    chunk_id: str                   # 规则: f"{doc_id}_{version:04d}_{index:05d}"  ← 版本段为硬约束
    chunk_index: int
    content: str
    start_offset: int; end_offset: int
    # 其余字段透传 DocMeta（doc_id/doc_title/source/file_path/version/…）
```

### 4.3 VectorStore（Append-Only 约束）

```python
# 允许：新增、payload 元数据原位更新、带过滤查询、健康检查
# 禁止：物理删除、替换向量（replace_embedding）；离线重建专用 _purge 仅 compaction 脚本可调
class VectorStore(Protocol):
    async def add(self, records: list[ChunkRecord], embeddings: list[list[float]]) -> None: ...
    async def set_metadata(self, chunk_ids: list[str], fields: dict) -> None: ...   # 软删除翻 is_valid、update_time
    async def query(self, embedding: list[float], *, top_k: int,
                    where: dict) -> list[Hit]: ...                                  # where 见 4.5 过滤构造
    async def health(self) -> bool: ...
```

> Append-Only 澄清：契约禁止的是"向量内容级"的删改（产生墓碑）；payload **元数据原位更新**（软删除标记、时间戳）不触碰图结构，允许。

### 4.4 BM25Store

```python
class BM25Store(Protocol):
    async def add(self, records: list[ChunkRecord]) -> None: ...
    async def search(self, query: str, *, top_k: int) -> list[Hit]: ...   # 过滤在调用方与向量侧对齐
    def rebuild(self) -> None: ...        # 全量重建（含 compaction 后）；业务路径不调用
    async def health(self) -> bool: ...
```

### 4.5 HybridRetriever（两路并行 + RRF，过滤下推）

```python
async def retrieve(query: str, *, tenant_id: str, permission: str,
                   include_expired: bool = False,
                   department: str | None = None) -> RetrieveOutcome: ...

class RetrieveOutcome(BaseModel):
    valid_hits: list[ChunkRecord]     # 供 Answer 使用
    expired_hits: list[ChunkRecord]   # 仅提示/用户确认后使用（F2.8）
    lanes: dict                       # {"vector": ..., "bm25": ...} 调试/日志用
    query: str
```

过滤构造规则（向量侧 where 与 BM25 侧代码过滤**必须一致**）：

```
必选:  tenant_id = X-Tenant-Id
      is_valid = True
      permission ∈ 调用方可见集合（服务端注入）
现行:  effective_time == 0 或 now <= effective_time   （include_expired=true 时放宽此项）
可选:  department / category
      doc_year ∈ 显式年份集合（F2.9 两段式裁决：年份过滤检索结果须与不限年份语义 top1 比对 doc_id 一致后采用，否则回退语义结果 + note）
```

Chroma where 示例：`{"$and": [{"tenant_id": {"$eq": "tenant_a"}}, {"is_valid": {"$eq": true}}, {"$or": [{"effective_time": {"$eq": 0}}, {"effective_time": {"$gte": now}}]}]}`。约定：**永久有效文档写入时 `effective_time` 一律写 `0`，字段不可缺失**——多数向量库对"无该字段"的记录不匹配操作符过滤，字段缺失会被永久排除在召回外。

### 4.6 LLM Gateway（统一出口）

```python
class LLMGateway(Protocol):
    async def generate(self, messages: list[dict], *, temperature: float = 0,
                       max_tokens: int = 1024) -> str: ...
    async def generate_stream(self, messages: list[dict], **kw) -> AsyncIterator[str]: ...
    async def generate_json(self, system: str, user: str,
                            schema: type[BaseModel], *, temperature: float = 0) -> BaseModel: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

- 所有外部调用经此网关：连接池、超时（connect 5s / read 60s）、指数退避重试 ≤3、Semaphore 限流（默认 10）都在网关内实现，**业务代码不得直连上游**
- `generate_json` 内部将 schema 序列化后注入 prompt，**禁止手写 f-string 拼 JSON 模板**（防花括号转义事故）；结构化输出统一走 pydantic 校验

### 4.7 DocRegistry（幂等登记）

```python
# 状态（唯一权威）：pending / processing / done / failed；主键 (tenant_id, doc_key)
class DocRegistry(Protocol):
    async def reserve(self, tenant_id: str, doc_key: str, content_hash: str) -> DocRecord | None: ...
        # CAS：doc_key 不存在则注册为 processing（version=1）；存在返回现有记录，
        # 调用方据 status + content_hash 分流：同 hash done→幂等 / 异 hash→版本化更新 / processing→冲突
    async def get(self, doc_id: str) -> DocRecord | None: ...
    async def mark(self, doc_id: str, status: str, *, error: dict | None = None) -> None: ...
    async def chunk_exists(self, chunk_id: str) -> bool: ...   # 断点续跑
```

### 4.8 入库编排（异步任务）

- 阶段常量与进度回写：`PHASES = ["parse", "chunk", "embed", "index"]`，每阶段完成回写 `progress.percent`
- 断点续跑（F1.8）：重试前查 `chunk_exists`，跳过已完成 chunk 的 embed / index
- 双索引写入顺序：先向量后 BM25，单侧失败 → 任务 failed 可重试（幂等保证不重复）

### 4.9 Agent 编排（LangGraph）

节点清单（固定）：`ingest → (chitchat/contact→direct_reply | 其余→rewrite→retrieve ─┬→ no_data └→ answer(合并意图)→verify) → finalize` + retry 回环（意图分类已合并进 answer 节点）

> **职责边界**：本系统只做**检索与披露**，不发起任何升级 / 转交动作——没有转人工节点，
> 不建工单、不转接人工、不指定责任人。"该找谁"一律以文本形式告知，由客户自行联系。

```python
class QAState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    rewritten_query: str
    intent: Literal["kb_qa", "chitchat", "contact_guidance"]
    retrieved: RetrieveOutcome          # 4.5
    answer: str
    citations: list[Citation]
    confidence: float
    retry_count: int                    # max 2，超限 → disclose（披露局限，不转人工）
    degraded: bool

class AgentResult(BaseModel):           # = done 事件 data（2.1 AssistantReply）
    answer: str; citations: list[Citation]
    confidence: float; degraded: bool
    intent: str; latency_ms: int; request_id: str
    notes: list[str] | None = None      # 可选过程说明（年份回退/降级/过期提示等，v1.3 补入）
```

- 分支逻辑（重试计数、置信度阈值、空检索短路、仅过期命中提示）在**条件边**实现；意图分类已合并进 answer 节点（同一次 LLM 调用产出 intent + answer），verify 只产出 grounded/confidence 判定，LLM 不决定流程走向
- 空检索短路（成本关键）：`retrieved` 为空 → `no_data` 节点如实告知缺失，**0 次 LLM**（旧设计会白烧 answer+verify 两轮再转人工）；同时落一条 `event=kb_gap` 结构化日志（query/rewritten_query/thread_id），**仅作离线线索**供知识库管理员聚类缺失主题，不进任何人工作队列、不触发工单
- 过期引用约束（F3.9）：citation.validity=expired 时 answer 内必须内嵌失效提示，Verify confidence 下调一档，纯过期支撑 → degraded=true
- 零引用判据（2026-09-15）：`citations` 为空 → verify **不调 LLM** 直接判 `grounded=false / confidence=0`（无可回查出处即为未达标），落回 `retry（≤2）→ disclose` 出口；此时披露后缀用更重的一档 `_NO_CITE_SUFFIX`（"未能在现有知识库中找到对应出处，请勿直接作为依据"）。依据：四类出口里凡"给出答案"的（① 有资料→答案+出处；② 资料不全→现有数据+出处）都必须带出处
- 历史文本预算（2026-09-15）：注入 prompt 的历史按「每条 `HISTORY_PER_MSG_CHARS` + 总量 `HISTORY_TOTAL_CHARS`」两级截断，**从最新往旧累积**（最新一轮必然保留、优先丢最旧）；历史被注入两次（rewrite 一跳 + answer 一跳），verify 不注入

---

## 5. 枚举字典（唯一来源，禁止散落硬编码）

| 枚举 | 取值 | 用途 |
|---|---|---|
| intent | `kb_qa` / `chitchat` / `contact_guidance` | QAState / AssistantReply |
| source | `wiki` / `pdf` / `markdown` / `database` / `web` | chunk 元数据 |
| block_type | `heading` / `paragraph` / `table` / `image` / `code` / `figure_transcript` | Block（解析内部；`figure_transcript` 为 VLM 红页转录块，F1.9） |
| permission | `public` / `internal` / `secret` | chunk 元数据 / 过滤 |
| validity | `valid` / `expired` | citation |
| task_status | `pending` / `processing` / `done` / `failed` | 任务状态 |
| task_phase | `parse` / `chunk` / `embed` / `index` | progress.phase |
| role | `user` / `assistant` / `system` | history / messages |
| sse_event | `ready` / `token` / `citation` / `done` / `error` / `ping` | SSE 事件名 |
| category / department | 自由字符串 | 建议维护内部字典，接口不强制枚举 |

> 扩展规则：枚举**只允许追加**，禁止改名、删值或复用旧值语义（会破坏存量数据过滤与历史消息解析）。

---

## 6. 契约治理

### 6.1 唯一权威与代码同步

- `app/models/*` 的 pydantic 模型字段、默认值、枚举与本文档逐条对应；新增/修改模型必须**同步更新本文档并递增版本**
- M4 交付时生成 `openapi.yaml` 作为机器可读副本（覆盖路径、请求体、错误码结构）；**人工评审以本文档为准，响应体字段以本文档第 2 节为准**；契约测试据此校验实际响应（pytest + fastapi TestClient）
- 评审清单（Code Review 必查）：
  1. 新字段是否 snake_case + 可选字段带默认值
  2. 枚举是否走第 5 节追加规则
  3. 错误路径是否使用第 3 节已登记 code
  4. 时间戳是否 int / Unix 秒
  5. 幂等入口（documents 上传）是否保留 `doc_key` 身份 + `content_hash` 指纹双层判重
  6. 检索过滤条件是否由服务端注入（客户端不可传 permission）

### 6.2 变更流程与兼容性

| 变更类型 | 要求 |
|---|---|
| 加可选字段 / 追加枚举 / 放宽校验 | 允许，版本号 +0.1，同步本文档 |
| 改字段类型 / 删字段 / 改必填 / 改枚举语义 | **破坏性**，必须走 `/v2` 或需求评审通过后同步迁移 |
| 新增错误码 | 先登记第 3 节再使用 |
| 修改内部模块接口 | 同步 4.x 契约 + 调用方，禁止只改实现不改契约 |
