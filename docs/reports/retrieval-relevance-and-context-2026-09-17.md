# 检索相关性判定 + 邻近上下文 + 文档管理闭环（①②③，2026-09-17）

> 本轮从「测试/修复」转向**功能补齐**：做了三件此前只有读取方、没有机制的能力。
> 第一件最有价值，也最反直觉——**结论是一个负结果**：单靠向量相似度**不可能**判断
> "库里到底有没有答案"，必须换信号。

## 一、为什么做这三件

| # | 缺口 | 代码证据 | 后果 |
|---|---|---|---|
| ① | 检索没有「无相关内容」判定 | `hybrid.py` 注释自认「语义检索恒返回 topN（**无距离阈值**）」 | `no_data`（0 LLM 如实告知缺失）**只在索引真空时触发**；"库里有 A 主题、用户问 B 主题"一律走 answer→verify→retry→disclose，**本该 0 成本的一句话烧了 2~3 轮 LLM**，且验收标准 3 靠模型自觉而非机制 |
| ② | 邻近 chunk 上下文未消费 | 需求 7.1 明写「前后 chunk 由 chunk_id 前缀 + chunk_index 相邻确定」；实际 `chunk_index` 只在切分时写入，检索/生成侧**零消费** | 段落被切在 chunk 边界时命中半段 → 答案"看着对但缺一条" |
| ③ | 文档列表缺失、`effective_time` 无写入口 | `/v1/documents` 只有 POST/DELETE；`DocumentMeta` 无 `effective_time`，全项目无 PATCH | 前端**刷新即丢**、拿不到 DELETE 所需 doc_id → 验收标准 12 无法成立；F2.8 过期链路**代码在、验不了**（验收标准 8） |

## 二、① 相关性判定（核心，含负结果）

### 2.1 标定实验：先证伪，再设计

在 `golden 55 条（语料覆盖）` vs `12 条（语料明确不覆盖）` 上实测两个候选信号的分布
（`scripts/calibrate_relevance.py` / `calibrate_coverage.py`）：

| 信号 | 组 A（应命中） | 组 B（应无资料） | 结论 |
|---|---|---|---|
| 向量 top1 余弦 | min **0.406** / med 0.639 | max **0.542** / med 0.422 | ❌ **大面积重叠，不可分** |
| 词表覆盖度（IDF 加权） | min 0.110 / med 0.725 | p75 0.505 | ❌ 口语改写会掉覆盖 |
| **复合**（覆盖 < C **且** 余弦 < T） | C=0.20 / T=0.40 → **55/55 保留** | 4/12 判出 | ✅ 零误杀 |

**这是本轮最重要的判断**：如果只按"相似度 < 0.5 就算无资料"，会**误杀 18% 的真实
可答题**（0.50 阈值下 A 组存活 0.818）——把库里有答案的问题回答成"没资料"，比多烧两轮
LLM **严重得多**。因此最终设计只做**高精度短路**：两个条件**必须同时**成立。

### 2.2 实现

- 新模块 `app/retrieval/relevance.py`：`CorpusVocabulary`（语料词表 + IDF）→
  `coverage = Σ idf(命中的内容词) / Σ idf(全部内容词)`；`RelevanceGate.assess()` 双证据判定。
- **字符级兜底**（实测踩坑后补）：jieba 分词是**上下文相关**的——同一串字在不同语境
  切法不同（语料里「员工年假制度…可跨年休」切成 `年/假/跨/年/休`，查询「年假可以跨年休吗」
  切成 `年/假/跨年/休`）。纯词级精确匹配会把**真实存在的主题**判成缺席。故 df 查不到时
  再看该词的**字符二元组是否都在语料中**（近似子串判定）。方向刻意偏保守：判定"存在"
  更容易 → 覆盖度偏高 → 更少触发"无资料"（**漏判可忍，误杀不可忍**）。
  效果：q027 覆盖度由 **0.110 → 0.680**，误杀风险随之消失。
- **mock 护栏**：`provider=mock` 或 `degraded` 时**不判**——mock 是哈希向量，余弦无语义，
  若照常判定会因"相似度恒低"把可答题判成无资料（合约测试/无 Key 环境全走这条路）。
- 落点：`HybridResult.relevance`（含 `coverage`/`top_vector_score`/`absent_terms`/`reason`）
  → `retrieve` 节点 → `AgentState.no_relevant` → `graph._route_after_retrieve` → `no_data`（0 LLM）。
- `log_kb_gap` 新增 `reason` 字段，区分 `empty_index`（索引真空）与 `relevance_gate`
  （"库里有 A 主题、用户问 B"）——缺口聚类时这两种是**完全不同的补库动作**。

### 2.3 实测

- **golden 55 条零误杀**（真实 embedding，全部 `no_relevant=False`）；
- 语料不覆盖 12 条中判出 **4 条**：量子纠错码 / 世界杯决赛 / 宠物狗握手 / 空气炸锅烤鸡；
- 真实 Agent 链路（真 embedding + 真 **LLM**）：「怎么训练宠物狗学会握手？」→ 返回
  无资料话术且 **LLM 调用 0 次**（此前该问题要烧 answer + verify + retry）。

**剩余 8 条判不出的是「近似主题」**（员工宿舍/班车/会议室/体检/办公用品/内部推荐……）——
它们与语料共享"申领/流程/标准/奖励"等词，词面与语义都够不上"完全不在库里"。这类只能
靠 LLM 路径（现状即最优）；**它们的正确解法是补文档，不是改阈值**。

## 三、② 邻近 chunk 上下文扩展

- `HybridResult.context_items` **与 `items` 分开放**：评估指标（recall@5 / MRR）读 `items`，
  邻居掺进去会让指标虚高、与历史基线不可比。
- `HybridRetriever._expand_context()`：取前 `CONTEXT_EXPAND_TOP`(3) 条命中的
  `chunk_index ±1`（同 doc + 同 version），去重后受 `CONTEXT_EXPAND_MAX`(4) 约束；
  `BM25Store.get_by_chunk_ids()` 强制套用**与检索路同一套隔离**（tenant + permission 白名单）
  ——邻居同样不能越权泄漏；**跳过过期邻块**（避免把过期内容当上下文引进来）。
- 节点合并 `retrieved = items + context_items`。**成本护栏**：邻居会进 answer 的 evidence
  （prompt 大头），4 条 ≈ +500 tok/问；verify 只按引用收窄证据，额外开销不落在 verify。
- `chunk_id` 反解用 `parse_chunk_id()`（`doc_id` 自身不含下划线，故右切两段即可，无需额外索引）。

## 四、③ 文档列表 + 有效期写入口

- `GET /v1/documents`（契约新增 §2.4b）：`include_deleted` / `limit`(上限 200) / `offset`，
  返回 `total`（过滤后总数，供前端分页器）；`DocRegistry.list_docs()` 按
  `updated_at DESC, doc_key ASC` 排序（最近操作在前 + 同秒稳定）。
- `PATCH /v1/documents/{doc_id}`（契约新增 §2.4c）：`effective_time` 的**唯一写入口**，
  一次写三处防漂移——① Registry（列表权威值）② Chroma chunk 元数据（向量路过滤）
  ③ BM25 列 + `meta_json`（词法路过滤与证据渲染）；只翻**当前版本**的 chunk。
- `DocumentMeta` **刻意不加 `effective_time`**：上传是异步任务，元数据要穿透任务队列写进
  每个 chunk；有效期改用 PATCH 单一入口（理由写在模型 docstring 里，避免又一处半死配置）。

## 五、验证

| 项 | 结果 |
|---|---|
| `verify_relevance_gate.py`（新） | **30/30**：词表语义 / Gate 四象限（**单低不判**）/ 上下文扩展 6 项 / **图级 0 LLM 短路** / 真实标定零误杀 / **真实 Agent 链路 0 LLM** |
| `verify_documents_api.py`（新） | **33/33**：空列表 → 上传可见 → 分页 → PATCH 三处同步 → **过期链路（主检索不再召回、放宽后成 expired 候选）** → 404/422 → 删除后列表同步 |
| 既有回归 | m1 45/0 · m3 60/0 · m4 35/0 · m6 20/0 · sse 24/0 · cli_ingest 14/0 · token 通过 · golden_anchors 55/55 · pdf_structure PASS · compileall OK |
| 真实单轮 eval | recall@5 **0.855** / MRR **0.791**（verdict=pass），见下节口径说明 |

**回归过程中修的一处夹具陈旧**：`verify_m3/_EmptyResult`、`verify_sse_streaming/_RetrievalResult`、
`verify_token_accounting/_RetrievalResult` 三个测试替身缺 `no_relevant`/`context_items`
→ AttributeError。节点按属性直接访问（与 `.items` 一致，保持显式），因此**同步替身**而非
在节点里加 `getattr` 兜底——接口长了，替身就该跟上。

## 六、评测口径发现（重要，影响以后怎么读数字）

改动后首次 eval 得 recall@5 **0.855**（此前基线 0.873），差 **1 题**。逐层归因：

1. A/B 探针（同进程开关新特性）发现 4 题 items 不同 → 疑似改坏命中口径；
2. **同臂连跑两次**却也有 1 题翻转（q034）→ 排查方向转向评测本身；
3. 直接测量：**DashScope embedding 对同一输入两次返回余弦 0.99927、最大分量差 4.4e-3**
   —— **embedding 非确定**；
4. q027 的期望块向量排名第 6（0.3473），与第 5（0.3501）**只差 0.0028** —— 正好卡在
   top-5 边界。两次 eval 连跑都得 0.855（抖动落点稳定），但**换个时刻就会是 0.873**。

**结论**：`recall@5` 在本题集上有 **±1 题 ≈ ±0.018 的噪声地板**（来自供应商 embedding
非确定性 + 边界平局）。**今后判断检索退化必须连跑 ≥3 次看分布，不能拿单次 0.018 的差异
当回归**。新特性不触碰命中口径（`items` 只由融合结果决定，`context_items` 是独立字段）。

## 七、残留

1. **① 识别率上限 = 1/3**（"话题完全不在库里"那一类）。另 2/3 是近似主题，需要 LLM 路径；
   这是**能力边界而非缺陷**，已在代码注释与本节写明。
2. 阈值 `RETRIEVAL_MIN_COVERAGE=0.20` / `RETRIEVAL_MIN_SIMILARITY=0.40` 是**小语料标定值**，
   语料规模或 embedding 模型更换后需重跑 `calibrate_relevance.py` 复标（脚本已留）。
3. 邻近上下文是"±1 邻块"级，非像素级版面还原；跨页/跨章节的语义续接（如"见下页附表"）未覆盖。
4. `category` / `department` 仍是半死配置：`DocumentMeta` 能传但没有真正的写入链路
   （`doc_meta_extra` 在 `pipeline` 里有参数、无调用方）。要么打通上传链路，要么明确弃用。
5. 冷启动首问 ~44s（Chroma HNSW 载入 + 连接池预热），**改动前即存在**（首次标定实验已复现
   44.9s），非本轮引入；属 NFR 待优化项。
