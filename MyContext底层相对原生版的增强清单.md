# MyContext 底层相对原生版的增强清单

**原生 MyContext 已经会抽事实、建实体、连图、存向量并检索。本项目最实质的底层增量，是让命中的事实能回到可核验的原句，让分散记录能沿有依据的关系一起被找到。** 写入侧的改动较小，集中在来源元数据、抽取约束和逐字引文。读取侧新增了三个查询模块，承担原文节选、参与者核验和跨来源查证。[图谱设计](kl-graph/docs/graph-design.md) · [事实证据](kl-graph/kl_graph/query/fact_evidence.py) · [来源与参与者](kl-graph/kl_graph/query/source_graph.py) · [共享实体查询](kl-graph/kl_graph/query/associated_sources.py)

本文将本地上游提交 `23c6f609fb1d392f922e19dae7bee174075a2569` 与当前工作树直接比较。当前工作树对应恢复后的 V5 版本，包含 Git 提交后的本地修改。底层范围包括 `kl_graph`、原生服务 `kl_server.py` 和配套配置。仅在 `project_demo` 中实现的流程单列在文末。[V5 评测源码清单](kl-graph/evaluation-artifacts/eval-v5-all20-2026-11-23/manifest.json)

## 一、写入时增强了什么

| 环节 | 原生版已有 | 当前实际增加 |
| --- | --- | --- |
| 群聊、文档、会议的来源定位 | 已有加载器、原始记录、片段及两者的映射。Wiki 已有记录 ID 和文档 URL，会议已有摘要及转写页。 | 加载器接受可选的 `source_record`。群聊可携带消息定位和来源元数据；Wiki 可采用来源的发生时间（原逻辑为 0），缺少文档 URL 时可用来源定位；会议新增 `minutes_summary_section`，使同一会议下分别编号的纪要段落保有各自 ID、时间和出处。[群聊](kl-graph/kl_graph/ingest/loaders/message_loader.py) · [文档](kl-graph/kl_graph/ingest/loaders/wiki_loader.py) · [会议](kl-graph/kl_graph/ingest/loaders/minutes_loader.py) |
| 事实抽取 | 已按目标内容提取原子主张、主体、客体、其他涉及实体、类型及置信度。已有回复上下文、职责类型和指代处理规则。 | 中英文提示都加强了三项要求：只有原文点名或当事人承诺才记执行人；区分交付物、接收方、执行者并保留绑定方向；输出保留条件和否定的逐字支持原句。结构中新增可选 `source_quote`，提示版本由 `source-aware-v6` 升为 `v7`，以区分抽取缓存。这些是抽取要求，无法保证每条事实都理解正确。[提示与输出字段](kl-graph/kl_graph/ingest/llm_extractor.py) · [提示版本](kl-graph/kl_graph/ingest/extraction_strategy.py) |
| 事实出处落库 | 已有 `Fact → STATES → Chunk`、事实来源 ID、参与实体与 `ABOUT` 边。 | 建边时检查模型给出的原句。只有当它在本次抽取目标正文中逐字且唯一匹配，才在**主来源的既有 `STATES` 边属性**上保存 `source_quote = {quote, start, end}`。匹配失败或有歧义时仍保留片段出处，不写精确位置。沿用了原表和边类型。[建边实现](kl-graph/kl_graph/ingest/pipeline.py) · [唯一匹配与主来源测试](kl-graph/tests/test_fact_source_evidence.py) |

例如，V5 的 `A04-S04` 中，一段原话支持两条不同事实：一条记录“还要比对模型选路和应用派发”，另一条记录“输入链路不完整时不指定唯一根因”。两条事实对应同一个 45 字区间 `[145, 190)`，均无 `ABOUT` 实体边。原生机制已能将一段内容拆成多条事实；这里能具体归给增强版的，是逐字保存并核验两条主张各自所依赖的原文位置。这也是**事实证据增强可以独立于实体关联产生用途**的实例。[原始材料](datasets/项目A_Demo应用增强数据_v5_自然表述版/normalized/application_focus.jsonl#L329) · [实库核查详解](V5事实抽取与建库详解.md)

## 二、查询时增强了什么

| 新增能力 | 实现细节及边界 |
| --- | --- |
| **从事实定位原句，决定能否安全节选** | 批量回查候选来源的事实。对长文档、纪要按完整句定位，结合已存引文、跨句指代和相邻限定，返回文本、字符区间及用于定位的事实 ID。任一合格事实定位失败，就放弃压缩该记录，交给调用方使用全文。短记录和群消息保留原文。旧库没有逐字引文时，仍可尝试从事实和正文定位。[节选与回退逻辑](kl-graph/kl_graph/query/fact_evidence.py) |
| **核对事实究竟属于哪条原文** | 用 `source_unit_id` 和 `chunk_units` 双向核对。旧事实缺少记录 ID 时，只有来源片段恰好对应一条记录才精确归属。多消息片段无法消除歧义时，单事实接口可保留片段级出处；构造问答候选时跳过无法归属到可见记录的事实。[来源核对](kl-graph/kl_graph/query/source_graph.py) · [片段级回退](kl-graph/kl_graph/query/fact_evidence.py) |
| **把参与关系和聊天指代落实到依据** | 对命中事实回读数据库已有的 `ABOUT` 边，包括有限查询视图裁掉的参与者，再用原句或可核验的发言身份检查端点。对“我补这两列”一类回复，沿显式回复关系补上文，保留各条消息的发言人和出处。缺失依赖时标记上下文不足，省略缺乏依据的参与关系。人员核验要求可见范围内的发言身份明确；这项改动没有新增全库同名消歧算法。[参与者回查](kl-graph/kl_graph/query/source_graph.py) · [回复与引文核验](kl-graph/kl_graph/query/fact_evidence.py) |
| **沿来源关系补材料并记录路径** | 可从命中记录找同文档或同线程材料、回复上文和附件，记录从哪条来源、经什么关系找到另一条。对项目归属不明的聊天，可核对其同线程回复链能否通向唯一且有明确项目登记的附件，返回依据。来源的可见范围和项目范围由调用方提供。这类关联主要依据记录结构，普通数据库同样可以实现。[来源扩展与附件归属](kl-graph/kl_graph/query/source_graph.py) |
| **沿共享实体查找另一条事实** | 从种子来源走“事实 A → 同一实体 ← 事实 B → 目标来源”。使用两条持久化 `ABOUT` 边，要求两端事实及各自原文都提到该实体，并复核参与关系。当前最多用 16 条种子来源，跳过人员及关联事实超过 64 条的宽泛实体，结合问句相关性和原文排名最多补 8 条目标来源。返回路径供审查，共享实体本身不证明因果、责任或审批关系。[共享实体查询实现](kl-graph/kl_graph/query/associated_sources.py) |

**哪些能力通过原生 API 自动可见？** 已有的 `POST /context` 查询单条事实时，现会多返回 `source_evidence`，可含原文、位置、回复依赖和支持状态。批量节选、附件归属和共享实体扩展则由 Demo 取得原生 `/ask` 候选后，显式调用上述 Python 查询模块。把它们写进原生查询库，意味着其他调用方可复用；单独调用原生 `/ask` 并不会自动执行这整套流程。[原生接口实现](kl-graph/kl_server.py) · [接口说明](kl-graph/docs/query-api.md#post-context) · [Demo 调用位置](kl-graph/project_demo/context.py)

## 三、向量检索和运行配套

- **向量检索**增加可选的 `KL_EMBED_QUERY_INSTRUCTION`。配置后，同步和异步单条查询会按模型所需格式拼接指令；用于建库的批量文本沿用原输入。原生版已有向量检索。Demo 选择 Qwen3-Embedding-0.6B 和 1,024 维属于部署选择；查询指令适配才是这一处底层代码差异。[向量实现](kl-graph/kl_graph/ingest/embedder.py) · [默认配置](kl-graph/config.default.yaml)
- **本地服务**允许显式传入 CLI 端口 0，让操作系统分配空闲端口，方便启动隔离的本地索引服务。这是运行便利性改动。[原生服务](kl-graph/kl_server.py)

## 四、怎样避免把原生能力或 Demo 效果算到底层增强里

原生版已经支持按来源构造抽取目标、聊天上下文、实体名称归一、确定性事实 ID、缓存和重试，以及 `Fact`、`Entity`、`Chunk`、`Scope`、`Community` 等节点。与上游基线对比，当前版本的图模型类型、存储模块和周期优化模块没有代码差异。当前 Demo 导入设置 `improve_mode=off`。此前实验增加的 `PROCEDURE` 方法事实和按计数题型补比较的逻辑均已回退。[图谱设计](kl-graph/docs/graph-design.md) · [Demo 导入](kl-graph/project_demo/runtime.py) · [当前实现基线](MyContext增强.md)

项目与截止时间筛选、B/C 的原文关键词与向量排名融合、模型分批初筛、全局及遗漏复核、证据组预算装入、最终作答和引用 ID 校验，归于 **Demo 应用层**。底层把可核验的原句、依赖和关联路径交给应用；哪些最终入选以及模型如何回答，还要经过上述应用流程。因此，V5 中 C 通过 17/20、B 通过 12/20 衡量的是端到端组合。更接近单项机制验证的是共享实体查询开关对照：X06 从 2/3 到 3/3，TR03 两组均为 2/3，每种条件各运行一次。[选证](kl-graph/project_demo/selection.py) · [融合与作答](kl-graph/project_demo/answer.py) · [V5 整体评测](kl-graph/evaluation-artifacts/eval-v5-all20-2026-11-23/report.md) · [关联查询开关对照](kl-graph/evaluation-artifacts/eval-v5-association-tr03-x06/report.md)

在仓库根目录可用以下只读命令复核差异。与工作树比较可包含恢复 17/20 版本时保留的本地修改。

```bash
git diff --name-status 23c6f609 -- kl-graph/kl_graph kl-graph/kl_server.py kl-graph/config.default.yaml
git diff 23c6f609 -- kl-graph/kl_graph/ingest/llm_extractor.py kl-graph/kl_graph/ingest/pipeline.py
git diff --stat 23c6f609 -- kl-graph/kl_graph/query kl-graph/kl_graph/models kl-graph/kl_graph/storage kl-graph/kl_graph/periodic
```
