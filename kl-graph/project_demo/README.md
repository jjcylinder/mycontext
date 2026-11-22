# 项目 A 进展核查 Demo

> 当前 V5 正式 20 题，B 整题通过 12/20，C 通过 17/20。逐要点分别为 51/64 和 62/64。[本轮逐题结果](../evaluation-artifacts/eval-v5-all20-2026-11-23/report.md) · [此前实体关联查询的开关对照](../../企业上下文方案C的客观评价.md)

项目 A 的本地核查工作台基于 MyContext `23c6f609`，支持查阅材料、比较 B/C 核查答卷和追溯引用。仓库提供 V4 和 V5 的模拟材料。

## 启动

从 `mycontext/kl-graph` 目录执行。已有虚拟环境可直接运行最后一条命令。

```bash
python3 -m venv .venv
python3 -m pip --python .venv install -e '.[test]'
.venv/bin/python -m project_demo.app \
  --dataset '../datasets/项目A_Demo应用增强数据_v5_自然表述版'
```

在 Windows 浏览器打开 `http://127.0.0.1:8765`，导入「应用核查」配置。该配置包含 208 条群聊、21 个会议段落和 120 个文档段落。没有模型时仍可查看资料、时间、回复、附件引用、原文位置，以及由原文标识和语句建立的工作图谱。页面会明确显示原生图谱提炼和最终回答模型的可用状态。

默认数据和逐次运行记录保存在 `kl-graph/data/project_demo`，可以用 `--data-dir` 指定另一个目录。每个运行数据目录同时由一个服务进程使用。`--port` 可更改页面端口。

### V5 自然表述版

V5 在 V4 长文基础上调整了 54 条正文中的业务称呼。使用 `--dataset '../datasets/项目A_Demo应用增强数据_v5_自然表述版'` 加载。归档的 B/C 成绩对应 V4。[数据说明与逐条对照](../../datasets/项目A_Demo应用增强数据_v5_自然表述版/README.md)

### V4 独立扩写版

扩写版路径为 `../datasets/项目A_Demo应用增强数据_v4_长文版`。它保留 V3 的业务记录身份、时间、编号关联、群聊及评测材料，只扩写模拟文档和会议正文。具体构建条件、适用边界和 Qwen embedding 的对照建议见[构建说明](../../datasets/扩写数据集构建说明.md)。验收统计见该数据集的 `audit/validation.json`。

```bash
.venv/bin/python -m project_demo.app --chatgpt \
  --dataset '../datasets/项目A_Demo应用增强数据_v4_长文版' \
  --data-dir data/project_demo/v4-longform-runtime --port 8767
```

在 `http://127.0.0.1:8767` 导入「应用核查」。新版本的数据标识含 `v4-expanded`，正文哈希与旧版不同，必须从扩写正文重新建立原文向量及事实图谱。不能复制旧版 `ready.json` 或图谱冒充新版处理结果。`full_v3` 作为兼容参数名，在新版目录中表示新版全量模拟材料。扩写不会改变默认五题和 BC 两组的回归范围。

## 接入模型

### 使用已登录的 ChatGPT 和本机向量模型

已使用 `codex login status` 登录 ChatGPT 的本机可以选择 `--chatgpt`。事实抽取、查询改写和最终回答通过官方 `codex exec` 启动独立请求，三处使用同一模型。调用计入该账户的 Codex 用量。CLI 默认模型可以用 `--codex-model` 显式指定。[Codex 非交互模式](https://developers.openai.com/codex/noninteractive) · [ChatGPT 套餐与 Codex](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)

```bash
# 先安装 CPU PyTorch，避免拉取 CUDA 依赖；已安装可跳过。
python3 -m pip --python .venv install 'torch==2.10.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
python3 -m pip --python .venv install -e '.[demo-local-models]'
.venv/bin/python -m project_demo.app --chatgpt \
  --dataset '/mnt/c/Users/jcyli/Downloads/项目A_Demo应用增强数据_v3'
```

本机向量模型已切换为 **`Qwen/Qwen3-Embedding-0.6B`，1024 维**，使用 CPU PyTorch 和 Transformers。约 1.2 GB 官方权重位于 `data/project_demo/models/Qwen3-Embedding-0.6B/`，固定版本 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。服务只加载本地权重，不在启动时自动下载，也不执行远程模型代码。[官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

新环境可在启动前执行以下下载命令；使用其他 `--data-dir` 时需相应调整 `local_dir`。当前 V4 运行目录的 `models` 已链接到共享模型目录。

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-Embedding-0.6B',
    revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
    local_dir='data/project_demo/models/Qwen3-Embedding-0.6B',
    allow_patterns=['*.json', '*.safetensors', '*.txt', 'README.md', 'LICENSE'],
)
PY
```

`--chatgpt` 的消息预算使用本地 `o200k_base` 估算。新环境可先缓存分词表；当前机器已下载并校验，服务启动会检查可加载性。

```bash
TIKTOKEN_CACHE_DIR=data/project_demo/tokenizers .venv/bin/python - <<'PYTOKEN'
import tiktoken
tiktoken.get_encoding('o200k_base')
PYTOKEN
```

模型本身支持 32K 上下文；当前 CPU 服务采用 **8192 token 的单条运行上限**、两条 CPU 计算线程、逐条推理以限制峰值内存。原生建库默认每批 2 条、并发 1、请求超时 120 秒。超过单条上限返回 HTTP 413，明确要求分段，不静默截断。所有向量使用最后一个有效 token 的表示并进行 L2 归一化。原生同步和异步查询均添加相同英文检索指令，原文、事实及实体的入库向量不添加查询指令。服务返回真实分词量。

旧 `BAAI/bge-small-zh-v1.5` 的 512 维索引保留，新模型使用不同配置指纹的目录。需重新导入建立 1024 维索引，不能对旧向量补零复用。B/C 共用新 embedding 和原文向量索引。更换模型不改变 32768 的 Demo 上下文或 8192 的最终证据预算；后者与 embedding 单条上限是不同参数。

首次 CPU 建库可能需要较长时间；`--chatgpt` 模式默认允许导入任务等待 7200 秒，可通过 `DEMO_INGEST_TIMEOUT_SECONDS` 调整。这是整次建库的等待上限，与单次 embedding 请求超时不同。重新导入可利用已缓存的处理进度；如果中断导致原生抽取以部分成功结束，需修复相应检查点后补齐抽取，不能将部分图谱标为就绪。

本机 V4「应用核查」已完成 1024 维索引重建：349 条原始记录、161 个原文片段、182 个实体、965 条事实和 3013 条关系。原文、实体、事实的向量数量分别为 161、182、965，与数据库一致；349 条原文均可溯源，抽取失败数为 0。服务地址为 `http://127.0.0.1:8767`。三类真实向量检索和页面访问均已通过验证，详见 `data/project_demo/qwen06-migration/deployment_validation.json`。模型接入验证与业务评测分开记录；最新五题结果见下方“跑题与验收”。

向量计算在本机完成，交给 Codex 的抽取和回答输入会发往 ChatGPT 云端。默认仅允许对标记为模拟的资料使用该组合。首次处理会按批次发起模型调用，进度可在导入面板查看。A 可在原文导入后运行。B 和 C 复用同一次导入建立的原文向量索引，需要等待原生处理完成并确保向量服务可查询。

### 使用自有兼容接口

在启动终端设置以下环境变量，示例端口对应两个可分别部署的 OpenAI 兼容服务。换成实际端点、模型名称、模型窗口、嵌入维度和凭据。自带相同 OpenAI 兼容入口的服务也可共用 URL。远程网关的密钥应通过环境或私有配置注入。

```bash
export KL_LLM_FLASH_PROVIDER=openai
export KL_LLM_FLASH_BASE_URL=http://127.0.0.1:8000/v1
export KL_LLM_FLASH_MODEL=your-extraction-model
export KL_EMBED_BASE_URL=http://127.0.0.1:8001/v1
export KL_EMBED_MODEL=your-embedding-model
export KL_EMBEDDING_DIM=1024
export KL_GRAPH_BACKEND=sqlite
export KL_VECTOR_BACKEND=zvec

export DEMO_ANSWER_BASE_URL=http://127.0.0.1:8000/v1
export DEMO_ANSWER_MODEL=your-answer-model
export DEMO_CONTEXT_WINDOW_TOKENS=32768
export DEMO_RETRIEVAL_TOKENS=8192
export DEMO_OUTPUT_TOKENS=1800
```

- `OPENAI_API_KEY` 提供给 OpenAI 兼容抽取的 LiteLLM 客户端，`KL_EMBED_API_KEY` 提供给向量服务，`DEMO_ANSWER_API_KEY` 提供给最终回答接口。按服务要求配置。
- `DEMO_TOKENIZE_URL` 可指向所选模型的聊天分词接口，例如兼容 vLLM 请求与 `count` 响应的 `/tokenize`。配置后，A 超窗时会直接标记不适用。也可用 `DEMO_TIKTOKEN_ENCODING` 设置经确认的编码，估算计入消息格式余量。`--chatgpt` 默认使用本地 `o200k_base` 对消息做分词估算，可用 `DEMO_TIKTOKEN_ENCODING` 指定其他编码；它不包含 Codex 额外运行指令，不声称等于服务端总用量。自有接口未配置分词时仍使用 UTF-8 字节上界。A 提交完整原文并按模型接口的实际反馈判断。页面显示计量方式，已生成答案的实际用量以模型响应为准。
- 可用 `DEMO_USD_INPUT_PER_MILLION` 和 `DEMO_USD_OUTPUT_PER_MILLION` 设置最终回答单价。图谱首轮的耗时、处理单元、抽取成功和失败数单独记录。完整抽取 Token 成本仍需从模型网关计量获取。
- 重启服务加载配置，再次导入同一资料以启动图谱处理。图谱数据按资料哈希和处理配置隔离，导入阶段会核对原始单元数及抽取失败数。
- `full_with_report` 含 57 条资料包标记为非模拟的原周报摘录。默认仅向本机模型发送这类摘录，有外发授权时可在启动命令中显式加 `--allow-original-report`。

## 核查方式

| 方案 | 传给同一个最终回答模型的输入 |
| --- | --- |
| A 全量原文 | 同一时间范围内的完整记录及原文位置 |
| B 原文检索 | 关键词和语义向量共同召回的原文，补读回复及附件引用 |
| C 企业上下文 | 在同一原文检索基础上，补充从图谱找到的相关原文、实体、事实和关系 |

B 和 C 共用中文与编号关键词排名，并将 MyContext `/search` 返回的原文 chunk 向量命中通过 `chunk_units` 映射到材料时间范围内可见的原文记录，再用倒数排名融合（RRF）合并两个排名。原文向量索引或服务不可用时，B 和 C 会显示原因。

### 数据层提供的证据与 Demo 保留的工作

C 从原生 `/ask` 取回事实和片段候选。Demo 按项目与截止时间选出可见材料，再调用原生查询模块，从已命中的事实回查 `ABOUT` 参与边，核实端点与引文，沿原文回复、附件、同文档或同线程补查来源。问答侧用来源原句表示待核查的事实。原生索引就绪时侧栏也展示原生实体、事实和关系；基于编号和句式的业务关系仅用于缺索引时浏览。召回种子、来源扩展路径和参与边数可查 `coverage.graph`。[来源与参与关系](../kl_graph/query/source_graph.py)

数据层为候选长文和会议返回精确节选；遇到跨句指代时保留前件及相邻限定，任一合格事实定位失败则整条使用全文。聊天保留完整消息，对依赖上文的原话附带已解析的回复链。单条事实的出处可经原生 `POST /context` 的 `source_evidence` 回查；线上 Demo 通过 Python 查询库批量取得 `projections`、`audit` 和 `dependencies`。新抽取可向主 `STATES` 边写入逐字匹配的引文，现有 V4 索引可直接使用查询时定位。[原生证据实现](../kl_graph/query/fact_evidence.py) · [接口字段](../docs/query-api.md#post-context)

与旧版相比，Demo 将读取事实参与边、按来源定位引文及解析事实的回复依赖交给 `kl_graph/query`。`project_demo/fact_evidence.py` 只为旧评测脚本转出同一个节选函数。构造 C 问答输入时已移除 Demo 按编号和句式推断业务关系的步骤。DeepSeek 实验阶段按角色生成答题清单、按职责改写展示的逻辑已随回退撤下。当前由回答模型根据所选来源作答，Demo 检查结果结构和引用 ID。[证据组选择](selection.py) · [回答与引用检查](answer.py)

B、C 均将完整候选原文分批交给同一个模型重排。模型根据问题给出直接证据、必要条件和背景三个相关性等级，并标注业务对象、核查事项和最小配套原文组。原文候选超过重排输入窗口时分批处理，每批最多返回20组。随后把推荐原文集中复核，统一对象及事项命名、检查遗漏与配套条件；C 还将图谱事实和明确关系的来源原文重新纳入这一步，并提供有对应原文支持的图谱线索。若集中复核仍超过窗口，改用前一轮完整阅读后形成的证据组索引卡进行有界复核，必要时分窗；共同对象目录用于统一命名，配套原文组保持完整。索引卡只用于排序，不作为最终回答证据。单个配套组本身仍无法容纳时明确报错，不再跳过全局复核。`global_review_method` 和 `global_review_batches` 记录实际路径。

B、C 共用按需遗漏复核。初筛标记为直接依据或必要条件的组，只要有来源在全局复核后落选，就将该组和已选原文一起对照。模型判断已选原文能否支持同一结论及其条件，将判定有必要补入的依据交给预算装入。若单组与已选原文合计仍超过复核窗口，该组标记为未复核，交由原有预算装入流程处理。背景组和来源已全部保留的组跳过这一步。覆盖信息分别列出遗漏组数、需要复核的组数及实际调用批数。

Demo 用相同分词方法和预算装入证据组。C 将数据层返回的回复依赖与目标消息一起计入预算，见 `coverage.evidence_selection.packed_dependencies`。原始 `content` 保存全文，`model_excerpt`、`excerpt_spans` 和 `projection_fact_ids` 记录实际节选、字符区间和用于定位的事实。要解释 C 的增益，可分别看 `coverage.fact_projection` 中的原文节选与 `used_graph_facts`、`used_graph_relations` 中最终另附的事实和关系。页上引用校验检查来源 ID，业务含义和限定是否完整由逐项评审核实。[预算装入与引用校验](answer.py)

运行覆盖信息中的 `evidence_selection` 保存分批和全局重排组、选取或预算排除原因、未获重排推荐的候选 ID，以及各次重排的输入计量和模型用量。相同模型、输出预算和完整请求内容的重排结果在当前服务内复用，最多缓存128份，失败不缓存。`cache_hit` 标记复用；实际 `usage` 与 `cached_usage` 分开保存。先跑 B 再跑 C 会让 C 复用部分共同请求，因此实测耗时不能直接解释为 C 单独运行更快。回答用量字段只统计最终回答调用，额外重排调用的用量保存在该覆盖信息中。

首轮只读取 `normalized/application_focus.jsonl`。另外两种配置可独立选择，`full_v3` 读取 497 条，`full_with_report` 读取 554 条。转换复用原生 IM、Wiki 和 Minutes，保留记录 ID、时间和证据定位。文档可信时间和会议段落导入为可选输入能力，旧的导出格式仍可读取。

每次回答保存原始模型输入、输出、覆盖、引用和可获取的用量。`/demo/runs/{run_id}` 返回运行状态和答复，实际输入保存在数据目录的 `runs/{run_id}.json`。引用校验确保答案指向实际提供给生成模型的原文，语义是否充分由评审确认。

### 实体识别与消歧边界

抽取使用上下文及名称归一。问答图中的人员关联还要求有原文发言身份依据，离线浏览原文时可使用显式编号和别名。登记中的`U21 应用林工`与`U26 模型林工`可区分，但不能据此保证裸称“林工”或跨组织同名对象被可靠消歧。原生实体ID按归一名称生成，未包含项目命名空间。

Demo导入设置`improve_mode=off`，未运行原生“候选召回＋多信号评分＋LLM判断”的周期实体消歧。当前V4图182实体、3013边，`ENTITY_SIMILAR`为0；原生消歧如启用会生成带置信度的关联边，而非直接硬合并。故本轮C正确率提升不能归因于完整实体消歧。[实现及边界说明](../../MyContext增强.md#当前实体消歧的启用范围)

### 原生事实依据的离线验证（2026-11-22）

V4 历史库 965 条事实的引文位置均与保存的来源正文吻合。487 条定位到文中句组，356 条保留来源全文，111 条能沿回复找到上下文，11 条标记为上下文不足。这些是出处定位统计，事实含义与责任归属仍需按原文核实。

固定 ChatGPT 历史候选和排序组，离线回放默认五题及 X05、LE03 的 B/C 装入。沿用最终排序时，14 份输入的列明参考证据组覆盖与原答卷一致。TR04 的应用承诺可追到两条前序消息；X05 的连续否定句保留；X01 的 C 输入装入 49 条来源，晨报旧快照及三条分工材料均在。X05 的 C 输入因补读原文共装入 62 条，保留旧 57 条中的 52 条。默认五题的历史遗漏组从 113 组筛出 106 组待复核。此数衡量证据组，实际调用与评分见下节的线上记录。[逐题离线结果](../evaluation-artifacts/native-evidence-offline-2026-11-22/checks.json)

## 跑题与验收

2026-11-23 在撤回题型定制、修复 SU04 共用选证和回答输入之后，按冻结口径重新完成 20 题、40 份答卷。默认五题 B 1/5、C 4/5，第二梯队 B 3/5、C 4/5，其余十题 B 8/10、C 9/10。下列历史成绩保留各自的版本与题组。[本轮验收](../evaluation-artifacts/eval-v5-all20-2026-11-23/report.md) · [修复与回退范围](../../MyContext_C方案诊断与优化.md)

**V5 修复后，两梯队共 10 题。** 按 `V5-rubric-1` 逐项核对首次 B/C 答卷。默认五题 B 通过 2/5（14/19 个要点），C 通过 5/5（19/19）。第二梯队两组均通过 4/5，各得 14/15 个要点。C 在 SU04 漏掉更换候选模型需复验，B 在 X05 漏掉问题包提交后的可训练性判断。合计 B 通过 6/10、C 通过 9/10，其余十题尚待本版本验证。[默认五题中的 AM03、TR04](../evaluation-artifacts/eval-v5-project-scope-am03-tr04/report.md) · [默认其余三题](../evaluation-artifacts/eval-v5-project-scope-po03-x01-x03/report.md) · [第二梯队](../evaluation-artifacts/eval-v5-project-scope-second-bc/report.md)

**V5 自然表述版修复前默认五题（2026-11-22）**　依据冻结口径 `V5-rubric-1`，B 整卷通过 1/5，必要要点 14/19；C 整卷通过 3/5，必要要点 17/19。C 在 PO03、X01 多通过两题，X03 两组均通过，TR04、AM03 两组均为部分答对。每题先 B 后 C，各取一次端到端答卷。[逐题报告](../evaluation-artifacts/eval-v5-natural-first-bc-2026-11-22/report.md) · [评分明细](../evaluation-artifacts/eval-v5-natural-first-bc-2026-11-22/review.csv)

**20 题归档总成绩（2026-11-22）**　按每题必要要点全部答全计分，默认五题 B 1/5、C 4/5，第二梯队 B 5/5、C 5/5，其余十题 B 7/10、C 8/10。合计 **B 13/20（65%）、C 17/20（85%）**，逐项得分 B 57/64、C 61/64。默认五题的端到端记录早于 PO03 计数修复。若单独用修复后固定候选的 PO03 C 答卷更新该题，C 为 **18/20（90%）**，其中 1 题采用定向验证。[默认五题](../evaluation-artifacts/eval-native-evidence-first-bc-2026-11-22/report.md) · [第二梯队](../evaluation-artifacts/eval-native-evidence-second-bc-2026-11-22/report.md) · [其余十题](../evaluation-artifacts/eval-native-evidence-remaining-bc-2026-11-22/report.md) · [PO03 定向验证](../evaluation-artifacts/po03-count-comparison-2026-11-22/report.md)

日常回归默认只跑以下固定五道题，每题由 B、C 各回答一次，共十份答卷，不运行 A。代码单元测试的数量不计入这五道业务评测题。调用超时等失败需要保留原记录，仅重试失败的题目与方案。表中的观察来自旧版 B/C 评测，当时 B 仅使用关键词检索，C 优先使用图谱返回的原文。当前实现的表现以新运行的答案和证据为准。

| 题目 | 检查点 |
| --- | --- |
| `TR04` 翻译演示协调 | B 把公函签署和晨报时间窗写进翻译事项。C 避开了误归属，但仍遗漏翻译分支回归和测试样本要求。 |
| `AM03` 晨报快照 | 两边都漏掉旧产物把 10 点写成确定安排的原文。检查能否找齐旧快照、消息进入索引、原产物未重跑的证据。 |
| `PO03` 润色通过范围 | 检查风格和数字用例能否合计。B 的重试已拿到正式包未升级的原文，仍把正式状态留待核实。 |
| `X01` 六类 Skill 汇总 | 两边都没有覆盖六类 Skill 和相应分工。检查每一类的已完成、未决、演示范围和责任是否齐全。 |
| `X03` 预演结果能否用于正式环境 | C 找到了项目归属的对照记录，但没有把它放入回答输入。检查关键证据能否保留，并区分项目 B 发布、项目 A 正式和预演。 |

### 当前版默认五组实测（2026-11-22）

V4 长文版沿用已有的 ChatGPT 事实库，向量模型为本地 Qwen3-Embedding-0.6B（1024 维）。十份答卷一次完成。X01 的六份职责材料都进入了两边的候选和排序；B 最终装入 28 条来源，六份均落选，C 装入 54 条，其中 38 条使用事实定位的节选，六份分工均被引用。该题 C 最终另附的事实和关系均为 0。PO03 中 C 已拿到“两组检查口径不同”的原文，答卷仍缺样本交集未知这一限定。

B 平均每题 154.0 秒，C 为 122.7 秒。C 复用了五次相同的选证请求，耗时对应每题先 B 后 C 的顺序。本版只验证第一梯队，下文保留各历史版本及其适用题组。[完整答卷与评分](../evaluation-artifacts/eval-native-evidence-first-bc-2026-11-22/report.md) · [逐题输入审计](../evaluation-artifacts/eval-native-evidence-first-bc-2026-11-22/audit.json)

**PO03 定向修复（已回退）。** 当时 C 在装入原文后，从原生查询层取得有出处的计数比较，分别说明检查项和成员去重依据。固定原排序后，只有 PO03 C 的最终输入发生变化。相同 14 条来源、同一系统提示的当次线上对照覆盖 2/3 个要点，增强答卷为 3/3。其他九份 B/C 输入与归档逐字一致。此结果单列为定向验证。[修复及验证报告](../evaluation-artifacts/po03-count-comparison-2026-11-22/report.md)

### 第二梯队：补充五题

2026-11-22 在 PO03 计数修复后的版本上逐题先 B 后 C 运行。按题库的 15 个必要要点，B、C 均为 5/5。复核整份答卷时，B 为 5/5，C 为 4/5。X07 的 C 正确区分了共享补丁的影响范围和实测范围，却额外写成缺少润色在候选包上的回归证据。相应证据已经重排识别，在最终装入时因预算落选。五题均未触发计数比较。[第二梯队本轮报告](../evaluation-artifacts/eval-native-evidence-second-bc-2026-11-22/report.md)

同日按题库差集补跑余下十题，B 按必要要点答全 7/10，C 为 8/10，整份答卷复核分别为 5/10 和 8/10。X04 中 C 保留了隐式漏调需比对选路与派发的证据。TR03 的正式语言配置、X06 的消息索引可见性均被两组遗漏。与上段第二梯队合计，同版本 15 题的必要要点整题通过数为 B 12、C 13，整份答卷通过数为 B 10、C 12。[剩余十题逐题报告](../evaluation-artifacts/eval-native-evidence-remaining-bc-2026-11-22/report.md)

第一梯队默认仍为TR04、AM03、PO03、X01、X03。第二梯队固定为以下五题，使用`--second-tier`显式启用，不与第一梯队重复，也不自动扩跑全部20题。

| 题目 | 代表性与检查点 |
| --- | --- |
| SU04 | 新文件、否定条款能否沿用摘要固定六例准入；检查范围外推。 |
| LE03 | 模板三个样例通过能否关闭全部公函缺陷；区分根因与未闭环项。 |
| WR04 | 去重之后周报应如何回归；覆盖取数周期、来源、风险内容与配套验证。 |
| X05 | QA-A27覆盖范围及应用林工/模型林工分工；检查样本与人员归属。 |
| X07 | ROUTE-03/LANG-02跨Skill影响；区分可能影响、实际验证和未覆盖范围。 |

名单按剩余15道eval题的能力类型预先固定，由3道单Skill题和2道跨Skill题组成，不根据本轮答卷换题。原20题曾用于历史评测，因此这组是对冻结实现的补充回归，不是全新独立盲测。B/C共用当前数据、模型和预算，沿用必要要点完整的评分标准。

```bash
.venv/bin/python -m project_demo.evaluate \
  --dataset '../datasets/项目A_Demo应用增强数据_v4_长文版' \
  --url http://127.0.0.1:8767 --second-tier --modes BC \
  --output data/project_demo/eval-v4-second-tier-bc-2026-11-21
```

复跑时更换为尚不存在的输出目录；加`--dry-run`可仅检查名单。`--second-tier`不能与`--all-questions`或`--split dev`混用。原始名单、选择理由与实现指纹保存在该次运行目录的`selection.json`和`implementation.json`。本次已完成并逐项复核，结果见文末“第二梯队实测”；不以成功返回JSON代替正确率。

用 `--split dev` 显式选择 12 道开发题。配置冻结后，可用 `--all-questions` 执行完整的 20 道验收题。只有问题文本进入查询请求，参考事实在运行全部完成后读取。评测脚本写出逐题运行记录和待人工核查表。

```bash
.venv/bin/python -m project_demo.evaluate \
  --dataset '/mnt/c/Users/jcyli/Downloads/项目A_Demo应用增强数据_v3' \
  --dry-run

.venv/bin/python -m project_demo.evaluate \
  --dataset '/mnt/c/Users/jcyli/Downloads/项目A_Demo应用增强数据_v3' \
  --output data/project_demo/eval-app-focus-regression

.venv/bin/python -m pytest -q tests/project_demo tests/test_loaders.py tests/test_e2e_mini_ingest.py
```

优化目标是让 C 的图谱、事实提取和实体关联在公平对照中带来可验证的增益。B、C 共用语料、截止时间、回答模型、关键词与原文向量检索、重排算法及最终输入和输出预算；不能通过弱化 B、向 C 泄露参考答案或另给 C 更大的最终输入预算来制造优势。额外图谱查询、重排调用及其总用量、耗时也应单独报告。

评审逐题比较完整答对、关键事实覆盖、引用支撑性和错误归属，并报告代价。不能仅凭使用了图谱、原文条数增加或有效 JSON 就判定 C 更强。若本轮仍有混合胜负或没有优势，应如实记录，再定位 C 的证据链缺口继续改进。固定五题支持本轮回归结论；推广到其他问题需要后续独立验证。项目数据为模拟材料，评测结果适用于相应资料、截止时间、模型和预算配置。

上一轮（仅统一重排、尚未加入来源补查与全局复核）的固定五题结果：B、C 各 5 题均为部分答对，尚不能判定 C 整体更强。PO03 B 首次 HTTP 504，已保留失败并单独同配置重试。C 平均入选原文恢复到 14.8 条，完整参考证据组覆盖为 B 3/19、C 4/19。详细答卷和逐题复核见 `data/project_demo/eval-v3-selection-bc-regression-2026-11-21/`，重试见 `data/project_demo/eval-v3-selection-po03-retry-2026-11-21/`。

上一轮 V3/BGE 关联来源补查版的固定五题：B 完整答对 0/5、部分 5/5；C 完整答对 2/5（PO03、X03）、部分 3/5。X01 的部分范围与分工仍是 B 更完整，因此只能报告本轮五题完整回答率的优势。18 项针对性代码测试通过，未验证 A。

答卷、逐题判定与验证记录见 `data/project_demo/eval-v3-linked-bc-regression-2026-11-21/`。其中 `effective_runs.jsonl` 汇集十份有效答卷：X03 C 采用 `eval-v3-linked-x03-global-retry2-2026-11-21/` 的窗口修复结果，冻结原候选和分批重排后仅补做全局复核与回答；其余九份执行路径未变。所有有效答卷最终输入上限仍为 8192。


### 修复前基线：V4扩写数据 + Qwen0.6B/1024维（2026-11-21）

默认五题已完成BC共10份答卷，未运行A，无技术失败、无重试；评测期间未改代码与预算。按全部必要要点齐全的同一口径，**B完整答对0/5、部分5/5；C完整答对0/5、部分5/5**。TR04、AM03、PO03、X01、X03均为部分答对，不能将正确的核心否定结论等同于完整答对。

除PO03外，两组其余四题均因Demo估算输入超过全局重排窗口而跳过全局复核。C五题最终使用的提炼事实分别为0、0、2、0、0；TR04两组最终模型输入完全相同。PO03两组都遗漏“两个样例集合是否重叠未知”，即使C已装入全部参考证据组，也未完整作答。本轮没有证明C优于B。

C平均耗时较短，但复用了26次B的重排缓存，不能据此认定冷启动更省。V4语料与embedding同时改变，不能把相对旧结果的变化单独归因于模型。详见[完整评测报告](../evaluation-artifacts/eval-v4-qwen06-bc-regression-2026-11-21/report.md)，逐题判定、原始答卷、完整模型输入和覆盖审计均保存在同目录。


### 第一梯队：C根因修复后的默认五题（2026-11-21）

同一V4/Qwen0.6B索引，默认五题BC各一次，无A、失败或重试。按上一轮必要要点齐全的标准，**B完整3/5（60%）、部分2/5；C完整5/5（100%）**。逐题TR04为完整/完整，AM03为部分/完整，PO03为完整/完整，X01为部分/完整，X03为完整/完整。助手语义评审，非独立盲评；本结论只适用于当前固定五题。

修复了中文输入按字节过度估算、全局复核超窗跳过、短句成本优先，以及C事实只能在完整原文装入后追加的问题。共同计数和排序修复同样用于B；C在预算装入前利用事实定位带字符出处的整句原文节选，全文仍可回查、聊天不压缩。10份均完成全局复核。

C平均最终消息4163.2 token，B5956.2，少约30.1%。X01中C用6743 token保留51条来源，B用8075保留29条；C补齐了B遗漏的格式局部通过、旧快照重跑、去重局部通过与分工。C定位节选使用348次事实、另附29条事实，最终显式关系文本仍为0，不能把改善全部归因于图谱推理。

24项相关测试及7个子测试通过，另有2项涉及A的测试排除。108次节选的122个字符区间与原文一致，引用及运行期代码指纹校验通过。两组仍使用相同8192最终消息预算；32768为Demo上下文配置，1800为相同请求输出预算，Codex桥接实际输出可能超过该请求值，不能解释为硬截断。C命中6次重排缓存，实际耗时更低不证明冷启动成本更低。详见[本轮完整报告](../evaluation-artifacts/eval-v4-c-repair-bc-2026-11-21/report.md)，内含逐要点判定、实际输入、用量、根因和剩余边界。


本轮问题与修复方法已同步到[诊断与优化文档](../../MyContext_C方案诊断与优化.md)和[增强说明及最新证据处理流程图](../../MyContext增强.md)。此节保留 2026-11-21 该版成绩。


### 第二梯队实测与两梯队合并（2026-11-21）

冻结第一梯队的检索、回答及事实节选实现，仅给评测脚本增加`--second-tier`。SU04、LE03、WR04、X05、X07各运行BC一次，10份均完成，无A、失败或重试。**第二梯队B完整5/5，C完整3/5、部分2/5；C本轮没有保持正确率优势。** SU04、WR04、X07两组均完整，LE03、X05为B完整/C部分。与第一梯队合并，不重复10题的完整回答率均为8/10。助手语义评审，非独立盲测。

两处遗漏已有可复查根因：LE03的C召回了正式模板未升级的ALE2-S02，但全局重排丢弃，最终仅保留项目归属不明的短聊天；输入仅2006/8192，非容量问题。X05的原生事实已正确抽出SEM-S09/POL-F08未入QA-A27，但节选匹配要求所有编号同句，漏掉含“该包”指代的后句，把已知未入包降成未知。这两处新问题未在本轮运行中修复或以重跑替换。

C第二梯队平均最终消息4047.4 token，B4943，少18.1%，但准确性较低；引用来源更多不能替代事实完整性。C复用6次缓存，耗时差不证明冷启动优势。原文位置校验全部通过，仍存在X05语义遗漏，因此后续应验证关键限定覆盖、跨句指代保留和不可靠节选回退。[完整报告](../evaluation-artifacts/eval-v4-second-tier-bc-2026-11-21/report.md)含逐题判定、实际输入、根因复现和两梯队合并统计。

### 最终保真修复v2：两梯队验证（2026-11-21）

**第一梯队B完整4/5、C完整5/5；第二梯队B完整5/5、C完整5/5；合并10题B完整9/10、C完整10/10。** 第一梯队X01为B部分/C完整，其余九题均完整。第二梯队LE03、X05的原有遗漏已修复。语料、既有索引、模型和B/C预算不变，各题各模式首次成功答卷，无A、失败或重试；最终版本24项针对性测试通过。原默认五题与`--second-tier`名单保持不变。

修复包括：C连续指代句定位和未匹配业务事实全文回退；B/C共同的项目范围提示及必要遗漏复核；预算装入取消事项标签数量加分。C平均最终消息5294.9 token，B5698.1，少7.1%。X01同预算C保留49条来源、B28条。首版修复B/C均4/5的回归失败同样保留归档。

本轮同题B/C并行，调用及缓存逐项记录。全链路成本包含选证和新增遗漏复核，不能只看最终消息量；实际桥接输出不保证1800硬截断。未独立证明关系推理增益，非未见题盲测。[总报告](../evaluation-artifacts/c-fidelity-repair-2026-11-21/report.md) · [第一梯队](../evaluation-artifacts/eval-v4-fidelity-v2-first-bc-2026-11-21/report.md) · [第二梯队](../evaluation-artifacts/eval-v4-fidelity-v2-second-bc-2026-11-21/report.md)。
