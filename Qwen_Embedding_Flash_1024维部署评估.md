# Qwen Embedding Flash 与 1024 维部署评估

核实日期：2026-11-21。

## 结论

`qwen3.7-text-embedding-flash` 是官方已列出的云端文本向量模型，默认输出 1024 维。以 API 方式接入当前 Demo，硬件和存储层面可行；实际账号权限、网络连通和模型返回尚未验证。当前进程没有配置 `DASHSCOPE_API_KEY`，本次没有发送云端模型请求。[官方模型页](https://help.aliyun.com/zh/model-studio/qwen3-7-text-embedding-flash)

未找到该 Flash 型号的官方可下载权重，不能据云端可用推断可以本地部署。若目标是本地运行并使用 1024 维，建议优先试验开源 `Qwen/Qwen3-Embedding-0.6B`，它有官方权重、最高 1024 维、32K 输入窗口，采用 Apache-2.0 许可。[官方模型卡与下载入口](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

此前“未查到 Qwen3.7-Embedding”的说法检索不充分：混淆了已查到的开源 Qwen3-Embedding 系列与百炼新型号，应以本次核实为准。0.6B、4B、8B 开源模型与 Flash API 型号不能按名称直接对应。

## Flash 云端规格及适配

| 项目 | 核实结果 |
| --- | --- |
| API 模型 ID | `qwen3.7-text-embedding-flash` |
| 向量维度 | 256、512、768、1024；默认 1024 |
| 接口 | OpenAI 兼容 Embeddings，可用于当前稠密向量检索 |
| 请求条数 | 官方表列最多 20 条；Demo 可先保留每批 10 条、并发 2 |
| 模型输入 | 模型页列 131072；API 表列 128000，实施时取更保守的接口约束 |
| 北京公开输入单价 | 每百万 token 0.125 元，不含其他模型和基础设施成本 |

上述规格见[官方模型页](https://help.aliyun.com/zh/model-studio/qwen3-7-text-embedding-flash)和[OpenAI 兼容接口说明](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)。官方不同页面对批次 token 上限的表述存在差异，初期采用小批次，并以账号所用接口实际返回验证，不把 128K 乘以批次条数作为已验证容量。

通过云端生成向量不需要本机承载模型权重。当前环境约 7.4 GiB 内存、未检测到 nvidia-smi，不妨碍 API 接入。1024 维 float32 原始向量每条 4096 字节，较 512 维增加一倍；这只是向量本体，不含索引、元数据和服务进程内存。本 Demo 规模下向量维度本身不是部署瓶颈。

## 已完成的本地验证

检查了原生 OpenAI 兼容向量客户端、可配置维度和 Zvec 存储。实际以 1024 维测试向量完成 chunks、facts、entities 三个集合的创建、写入、检索、向量读取及关闭后重开，均通过。测试库位于独立临时目录，完成后清理，没有改动原业务索引。

[验证记录](kl-graph/evaluation-artifacts/qwen-flash-1024-assessment/storage_validation.json)。这是存储兼容性验证，不是 Flash 模型效果或云端连通性测试。

## 当前代码仍需修改的地方

`project_demo/model_setup.py` 的 `--chatgpt` 路径会加载本地 BGE，并覆盖 embedding 地址、模型名、密钥、512 维配置及是否发送 dimensions。仅在启动前设置云端环境变量会被覆盖，不能直接视为完成接入。

应让 ChatGPT 回答/抽取服务与 embedding 服务独立配置：保留 ChatGPT，embedding 可选择本地或外部接口。密钥分别使用，不能把 ChatGPT 桥接的本机令牌传给百炼。使用云端时不应再强制加载本地 BGE。

适配后的目标配置如下，目前不是可直接配合现有 `--chatgpt` 生效的启动命令：

```text
KL_EMBED_MODEL=qwen3.7-text-embedding-flash
KL_EMBEDDING_DIM=1024
KL_EMBED_SEND_DIMENSIONS=1
KL_EMBED_BATCH_SIZE=10
KL_EMBED_CONCURRENCY=2
KL_EMBED_BASE_URL=https://<业务空间ID>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
KL_EMBED_API_KEY=<通过环境安全注入的对应地域密钥>
```

官方支持 dimensions 参数，但部分旧示例的注释没有列出 Flash。上线前需验证显式请求 1024 维后，实际返回长度确为 1024；该型号默认也是 1024。[同步接口参数](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)

更换模型及维度后，原文、事实、实体和查询必须使用相同新模型。旧 512 维向量不能补零后继续使用。当前 Demo 按模型配置划分索引目录，可以隔离旧索引；V4 已使用选定的本地 0.6B/1024 维配置建立新版索引，见文末验证记录。

## 如果要求本地运行

本地方案已选择 Qwen3-Embedding-0.6B，1024 维，并完成下载与实际 CPU 推理验证，详见文末执行记录。当前使用逐条推理和每批两条请求；测试进程峰值内存约 3.8 GiB。模型支持 32K 不等于本机适合处理满窗口：已验证的 V4 最长片段为 938 token，尚未进行满 8192-token 输入的压力测试。

无论采用 Flash 云端还是 0.6B 本地，都应让 B/C 统一切换并重建向量索引。更长输入可以避开当前 512-token 上限，但换模型后的 token 数仍应按新模型核查。检索效果提升和 C 的优势需继续用 TR04、AM03、PO03、X01、X03 五题验证；本次未运行题目，也未运行 A。


## 后续执行：本地 0.6B 已下载并接入（2026-11-21）

用户选择本地方案后，已下载官方 Qwen3-Embedding-0.6B，并将 Demo 切换为 1024 维。CPU PyTorch 2.10.0 与 Transformers 4.57.6 已安装；权重版本、SHA256、实际 API 向量及资源测量见[验证记录](kl-graph/evaluation-artifacts/qwen06-migration/model_validation.json)。本机采用 8192-token 单条运行上限，遇到超限会拒绝请求，不截断。V4 的 161 个检索 chunk 最长 938 token，均符合上限。Flash 云端未接入。

V4 应用核查已完成建库及检索验证：349 条原文全部抽取成功，161 个原文片段向量、182 个实体向量、965 个事实向量与数据库数量一致，并保留旧索引。[完整部署验证](kl-graph/evaluation-artifacts/qwen06-migration/deployment_validation.json)。CPU 首次建库较慢，本次事实向量化约 15 分钟；模型可在本机运行，但该耗时不适合作为高并发部署能力证明。
