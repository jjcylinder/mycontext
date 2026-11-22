# 项目 A 核查数据 V5 自然表述版

本版沿用 V4 长文材料，调整 32 条群消息、15 个会议段落和 7 个文档段落中的称呼方式。晨报和周报讨论按消息时间、业务内容或上下文指代来源。六份联调会议按业务名称引入议题，并用会议语言记录验证结果和待办，分工说明按岗位称呼人员。运行日志、测试登记和发布记录继续提供技术标识，便于核对业务说法。

[逐条修改前后对照](audit/正文变更.jsonl) · [V4 长文版](../项目A_Demo应用增强数据_v4_长文版/README.md)

应用核查范围为 349 条。按大写连字符编码及人员编号统计，含此类标识的正文从 177 条降至 127 条，出现次数从 356 次降至 270 次。统计只读取正文。文件内的记录 ID、引用位置、回复、附件、时间和角色元数据沿用 V4。

三个范围均已用 Demo 加载并导出 DWS，逐条核对了导出正文和引用的上文。20 道评测题使用独立冻结的 V5 评分口径，保留 64 个必要要点。每项记录业务条件、等价表达或缺项边界及来源。修复后按同一口径验证了前两个梯队。默认五题 B 整卷通过 2/5、C 通过 5/5，第二梯队 B、C 各通过 4/5。合计 B 通过 6/10、C 通过 9/10。另十题尚待本版本验证。[第一梯队 AM03、TR04](../../kl-graph/evaluation-artifacts/eval-v5-project-scope-am03-tr04/report.md) · [第一梯队其余三题](../../kl-graph/evaluation-artifacts/eval-v5-project-scope-po03-x01-x03/report.md) · [第二梯队](../../kl-graph/evaluation-artifacts/eval-v5-project-scope-second-bc/report.md) · [修复前默认五题](../../kl-graph/evaluation-artifacts/eval-v5-natural-first-bc-2026-11-22/report.md) · [逐题评分口径](evaluation/应用问题及参考依据.md) · [评分版本与文件校验值](evaluation/rubric_lock.json)

## 应用核查库

2026-11-22 已从 V5 的 208 条群聊、21 个会议段落和 120 个文档段落独立建库。349 条原文全部抽取成功，形成 161 个原文片段、926 条事实、190 个实体和 2,663 条关系。每条事实均可定位到唯一的原文记录。原文、事实和实体分别有 161、926、190 条 1024 维向量，使用“模型构建登记”检索三个索引均返回了可回查的结果。[建库记录](../../kl-graph/data/project_demo/v5-natural-runtime/datasets/project-a-v5-natural-app_focus-4f98497fe04be748/graph-3c10b57b3e8ace49/ready.json) · [索引与溯源核验](../../kl-graph/data/project_demo/v5-natural-runtime/datasets/project-a-v5-natural-app_focus-4f98497fe04be748/graph-3c10b57b3e8ace49/import_validation.json)

本机 V5 Demo 使用端口 `8768`。需要重新启动时，在 `mycontext/kl-graph` 目录运行以下命令，即可加载同一份库。

```bash
KL_GRAPH_BACKEND=sqlite KL_VECTOR_BACKEND=zvec .venv/bin/python -m project_demo.app \
  --chatgpt --port 8768 --data-dir data/project_demo/v5-natural-runtime \
  --dataset '../../datasets/项目A_Demo应用增强数据_v5_自然表述版'
```

GitHub 仓库收录模拟材料，可加载 `app_focus` 和 `full_v3`。包含非模拟原周报的 `full_with_report` 仅在持有原始资料的本地数据集中可用。
