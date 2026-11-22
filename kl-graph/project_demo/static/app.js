"use strict";

const projectId = "项目A";
const pageSize = 30;
const modes = {
  A: { name: "全量原文" },
  B: {
    name: "原文检索",
    description: "结合关键词和语义向量查找原文，补读回复和附件关联。",
    nativeRequirement: "当前服务未配置原文向量索引。",
  },
  C: {
    name: "企业上下文",
    description: "在同一原文检索基础上，补充图谱中的实体、事实和关系。",
    nativeRequirement: "当前服务未配置原文向量索引和工作图谱。",
  },
};
const comparisonModes = ["B", "C"];
const categories = {
  progress: "关键进展",
  changes: "重要变更",
  open_issues: "未决问题",
  coordination: "待协调事项",
};
const sourceTypes = { chat: "群聊", message: "群聊", meeting: "会议记录", document: "工作资料" };
const jobLabels = {
  queued: "等待处理",
  running: "处理中",
  completed: "已完成",
  failed: "失败",
  unavailable: "不适用",
};
const stageLabels = {
  queued: "等待处理",
  parsing: "解析资料",
  loading: "读取资料",
  raw_ready: "原文资料就绪",
  indexing: "建立原文索引",
  extracting: "加工工作图谱",
  phase_a: "建立原文索引",
  phase_b: "提炼事实与实体",
  finalize: "核对索引和原文",
  preparing: "组织核查依据",
  retrieving: "检索原文",
  retrieving_graph: "检索图谱关联",
  preparing_selection: "准备证据候选",
  candidate_ranking: "分批筛选证据",
  global_ranking: "全局复核证据",
  coverage_audit: "检查遗漏证据",
  packing_evidence: "组织最终回答依据",
  answering: "生成答案",
  checking_citations: "校验引用",
  validating: "校验引用",
  completed: "已完成",
};
const graphLabels = { ready: "已就绪", completed: "已就绪", pending: "等待处理", not_started: "尚未加工", not_built: "尚未加工", not_indexed: "尚未加工", unavailable: "当前不可用", failed: "加工失败", running: "加工中", indexing: "加工中" };
const coverageLabels = {
  record_count: "输入记录数",
  available_records: "可见记录数",
  total_records: "可见记录数",
  selected_records: "选取记录数",
  evidence_count: "证据数",
  truncated: "发生截断",
  cutoff: "材料截点",
  notes: "范围说明",
  limitation: "覆盖限制",
  visible_records: "可见记录数",
  used_records: "实际输入记录数",
  all_records_included: "包含全部可见原文",
  used_graph_facts: "实际使用事实数",
  used_graph_relations: "实际使用关联数",
};
const origins = { metadata: "输入元数据", mycontext: "自动提取", source_reference: "原文引用关系" };
const contextTypes = {
  project: "项目", person: "人员", skill: "Skill", task: "任务", demo_task: "演示任务",
  application_config: "应用配置", environment: "环境", model_version: "模型版本", issue: "问题",
  case_set: "用例集", component: "组件", artifact: "产物", source_record: "来源记录",
  run: "运行记录", snapshot: "快照", co_mentioned: "共同出现", deployment_binding: "部署关系",
  model_reference: "模型调用关联", skill_task: "Skill 任务", verification_reference: "验证关联",
  coordination_reference: "协作关联", project_affiliation: "项目归属", about: "涉及",
};
const scopeLabels = { requested: "项目A", other: "其他项目", comparison: "跨项目对照", unknown: "归属待核实" };
const certaintyLabels = { native_participant: "原生事实涉及的实体", inferred: "依据同处一段原文", clause_reference: "原文明确关联" };

const nodes = Object.fromEntries([
  "profile-select", "profile-status", "import-button", "source-count",
  "source-list", "more-sources", "check-form", "check-button", "question", "mode-select",
  "check-result", "context-search", "context-status", "context-list", "context-detail", "context-graph",
  "more-context", "compare-form", "compare-question", "compare-button",
  "compare-results", "evidence-dialog", "evidence-content", "back-evidence", "close-evidence",
].map((id) => [id, document.getElementById(id)]));
const state = {
  profiles: [], profileId: "", answerModel: null, graphConfigured: false,
  importJob: null, importingProfile: null, checkPending: false, comparePending: false,
  activeTab: "check", checkKey: null, jobs: new Map(), activeChecks: new Map(), sourceOffset: 0, sources: [],
  contextView: "graph", contextOffset: 0, contextItems: [], contextSelection: null, contextGraphFocus: null,
  contextGraphs: new Map(),
  sourceRequest: 0, contextRequest: 0, drawerRequest: 0, returnToSource: null,
};

function element(tag, className, label) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (label !== undefined) node.textContent = label;
  return node;
}

function button(label, className, onClick) {
  const node = element("button", className, label);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

function asArray(value) { return Array.isArray(value) ? value : []; }
function currentProfile() { return state.profiles.find((profile) => profile.id === state.profileId); }
function currentDataset() { return currentProfile()?.dataset_id; }
function labelForSource(value) { return sourceTypes[value] || value || "原文"; }

function formatValue(value) {
  if (value == null || value === "") return "未获取";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) return value.map(formatValue).join("、") || "无";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function formatDate(value) {
  if (!value) return "";
  return String(value).replace("T", " ").replace(/:00(?=[+-]\d\d:\d\d$)/, "");
}

function errorText(error) {
  if (!error) return "";
  if (typeof error === "string") return error;
  if (Array.isArray(error)) return error.map(errorText).join("；");
  return error.message || error.msg || error.detail || JSON.stringify(error);
}

async function request(path, body) {
  const options = body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorText(payload?.detail || payload?.error) || `服务请求失败（HTTP ${response.status}）`);
  if (!payload) throw new Error("服务未返回数据。");
  return payload;
}

function contextPath(datasetId, view, offset, q = "", limit = pageSize, groupRelations = false) {
  const params = new URLSearchParams({ dataset_id: datasetId, view, offset, limit, q });
  if (groupRelations) params.set("group_relations", "true");
  return `/demo/context?${params}`;
}

function jobKey(datasetId, question, mode) { return JSON.stringify([datasetId, question, mode, state.answerModel]); }
function loadingJob() { return { status: "queued" }; }

function jobMessage(job) {
  const stage = stageLabels[job.stage] || job.stage;
  if (job.error) {
    const failedStage = stageLabels[job.failed_stage];
    return failedStage ? `${failedStage}失败 · ${errorText(job.error)}` : errorText(job.error);
  }
  const parts = [stage || jobLabels[job.status] || job.status];
  if (job.status === "running" && job.kind === "check") {
    if (job.stage_progress) {
      const { batch, total } = job.stage_progress;
      parts.push(`第 ${batch}/${total} 批`);
    }
    const elapsed = Math.floor((Date.now() - Date.parse(job.created_at)) / 1000);
    if (elapsed >= 0) {
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      parts.push(`已用时 ${minutes ? `${minutes} 分 ` : ""}${seconds} 秒`);
    }
  }
  return parts.join(" · ");
}

function statusMessage(text, status = "") { return element("p", `status-message ${status}`, text); }
function badge(status, label = jobLabels[status]) { return element("span", `tag ${status}`, label || status); }

function metaList(rows) {
  const list = element("dl", "meta-list");
  for (const [label, value] of rows) list.append(element("dt", null, label), element("dd", null, formatValue(value)));
  return list;
}

async function followJob(path, body, onUpdate) {
  onUpdate(loadingJob());
  try {
    let job = await request(path, body);
    onUpdate(job);
    while (job.status === "queued" || job.status === "running") {
      if (!job.run_id) throw new Error("服务未提供运行编号。");
      await new Promise((resolve) => setTimeout(resolve, 850));
      job = await request(`/demo/runs/${encodeURIComponent(job.run_id)}`);
      onUpdate(job);
    }
    return job;
  } catch (error) {
    const job = { status: "failed", error: errorText(error) };
    onUpdate(job);
    return job;
  }
}

function updateControls() {
  const profile = currentProfile();
  nodes["profile-select"].disabled = !state.profiles.length || !!state.importingProfile;
  nodes["import-button"].disabled = !profile || !!state.importingProfile;
  nodes["import-button"].textContent = state.importingProfile ? "正在导入…" : profile?.loaded ? "重新读取资料" : "导入资料";
  nodes["check-button"].disabled = !profile?.loaded || !currentDataset() || state.checkPending;
  nodes["check-button"].textContent = state.checkPending ? "正在核查…" : "开始核查";
  nodes["compare-button"].disabled = !profile?.loaded || !currentDataset() || state.comparePending;
  nodes["compare-button"].textContent = state.comparePending ? "正在对比…" : "运行可用方案";
}

function renderScope() {
  const profile = currentProfile();
  const info = [];
  if (profile?.record_count != null) info.push(`${profile.loaded ? "已导入" : "待导入"} ${profile.record_count} 条原文记录`);
  if (profile?.description) info.push(profile.description);
  if (state.importJob) info.push(jobMessage(state.importJob));
  if (state.importJob?.result?.graph_reason) info.push(state.importJob.result.graph_reason);
  else if (profile?.loaded && profile.graph_status) info.push(`图谱${graphLabels[profile.graph_status] || profile.graph_status}`);
  nodes["profile-status"].textContent = info.join(" · ");
  updateControls();
}

async function loadProfiles() {
  const payload = await request("/demo/profiles");
  state.profiles = asArray(payload.profiles).filter((profile) => profile.id === "app_focus");
  state.answerModel = payload.answer_model;
  state.graphConfigured = payload.graph_configured;
  state.profileId = currentProfile()?.id || state.profiles[0]?.id || "";
  const options = state.profiles.map((profile) => {
    const option = element("option", null, profile.label || profile.id);
    option.value = profile.id;
    return option;
  });
  nodes["profile-select"].replaceChildren(...options);
  nodes["profile-select"].value = state.profileId;
  if (!state.profiles.length) nodes["profile-select"].append(element("option", null, "暂无数据配置"));
  renderScope();
}

function citationGroup(ids, datasetId, onSelect = (id) => openEvidence(id, datasetId)) {
  const group = element("div", "citation-group");
  for (const id of new Set(asArray(ids))) {
    if (typeof id === "string" && id) {
      group.append(button(id, "citation", (event) => onSelect(id, event.currentTarget)));
    }
  }
  return group;
}

function renderEvidenceDrawer(children, returnToSource = null) {
  const drawerRequest = ++state.drawerRequest;
  const drawer = nodes["evidence-dialog"];
  state.returnToSource = returnToSource;
  nodes["back-evidence"].hidden = !returnToSource;
  nodes["evidence-content"].replaceChildren(...children);
  if (!drawer.open) drawer.showModal();
  drawer.scrollTop = 0;
  return drawerRequest;
}

async function openEvidence(id, datasetId = currentDataset(), returnToSource = null) {
  const content = nodes["evidence-content"];
  const drawerRequest = renderEvidenceDrawer([element("p", "muted", "正在读取原文…")], returnToSource);
  if (returnToSource) nodes["back-evidence"].focus();
  try {
    const params = new URLSearchParams({ dataset_id: datasetId });
    const evidence = await request(`/demo/evidence/${encodeURIComponent(id)}?${params}`);
    if (drawerRequest !== state.drawerRequest) return;
    const details = [
      ["证据编号", evidence.id || evidence.record_id || id],
      ["来源类型", labelForSource(evidence.source_type)],
      ["发生时间", formatDate(evidence.occurred_at)],
      ["原文位置", evidence.source_locator],
    ];
    if (evidence.speaker) details.push(["发言人", evidence.speaker]);
    content.replaceChildren(
      element("h3", null, evidence.title || "原文记录"),
      metaList(details),
      element("pre", "raw-content", evidence.content || "当前记录没有可显示的原文。"),
    );
    const reply = evidence.reply_to_message_id || evidence.reply_to;
    if (reply) {
      content.append(
        element("p", "muted", "回复所引用的消息"),
        citationGroup([reply], datasetId, (replyId) => openEvidence(replyId, datasetId, returnToSource)),
      );
    }
  } catch (error) {
    if (drawerRequest === state.drawerRequest) content.replaceChildren(statusMessage(errorText(error), "failed"));
  }
}

function showSource(source, datasetId) {
  const directId = source.record_id || source.id;
  const evidenceIds = asArray(source.evidence_ids || source.record_ids);
  if (evidenceIds.length < 2) {
    openEvidence(evidenceIds[0] || directId, datasetId);
    return;
  }
  const drawer = nodes["evidence-dialog"];
  const sourceView = [
    element("h3", null, source.title || source.name || directId),
    metaList([["来源类型", labelForSource(source.source_type || source.type)], ["原文记录", evidenceIds.length]]),
    citationGroup(evidenceIds, datasetId, (id, trigger) => {
      const scrollTop = drawer.scrollTop;
      openEvidence(id, datasetId, () => {
        renderEvidenceDrawer(sourceView);
        drawer.scrollTop = scrollTop;
        trigger.focus({ preventScroll: true });
      });
    }),
  ];
  renderEvidenceDrawer(sourceView);
}

async function loadSources(reset) {
  const requestId = ++state.sourceRequest;
  const datasetId = currentDataset();
  if (reset) {
    state.sources = [];
    state.sourceOffset = 0;
    nodes["source-count"].textContent = "";
    nodes["more-sources"].hidden = true;
    nodes["source-list"].replaceChildren(element("p", "muted", datasetId ? "正在读取来源…" : "导入后可浏览来源。"));
  }
  if (!datasetId) return;
  try {
    const payload = await request(contextPath(datasetId, "sources", state.sourceOffset, "", 500));
    if (requestId !== state.sourceRequest) return;
    state.sources.push(...asArray(payload.items));
    state.sourceOffset = state.sources.length;
    const total = payload.total ?? state.sourceOffset;
    nodes["source-count"].textContent = `${total} 条记录`;
    nodes["more-sources"].hidden = state.sourceOffset >= total || !asArray(payload.items).length;
    const grouped = new Map();
    for (const source of groupSourceRecords(state.sources)) {
      const kind = labelForSource(source.source_type || source.type);
      if (!grouped.has(kind)) grouped.set(kind, []);
      grouped.get(kind).push(source);
    }
    const containers = [];
    for (const [kind, sources] of grouped) {
      const details = element("details", "source-group");
      details.open = containers.length === 0;
      const links = element("div", "source-items");
      for (const source of sources) {
        const name = source.title || source.name || source.id;
        links.append(button(name, "source-link", () => showSource(source, datasetId)));
      }
      details.append(element("summary", null, kind), links);
      containers.push(details);
    }
    if (!containers.length) containers.push(element("p", "muted", "当前数据范围没有资料来源。"));
    nodes["source-list"].replaceChildren(...containers);
  } catch (error) {
    if (requestId === state.sourceRequest) nodes["source-list"].replaceChildren(statusMessage(errorText(error), "failed"));
  }
}

function groupSourceRecords(records) {
  const groups = new Map();
  for (const record of records) {
    const groupId = record.source_id || record.id;
    if (!groups.has(groupId)) groups.set(groupId, { ...record, record_ids: [] });
    const group = groups.get(groupId);
    group.record_ids.push(...asArray(record.evidence_ids || record.record_ids || [record.id]));
  }
  return groups.values();
}

function itemName(item) {
  if (item.source_name && item.target_name) {
    const source = item.source_text || item.source_name;
    const arrow = item.type === "co_mentioned" ? "↔" : "→";
    return `${source} ${arrow} ${item.target_name}`;
  }
  return item.name || item.title || item.text || item.id || "未命名条目";
}

function contextType(item) { return contextTypes[item.type?.toLowerCase()] || item.type; }

function renderContextDetail(item, datasetId) {
  state.contextSelection = item.id || itemName(item);
  const fields = [];
  if (item.type) fields.push(["类型", contextType(item)]);
  if (item.origin) fields.push(["来源", origins[item.origin] || item.origin]);
  const scopes = item.scopes || (item.scope ? [item.scope] : []);
  if (scopes.length) fields.push(["适用范围", scopes.map((scope) => scopeLabels[scope] || scope)]);
  if (item.occurred_at) fields.push(["时间", formatDate(item.occurred_at)]);
  if (item.certainty) fields.push(["确认语义", certaintyLabels[item.certainty] || item.certainty]);
  if (item.confidence != null) fields.push(["提炼置信度", item.confidence]);
  if (asArray(item.aliases).length) fields.push(["别名", item.aliases]);
  const detail = nodes["context-detail"];
  detail.replaceChildren(element("h2", null, itemName(item)));
  if (fields.length) detail.append(metaList(fields));
  if (item.text && item.text !== itemName(item)) detail.append(element("p", null, item.text));
  for (const part of asArray(item.support?.quotes)) {
    const author = part.speaker ? ` · ${part.speaker}` : "";
    detail.append(element("p", null, `${part.unit_id}${author}　${part.quote}`));
  }
  if (item.support?.status === "incomplete_context") detail.append(element("p", "muted", "可见回复链不足，指代仍需核实。"));
  if (asArray(item.evidence_ids).length) detail.append(element("p", "muted", "原文依据"), citationGroup(item.evidence_ids, datasetId));
  else detail.append(element("p", "muted", "当前条目未返回原文依据。"));
  for (const node of document.querySelectorAll("#context-list [data-id], #context-graph [data-id]")) {
    node.setAttribute("aria-pressed", String(node.dataset.id === state.contextSelection));
  }
}

function contextGraphData(datasetId) {
  if (state.contextGraphs.has(datasetId)) return state.contextGraphs.get(datasetId);
  async function readView(view) {
    const items = [];
    let payload;
    do {
      payload = await request(contextPath(datasetId, view, items.length, "", 500));
      if (!payload.items.length) break;
      items.push(...payload.items);
    } while (items.length < payload.total);
    return items;
  }
  const pending = Promise.all(["entities", "facts", "relations"].map(readView)).then(([entities, facts, relations]) => {
    const entries = new Map();
    for (const item of entities) entries.set(item.id, { item, kind: "实体", links: [] });
    for (const item of facts) entries.set(`fact:${item.id}`, { item, kind: "事实", links: [] });
    for (const relation of relations) {
      const source = entries.get(relation.source_id);
      const target = entries.get(relation.target_id);
      if (!source || !target) continue;
      source.links.push({ neighbor: target, relation, incoming: false });
      if (source !== target) target.links.push({ neighbor: source, relation, incoming: true });
    }
    return [...entries.values()].sort((a, b) =>
      Number(b.kind === "实体") - Number(a.kind === "实体") ||
      b.links.length - a.links.length || itemName(a.item).localeCompare(itemName(b.item), "zh-CN"));
  }).catch((error) => {
    if (state.contextGraphs.get(datasetId) === pending) state.contextGraphs.delete(datasetId);
    throw error;
  });
  state.contextGraphs.set(datasetId, pending);
  return pending;
}

let graphResize;
function drawContextLinks(canvas, network, center, branches) {
  const area = network.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.ceil(area.width * scale);
  canvas.height = Math.ceil(area.height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx || !area.width || !area.height) return;
  ctx.scale(scale, scale);
  ctx.strokeStyle = ctx.fillStyle = "#788ba9";
  ctx.lineWidth = 1.3;
  const root = center.getBoundingClientRect();
  const x1 = root.right - area.left;
  const y1 = root.top + root.height / 2 - area.top;
  for (const { node, link } of branches) {
    const target = node.getBoundingClientRect();
    const x2 = target.left - area.left;
    const y2 = target.top + target.height / 2 - area.top;
    const mid = (x1 + x2) / 2;
    ctx.setLineDash(link.relation.certainty === "inferred" ? [4, 3] : []);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(mid, y1, mid, y2, x2, y2);
    ctx.stroke();
    if (link.relation.type === "co_mentioned") continue;
    const tipX = link.incoming ? x1 : x2;
    const tipY = link.incoming ? y1 : y2;
    const baseX = tipX + (link.incoming ? 6 : -6);
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(baseX, tipY - 3.5);
    ctx.lineTo(baseX, tipY + 3.5);
    ctx.fill();
  }
}

function renderContextGraph(entry, datasetId, page = 0, focusControl = null) {
  state.contextGraphFocus = entry;
  const section = nodes["context-graph"];
  const makeNode = (value, onClick) => {
    const label = itemName(value.item);
    const node = button("", "graph-node", onClick);
    node.dataset.id = value.item.id;
    node.dataset.kind = value.kind;
    node.title = label;
    node.append(element("span", "graph-node-kind", value.kind), element("span", "graph-node-label", label));
    return node;
  };
  const heading = element("div", "graph-heading");
  heading.append(element("h2", null, "直接关联"), element("span", "muted",
    entry.links.length ? `${entry.links.length} 条关系` : "这个节点暂无直接关联"));
  const network = element("div", "graph-network");
  const canvas = element("canvas");
  canvas.setAttribute("aria-hidden", "true");
  const root = makeNode(entry, () => renderContextDetail(entry.item, datasetId));
  root.setAttribute("aria-label", `当前${entry.kind}${itemName(entry.item)}`);
  const neighbors = element("div", "graph-branches");
  const perPage = 8;
  const pageCount = Math.ceil(entry.links.length / perPage);
  const branches = entry.links.slice(page * perPage, (page + 1) * perPage).map((link) => {
    const row = element("div", "graph-branch");
    const neighbor = makeNode(link.neighbor, () => renderContextGraph(link.neighbor, datasetId, 0, "root"));
    const from = link.incoming ? link.neighbor : entry;
    const to = link.incoming ? entry : link.neighbor;
    const relation = { ...link.relation, source_name: itemName(from.item), target_name: itemName(to.item) };
    const direction = relation.type === "co_mentioned" ? "↔" : link.incoming ? "←" : "→";
    const edge = button(`${direction} ${contextType(relation)}`, "graph-edge",
      () => renderContextDetail(relation, datasetId));
    edge.dataset.id = relation.id;
    edge.title = itemName(relation);
    row.append(edge, neighbor);
    neighbors.append(row);
    return { node: neighbor, link };
  });
  network.append(canvas, root, neighbors);
  section.replaceChildren(heading, network);
  if (pageCount > 1) {
    const pager = element("div", "graph-pagination");
    const prev = button("上一组", "button quiet", () => renderContextGraph(entry, datasetId, page - 1, "prev"));
    const next = button("下一组", "button quiet", () => renderContextGraph(entry, datasetId, page + 1, "next"));
    prev.disabled = page === 0;
    next.disabled = page === pageCount - 1;
    pager.append(prev, element("span", "muted", `${page + 1} / ${pageCount}`), next);
    section.append(pager);
    if (focusControl === "prev") (prev.disabled ? next : prev).focus();
    if (focusControl === "next") (next.disabled ? prev : next).focus();
  }
  graphResize?.disconnect();
  graphResize = new ResizeObserver(() => drawContextLinks(canvas, network, root, branches));
  graphResize.observe(network);
  renderContextDetail(entry.item, datasetId);
  if (focusControl === "root") root.focus();
}

function contextListButton(item, text, onClick) {
  const node = button(itemName(item), "context-item", onClick);
  node.dataset.id = item.id || itemName(item);
  node.setAttribute("aria-pressed", String(node.dataset.id === state.contextSelection));
  if (text) node.append(element("span", "item-date", text));
  return node;
}

async function loadContext(reset) {
  const requestId = ++state.contextRequest;
  const datasetId = currentDataset();
  const list = nodes["context-list"];
  const status = nodes["context-status"];
  if (reset) {
    graphResize?.disconnect();
    nodes["context-graph"].hidden = true;
    nodes["context-graph"].replaceChildren();
    state.contextGraphFocus = null;
    state.contextOffset = 0;
    state.contextItems = [];
    state.contextSelection = null;
    list.replaceChildren();
    nodes["context-detail"].replaceChildren(element("p", "muted", "选择一项查看详情和依据。"));
    nodes["more-context"].hidden = true;
    status.textContent = datasetId ? "正在读取上下文…" : "导入资料后可读取已加工的上下文。";
  }
  if (!datasetId) return;
  try {
    const q = nodes["context-search"].value.trim();
    if (state.contextView === "graph") {
      const entries = await contextGraphData(datasetId);
      if (requestId !== state.contextRequest) return;
      const needle = q.toLocaleLowerCase();
      const matches = entries.filter(({ item }) =>
        [itemName(item), item.id, ...asArray(item.aliases), ...asArray(item.evidence_ids)]
          .some((value) => value.toLocaleLowerCase().includes(needle)));
      state.contextOffset += pageSize;
      status.textContent = q ? `匹配 ${matches.length} 个节点` : `${entries.length} 个可浏览节点`;
      list.replaceChildren(...matches.slice(0, state.contextOffset).map((entry) =>
        contextListButton(entry.item, `${entry.kind} · ${entry.links.length} 条关系`,
          () => renderContextGraph(entry, datasetId))));
      nodes["more-context"].hidden = state.contextOffset >= matches.length;
      if (!matches.length) list.append(element("p", "muted", q ? "没有匹配的节点。" : "当前暂无节点。"));
      if (matches.length && !state.contextGraphFocus) {
        nodes["context-graph"].hidden = false;
        renderContextGraph(matches[0], datasetId);
      }
      return;
    }
    const payload = await request(contextPath(datasetId, state.contextView, state.contextOffset, q, pageSize,
      state.contextView === "relations"));
    if (requestId !== state.contextRequest) return;
    state.contextItems.push(...asArray(payload.items));
    state.contextOffset = state.contextItems.length;
    const total = payload.total ?? state.contextOffset;
    const graph = graphLabels[payload.graph_status] || payload.graph_status;
    status.textContent = payload.graph_status === "ready"
      ? `工作图谱已就绪 · ${total} 项`
      : `原文工作图谱 · ${total} 项 · 提炼事实${graph || "尚未加工"}`;
    nodes["more-context"].hidden = state.contextOffset >= total || !asArray(payload.items).length;
    const buttons = state.contextItems.map((item) =>
      contextListButton(item, contextType(item), () => renderContextDetail(item, datasetId)));
    list.replaceChildren(...buttons);
    if (!buttons.length) list.append(element("p", "muted", q ? "没有匹配的条目。" : "当前暂无条目。"));
  } catch (error) {
    if (requestId === state.contextRequest) status.textContent = errorText(error);
  }
}

function resultCards(result, datasetId) {
  const grid = element("div", "result-grid");
  for (const [field, title] of Object.entries(categories)) {
    const card = element("section", "card result-card");
    card.append(element("h3", null, title));
    const items = asArray(result[field]);
    if (!items.length) card.append(element("p", "muted", "本次未返回条目。"));
    else {
      const list = element("ul", "result-list");
      for (const item of items) {
        const row = element("li");
        row.append(element("span", null, typeof item === "string" ? item : item.text));
        if (item.occurred_at) row.append(element("p", "item-date", formatDate(item.occurred_at)));
        if (asArray(item.evidence_ids).length) row.append(citationGroup(item.evidence_ids, datasetId));
        list.append(row);
      }
      card.append(list);
    }
    grid.append(card);
  }
  return grid;
}

function resultDetail(title, ...content) {
  const details = element("details", "card result-extra");
  details.append(element("summary", null, title), ...content);
  return details;
}

function resultContent(job, datasetId) {
  const result = job.result;
  if (!result) return [statusMessage("服务未返回核查结果。", "failed")];
  const segments = [resultCards(result, datasetId)];
  if (asArray(result.uncertainties).length) {
    const list = element("ul", "result-list");
    for (const item of result.uncertainties) list.append(element("li", null, formatValue(item)));
    segments.push(resultDetail("待核实", list));
  }
  const suppliedIds = asArray(result.evidence).map((entry) => entry.id || entry.record_id);
  if (suppliedIds.length || result.context) {
    const details = [];
    if (suppliedIds.length) details.push(element("p", "muted", `本次提供的 ${suppliedIds.length} 条原文`), citationGroup(suppliedIds, datasetId));
    const graphInput = result.context || {};
    for (const [key, label] of Object.entries({ entities: "本次关联的实体", facts: "本次使用的事实", relations: "本次使用的关系" })) {
      const items = asArray(graphInput[key]);
      if (items.length) {
        const list = element("ul", "result-list");
        for (const item of items) list.append(element("li", null, typeof item === "string" ? item : itemName(item)));
        details.push(element("p", "muted", label), list);
      }
    }
    if (details.length) segments.push(resultDetail("本次使用的依据", ...details));
  }
  const usage = result.usage || {};
  const runInfo = [
    ["耗时", usage.elapsed_ms == null ? null : `${usage.elapsed_ms} ms`],
    ...(result.cutoff ? [["材料截点", formatDate(result.cutoff)]] : []),
    ...(job.run_id ? [["运行编号", job.run_id]] : []),
  ];
  const coverage = Object.entries(coverageLabels)
    .filter(([key]) => Object.hasOwn(result.coverage || {}, key))
    .map(([key, label]) => [label, result.coverage[key]]);
  segments.push(resultDetail("运行详情与覆盖范围", metaList([...runInfo, ...coverage])));
  return segments;
}

function renderCheck() {
  const panel = nodes["check-result"];
  const job = state.jobs.get(state.checkKey);
  if (!job) { panel.replaceChildren(); return; }
  const [datasetId, question, mode] = JSON.parse(state.checkKey);
  const heading = element("div", "result-heading");
  heading.append(element("h2", null, question), badge(job.status, `${modes[mode].name} · ${jobLabels[job.status] || job.status}`));
  panel.replaceChildren(heading);
  if (job.status === "completed") panel.append(...resultContent(job, datasetId));
  else panel.append(statusMessage(jobMessage(job), job.status));
  if (job.status === "completed") panel.append(button("比较其他方案", "button quiet", () => {
    nodes["compare-question"].value = question;
    setTab("compare");
  }));
}

function renderCompare() {
  const datasetId = currentDataset();
  const question = nodes["compare-question"].value.trim();
  const cards = comparisonModes.map((mode) => {
    const info = modes[mode];
    const job = state.jobs.get(jobKey(datasetId, question, mode));
    const card = element("article", "card compare-card");
    card.append(element("h2", null, info.name), element("p", "muted", info.description));
    if (job) {
      card.append(badge(job.status));
      if (job.status === "completed") card.append(...resultContent(job, datasetId));
      else card.append(statusMessage(jobMessage(job), job.status));
    } else if (datasetId && info.nativeRequirement && !state.graphConfigured) {
      card.append(badge("unavailable"), element("p", "muted", info.nativeRequirement));
    } else card.append(badge("idle", "未运行"));
    return card;
  });
  nodes["compare-results"].replaceChildren(...cards);
}

function setTab(name, focus = false) {
  state.activeTab = name;
  for (const tab of document.querySelectorAll("[role=tab]")) {
    const active = tab.id === `tab-${name}`;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
    if (active && focus) tab.focus();
  }
  if (name === "context") loadContext(true);
  if (name === "compare") renderCompare();
}

async function runCheck(datasetId, question, mode) {
  const key = jobKey(datasetId, question, mode);
  if (state.activeChecks.has(key)) return state.activeChecks.get(key);
  const pending = followJob("/demo/check", { dataset_id: datasetId, project_id: projectId, question, mode }, (job) => {
    state.jobs.set(key, job);
    if (key === state.checkKey) renderCheck();
    if (state.activeTab === "compare") renderCompare();
  });
  state.activeChecks.set(key, pending);
  try {
    return await pending;
  } finally {
    state.activeChecks.delete(key);
  }
}

nodes["profile-select"].addEventListener("change", () => {
  state.profileId = nodes["profile-select"].value;
  state.importJob = null;
  state.checkKey = null;
  renderScope();
  renderCheck();
  loadSources(true);
  if (state.activeTab === "context") loadContext(true);
  if (state.activeTab === "compare") renderCompare();
});

nodes["import-button"].addEventListener("click", async () => {
  if (!state.profileId || state.importingProfile) return;
  state.importingProfile = state.profileId;
  const job = await followJob("/demo/import", { profile: state.importingProfile }, (update) => {
    state.importJob = update;
    renderScope();
  });
  state.importingProfile = null;
  if (job.status === "completed" || job.dataset_id) {
    try {
      state.contextGraphs.delete(job.result?.dataset_id || job.dataset_id);
      await loadProfiles();
      loadSources(true);
      if (state.activeTab === "context") loadContext(true);
      if (state.activeTab === "compare") renderCompare();
    } catch (error) {
      state.importJob = { status: "failed", error: errorText(error) };
    }
  }
  renderScope();
});

nodes["check-form"].addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = nodes.question.value.trim();
  const datasetId = currentDataset();
  if (!question || !datasetId || state.checkPending) return;
  const mode = nodes["mode-select"].value;
  state.checkPending = true;
  state.checkKey = jobKey(datasetId, question, mode);
  nodes["compare-question"].value = question;
  updateControls();
  await runCheck(datasetId, question, mode);
  state.checkPending = false;
  updateControls();
});

nodes["compare-form"].addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = nodes["compare-question"].value.trim();
  const datasetId = currentDataset();
  if (!question || !datasetId || state.comparePending) return;
  state.comparePending = true;
  updateControls();
  for (const mode of comparisonModes) {
    if (modes[mode].nativeRequirement && !state.graphConfigured) continue;
    if (state.jobs.get(jobKey(datasetId, question, mode))?.status === "completed") continue;
    await runCheck(datasetId, question, mode);
  }
  state.comparePending = false;
  updateControls();
});

for (const example of document.querySelectorAll("[data-question]")) {
  example.addEventListener("click", () => {
    nodes.question.value = example.dataset.question;
    nodes.question.focus();
  });
}

const tabs = [...document.querySelectorAll("[role=tab]")];
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => setTab(tab.id.slice(4)));
  tab.addEventListener("keydown", (event) => {
    const offsets = { ArrowLeft: index - 1, ArrowRight: index + 1, Home: 0, End: tabs.length - 1 };
    if (offsets[event.key] == null) return;
    event.preventDefault();
    const next = tabs[(offsets[event.key] + tabs.length) % tabs.length];
    setTab(next.id.slice(4), true);
  });
}

for (const control of document.querySelectorAll("[data-view]")) {
  control.addEventListener("click", () => {
    state.contextView = control.dataset.view;
    for (const peer of document.querySelectorAll("[data-view]")) peer.setAttribute("aria-pressed", String(peer === control));
    loadContext(true);
  });
}

let searchTimer;
nodes["context-search"].addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadContext(true), 220);
});
nodes["more-sources"].addEventListener("click", () => loadSources(false));
nodes["more-context"].addEventListener("click", () => loadContext(false));
nodes["compare-question"].addEventListener("input", renderCompare);
nodes["back-evidence"].addEventListener("click", () => state.returnToSource?.());
nodes["close-evidence"].addEventListener("click", () => nodes["evidence-dialog"].close());
nodes["evidence-dialog"].addEventListener("close", () => {
  if (nodes["evidence-dialog"].open) return;
  state.drawerRequest++;
  state.returnToSource = null;
});
nodes["evidence-dialog"].addEventListener("click", (event) => {
  if (event.target === nodes["evidence-dialog"]) nodes["evidence-dialog"].close();
});

renderCompare();
loadProfiles().then(() => {
  loadSources(true);
  if (state.activeTab === "context") loadContext(true);
}).catch((error) => {
  nodes["profile-status"].textContent = `本地服务连接失败 · ${errorText(error)}`;
});
