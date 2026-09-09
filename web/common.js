/* Enterprise-QA-Agent 前端共享逻辑：API 客户端 + SSE 解析 + 工具 */
(function (global) {
  "use strict";

  const API_BASE = "/api/v1";

  // 当前 tenant（与后端 DEFAULT_TENANT_ID 对齐，默认 tenant_demo）
  function tenantId() {
    return localStorage.getItem("qa_tenant") || "tenant_demo";
  }
  function threadId() {
    // 每个会话一个固定 thread，可手动重置
    let t = localStorage.getItem("qa_thread");
    if (!t) {
      t = tenantId() + ":" + "web_" + Math.random().toString(36).slice(2, 10);
      localStorage.setItem("qa_thread", t);
    }
    return t;
  }
  function resetThread() {
    localStorage.removeItem("qa_thread");
    return threadId();
  }

  // 健康检查
  async function checkHealth() {
    try {
      const r = await fetch(API_BASE + "/health");
      return { http: r.status, data: await r.json() };
    } catch (e) {
      return { http: 0, data: null, error: String(e) };
    }
  }

  // ── SSE 问答流（基于 fetch + ReadableStream 解析 text/event-stream）────
  // onEvent: ({type, data}) => void; type ∈ ready|token|citation|done|ping|error
  async function streamChat(question, { includeExpired = false, stream = true, onEvent }) {
    if (!stream) {
      const r = await fetch(API_BASE + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: threadId(), question, stream: false, include_expired: includeExpired,
        }),
      });
      const body = await r.json();
      if (!r.ok) throw Object.assign(new Error(body.message || "请求失败"), { body, status: r.status });
      onEvent && onEvent({ type: "done", data: body });
      return body;
    }

    const r = await fetch(API_BASE + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: threadId(), question, stream: true, include_expired: includeExpired,
      }),
    });
    if (!r.ok) {
      let body = null;
      try { body = await r.json(); } catch (e) {}
      throw Object.assign(new Error((body && body.message) || "请求失败"), { body, status: r.status });
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const rawEvent = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const ev = parseSSEEvent(rawEvent);
        if (ev) onEvent && onEvent(ev);
      }
    }
  }

  // 解析单个 SSE 事件块（"event: X\ndata: {...}"）
  function parseSSEEvent(block) {
    let type = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) type = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) return null;
    try { data = JSON.parse(data); } catch (e) { /* 非 JSON 数据原样 */ }
    return { type, data };
  }

  // ── 文档上传（multipart）────────────────────────────────────────
  async function uploadDocument(file, meta) {
    const fd = new FormData();
    fd.append("file", file);
    if (meta) fd.append("meta", JSON.stringify(meta));
    const r = await fetch(API_BASE + "/documents", { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) throw Object.assign(new Error(body.message || "上传失败"), { body, status: r.status });
    return body; // { task_id, doc_id, version, status, duplicated }
  }

  // 任务状态查询
  async function getTask(taskId) {
    const r = await fetch(API_BASE + "/tasks/" + encodeURIComponent(taskId));
    const body = await r.json();
    if (!r.ok) throw Object.assign(new Error(body.message || "查询失败"), { body, status: r.status });
    return body;
  }

  // 对话历史
  async function getHistory(limit = 50) {
    const r = await fetch(API_BASE + "/threads/" + encodeURIComponent(threadId()) + "/history?limit=" + limit);
    const body = await r.json();
    if (!r.ok) throw Object.assign(new Error(body.message || "历史查询失败"), { body, status: r.status });
    return body.messages || [];
  }

  // 调试检索
  async function debugRetrieve(query, opts = {}) {
    const r = await fetch(API_BASE + "/debug/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: opts.top_k || 8, include_expired: opts.includeExpired || false, department: opts.department || null }),
    });
    const body = await r.json();
    if (!r.ok) throw Object.assign(new Error(body.message || "检索失败"), { body, status: r.status });
    return body;
  }

  // 工具：转义 HTML
  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // 工具：格式化字节
  function fmtBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / 1024 / 1024).toFixed(2) + " MB";
  }

  // 工具：格式化耗时
  function fmtMs(ms) {
    if (ms == null) return "";
    if (ms < 1000) return ms + "ms";
    return (ms / 1000).toFixed(2) + "s";
  }

  // ── 元数据下拉选项（集中定义，后续增删只在代码里改，不给客户自由填写）────
  // 值 = 传给后端的 department / category；label = 下拉显示文案
  const DEPT_OPTIONS = [
    { value: "", label: "无" },
    { value: "财务部", label: "财务部" },
    { value: "人事部", label: "人事部" },
    { value: "行政部", label: "行政部" },
    { value: "技术部", label: "技术部" },
  ];
  const CATEGORY_OPTIONS = [
    { value: "", label: "无" },
    { value: "规章制度", label: "规章制度" },
    { value: "报销流程", label: "报销流程" },
    { value: "考勤请假", label: "考勤请假" },
    { value: "薪酬福利", label: "薪酬福利" },
  ];

  global.QA = {
    API_BASE, tenantId, threadId, resetThread, checkHealth,
    streamChat, uploadDocument, getTask, getHistory, debugRetrieve,
    esc, fmtBytes, fmtMs,
    DEPT_OPTIONS, CATEGORY_OPTIONS,
  };
})(window);
