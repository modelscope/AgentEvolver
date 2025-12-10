// ---------------- Config/Start Game ----------------
async function fetchOptionsAndInitConfigForm() {
  try {
    const resp = await fetch("/api/options");
    if (!resp.ok) throw new Error("Failed to fetch options");
    const opts = await resp.json();

    // 填充 mode
    const modeSel = document.getElementById("config-mode");
    if (modeSel) {
      modeSel.innerHTML = "";
      ["observe", "participate"].forEach(m => {
        const o = document.createElement("option");
        o.value = m;
        o.textContent = m.charAt(0).toUpperCase() + m.slice(1);
        modeSel.appendChild(o);
      });
      modeSel.value = opts.defaults.mode;
    }

    // 填充 human_power
    const hpSel = document.getElementById("config-human-power");
    if (hpSel) {
      hpSel.innerHTML = "";
      opts.powers.forEach(p => {
        const o = document.createElement("option");
        o.value = p;
        o.textContent = p;
        hpSel.appendChild(o);
      });
      hpSel.value = opts.defaults.human_power;
    }

    // 填充 model
    const modelSel = document.getElementById("config-model");
    if (modelSel) {
      modelSel.innerHTML = "";
      opts.models.forEach(m => {
        const o = document.createElement("option");
        o.value = m;
        o.textContent = m;
        modelSel.appendChild(o);
      });
      modelSel.value = opts.defaults.model_name;
    }

    // 填充 language
    const langSel = document.getElementById("config-language");
    if (langSel) {
      langSel.innerHTML = "";
      ["en", "cn"].forEach(l => {
        const o = document.createElement("option");
        o.value = l;
        o.textContent = l;
        langSel.appendChild(o);
      });
      langSel.value = opts.defaults.language;
    }

    // max_phases
    const maxPhasesInput = document.getElementById("config-max-phases");
    if (maxPhasesInput) maxPhasesInput.value = opts.defaults.max_phases;

    // negotiation_rounds
    const negoInput = document.getElementById("config-negotiation-rounds");
    if (negoInput) negoInput.value = opts.defaults.negotiation_rounds;

    // 参与模式才显示 human_power，否则禁用
    if (modeSel && hpSel) {
      modeSel.addEventListener("change", function () {
        hpSel.disabled = (this.value !== "participate");
      });
      hpSel.disabled = (modeSel.value !== "participate");
    }
  } catch (e) {
    console.error("Failed to init config form", e);
  }
}

const configForm = document.getElementById("config-form");
if (configForm) {
  configForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const mode = document.getElementById("config-mode")?.value || "observe";
    const human_power = document.getElementById("config-human-power")?.value || "";
    const model_name = document.getElementById("config-model")?.value || "qwen-plus";
    const max_phases = parseInt(document.getElementById("config-max-phases")?.value) || 20;
    const negotiation_rounds = parseInt(document.getElementById("config-negotiation-rounds")?.value) || 3;
    const language = document.getElementById("config-language")?.value || "en";

    const payload = {
      mode,
      model_name,
      human_power: mode === "participate" ? human_power : null,
      max_phases,
      negotiation_rounds,
      language,
    };

    try {
      const resp = await fetch("/api/start-game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await resp.json();
      if (!resp.ok) throw new Error(result.detail || "Failed to start game");
      // 隐藏配置栏
      configForm.style.display = "none";
    } catch (e) {
      alert("Failed to start game: " + e.message);
    }
  });
}

// 页面加载时初始化
fetchOptionsAndInitConfigForm();
// =============================
// main.js (FULL REPLACEMENT)
// Only 2 modes:
//   - observe     : no human_power -> render ONLY accumulated obs_log_entry (obsBuffer)
//   - participate : has human_power -> render ONLY logs
// No mode switch button.
// Phase entries ALWAYS show regardless of filter.
// =============================

const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

// If you still want to filter history items by kind, keep this set.
// Otherwise you can remove this and show all history.
const ALLOWED_HISTORY_KINDS = new Set(["orders", "order", "result", "init"]); // e.g. new Set(["orders", "order", "result", "init"]);

let socket = null;

let latestState = {};
let isViewingHistory = false;
let knownHistorySig = [];

let currentFilter = "all";
let currentRenderData = {};
let currentPrompt = null;
let currentInputAgentId = null;

// ---------------- Observe buffer (accumulate obs_log_entry) ----------------
let obsBuffer = [];              // array of renderable entries
let obsSeen = new Set();         // dedup signatures

function obsSig(x) {
  if (typeof x === "string") return x;
  if (x && typeof x === "object") return x.content ?? x.text ?? JSON.stringify(x);
  return String(x);
}

// Accept obs_log_entry of types: string | array | object
// Convert to array of entries and accumulate them
function appendObsEntry(obsEntry) {
  const entries = normalizeObsEntry(obsEntry); // always array
  for (const e of entries) {
    const sig = obsSig(e).trim();
    if (!sig) continue;
    if (obsSeen.has(sig)) continue;
    obsSeen.add(sig);
    obsBuffer.push(e);
  }
}

// ---------------- DOM ----------------
const phaseSelect = document.getElementById("phase-select");
const logFilter = document.getElementById("log-filter");

const logsContainer = document.getElementById("logs-container");
const mapContainer = document.getElementById("map-container");

const inputArea = document.getElementById("user-input-area");
const sendBtn = document.getElementById("submit-input");
const userInput = document.getElementById("user-input");
const promptEl = document.getElementById("input-prompt");

// ---------------- Helpers ----------------
function getField(state, key, fallback = undefined) {
  if (!state) return fallback;
  if (state[key] !== undefined && state[key] !== null) return state[key];
  if (state.meta && state.meta[key] !== undefined && state.meta[key] !== null) return state.meta[key];
  return fallback;
}

function normalizeHistorySnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== "object") return { meta: {} };
  const state = { ...snapshot };
  state.meta = (snapshot.meta && typeof snapshot.meta === "object") ? snapshot.meta : {};

  const keysToLift = ["map_svg", "logs", "obs_log_entry", "human_power", "status", "phase", "round", "kind", "timestamp"];
  keysToLift.forEach((k) => {
    if (state[k] === undefined && state.meta[k] !== undefined) state[k] = state.meta[k];
  });

  return state;
}

function isMessageLike(obj) {
  if (!obj) return false;
  if (typeof obj === "string") return true;
  if (typeof obj !== "object") return false;

  // 标准 message
  if (obj.type === "message") return true;

  // agentscope/msg 风格常见字段
  if (obj.content !== undefined || obj.text !== undefined) return true;
  if (obj.name !== undefined || obj.sender !== undefined) return true;

  return false;
}

function messageToLogEntry(msg) {
  // 统一转换成你现有渲染能吃的 entry（string 或 object）
  if (typeof msg === "string") return msg;

  const content =
    msg.content ?? msg.text ?? msg.message ?? msg.data ?? JSON.stringify(msg);

  const sender =
    msg.sender ?? msg.name ?? msg.from ?? "System";

  const role = msg.role ?? "assistant";
  const ts = msg.timestamp ?? null;

  // 你现有 logText() 会优先 content/text，所以这里用 content
  // 也保留 sender 方便你未来做样式
  const line = sender ? `${sender}: ${String(content)}` : String(content);

  return {
    message_type: "chat_message", // 新类型，可选
    sender,
    role,
    timestamp: ts,
    content: line,
  };
}

function appendToParticipateLogs(entry) {
  const prevLogs = Array.isArray(latestState.logs) ? latestState.logs : normalizeLogs(latestState.logs);

  const sig = (x) => (typeof x === "string" ? x : JSON.stringify(x));
  const last = prevLogs.length ? prevLogs[prevLogs.length - 1] : null;

  if (!last || sig(last) !== sig(entry)) {
    prevLogs.push(entry);
  }
  latestState.logs = prevLogs;
}


function normalizeGameStateMessage(message) {
  const state = {};
  if (message && message.data && typeof message.data === "object") {
    Object.assign(state, message.data);
  } else if (message && typeof message === "object") {
    Object.assign(state, message);
  }

  const meta =
    (state.meta && typeof state.meta === "object" && state.meta) ||
    (message.meta && typeof message.meta === "object" && message.meta) ||
    (message.data && message.data.meta && typeof message.data.meta === "object" && message.data.meta) ||
    {};
  state.meta = meta;

  delete state.type;
  delete state.data;
  return state;
}

// logs can be: array | string | object-of-arrays
function normalizeLogs(logs) {
  if (Array.isArray(logs)) return logs;

  if (logs && typeof logs === "object") {
    // { Agent_ITALY: [...], Agent_FRANCE: [...] } -> flatten
    const out = [];
    for (const [agentName, arr] of Object.entries(logs)) {
      if (!Array.isArray(arr)) continue;
      for (const item of arr) {
        if (item && typeof item === "object") {
          out.push({ sender: agentName.replace(/^Agent_/, ""), ...item });
        } else {
          out.push(String(item));
        }
      }
    }
    return out;
  }

  if (typeof logs === "string") return [logs];
  return [];
}

function logText(entry) {
  if (typeof entry === "string") return entry;
  if (entry && typeof entry === "object") return entry.content ?? entry.text ?? JSON.stringify(entry);
  return String(entry);
}

// Convert obs_log_entry into renderable log entries
function normalizeObsEntry(obsEntry) {
  if (!obsEntry) return [];
  if (Array.isArray(obsEntry)) return obsEntry;

  if (typeof obsEntry === "string") {
    // split by lines -> each line is an observer_log entry
    return obsEntry
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(Boolean)
      .map(line => ({ message_type: "observer_log", content: line }));
  }

  if (typeof obsEntry === "object") return [obsEntry];
  return [{ message_type: "observer_log", content: String(obsEntry) }];
}

// 4 styles: phase / negotiation / orders / system
function getLogType(entry) {
  const t = logText(entry);

  // 1) phase must be highest priority
  if (/---\s*Phase:/i.test(t)) return "phase";

  // 2) orders (must be before system)
  const ORDER_LINE = /^(\[[A-Z_]+\]\s*)?([A-Z][A-Z_]+)\s+orders\s*:/;
  if (ORDER_LINE.test(t) || /^\[Moderator\]/.test(t)) return "orders";

  // 3) system prompt / observer_log
  if (
    /^\[System\]/.test(t) ||
    /---COUNTRY_SYSTEM---|---FEW_SHOT---|---PHASE_INSTRUCTIONS---|---SITUATION---|Your power is/i.test(t) ||
    (entry && typeof entry === "object" && entry.role === "system") 
  ) return "system";

  // 4) negotiation
  if (
    /\[Negotiation\]/i.test(t) ||
    /^\s*From\b/i.test(t) ||
    /^\s*To\b/i.test(t)
  ) return "negotiation";

  return "negotiation";
}

function renderOneLog(entry) {
  const el = document.createElement("div");
  const t = logText(entry);
  const type = getLogType(entry);

  el.classList.add("log-entry");
  if (type === "phase") el.classList.add("phase");
  if (type === "negotiation") el.classList.add("log-negotiation");
  if (type === "orders") el.classList.add("log-orders");
  if (type === "system") el.classList.add("log-system");

  el.textContent = t;
  return el;
}

// phase always show
function wantByFilter(entry) {
  const type = getLogType(entry);
  if (type === "phase") return true;

  if (currentFilter === "all") return true;
  if (currentFilter === "negotiation") return type === "negotiation";
  if (currentFilter === "orders") return type === "orders";
  if (currentFilter === "system") return type === "system";
  return true;
}

function ensurePhaseLog(logs, state) {
  const hasPhase = logs.some(e => getLogType(e) === "phase");
  if (hasPhase) return logs;

  const phase = getField(state, "phase", null);
  if (!phase) return logs;

  const round = getField(state, "round", undefined);
  const line = `--- Phase: ${phase}${round !== undefined ? ` (Round ${round})` : ""} ---`;
  return [line, ...logs];
}

// ---------------- Filters / Phase select ----------------
if (logFilter) {
  logFilter.addEventListener("change", function () {
    currentFilter = this.value;
    renderState(currentRenderData, isViewingHistory);
  });
}

if (phaseSelect) {
  phaseSelect.addEventListener("change", function () {
    const value = this.value;
    if (value === "latest") {
      isViewingHistory = false;
      renderState(latestState, false);
    } else {
      isViewingHistory = true;
      fetchHistoryState(parseInt(value, 10) - 1);
    }
  });
}

// ---------------- History ----------------
async function fetchHistoryList() {
  try {
    const response = await fetch("/api/history");
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    let history = await response.json();

    if (ALLOWED_HISTORY_KINDS) {
      history = history.filter(h => ALLOWED_HISTORY_KINDS.has(h.kind));
    }

    const newSig = history.map((h) => `${h.index}|${h.phase}|${h.round}|${h.kind || ""}`);
    if (JSON.stringify(newSig) === JSON.stringify(knownHistorySig)) return;
    knownHistorySig = newSig;

    if (!phaseSelect) return;

    const currentVal = phaseSelect.value || "latest";
    phaseSelect.innerHTML = '<option value="latest">Latest</option>';

    history.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.index;
      const kind = item.kind ? `(${item.kind})` : "";
      option.textContent = `${item.phase} R${item.round} ${kind}`.trim();
      phaseSelect.appendChild(option);
    });

    if (currentVal !== "latest" && history.some(h => String(h.index) === currentVal)) {
      phaseSelect.value = currentVal;
    } else {
      phaseSelect.value = "latest";
    }
  } catch (e) {
    console.error("Failed to fetch history list", e);
  }
}

async function fetchHistoryState(index) {
  try {
    const response = await fetch(`/api/history/${index}`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const snapshot = await response.json();
    const state = normalizeHistorySnapshot(snapshot);
    renderState(state, true);
  } catch (e) {
    console.error("Failed to fetch history state", e);
  }
}

// ---------------- WebSocket ----------------
function connect() {
  socket = new WebSocket(wsUrl);

  socket.onopen = function () {
    console.log("[WebSocket] Connection established");
    fetchHistoryList();
  };

  socket.onmessage = function (event) {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (e) {
      console.error("Invalid JSON:", event.data);
      return;
    }

    if (message.type === "game_state") {
      const state = normalizeGameStateMessage(message);
      updateState(state);
      return;
    }

    if (message.type === "input_request" || message.type === "user_input_request") {
      handleInputRequest(message);
      return;
    }

    // ✅ NEW: 兼容后端直接 broadcast 的 Msg / message
    if (isMessageLike(message)) {
      // 只有 participate 才把 message 追加进 logs
      const human_power = getField(latestState, "human_power", "");
      const isParticipate = !!human_power;

      if (isParticipate) {
        const entry = messageToLogEntry(message);
        appendToParticipateLogs(entry);
        if (!isViewingHistory) renderState(latestState, false);
      }
      return;
    }

    // 其他未知消息类型：忽略或打印
    console.log("[WebSocket] Unknown message:", message);
  };


  socket.onclose = function () {
    setTimeout(connect, 1500);
  };

  socket.onerror = function (error) {
    console.log("[WebSocket] Error:", error.message);
  };
}

// Accumulate logs for participate mode, and accumulate obs_log_entry for observe mode
function updateState(incoming) {
  const prevMap = getField(latestState, "map_svg", "");
  const nextMap = getField(incoming, "map_svg", "");

  // ----- merge logs incrementally -----
  const incomingLogsRaw = getField(incoming, "logs", undefined);
  if (incomingLogsRaw !== undefined) {
    const incomingLogs = normalizeLogs(incomingLogsRaw);
    const prevLogs = Array.isArray(latestState.logs) ? latestState.logs : normalizeLogs(latestState.logs);

    for (const item of incomingLogs) {
      const last = prevLogs.length ? prevLogs[prevLogs.length - 1] : null;
      const sig = (x) => (typeof x === "string" ? x : JSON.stringify(x));
      if (!last || sig(last) !== sig(item)) prevLogs.push(item);
    }
    latestState.logs = prevLogs;
  }

  // ----- accumulate obs_log_entry -----
  const incomingObs = getField(incoming, "obs_log_entry", undefined);
  if (incomingObs !== undefined) {
    latestState.obs_log_entry = incomingObs;
    // ✅ append into observe buffer (dedup)
    appendObsEntry(incomingObs);
  }

  // ----- merge other fields -----
  for (const key in incoming) {
    if (key === "logs" || key === "obs_log_entry") continue;
    latestState[key] = incoming[key];
  }
  latestState.meta = latestState.meta || {};

  if (nextMap && nextMap !== prevMap) fetchHistoryList();

  if (!isViewingHistory) renderState(latestState, false);
}

// ---------------- Rendering ----------------
function renderState(data, isHistory = false) {
  currentRenderData = data;

  const meta = data.meta || {};

  const map_svg = data.map_svg ?? meta.map_svg ?? "";
  const human_power = data.human_power ?? meta.human_power ?? "";
  const logsRaw = data.logs ?? meta.logs ?? [];
  const obsEntry = data.obs_log_entry ?? meta.obs_log_entry ?? null;

  // participate: show ONLY logs
  // observe: show ONLY accumulated obsBuffer (not just latest obsEntry)
  const isParticipate = !!human_power;
  let logs;

  if (isParticipate) {
    logs = normalizeLogs(logsRaw);
  } else {
    // ensure current snapshot obsEntry is included too (especially when viewing history snapshot)
    // but still keep accumulation behavior for live observe
    const snapObs = normalizeObsEntry(obsEntry);
    // build a merged list: obsBuffer + snapshot entries (dedup by obsSeen)
    logs = obsBuffer.slice();
    for (const e of snapObs) {
      const sig = obsSig(e).trim();
      if (!sig) continue;
      if (obsSeen.has(sig)) continue;
      // do not mutate obsBuffer in history view; just include for rendering
      logs.push(e);
    }
  }

  logs = ensurePhaseLog(logs, data);

  // ----- header/status -----
  const phaseEl = document.getElementById("phase");
  const roundEl = document.getElementById("round");
  const statusEl = document.getElementById("status");

  const phase = getField(data, "phase", "");
  const round = getField(data, "round", undefined);
  const status = getField(data, "status", "");

  if (phaseEl && phase) phaseEl.textContent = `Phase: ${phase}`;
  if (roundEl && round !== undefined) roundEl.textContent = `Round: ${round}`;
  if (statusEl && status) statusEl.textContent = `Status: ${status}${isHistory ? " (History View)" : ""}`;

  // ----- map -----
  if (mapContainer && map_svg) {
    if (mapContainer.innerHTML !== map_svg) mapContainer.innerHTML = map_svg;
  }

  // ----- input area -----
  if (inputArea) {
    inputArea.style.display = (!isHistory && isParticipate) ? "block" : "none";
  }

  // ----- logs -----
  if (!logsContainer) return;

  logsContainer.innerHTML = "";
  logs.filter(wantByFilter).forEach((entry) => {
    logsContainer.appendChild(renderOneLog(entry));
  });
  logsContainer.scrollTop = logsContainer.scrollHeight;
}

// ---------------- Human input ----------------
if (sendBtn && userInput) {
  sendBtn.addEventListener("click", sendInput);
  userInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendInput();
    }
  });
}

function sendInput() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  const text = (userInput.value || "").trim();
  if (!text) return;

  const payload = { type: "user_input", content: text };
  if (currentInputAgentId !== null && currentInputAgentId !== undefined) {
    payload.agent_id = currentInputAgentId;
  }

  socket.send(JSON.stringify(payload));

  userInput.value = "";
  currentPrompt = null;
  currentInputAgentId = null;

  if (promptEl) {
    promptEl.innerText = "";
    promptEl.style.display = "none";
  }

  if (!isViewingHistory) renderState(latestState, false);
}

function handleInputRequest(message) {
  const req = (message && message.data && typeof message.data === "object") ? message.data : message;

  currentPrompt = req.prompt || "Please enter your orders or message:";
  currentInputAgentId = req.agent_id ?? null;

  // live view
  isViewingHistory = false;
  if (phaseSelect) phaseSelect.value = "latest";

  if (inputArea) inputArea.style.display = "block";
  if (promptEl) {
    promptEl.innerText = currentPrompt;
    promptEl.style.display = "block";
  }

  renderState(latestState, false);

  if (userInput) {
    userInput.value = "";
    userInput.focus();
  }
}

// ---------------- Start ----------------
connect();
