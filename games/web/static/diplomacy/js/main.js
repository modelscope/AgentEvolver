// =============================
// Diplomacy main.js (Unified version)
// Config is handled by index.html, this file only handles game display
// =============================

// 清除页面缓存：当离开游戏页面时清除游戏数据
window.addEventListener('beforeunload', () => {
    const keysToKeep = ['gameConfig', 'selectedPortraits', 'gameLanguage'];
    Object.keys(sessionStorage).forEach(key => {
        if (!keysToKeep.includes(key)) {
            sessionStorage.removeItem(key);
        }
    });
});

// 强制不使用浏览器的 bfcache（后退/前进缓存）
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        window.location.reload();
    }
});

// 应用语言类到 body
const gameLanguage = sessionStorage.getItem('gameLanguage') || 'en';
document.body.classList.add(`lang-${gameLanguage}`);

const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

const ALLOWED_HISTORY_KINDS = new Set(["orders", "order", "result", "init"]);

let socket = null;
let latestState = {};
let isViewingHistory = false;
let knownHistorySig = [];

let currentFilter = "all";
let currentRenderData = {};
let currentPrompt = null;
let currentInputAgentId = null;

// Observe buffer (accumulate obs_log_entry)
let obsBuffer = [];
let obsSeen = new Set();

function obsSig(x) {
  if (typeof x === "string") return x;
  if (x && typeof x === "object") return x.content ?? x.text ?? JSON.stringify(x);
  return String(x);
}

function appendObsEntry(obsEntry) {
  const entries = normalizeObsEntry(obsEntry);
  for (const e of entries) {
    const sig = obsSig(e).trim();
    if (!sig) continue;
    if (obsSeen.has(sig)) continue;
    obsSeen.add(sig);
    obsBuffer.push(e);
  }
}

// DOM elements
const phaseSelect = document.getElementById("phase-select");
const logFilter = document.getElementById("log-filter");
const logsContainer = document.getElementById("logs-container");
const mapContainer = document.getElementById("map-container");
const inputArea = document.getElementById("user-input-area");
const sendBtn = document.getElementById("submit-input");
const userInput = document.getElementById("user-input");
const promptEl = document.getElementById("input-prompt");
const backExitButton = document.getElementById("back-exit-button");

// Helpers
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
  if (obj.type === "message") return true;
  if (obj.content !== undefined || obj.text !== undefined) return true;
  if (obj.name !== undefined || obj.sender !== undefined) return true;
  return false;
}

function messageToLogEntry(msg) {
  if (typeof msg === "string") return msg;

  const content = msg.content ?? msg.text ?? msg.message ?? msg.data ?? JSON.stringify(msg);
  const sender = msg.sender ?? msg.name ?? msg.from ?? "System";
  const role = msg.role ?? "assistant";
  const ts = msg.timestamp ?? null;
  const line = String(content);

  return {
    message_type: "chat_message",
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

function normalizeLogs(logs) {
  if (Array.isArray(logs)) return logs;

  if (logs && typeof logs === "object") {
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

function normalizeObsEntry(obsEntry) {
  if (!obsEntry) return [];
  if (Array.isArray(obsEntry)) return obsEntry;

  if (typeof obsEntry === "string") {
    return obsEntry
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(Boolean)
      .map(line => ({ message_type: "observer_log", content: line }));
  }

  if (typeof obsEntry === "object") return [obsEntry];
  return [{ message_type: "observer_log", content: String(obsEntry) }];
}

function getLogType(entry) {
  const t = logText(entry);

  if (/---\s*Phase:/i.test(t)) return "phase";

  const ORDER_LINE = /^(\[[A-Z_]+\]\s*)?([A-Z][A-Z_]+)\s+orders\s*:/;
  if (ORDER_LINE.test(t) || /^\[Moderator\]/.test(t)) return "orders";

  if (
    /^\[System\]/.test(t) ||
    /---COUNTRY_SYSTEM---|---FEW_SHOT---|---PHASE_INSTRUCTIONS---|---SITUATION---|Your power is/i.test(t) ||
    (entry && typeof entry === "object" && entry.role === "system") 
  ) return "system";

  if (
    /\[Negotiation\]/i.test(t) ||
    /^\s*From\b/i.test(t) ||
    /^\s*To\b/i.test(t)
  ) return "negotiation";

  return "negotiation";
}

// Power name to portrait ID mapping
const POWER_PORTRAITS = {
  england: 1, france: 2, germany: 3, italy: 4,
  austria: 5, russia: 6, turkey: 7
};

function getPowerFromSender(sender) {
  if (!sender) return null;
  const s = sender.toLowerCase();
  for (const power of Object.keys(POWER_PORTRAITS)) {
    if (s.includes(power)) return power;
  }
  return null;
}

// 从早期初始化脚本或 sessionStorage 读取选择的头像映射
let selectedPortraits = [];
if (window.__EARLY_INIT__ && window.__EARLY_INIT__.portraits) {
  selectedPortraits = window.__EARLY_INIT__.portraits;
} else {
  try {
    const stored = sessionStorage.getItem('selectedPortraits');
    if (stored) selectedPortraits = JSON.parse(stored);
  } catch (e) {}
}

// 从gameConfig读取power_names顺序
let powerNamesOrder = null;
try {
  const gameConfigStr = sessionStorage.getItem('gameConfig');
  if (gameConfigStr) {
    const gameConfig = JSON.parse(gameConfigStr);
    if (gameConfig.power_names && Array.isArray(gameConfig.power_names)) {
      powerNamesOrder = gameConfig.power_names;
    }
  }
} catch (e) {}

// 根据power名称获取对应的头像
function getPortraitSrcByPower(powerName) {
  if (!powerName) return `/static/portraits/portrait_1.png`;
  
  // 如果有power_names顺序和selectedPortraits，使用它们
  if (powerNamesOrder && selectedPortraits.length > 0) {
    const powerIndex = powerNamesOrder.findIndex(p => 
      p.toUpperCase() === powerName.toUpperCase() || 
      p.toLowerCase().includes(powerName.toLowerCase()) ||
      powerName.toLowerCase().includes(p.toLowerCase())
    );
    if (powerIndex !== -1 && powerIndex < selectedPortraits.length) {
      const portraitId = selectedPortraits[powerIndex];
      return `/static/portraits/portrait_${portraitId}.png`;
    }
  }
  
  // 回退到旧的硬编码映射
  const powerLower = powerName.toLowerCase();
  const portraitId = POWER_PORTRAITS[powerLower] || 1;
  return `/static/portraits/portrait_${portraitId}.png`;
}

function getPortraitSrc(playerId) {
  // 防止 NaN，确保 playerId 是有效数字
  const validId = (typeof playerId === 'number' && !isNaN(playerId)) ? playerId : 1;
  
  // 如果有选择的头像，使用映射（diplomacy的playerId从1开始）
  if (selectedPortraits.length >= validId && validId > 0) {
    const portraitId = selectedPortraits[validId - 1];
    return `/static/portraits/portrait_${portraitId}.png`;
  }
  
  // 否则使用默认映射
  const id = ((validId - 1) % 15) + 1;
  return `/static/portraits/portrait_${id}.png`;
}

function renderOneLog(entry) {
  const t = logText(entry);
  const type = getLogType(entry);
  
  // For phase markers, keep simple style
  if (type === "phase") {
    const el = document.createElement("div");
    el.classList.add("log-entry", "phase");
    el.textContent = t;
    return el;
  }
  
  // For system logs, keep simple style but with icon
  if (type === "system") {
    const el = document.createElement("div");
    el.classList.add("log-entry", "log-system");
    el.textContent = t;
    return el;
  }
  
  // For orders and negotiation, use chat bubble style
  const el = document.createElement("div");
  el.classList.add("chat-message");
  // 给不同类型的气泡打标，便于 CSS 区分背景样式
  // 目前只有 orders / negotiation 会进入这个分支，但这里写得更通用
  if (type) {
    el.classList.add(`chat-${type}`);
  }
  
  // Extract sender info
  let sender = "System";
  let power = null;
  let content = t;
  
  if (entry && typeof entry === "object") {
    sender = entry.sender || entry.name || entry.from || "System";
    content = entry.content || entry.text || t;
  }
  
  // Try to extract power from content patterns like "[ENGLAND]" or "ENGLAND orders:"
  const powerMatch = t.match(/^\[?([A-Z_]+)\]?\s*(orders:|:)?/i);
  if (powerMatch) {
    const extracted = powerMatch[1].toLowerCase().replace(/_/g, '');
    for (const p of Object.keys(POWER_PORTRAITS)) {
      if (extracted.includes(p)) {
        power = p;
        sender = p.charAt(0).toUpperCase() + p.slice(1);
        break;
      }
    }
  }
  
  if (!power) {
    power = getPowerFromSender(sender);
  }
  
  // Avatar
  let avatarHtml;
  if (power) {
    // 尝试从sender或power中提取准确的power名称
    let powerNameForPortrait = power;
    if (powerNamesOrder) {
      // 首先尝试精确匹配power（小写）
      let matchedPower = powerNamesOrder.find(p => 
        p.toLowerCase() === power.toLowerCase()
      );
      
      // 如果没找到，尝试从sender中匹配
      if (!matchedPower && sender && typeof sender === 'string') {
        const senderUpper = sender.toUpperCase();
        matchedPower = powerNamesOrder.find(p => 
          p.toUpperCase() === senderUpper || 
          senderUpper.includes(p.toUpperCase()) ||
          p.toUpperCase().includes(senderUpper)
        );
      }
      
      if (matchedPower) {
        powerNameForPortrait = matchedPower;
      }
    }
    const portraitSrc = getPortraitSrcByPower(powerNameForPortrait);
    avatarHtml = `<div class="chat-avatar ${power}"><img src="${portraitSrc}" alt="${sender}"></div>`;
  } else if (type === "orders") {
    avatarHtml = `<div class="chat-avatar system">📜</div>`;
  } else {
    avatarHtml = `<div class="chat-avatar system">💬</div>`;
  }
  
  const senderClass = power || (type === "orders" ? "system" : "");
  
  el.innerHTML = `
    ${avatarHtml}
    <div class="chat-bubble">
      <div class="chat-header">
        <span class="chat-sender ${senderClass}">${escapeHtml(sender)}</span>
      </div>
      <div class="chat-content">${escapeHtml(content)}</div>
    </div>
  `;
  
  return el;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

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

// Filters / Phase select
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

// History
async function fetchHistoryList() {
  try {
    const response = await fetch("/api/history");
    if (!response.ok) {
      // History not available (e.g., game not started yet)
      return;
    }
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
      const phase = item.phase || "Init";
      const round = (item.round !== undefined && item.round !== null) ? item.round : 0;
      const kind = item.kind ? `(${item.kind})` : "";
      option.textContent = `${phase} R${round} ${kind}`.trim();
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

// WebSocket
function connect() {
  socket = new WebSocket(wsUrl);

  socket.onopen = function () {
    console.log("[WebSocket] Connection established");
    fetchHistoryList();
    
    // 使用早期初始化的配置（首次启动）
    if (window.__EARLY_INIT__ && window.__EARLY_INIT__.hasGameConfig && window.__EARLY_INIT__.config) {
      console.log('Found game config from early init, starting game automatically...');
      
      const config = window.__EARLY_INIT__.config;
      
      // 清除 sessionStorage 中的 gameConfig
      sessionStorage.removeItem('gameConfig');
      // 设置游戏正在运行标记（用于刷新后重连）
      sessionStorage.setItem('gameRunning', 'true');
      // 清除早期初始化标记，防止重复启动
      window.__EARLY_INIT__.hasGameConfig = false;
      
      // 启动游戏
      fetch('/api/start-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      }).then(resp => {
        if (resp.ok) {
          console.log('Game started successfully');
        } else {
          console.error('Failed to start game');
        }
      });
    }
    // 刷新后重连（游戏已在运行）
    else if (window.__EARLY_INIT__ && window.__EARLY_INIT__.isGameRunning) {
      console.log('Game was running, reconnecting...');
      // 不需要启动游戏，只需要等待服务器发送状态
      window.__EARLY_INIT__.isGameRunning = false;
    }
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

    if (message.type === "mode_info") {
      // 处理模式信息，允许 game 和 mode 为 null（游戏未启动时）
      // 不进行任何操作，仅静默接收，避免打印警告
      return;
    }

    if (isMessageLike(message)) {
      const human_power = getField(latestState, "human_power", "");
      const isParticipate = !!human_power;

      const entry = messageToLogEntry(message);
      if (isParticipate) {
        appendToParticipateLogs(entry);
      } else {
        appendObsEntry(entry); // 观战模式也显示聊天
      }
      if (!isViewingHistory) renderState(latestState, false);
      return;
    }

    console.log("[WebSocket] Unknown message:", message);
  };

  socket.onclose = function () {
    setTimeout(connect, 1500);
  };

  socket.onerror = function (error) {
    console.log("[WebSocket] Error:", error.message);
  };
}

function updateState(incoming) {
  const prevMap = getField(latestState, "map_svg", "");
  const nextMap = getField(incoming, "map_svg", "");

  // Merge logs incrementally
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

  // Accumulate obs_log_entry
  const incomingObs = getField(incoming, "obs_log_entry", undefined);
  if (incomingObs !== undefined) {
    latestState.obs_log_entry = incomingObs;
    appendObsEntry(incomingObs);
  }

  // Merge other fields
  for (const key in incoming) {
    if (key === "logs" || key === "obs_log_entry") continue;
    latestState[key] = incoming[key];
  }
  latestState.meta = latestState.meta || {};

  if (nextMap && nextMap !== prevMap) fetchHistoryList();

  if (!isViewingHistory) renderState(latestState, false);
}

// Rendering
function renderState(data, isHistory = false) {
  currentRenderData = data;

  const meta = data.meta || {};

  const map_svg = data.map_svg ?? meta.map_svg ?? "";
  const human_power = data.human_power ?? meta.human_power ?? "";
  const logsRaw = data.logs ?? meta.logs ?? [];
  const obsEntry = data.obs_log_entry ?? meta.obs_log_entry ?? null;

  const isParticipate = !!human_power;
  let logs;

  if (isParticipate) {
    logs = normalizeLogs(logsRaw);
  } else {
    const snapObs = normalizeObsEntry(obsEntry);
    logs = obsBuffer.slice();
    for (const e of snapObs) {
      const sig = obsSig(e).trim();
      if (!sig) continue;
      if (obsSeen.has(sig)) continue;
      logs.push(e);
    }
  }

  logs = ensurePhaseLog(logs, data);

  // Header/status
  const phaseEl = document.getElementById("phase");
  const roundEl = document.getElementById("round");
  const statusEl = document.getElementById("status");

  const phase = getField(data, "phase", "");
  const round = getField(data, "round", undefined);
  const status = getField(data, "status", "");

  if (phaseEl && phase) phaseEl.textContent = `Phase: ${phase}`;
  if (roundEl && round !== undefined) roundEl.textContent = `Round: ${round}`;
  if (statusEl && status) statusEl.textContent = `Status: ${status}${isHistory ? " (History View)" : ""}`;

  // Map
  if (mapContainer && map_svg) {
    // Remove placeholder if exists
    const placeholder = document.getElementById("map-placeholder");
    if (placeholder) placeholder.remove();
    
    if (mapContainer.innerHTML !== map_svg) mapContainer.innerHTML = map_svg;
  }

  // Input area
  if (inputArea) {
    inputArea.style.display = (!isHistory && isParticipate) ? "block" : "none";
  }

  // Logs
  if (!logsContainer) return;

  logsContainer.innerHTML = "";
  logs.filter(wantByFilter).forEach((entry) => {
    logsContainer.appendChild(renderOneLog(entry));
  });
  logsContainer.scrollTop = logsContainer.scrollHeight;

  // Update back/exit button based on game status
  updateBackExitButton(status);
}

function updateBackExitButton(gameStatus) {
  if (!backExitButton) return;
  const goHome = () => { window.location.href = '/'; };
  
  if (gameStatus === 'running') {
    backExitButton.textContent = 'Exit';
    backExitButton.title = 'Exit Game';
    backExitButton.href = '#';
    backExitButton.onclick = async (e) => {
      e.preventDefault();
      try {
        await fetch('/api/stop-game', { method: 'POST' });
      } catch (error) {
        console.error('Error stopping game:', error);
      }
      sessionStorage.removeItem('gameRunning');  // 清除游戏运行标记
      goHome();
    };
  } else {
    // 游戏不在运行状态时，清除 gameRunning 标记
    if (gameStatus === 'stopped' || gameStatus === 'finished' || gameStatus === 'waiting') {
      sessionStorage.removeItem('gameRunning');
    }
    backExitButton.textContent = '← Back';
    backExitButton.title = 'Back to Home';
    backExitButton.href = '/';
    backExitButton.onclick = (e) => { e.preventDefault(); goHome(); };
  }
}

// Human input
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

// Start
connect();
