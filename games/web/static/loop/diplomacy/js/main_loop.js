// Diplomacy observe auto-loop (observer-only)

window.addEventListener('beforeunload', () => {
  const keysToKeep = ['gameConfig', 'selectedPortraits', 'gameLanguage', 'gameRunning'];
  Object.keys(sessionStorage).forEach((key) => {
    if (!keysToKeep.includes(key)) {
      sessionStorage.removeItem(key);
    }
  });
});

window.addEventListener('pageshow', (event) => {
  if (event.persisted) {
    window.location.reload();
  }
});

const BACK_HOME_URL = '/static/loop/index.html';
const gameLanguage = sessionStorage.getItem('gameLanguage') || 'en';
document.body.classList.add(`lang-${gameLanguage}`);

const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

const ALLOWED_HISTORY_KINDS = new Set(['orders', 'order', 'result', 'init']);

let socket = null;
let latestState = {};
let isViewingHistory = false;
let knownHistorySig = [];
let currentFilter = 'all';
let currentRenderData = {};
let currentPrompt = null;
let currentInputAgentId = null;
let obsBuffer = [];
let obsSeen = new Set();
let restartScheduled = false;
let lastStatus = null;
let isStarting = false;
let baseConfig = null;
let selectedPortraits = [];
let powerNamesOrder = null;

function deepClone(obj) {
  return obj ? JSON.parse(JSON.stringify(obj)) : obj;
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function loadBaseConfig() {
  let cfg = null;
  if (window.__EARLY_INIT__ && window.__EARLY_INIT__.config) {
    cfg = deepClone(window.__EARLY_INIT__.config);
  } else {
    try {
      const gameConfigStr = sessionStorage.getItem('gameConfig');
      if (gameConfigStr) cfg = JSON.parse(gameConfigStr);
    } catch (e) {}
  }
  cfg = cfg || {};
  cfg.game = 'diplomacy';
  cfg.mode = 'observe';
  return cfg;
}

function persistSession(cfg) {
  sessionStorage.setItem('gameConfig', JSON.stringify(cfg));
  sessionStorage.setItem('selectedPortraits', JSON.stringify(cfg.selected_portrait_ids || []));
  sessionStorage.setItem('gameLanguage', cfg.language || 'en');
  sessionStorage.setItem('gameRunning', 'true');
}

function buildNextConfig(reason = 'loop') {
  const cfg = deepClone(baseConfig || {});
  cfg.game = 'diplomacy';
  cfg.mode = 'observe';
  cfg.language = cfg.language || 'en';

  let powerNames = [];
  if (cfg.power_names && Array.isArray(cfg.power_names) && cfg.power_names.length) {
    powerNames = cfg.power_names.slice();
  } else {
    powerNames = ['England', 'France', 'Germany', 'Italy', 'Austria', 'Russia', 'Turkey'];
  }

  if (!selectedPortraits.length && cfg.selected_portrait_ids) {
    selectedPortraits = cfg.selected_portrait_ids.slice();
  }
  if (!powerNamesOrder && cfg.power_names) {
    powerNamesOrder = cfg.power_names.slice();
  }

  const portraits = (cfg.selected_portrait_ids && cfg.selected_portrait_ids.length)
    ? cfg.selected_portrait_ids.slice()
    : selectedPortraits.slice();

  const fallbackPortraits = portraits.length
    ? portraits
    : Array.from({ length: 15 }, (_, i) => i + 1);

  const pairs = powerNames.map((p, idx) => ({
    power: p,
    portrait: fallbackPortraits[idx % fallbackPortraits.length],
  }));

  shuffleInPlace(pairs);
  cfg.power_names = pairs.map((x) => x.power);
  cfg.selected_portrait_ids = pairs.map((x) => x.portrait);

  persistSession(cfg);
  return cfg;
}

baseConfig = loadBaseConfig();

if (window.__EARLY_INIT__ && window.__EARLY_INIT__.portraits) {
  selectedPortraits = window.__EARLY_INIT__.portraits;
} else {
  try {
    const stored = sessionStorage.getItem('selectedPortraits');
    if (stored) selectedPortraits = JSON.parse(stored);
  } catch (e) {}
}

try {
  const gameConfigStr = sessionStorage.getItem('gameConfig');
  if (gameConfigStr) {
    const gameConfig = JSON.parse(gameConfigStr);
    if (gameConfig.power_names && Array.isArray(gameConfig.power_names)) {
      powerNamesOrder = gameConfig.power_names;
    }
  }
} catch (e) {}

function obsSig(x) {
  if (typeof x === 'string') return x;
  if (x && typeof x === 'object') return x.content ?? x.text ?? JSON.stringify(x);
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

const phaseSelect = document.getElementById('phase-select');
const logFilter = document.getElementById('log-filter');
const logsContainer = document.getElementById('logs-container');
const mapContainer = document.getElementById('map-container');
const inputArea = document.getElementById('user-input-area');
const sendBtn = document.getElementById('submit-input');
const userInput = document.getElementById('user-input');
const promptEl = document.getElementById('input-prompt');
const backExitButton = document.getElementById('back-exit-button');

function getField(state, key, fallback = undefined) {
  if (!state) return fallback;
  if (state[key] !== undefined && state[key] !== null) return state[key];
  if (state.meta && state.meta[key] !== undefined && state.meta[key] !== null) return state.meta[key];
  return fallback;
}

function normalizeHistorySnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') return { meta: {} };
  const state = { ...snapshot };
  state.meta = snapshot.meta && typeof snapshot.meta === 'object' ? snapshot.meta : {};
  const keysToLift = ['map_svg', 'logs', 'obs_log_entry', 'human_power', 'status', 'phase', 'round', 'kind', 'timestamp'];
  keysToLift.forEach((k) => {
    if (state[k] === undefined && state.meta[k] !== undefined) state[k] = state.meta[k];
  });
  return state;
}

function isMessageLike(obj) {
  if (!obj) return false;
  if (typeof obj === 'string') return true;
  if (typeof obj !== 'object') return false;
  if (obj.type === 'message') return true;
  if (obj.content !== undefined || obj.text !== undefined) return true;
  if (obj.name !== undefined || obj.sender !== undefined) return true;
  return false;
}

function messageToLogEntry(msg) {
  if (typeof msg === 'string') return msg;
  const content = msg.content ?? msg.text ?? msg.message ?? msg.data ?? JSON.stringify(msg);
  const sender = msg.sender ?? msg.name ?? msg.from ?? 'System';
  const role = msg.role ?? 'assistant';
  const ts = msg.timestamp ?? null;
  const line = String(content);
  return { message_type: 'chat_message', sender, role, timestamp: ts, content: line };
}

function appendToParticipateLogs(entry) {
  const prevLogs = Array.isArray(latestState.logs) ? latestState.logs : normalizeLogs(latestState.logs);
  const sig = (x) => (typeof x === 'string' ? x : JSON.stringify(x));
  const last = prevLogs.length ? prevLogs[prevLogs.length - 1] : null;
  if (!last || sig(last) !== sig(entry)) {
    prevLogs.push(entry);
  }
  latestState.logs = prevLogs;
}

function normalizeGameStateMessage(message) {
  const state = {};
  if (message && message.data && typeof message.data === 'object') {
    Object.assign(state, message.data);
  } else if (message && typeof message === 'object') {
    Object.assign(state, message);
  }
  const meta =
    (state.meta && typeof state.meta === 'object' && state.meta) ||
    (message.meta && typeof message.meta === 'object' && message.meta) ||
    (message.data && message.data.meta && typeof message.data.meta === 'object' && message.data.meta) ||
    {};
  state.meta = meta;
  delete state.type;
  delete state.data;
  return state;
}

function normalizeLogs(logs) {
  if (Array.isArray(logs)) return logs;
  if (logs && typeof logs === 'object') {
    const out = [];
    for (const [agentName, arr] of Object.entries(logs)) {
      if (!Array.isArray(arr)) continue;
      for (const item of arr) {
        if (item && typeof item === 'object') {
          out.push({ sender: agentName.replace(/^Agent_/, ''), ...item });
        } else {
          out.push(String(item));
        }
      }
    }
    return out;
  }
  if (typeof logs === 'string') return [logs];
  return [];
}

function logText(entry) {
  if (typeof entry === 'string') return entry;
  if (entry && typeof entry === 'object') return entry.content ?? entry.text ?? JSON.stringify(entry);
  return String(entry);
}

function normalizeObsEntry(obsEntry) {
  if (!obsEntry) return [];
  if (Array.isArray(obsEntry)) return obsEntry;
  if (typeof obsEntry === 'string') {
    return obsEntry
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((line) => ({ message_type: 'observer_log', content: line }));
  }
  if (typeof obsEntry === 'object') return [obsEntry];
  return [{ message_type: 'observer_log', content: String(obsEntry) }];
}

function getLogType(entry) {
  const t = logText(entry);
  if (/---\s*Phase:/i.test(t)) return 'phase';
  const ORDER_LINE = /^(\[[A-Z_]+\]\s*)?([A-Z][A-Z_]+)\s+orders\s*:/;
  if (ORDER_LINE.test(t) || /^\[Moderator\]/.test(t)) return 'orders';
  if (/^\[System\]/.test(t) || /---COUNTRY_SYSTEM---|---FEW_SHOT---|---PHASE_INSTRUCTIONS---|---SITUATION---|Your power is/i.test(t) || (entry && typeof entry === 'object' && entry.role === 'system')) return 'system';
  if (/\[Negotiation\]/i.test(t) || /^\s*From\b/i.test(t) || /^\s*To\b/i.test(t)) return 'negotiation';
  return 'negotiation';
}

const POWER_PORTRAITS = { england: 1, france: 2, germany: 3, italy: 4, austria: 5, russia: 6, turkey: 7 };

function getPowerFromSender(sender) {
  if (!sender) return null;
  const s = sender.toLowerCase();
  for (const power of Object.keys(POWER_PORTRAITS)) {
    if (s.includes(power)) return power;
  }
  return null;
}

function getPortraitSrcByPower(powerName) {
  if (!powerName) return '/static/portraits/portrait_1.png';
  if (powerNamesOrder && selectedPortraits.length > 0) {
    const powerIndex = powerNamesOrder.findIndex(
      (p) => p.toUpperCase() === powerName.toUpperCase() || p.toLowerCase().includes(powerName.toLowerCase()) || powerName.toLowerCase().includes(p.toLowerCase()),
    );
    if (powerIndex !== -1 && powerIndex < selectedPortraits.length) {
      const portraitId = selectedPortraits[powerIndex];
      return `/static/portraits/portrait_${portraitId}.png`;
    }
  }
  const powerLower = powerName.toLowerCase();
  const portraitId = POWER_PORTRAITS[powerLower] || 1;
  return `/static/portraits/portrait_${portraitId}.png`;
}

function getPortraitSrc(playerId) {
  const validId = typeof playerId === 'number' && !isNaN(playerId) ? playerId : 1;
  if (selectedPortraits.length >= validId && validId > 0) {
    const portraitId = selectedPortraits[validId - 1];
    return `/static/portraits/portrait_${portraitId}.png`;
  }
  const id = ((validId - 1) % 15) + 1;
  return `/static/portraits/portrait_${id}.png`;
}

function renderOneLog(entry) {
  const t = logText(entry);
  const type = getLogType(entry);
  if (type === 'phase') {
    const el = document.createElement('div');
    el.classList.add('log-entry', 'phase');
    el.textContent = t;
    return el;
  }
  if (type === 'system') {
    const el = document.createElement('div');
    el.classList.add('log-entry', 'log-system');
    el.textContent = t;
    return el;
  }
  const el = document.createElement('div');
  el.classList.add('chat-message');
  if (type) {
    el.classList.add(`chat-${type}`);
  }
  let sender = 'System';
  let power = null;
  let content = t;
  if (entry && typeof entry === 'object') {
    sender = entry.sender || entry.name || entry.from || 'System';
    content = entry.content || entry.text || t;
  }
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
  let avatarHtml;
  if (power) {
    let powerNameForPortrait = power;
    if (powerNamesOrder) {
      let matchedPower = powerNamesOrder.find((p) => p.toLowerCase() === power.toLowerCase());
      if (!matchedPower && sender && typeof sender === 'string') {
        const senderUpper = sender.toUpperCase();
        matchedPower = powerNamesOrder.find((p) => p.toUpperCase() === senderUpper || senderUpper.includes(p.toUpperCase()) || p.toUpperCase().includes(senderUpper));
      }
      if (matchedPower) {
        powerNameForPortrait = matchedPower;
      }
    }
    const portraitSrc = getPortraitSrcByPower(powerNameForPortrait);
    avatarHtml = `<div class="chat-avatar ${power}"><img src="${portraitSrc}" alt="${sender}"></div>`;
  } else if (type === 'orders') {
    avatarHtml = '<div class="chat-avatar system">📜</div>';
  } else {
    avatarHtml = '<div class="chat-avatar system">💬</div>';
  }
  const senderClass = power || (type === 'orders' ? 'system' : '');
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
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function wantByFilter(entry) {
  const type = getLogType(entry);
  if (type === 'phase') return true;
  if (currentFilter === 'all') return true;
  if (currentFilter === 'negotiation') return type === 'negotiation';
  if (currentFilter === 'orders') return type === 'orders';
  if (currentFilter === 'system') return type === 'system';
  return true;
}

function ensurePhaseLog(logs, state) {
  const hasPhase = logs.some((e) => getLogType(e) === 'phase');
  if (hasPhase) return logs;
  const phase = getField(state, 'phase', null);
  if (!phase) return logs;
  const round = getField(state, 'round', undefined);
  const line = `--- Phase: ${phase}${round !== undefined ? ` (Round ${round})` : ''} ---`;
  return [line, ...logs];
}

if (logFilter) {
  logFilter.addEventListener('change', function () {
    currentFilter = this.value;
    renderState(currentRenderData, isViewingHistory);
  });
}

if (phaseSelect) {
  phaseSelect.addEventListener('change', function () {
    const value = this.value;
    if (value === 'latest') {
      isViewingHistory = false;
      renderState(latestState, false);
    } else {
      isViewingHistory = true;
      fetchHistoryState(parseInt(value, 10) - 1);
    }
  });
}

async function fetchHistoryList() {
  try {
    const response = await fetch('/api/history');
    if (!response.ok) {
      return;
    }
    let history = await response.json();
    if (ALLOWED_HISTORY_KINDS) {
      history = history.filter((h) => ALLOWED_HISTORY_KINDS.has(h.kind));
    }
    const newSig = history.map((h) => `${h.index}|${h.phase}|${h.round}|${h.kind || ''}`);
    if (JSON.stringify(newSig) === JSON.stringify(knownHistorySig)) return;
    knownHistorySig = newSig;
    if (!phaseSelect) return;
    const currentVal = phaseSelect.value || 'latest';
    phaseSelect.innerHTML = '<option value="latest">Latest</option>';
    history.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.index;
      const phase = item.phase || 'Init';
      const round = item.round !== undefined && item.round !== null ? item.round : 0;
      const kind = item.kind ? `(${item.kind})` : '';
      option.textContent = `${phase} R${round} ${kind}`.trim();
      phaseSelect.appendChild(option);
    });
    if (currentVal !== 'latest' && history.some((h) => String(h.index) === currentVal)) {
      phaseSelect.value = currentVal;
    } else {
      phaseSelect.value = 'latest';
    }
  } catch (e) {
    console.error('Failed to fetch history list', e);
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
    console.error('Failed to fetch history state', e);
  }
}

async function startGameWithConfig(cfg) {
  try {
    isStarting = true;
    await fetch('/api/start-game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    sessionStorage.setItem('gameRunning', 'true');
  } catch (error) {
    console.error('Error starting game:', error);
  } finally {
    // Running status will clear this; keep a safety clear to avoid deadlock.
    setTimeout(() => { isStarting = false; }, 5000);
  }
}

function queueRestart(reason) {
  if (restartScheduled || isStarting) return;
  restartScheduled = true;
  setTimeout(() => {
    const nextCfg = buildNextConfig(reason);
    startGameWithConfig(nextCfg).finally(() => {
      restartScheduled = false;
    });
  }, 150);
}

function refreshLeaderboard() {
  fetch('/api/leaderboard/diplomacy').catch(() => {});
}

function connect() {
  socket = new WebSocket(wsUrl);
  socket.onopen = function () {
    console.log('[WebSocket] Connection established');
    fetchHistoryList();
    if (window.__EARLY_INIT__ && window.__EARLY_INIT__.hasGameConfig && window.__EARLY_INIT__.config) {
      const cfg = buildNextConfig('early');
      window.__EARLY_INIT__.hasGameConfig = false;
      startGameWithConfig(cfg);
      return;
    }
    if (window.__EARLY_INIT__ && window.__EARLY_INIT__.isGameRunning) {
      window.__EARLY_INIT__.isGameRunning = false;
      return;
    }
    queueRestart('connect');
  };

  socket.onmessage = function (event) {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (e) {
      console.error('Invalid JSON:', event.data);
      return;
    }
    if (message.type === 'game_state') {
      const state = normalizeGameStateMessage(message);
      updateState(state);
      return;
    }
    if (message.type === 'input_request' || message.type === 'user_input_request') {
      handleInputRequest(message);
      return;
    }
    if (message.type === 'mode_info') {
      return;
    }
    if (isMessageLike(message)) {
      const human_power = getField(latestState, 'human_power', '');
      const isParticipate = !!human_power;
      const entry = messageToLogEntry(message);
      if (isParticipate) {
        appendToParticipateLogs(entry);
      } else {
        appendObsEntry(entry);
      }
      if (!isViewingHistory) renderState(latestState, false);
      return;
    }
    console.log('[WebSocket] Unknown message:', message);
  };

  socket.onclose = function () {
    setTimeout(connect, 1500);
  };

  socket.onerror = function (error) {
    console.log('[WebSocket] Error:', error.message);
  };
}

function updateState(incoming) {
  const prevMap = getField(latestState, 'map_svg', '');
  const nextMap = getField(incoming, 'map_svg', '');
  const incomingLogsRaw = getField(incoming, 'logs', undefined);
  if (incomingLogsRaw !== undefined) {
    const incomingLogs = normalizeLogs(incomingLogsRaw);
    const prevLogs = Array.isArray(latestState.logs) ? latestState.logs : normalizeLogs(latestState.logs);
    for (const item of incomingLogs) {
      const last = prevLogs.length ? prevLogs[prevLogs.length - 1] : null;
      const sig = (x) => (typeof x === 'string' ? x : JSON.stringify(x));
      if (!last || sig(last) !== sig(item)) prevLogs.push(item);
    }
    latestState.logs = prevLogs;
  }
  const incomingObs = getField(incoming, 'obs_log_entry', undefined);
  if (incomingObs !== undefined) {
    latestState.obs_log_entry = incomingObs;
    appendObsEntry(incomingObs);
  }
  for (const key in incoming) {
    if (key === 'logs' || key === 'obs_log_entry') continue;
    latestState[key] = incoming[key];
  }
  latestState.meta = latestState.meta || {};
  if (nextMap && nextMap !== prevMap) fetchHistoryList();
  if (!isViewingHistory) renderState(latestState, false);
}

function renderState(data, isHistory = false) {
  currentRenderData = data;
  const meta = data.meta || {};
  const map_svg = data.map_svg ?? meta.map_svg ?? '';
  const human_power = data.human_power ?? meta.human_power ?? '';
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
  const phaseEl = document.getElementById('phase');
  const roundEl = document.getElementById('round');
  const statusEl = document.getElementById('status');
  const phase = getField(data, 'phase', '');
  const round = getField(data, 'round', undefined);
  const status = getField(data, 'status', '');
  if (phaseEl && phase) phaseEl.textContent = `Phase: ${phase}`;
  if (roundEl && round !== undefined) roundEl.textContent = `Round: ${round}`;
  if (statusEl && status) statusEl.textContent = `Status: ${status}${isHistory ? ' (History View)' : ''}`;
  if (mapContainer && map_svg) {
    const placeholder = document.getElementById('map-placeholder');
    if (placeholder) placeholder.remove();
    if (mapContainer.innerHTML !== map_svg) mapContainer.innerHTML = map_svg;
  }
  if (inputArea) {
    inputArea.style.display = !isHistory && isParticipate ? 'block' : 'none';
  }
  if (!logsContainer) return;
  logsContainer.innerHTML = '';
  logs.filter(wantByFilter).forEach((entry) => {
    logsContainer.appendChild(renderOneLog(entry));
  });
  logsContainer.scrollTop = logsContainer.scrollHeight;
  updateBackExitButton(status);
  handleStatusChange(status);
}

function handleStatusChange(status) {
  if (!status) return;
  if (status === 'running') {
    restartScheduled = false;
    isStarting = false;
  }
  const isEnd = ['stopped', 'finished', 'waiting'].includes(status);
  // Restart on any end status if nothing is starting/scheduled.
  const shouldRestart = isEnd && !isStarting && !restartScheduled;
  if (shouldRestart) {
    queueRestart(`status:${status}`);
    refreshLeaderboard();
  }
  lastStatus = status;
}

function updateBackExitButton(gameStatus) {
  if (!backExitButton) return;
  const goHome = () => {
    window.location.href = BACK_HOME_URL;
  };
  backExitButton.textContent = '← Back';
  backExitButton.title = 'Back to Home';
  backExitButton.href = BACK_HOME_URL;
  backExitButton.onclick = (e) => {
    e.preventDefault();
    goHome();
  };
}

if (sendBtn && userInput) {
  sendBtn.addEventListener('click', sendInput);
  userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendInput();
    }
  });
}

function sendInput() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  const text = (userInput.value || '').trim();
  if (!text) return;
  const payload = { type: 'user_input', content: text };
  if (currentInputAgentId !== null && currentInputAgentId !== undefined) {
    payload.agent_id = currentInputAgentId;
  }
  socket.send(JSON.stringify(payload));
  userInput.value = '';
  currentPrompt = null;
  currentInputAgentId = null;
  if (promptEl) {
    promptEl.innerText = '';
    promptEl.style.display = 'none';
  }
  if (!isViewingHistory) renderState(latestState, false);
}

function handleInputRequest(message) {
  const req = message && message.data && typeof message.data === 'object' ? message.data : message;
  currentPrompt = req.prompt || 'Please enter your orders or message:';
  currentInputAgentId = req.agent_id ?? null;
  isViewingHistory = false;
  if (phaseSelect) phaseSelect.value = 'latest';
  if (inputArea) inputArea.style.display = 'block';
  if (promptEl) {
    promptEl.innerText = currentPrompt;
    promptEl.style.display = 'block';
  }
  renderState(latestState, false);
  if (userInput) {
    userInput.value = '';
    userInput.focus();
  }
}

connect();
