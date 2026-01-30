// Auto-loop observe mode for Avalon

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
const wsClient = new WebSocketClient();
const messagesContainer = document.getElementById('messages-container');
const phaseDisplay = document.getElementById('phase-display');
const missionDisplay = document.getElementById('mission-display');
const roundDisplay = document.getElementById('round-display');
const statusDisplay = document.getElementById('status-display');
const gameSetup = document.getElementById('game-setup');
const startGameBtn = document.getElementById('start-game-btn');
const numPlayersSelect = document.getElementById('num-players');
const languageSelect = document.getElementById('language');
const tablePlayers = document.getElementById('table-players');
const backButton = document.getElementById('back-exit-button');
const missionProgressEl = document.getElementById('mission-progress');

let messageCount = 0;
let gameStarted = false;
let numPlayers = 5;
let restartScheduled = false;
let isStarting = false;
let lastStatus = null;
let currentMissionId = 1;
let currentRoundId = null;

let missionNodes = [];
let missionData = Array.from({ length: 5 }, () => ({
  status: 'pending',
  resultDetail: 'Pending',
}));

const gameLanguage = sessionStorage.getItem('gameLanguage') || 'en';
document.body.classList.add(`lang-${gameLanguage}`);

let selectedPortraits = [];
if (window.__EARLY_INIT__ && window.__EARLY_INIT__.portraits) {
  selectedPortraits = window.__EARLY_INIT__.portraits;
} else {
  try {
    const stored = sessionStorage.getItem('selectedPortraits');
    if (stored) selectedPortraits = JSON.parse(stored);
  } catch (e) {}
}

let agentConfigs = {};
let baseConfig = null;

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
  cfg.game = 'avalon';
  cfg.mode = 'observe';
  if (cfg.num_players) {
    numPlayers = typeof cfg.num_players === 'number' ? cfg.num_players : parseInt(cfg.num_players, 10);
  }
  if (cfg.agent_configs) {
    agentConfigs = cfg.agent_configs;
  }
  return cfg;
}

baseConfig = loadBaseConfig();

if (!selectedPortraits.length && baseConfig.selected_portrait_ids) {
  selectedPortraits = baseConfig.selected_portrait_ids.slice();
}

function persistSession(cfg) {
  sessionStorage.setItem('gameConfig', JSON.stringify(cfg));
  sessionStorage.setItem('selectedPortraits', JSON.stringify(cfg.selected_portrait_ids || []));
  sessionStorage.setItem('gameLanguage', cfg.language || 'en');
  sessionStorage.setItem('gameRunning', 'true');
}

function buildNextConfig(reason = 'loop') {
  const cfg = deepClone(baseConfig);
  cfg.game = 'avalon';
  cfg.mode = 'observe';
  cfg.num_players = cfg.num_players || numPlayers || 5;
  cfg.language = cfg.language || languageSelect?.value || 'en';
  // Always re-randomize portraits from full index pool each game start.
  const defaultIds = Array.from({ length: 15 }, (_, i) => i + 1);
  shuffleInPlace(defaultIds);
  const portraits = defaultIds.slice(0, cfg.num_players);
  cfg.selected_portrait_ids = portraits;
  selectedPortraits = portraits.slice();

  if (Array.isArray(cfg.preset_roles)) {
    cfg.preset_roles = shuffleInPlace(cfg.preset_roles.slice());
  }

  persistSession(cfg);
  return cfg;
}

function getPortraitSrc(playerId) {
  const validId = typeof playerId === 'number' && !isNaN(playerId) ? playerId : 0;
  if (selectedPortraits && selectedPortraits.length > validId) {
    const portraitId = selectedPortraits[validId];
    return `/static/portraits/portrait_${portraitId}.png`;
  }
  const id = (validId % 15) + 1;
  return `/static/portraits/portrait_${id}.png`;
}

function getModelName(playerId) {
  const validId = typeof playerId === 'number' && !isNaN(playerId) ? playerId : 0;
  let portraitId = null;
  if (selectedPortraits && selectedPortraits.length > validId) {
    portraitId = selectedPortraits[validId];
  } else {
    portraitId = (validId % 15) + 1;
  }
  if (portraitId && agentConfigs) {
    const config = agentConfigs[portraitId] || agentConfigs[String(portraitId)];
    if (config && config.base_model) {
      return config.base_model;
    }
    if (config && config.model && config.model.model_name) {
      return config.model.model_name;
    }
  }
  return `Unknown`;
}

function polarPositions(count, radiusX, radiusY) {
  return Array.from({ length: count }).map((_, i) => {
    const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
    return { x: radiusX * Math.cos(angle), y: radiusY * Math.sin(angle) };
  });
}

function setupTablePlayers(count) {
  numPlayers = count;
  tablePlayers.innerHTML = '';

  const rect = tablePlayers.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const radiusX = Math.min(300, Math.max(160, rect.width * 0.45));
  const radiusY = Math.min(180, Math.max(100, rect.height * 0.40));
  const positions = polarPositions(count, radiusX, radiusY);

  for (let i = 0; i < count; i++) {
    const seat = document.createElement('div');
    seat.className = 'seat';
    seat.dataset.playerId = String(i);
    const modelName = getModelName(i);
    seat.innerHTML = `
      <div class="seat-label"></div>
      <span class="id-tag">P${i}</span>
      <img src="${getPortraitSrc(i)}" alt="Player ${i}">
      <span class="name-tag">${modelName}</span>
      <div class="speech-bubble">💬</div>
    `;
    seat.style.left = `${cx + positions[i].x - 34}px`;
    seat.style.top = `${cy + positions[i].y - 34}px`;
    const baseRotation = (i % 2 ? 1 : -1) * 2;
    seat.style.setProperty('--base-rotation', `${baseRotation}deg`);
    seat.style.transform = `rotate(var(--base-rotation, 0deg))`;
    tablePlayers.appendChild(seat);
  }

  updateMissionProgressPosition();
}

function highlightSpeaker(playerId) {
  document.querySelectorAll('.seat').forEach((seat) => {
    const isSpeaking = seat.dataset.playerId === String(playerId);
    if (isSpeaking && !seat.classList.contains('speaking')) {
      const bubble = seat.querySelector('.speech-bubble');
      if (bubble) {
        bubble.style.animation = 'none';
        bubble.offsetHeight;
        bubble.style.animation = '';
      }
    }
    seat.classList.toggle('speaking', isSpeaking);
  });
}

function formatTime(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addMessage(message) {
  messageCount++;
  if (messageCount === 1) {
    messagesContainer.innerHTML = '';
  }
  const messageDiv = document.createElement('div');
  messageDiv.className = 'chat-message';

  let senderType = 'system';
  let avatarHtml = '<div class="chat-avatar system">🎭</div>';
  let senderName = message.sender || 'System';
  let playerId = null;

  if (message.sender === 'Moderator') {
    senderType = 'moderator';
    avatarHtml = '<div class="chat-avatar system">⚔</div>';
  } else if (message.sender && message.sender.startsWith('Player')) {
    senderType = 'agent';
    const match = message.sender.match(/Player\s*(\d+)/);
    if (match) {
      playerId = parseInt(match[1], 10);
      const portraitSrc = getPortraitSrc(playerId);
      avatarHtml = `<div class="chat-avatar"><img src="${portraitSrc}" alt="${senderName}"></div>`;
      highlightSpeaker(playerId);
    } else {
      avatarHtml = '<div class="chat-avatar system">🎭</div>';
    }
  }

  messageDiv.innerHTML = `
    ${avatarHtml}
    <div class="chat-bubble">
      <div class="chat-header">
        <span class="chat-sender ${senderType}">${escapeHtml(senderName)}</span>
        <span class="chat-time">${formatTime(message.timestamp)}</span>
      </div>
      <div class="chat-content">${escapeHtml(message.content || '')}</div>
    </div>
  `;

  messagesContainer.appendChild(messageDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateGameState(state) {
  if (state && typeof state.mission_id === 'number') {
    currentMissionId = state.mission_id;
  }
  if (state && state.round_id !== undefined && state.round_id !== null) {
    currentRoundId = state.round_id;
  }

  if (phaseDisplay) {
    const phases = ['Team Selection', 'Team Voting', 'Quest Voting', 'Assassination'];
    const phaseName = state.phase !== null && state.phase !== undefined ? phases[state.phase] || 'Unknown' : '-';
    phaseDisplay.textContent = `Phase: ${phaseName}`;
  }
  if (missionDisplay) {
    missionDisplay.textContent = `Mission: ${state.mission_id ?? '-'}`;
  }
  if (roundDisplay) {
    roundDisplay.textContent = `Round: ${state.round_id ?? '-'}`;
  }
  if (statusDisplay) {
    statusDisplay.textContent = `Status: ${state.status ?? 'Waiting'}`;
  }

  if (state.num_players && state.num_players !== numPlayers) {
    setupTablePlayers(state.num_players);
  }

  const status = state.status || '';
  if (status === 'running') {
    restartScheduled = false;
    isStarting = false;
  }

  const isEndStatus = status && ['stopped', 'finished', 'waiting'].includes(status);
  // Restart on any end status if not already starting or scheduled.
  const shouldAutoRestart = isEndStatus && !isStarting && !restartScheduled;
  if (shouldAutoRestart) {
    queueRestart(`status:${status}`);
  }

  updateMissionProgressPosition();

  if (status) {
    lastStatus = status;
  }
}

function ensureMissionData(index) {
  if (!missionData[index]) {
    missionData[index] = { status: 'pending', resultDetail: 'Pending' };
  }
  return missionData[index];
}

function initMissionProgress() {
  if (!missionProgressEl) return;
  missionProgressEl.innerHTML = '';
  missionNodes = [];
  missionData = Array.from({ length: 5 }, () => ({
    status: 'pending',
    resultDetail: 'Pending',
  }));

  const rail = document.createElement('div');
  rail.className = 'mission-rail';
  missionProgressEl.appendChild(rail);

  for (let i = 0; i < 5; i++) {
    const node = document.createElement('div');
    node.className = 'mission-node pending';
    node.dataset.index = String(i);
    node.innerHTML = `
      <div class="mission-icon">${i + 1}</div>
      <div class="mission-tooltip">
        <span class="mission-title">Mission ${i + 1}</span>
        <span class="mission-detail">Pending</span>
      </div>
    `;
    missionProgressEl.appendChild(node);
    missionNodes.push(node);
  }
}

function buildMissionDetail(index) {
  const data = missionData[index] || {};
  return data.resultDetail || 'Pending';
}

function updateMissionNode(index) {
  const node = missionNodes[index];
  if (!node) return;
  const data = ensureMissionData(index);
  node.classList.remove('pending', 'success', 'fail');
  node.classList.add(data.status || 'pending');
  const detailEl = node.querySelector('.mission-detail');
  if (detailEl) {
    detailEl.textContent = buildMissionDetail(index) || 'Pending';
  }
}

function resetMissionProgress() {
  if (!missionNodes.length) {
    initMissionProgress();
  }
  missionData = missionData.map(() => ({ status: 'pending', resultDetail: 'Pending' }));
  missionNodes.forEach((_, idx) => updateMissionNode(idx));
  currentMissionId = 1;
  currentRoundId = null;
  updateMissionProgressPosition();
}

function updateMissionProgressPosition() {
  if (!missionProgressEl || !tablePlayers) return;
  const seat = tablePlayers.querySelector('.seat[data-player-id="0"]');
  const sceneRect = tablePlayers.getBoundingClientRect();
  if (!seat || !sceneRect.width) return;
  const seatRect = seat.getBoundingClientRect();
  const topOffset = 100;
  const left = seatRect.left - sceneRect.left + seatRect.width / 2;
  const top = Math.max(8, seatRect.top - sceneRect.top - topOffset);
  missionProgressEl.style.left = `${left}px`;
  missionProgressEl.style.top = `${top}px`;
}

function handleQuestResultMessage(text) {
  if (!text) return;
  const normalized = String(text).replace(/\s+/g, ' ').trim();
  const en = normalized.match(/Quest result:\s*Mission\s*(\d+)\s+(succeeded|failed)\.\s*The team was\s*([^\.]+)\.\s*Number of fails:\s*([0-9]+)/i);
  const zh = normalized.match(/任务结果：任务\s*(\d+)\s*([^\s。]+)[^。]*。?\s*团队[是为]\s*([^。.]+)[。\. ]*失败票数[:：]\s*([0-9]+)/i);
  let missionIndex = null;
  let status = 'pending';
  let fails = null;
  let teamText = '';
  if (en) {
    missionIndex = parseInt(en[1], 10) - 1;
    status = en[2].toLowerCase() === 'succeeded' ? 'success' : 'fail';
    teamText = en[3].trim();
    fails = en[4];
  } else if (zh) {
    missionIndex = parseInt(zh[1], 10) - 1;
    const outcomeToken = (zh[2] || '').toLowerCase();
    status = (outcomeToken === '成功' || outcomeToken === 'succeeded') ? 'success' : 'fail';
    teamText = zh[3].trim();
    fails = zh[4];
  } else {
    // Fallback: if we see "任务结果" and a fail count, try a generic parse
    if (normalized.includes('任务结果') && normalized.includes('失败票数')) {
      const failMatch = normalized.match(/失败票数[:：]\s*([0-9]+)/);
      const idMatch = normalized.match(/任务\s*(\d+)/);
      const teamMatch = normalized.match(/团队[是为]\s*([^\.。]+)/);
      missionIndex = idMatch ? parseInt(idMatch[1], 10) - 1 : (currentMissionId ? currentMissionId - 1 : 0);
      fails = failMatch ? failMatch[1] : '?';
      status = normalized.includes('成功') || normalized.toLowerCase().includes('succeeded') ? 'success' : 'fail';
      if (teamMatch) {
        teamText = teamMatch[1].trim();
      }
    }
  }
  if (missionIndex === null || Number.isNaN(missionIndex)) return;
  missionIndex = Math.min(missionData.length - 1, Math.max(0, missionIndex));
  const data = ensureMissionData(missionIndex);
  data.status = status;
  const teamPart = teamText ? `Team ${teamText} | ` : '';
  data.resultDetail = `${teamPart}${status === 'success' ? `Success` : `Fail(${fails})`}`;
  updateMissionNode(missionIndex);
}

function handleGameMessage(text) {
  handleQuestResultMessage(text);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function updateRoleLabels(roles) {
  if (!roles || !Array.isArray(roles)) return;
  roles.forEach((roleInfo, playerId) => {
    const roleName = roleInfo.role_name || roleInfo[1] || '';
    if (!roleName) return;
    const seat = tablePlayers.querySelector(`.seat[data-player-id="${playerId}"]`);
    if (!seat) return;
    const label = seat.querySelector('.seat-label');
    if (!label) return;
    label.textContent = roleName;
    seat.classList.add('has-label');
  });
}

wsClient.onMessage('message', (message) => {
  addMessage(message);
  handleGameMessage(message?.content || '');
});

wsClient.onMessage('game_state', (state) => {
  updateGameState(state);
  if (state.roles && Array.isArray(state.roles)) {
    updateRoleLabels(state.roles);
  }
  if (state.status === 'running' && !gameStarted) {
    resetMissionProgress();
    gameSetup.style.display = 'none';
    messagesContainer.style.display = 'flex';
    gameStarted = true;
  }
  if (state.status === 'stopped' || state.status === 'finished' || state.status === 'waiting') {
    gameStarted = false;
    messageCount = 0;
    resetMissionProgress();
  }
});

wsClient.onMessage('mode_info', (info) => {
  console.log('Mode info:', info);
});

wsClient.onMessage('error', (error) => {
  console.error('Error from server:', error);
  addMessage({ sender: 'System', content: `Error: ${error.message || 'Unknown error'}`, timestamp: new Date().toISOString() });
});

numPlayersSelect.addEventListener('change', () => {
  setupTablePlayers(parseInt(numPlayersSelect.value, 10));
  baseConfig.num_players = parseInt(numPlayersSelect.value, 10);
});

async function startGame(configOverride = null) {
  const cfg = configOverride ? deepClone(configOverride) : buildNextConfig('manual');
  try {
    isStarting = true;
    startGameBtn.disabled = true;
    startGameBtn.textContent = 'Starting...';
    const response = await fetch('/api/start-game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    const result = await response.json().catch(() => ({}));
    if (response.ok) {
      gameSetup.style.display = 'none';
      messagesContainer.style.display = 'flex';
      gameStarted = true;
      sessionStorage.setItem('gameRunning', 'true');
      resetMissionProgress();
    } else {
      isStarting = false;
      console.error('Failed to start game:', result.detail || 'unknown');
      startGameBtn.disabled = false;
      startGameBtn.textContent = 'Start Observing';
    }
  } catch (error) {
    console.error('Error starting game:', error);
    isStarting = false;
    startGameBtn.disabled = false;
    startGameBtn.textContent = 'Start Observing';
  }
}

function queueRestart(reason) {
  if (restartScheduled || isStarting) return;
  restartScheduled = true;
  setTimeout(() => {
    const nextCfg = buildNextConfig(reason);
    startGame(nextCfg).finally(() => {
      startGameBtn.textContent = 'Start Observing';
      restartScheduled = false;
    });
  }, 150);
}

startGameBtn.addEventListener('click', () => startGame());

wsClient.onConnect(() => {
  gameStarted = false;
  messageCount = 0;

  if (window.__EARLY_INIT__ && window.__EARLY_INIT__.hasGameConfig && window.__EARLY_INIT__.config) {
    const cfg = buildNextConfig('early');
    window.__EARLY_INIT__.hasGameConfig = false;
    isStarting = true;
    startGame(cfg);
    return;
  }
  if (window.__EARLY_INIT__ && window.__EARLY_INIT__.isGameRunning) {
    window.__EARLY_INIT__.isGameRunning = false;
    return;
  }
  // If nothing is running, kick off a game immediately.
  queueRestart('connect');
});

wsClient.onDisconnect(() => {
  console.log('Disconnected from game server');
});

function initializeObserve() {
  initMissionProgress();
  setupTablePlayers(numPlayers);
  if (backButton) {
    backButton.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = BACK_HOME_URL;
    });
  }
  wsClient.connect();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeObserve);
} else {
  initializeObserve();
}

window.addEventListener('resize', () => {
  setupTablePlayers(numPlayers);
});
