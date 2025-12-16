// Participate mode JavaScript - Pixel Town Style

// 清除页面缓存：当离开游戏页面时清除游戏数据
window.addEventListener('beforeunload', () => {
    // 保留必要的配置数据，清除可能过期的游戏状态数据
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
        // 页面从 bfcache 恢复，强制重新加载
        window.location.reload();
    }
});

const wsClient = new WebSocketClient();
const messagesContainer = document.getElementById('messages-container');
const phaseDisplay = document.getElementById('phase-display');
const missionDisplay = document.getElementById('mission-display');
const roundDisplay = document.getElementById('round-display');
const statusDisplay = document.getElementById('status-display');
const userInputElement = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const userInputRequest = document.getElementById('user-input-request');
const inputPrompt = document.getElementById('input-prompt');
const gameSetup = document.getElementById('game-setup');
const startGameBtn = document.getElementById('start-game-btn');
const numPlayersSelect = document.getElementById('num-players');
const userAgentIdSelect = document.getElementById('user-agent-id');
const languageSelect = document.getElementById('language');
const backExitButton = document.getElementById('back-exit-button');
const inputContainer = document.querySelector('.input-container');
const tablePlayers = document.getElementById('table-players');

let messageCount = 0;
let currentAgentId = null;  // 数字 player ID (0, 1, 2...)
let currentAgentStringId = null;  // agentscope 的字符串 agent.id
let waitingForInput = false;
let gameStarted = false;
let numPlayers = 5;

// 应用语言类到 body
const gameLanguage = sessionStorage.getItem('gameLanguage') || 'en';
document.body.classList.add(`lang-${gameLanguage}`);

// 从早期初始化脚本或 sessionStorage 读取配置
// __EARLY_INIT__ 在 HTML <head> 中的脚本设置
let selectedPortraits = [];
if (window.__EARLY_INIT__ && window.__EARLY_INIT__.portraits) {
    selectedPortraits = window.__EARLY_INIT__.portraits;
} else {
    try {
        const stored = sessionStorage.getItem('selectedPortraits');
        if (stored) selectedPortraits = JSON.parse(stored);
    } catch (e) {}
}

// 从早期初始化或 sessionStorage 读取 gameConfig
let agentConfigs = {};
if (window.__EARLY_INIT__ && window.__EARLY_INIT__.config) {
    const config = window.__EARLY_INIT__.config;
    if (config.user_agent_id !== undefined) {
        currentAgentId = typeof config.user_agent_id === 'number'
            ? config.user_agent_id
            : parseInt(config.user_agent_id, 10);
    }
    if (config.num_players) {
        numPlayers = typeof config.num_players === 'number'
            ? config.num_players
            : parseInt(config.num_players, 10);
    }
    if (config.agent_configs) {
        agentConfigs = config.agent_configs;
    }
} else {
    try {
        const gameConfigStr = sessionStorage.getItem('gameConfig');
        if (gameConfigStr) {
            const gameConfig = JSON.parse(gameConfigStr);
            if (gameConfig.user_agent_id !== undefined) {
                currentAgentId = typeof gameConfig.user_agent_id === 'number'
                    ? gameConfig.user_agent_id
                    : parseInt(gameConfig.user_agent_id, 10);
            }
            if (gameConfig.num_players) {
                numPlayers = typeof gameConfig.num_players === 'number'
                    ? gameConfig.num_players
                    : parseInt(gameConfig.num_players, 10);
            }
            if (gameConfig.agent_configs) {
                agentConfigs = gameConfig.agent_configs;
            }
        }
    } catch (e) {}
}

// Portrait helper - 使用选择的头像映射
function getPortraitSrc(playerId) {
    // 确保 playerId 转换为数字
    const validId = (typeof playerId === 'number' && !isNaN(playerId)) 
        ? playerId 
        : (typeof playerId === 'string' ? parseInt(playerId, 10) : 0);
    
    // 确保 currentAgentId 也是数字类型进行比较
    const humanId = (currentAgentId !== null && currentAgentId !== undefined) 
        ? (typeof currentAgentId === 'number' ? currentAgentId : parseInt(currentAgentId, 10))
        : null;

    // Participate 模式：人类玩家固定使用 portrait_human.png
    if (humanId !== null && !isNaN(humanId) && !isNaN(validId) && validId === humanId) {
        return `/static/portraits/portrait_human.png`;
    }
    
    // AI 头像：selectedPortraits 是用户选择的 AI 头像列表
    // 在 participate 模式下，selectedPortraits.length = numPlayers - 1（不包括人类玩家）
    // 映射规则：AI 玩家按顺序使用 selectedPortraits，跳过人类玩家位置
    if (selectedPortraits && selectedPortraits.length > 0) {
        let idx = validId;
        // 如果当前玩家在人类玩家之后，索引需要减1
        if (humanId !== null && !isNaN(humanId) && validId > humanId) {
            idx = validId - 1;
        }
        
        // 确保索引在有效范围内
        if (idx >= 0 && idx < selectedPortraits.length) {
            const portraitId = selectedPortraits[idx];
            return `/static/portraits/portrait_${portraitId}.png`;
        }
    }
    
    // 回退：使用默认映射
    const id = (validId % 15) + 1;
    return `/static/portraits/portrait_${id}.png`;
}

// 获取模型名字
function getModelName(playerId) {
    const validId = (typeof playerId === 'number' && !isNaN(playerId)) 
        ? playerId 
        : (typeof playerId === 'string' ? parseInt(playerId, 10) : 0);
    
    // 确保 currentAgentId 也是数字类型进行比较
    const humanId = (currentAgentId !== null && currentAgentId !== undefined) 
        ? (typeof currentAgentId === 'number' ? currentAgentId : parseInt(currentAgentId, 10))
        : null;
    
    // Participate 模式：人类玩家显示 "You"
    if (humanId !== null && !isNaN(humanId) && !isNaN(validId) && validId === humanId) {
        return 'You';
    }
    
    // 根据 playerId 找到对应的 portraitId
    let portraitId = null;
    if (selectedPortraits && selectedPortraits.length > 0) {
        let idx = validId;
        // 如果当前玩家在人类玩家之后，索引需要减1
        if (humanId !== null && !isNaN(humanId) && validId > humanId) {
            idx = validId - 1;
        }
        
        // 确保索引在有效范围内
        if (idx >= 0 && idx < selectedPortraits.length) {
            portraitId = selectedPortraits[idx];
        }
    }
    
    // 如果没有找到，使用默认映射
    if (!portraitId) {
        portraitId = (validId % 15) + 1;
    }
    
    // 从 agent_configs 中获取模型名字（键可能是字符串或数字）
    if (portraitId && agentConfigs) {
        const config = agentConfigs[portraitId] || agentConfigs[String(portraitId)];
        if (config && config.base_model) {
            return config.base_model;
        }
    }
    
    // 如果没有配置，返回默认值
    return 'Unknown';
}

// Polar positions for table seating
function polarPositions(count, radiusX, radiusY) {
    return Array.from({ length: count }).map((_, i) => {
        const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
        return { x: radiusX * Math.cos(angle), y: radiusY * Math.sin(angle) };
    });
}

// Setup table players
function setupTablePlayers(count) {
    numPlayers = count;
    tablePlayers.innerHTML = '';
    
    const rect = tablePlayers.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    // 增大分布半径，让人物分布更分散
    const radiusX = Math.min(300, Math.max(160, rect.width * 0.45)); // 从0.34增大到0.45，最大值从210增大到300
    const radiusY = Math.min(180, Math.max(100, rect.height * 0.40)); // 从0.30增大到0.40，最大值从120增大到180
    const positions = polarPositions(count, radiusX, radiusY);
    
    for (let i = 0; i < count; i++) {
        const seat = document.createElement('div');
        seat.className = 'seat';
        seat.dataset.playerId = String(i);
        
        // 确保类型一致进行比较
        const humanId = (currentAgentId !== null && currentAgentId !== undefined) 
            ? (typeof currentAgentId === 'number' ? currentAgentId : parseInt(currentAgentId, 10))
            : null;
        const isHuman = (humanId !== null && !isNaN(humanId) && i === humanId);
        const portraitSrc = getPortraitSrc(i);
        const modelName = getModelName(i);
        
        seat.innerHTML = `
            <span class="id-tag">P${i}</span>
            <img src="${portraitSrc}" alt="Player ${i}">
            <span class="name-tag">${modelName}</span>
            <div class="speech-bubble">💬</div>
        `;
        seat.style.left = `${cx + positions[i].x - 34}px`;
        seat.style.top = `${cy + positions[i].y - 34}px`;
        // 使用 CSS 变量保存基础旋转角度，让动画可以叠加抖动效果
        const baseRotation = (i % 2 ? 1 : -1) * 2;
        seat.style.setProperty('--base-rotation', `${baseRotation}deg`);
        seat.style.transform = `rotate(var(--base-rotation, 0deg))`;
        tablePlayers.appendChild(seat);
    }
}

// Highlight speaking player with bubble animation
function highlightSpeaker(playerId) {
    document.querySelectorAll('.seat').forEach(seat => {
        const seatPlayerId = seat.dataset.playerId;
        const isSpeaking = seatPlayerId === String(playerId);
        const wasSpeaking = seat.classList.contains('speaking');
        
        if (isSpeaking && !wasSpeaking) {
            // 开始说话：添加 speaking 类并触发气泡动画
            const bubble = seat.querySelector('.speech-bubble');
            if (bubble) {
                // 先移除 speaking 类（如果存在），重置动画
                seat.classList.remove('speaking');
                bubble.style.animation = 'none';
                bubble.style.opacity = '0';
                
                // 使用 requestAnimationFrame 确保 DOM 更新后再添加类
                requestAnimationFrame(() => {
                    seat.classList.add('speaking');
                    // 再次强制触发动画
                    bubble.offsetHeight; // 强制 reflow
                    bubble.style.animation = 'bubble-pop 2s ease-out forwards';
                });
            } else {
                seat.classList.add('speaking');
            }
        } else if (!isSpeaking && wasSpeaking) {
            // 停止说话：移除 speaking 类
            seat.classList.remove('speaking');
            const bubble = seat.querySelector('.speech-bubble');
            if (bubble) {
                // 立即隐藏气泡
                bubble.style.animation = 'none';
                bubble.style.opacity = '0';
            }
        }
    });
}

// 清除所有玩家的 speaking 状态（用于主持人发言时）
function clearAllSpeaking() {
    document.querySelectorAll('.seat').forEach(seat => {
        seat.classList.remove('speaking');
        const bubble = seat.querySelector('.speech-bubble');
        if (bubble) {
            bubble.style.animation = 'none';
            bubble.style.opacity = '0';
        }
    });
}

function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addMessage(message) {
    messageCount++;
    
    // Clear "waiting" message if this is the first message
    if (messageCount === 1) {
        messagesContainer.innerHTML = '';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message';
    
    // Determine sender type and get avatar
    let senderType = 'system';
    let avatarHtml = '<div class="chat-avatar system">🎭</div>';
    let senderName = message.sender || 'System';
    let playerId = null;
    
    if (message.sender === 'Moderator') {
        senderType = 'moderator';
        avatarHtml = '<div class="chat-avatar system">⚔</div>';
        // 主持人发言时，清除所有玩家的 speaking 状态
        clearAllSpeaking();
    } else if (message.sender && message.sender.startsWith('Player')) {
        senderType = 'agent';
        // 支持 "Player0", "Player 0", "Player1" 等格式
        const match = message.sender.match(/Player\s*(\d+)/);
        if (match) {
            playerId = parseInt(match[1], 10);
            console.log(`Parsed playerId from sender "${message.sender}": ${playerId}`);
            const portraitSrc = getPortraitSrc(playerId);
            console.log(`Using portrait for Player${playerId}: ${portraitSrc}`);
            avatarHtml = `<div class="chat-avatar"><img src="${portraitSrc}" alt="${senderName}"></div>`;
            // Highlight this player at the table
            highlightSpeaker(playerId);
        } else {
            console.warn(`Failed to parse playerId from sender: "${message.sender}"`);
            avatarHtml = '<div class="chat-avatar system">🎭</div>';
        }
    } else if (message.sender === 'You' || message.role === 'user') {
        senderType = 'user';
        messageDiv.classList.add('own');
        avatarHtml = `<div class="chat-avatar"><img src="${getPortraitSrc(currentAgentId || 0)}" alt="You"></div>`;
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
    
    // Auto-scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateGameState(state) {
    // Status bar
    if (phaseDisplay) {
        const phases = ['Team Selection', 'Team Voting', 'Quest Voting', 'Assassination'];
        const phaseName = (state.phase !== null && state.phase !== undefined) ? (phases[state.phase] || 'Unknown') : '-';
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
    
    // Update table if num_players changed
    if (state.num_players && state.num_players !== numPlayers) {
        setupTablePlayers(state.num_players);
    }
}

function showInputRequest(agentId, prompt) {
    // 保存 agentscope 的字符串 agent.id（用于发送消息）
    currentAgentStringId = agentId;
    waitingForInput = true;
    inputPrompt.textContent = prompt;
    userInputRequest.style.display = 'block';
    userInputElement.disabled = false;
    sendButton.disabled = false;
    userInputElement.focus();
}

function hideInputRequest() {
    waitingForInput = false;
    userInputRequest.style.display = 'none';
    userInputElement.disabled = true;
    sendButton.disabled = true;
    userInputElement.value = '';
}

function sendUserInput() {
    const content = userInputElement.value.trim();
    if (!content) return;
    
    if (!currentAgentStringId) {
        alert('Error: Agent ID not set. Please refresh the page.');
        return;
    }
    
    wsClient.sendUserInput(currentAgentStringId, content);
    hideInputRequest();
    
    // Show user's input in messages and trigger speaking animation
    addMessage({
        sender: 'You',
        content: content,
        role: 'user',
        timestamp: new Date().toISOString()
    });
    
    // Trigger speaking animation for human player
    if (currentAgentId !== null && currentAgentId !== undefined) {
        highlightSpeaker(currentAgentId);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event listeners
sendButton.addEventListener('click', sendUserInput);

userInputElement.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendUserInput();
    }
});

// WebSocket message handlers
wsClient.onMessage('message', (message) => {
    addMessage(message);
});

wsClient.onMessage('game_state', (state) => {
    updateGameState(state);
    // Show messages container when game starts
    if (state.status === 'running' && !gameStarted) {
        gameSetup.style.display = 'none';
        messagesContainer.style.display = 'flex';
        inputContainer.style.display = 'flex';
        gameStarted = true;
        updateBackExitButton('running');
    }
    // Handle game stopped
    if (state.status === 'stopped') {
        gameStarted = false;
        sessionStorage.removeItem('gameRunning');  // 清除游戏运行标记
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
        inputContainer.style.display = 'none';
        hideInputRequest();
        updateBackExitButton('stopped');
        messageCount = 0;
        messagesContainer.innerHTML = '<p style="text-align: center; color: var(--muted); padding: 20px; font-size: 9px;">Game stopped. You can start a new game.</p>';
    }
    // Handle game finished
    if (state.status === 'finished') {
        gameStarted = false;
        sessionStorage.removeItem('gameRunning');  // 清除游戏运行标记
        updateBackExitButton('finished');
    }
    // Handle waiting state
    if (state.status === 'waiting') {
        gameStarted = false;
        sessionStorage.removeItem('gameRunning');  // 清除游戏运行标记
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
        inputContainer.style.display = 'none';
        hideInputRequest();
        updateBackExitButton('waiting');
    }
});

wsClient.onMessage('user_input_request', (request) => {
    showInputRequest(request.agent_id, request.prompt);
});

wsClient.onMessage('mode_info', (info) => {
    console.log('Mode info:', info);
    if (info.mode !== 'participate') {
        console.warn('Expected participate mode, got:', info.mode);
    }
    // 只有当 currentAgentId 还没有设置时，才从 mode_info 更新
    // 防止覆盖已经正确设置的值
    if (info.user_agent_id !== undefined && currentAgentId === null) {
        currentAgentId = typeof info.user_agent_id === 'number'
            ? info.user_agent_id
            : parseInt(info.user_agent_id, 10);
        console.log('Setting currentAgentId from mode_info:', info.user_agent_id, '->', currentAgentId);
        // 只有在这种情况下才需要重新设置桌面
        setupTablePlayers(numPlayers);
    }
});

wsClient.onMessage('error', (error) => {
    console.error('Error from server:', error);
    addMessage({
        sender: 'System',
        content: `Error: ${error.message || 'Unknown error'}`,
        timestamp: new Date().toISOString()
    });
});

// Update user agent ID options based on num players
numPlayersSelect.addEventListener('change', () => {
    const np = parseInt(numPlayersSelect.value);
    userAgentIdSelect.innerHTML = '';
    for (let i = 0; i < np; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = i;
        userAgentIdSelect.appendChild(option);
    }
    setupTablePlayers(np);
});

async function startGame() {
    const np = parseInt(numPlayersSelect.value);
    const userAgentId = parseInt(userAgentIdSelect.value);
    const language = languageSelect.value;
    
    try {
        startGameBtn.disabled = true;
        startGameBtn.textContent = 'Starting...';
        
        const response = await fetch('/api/start-game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game: 'avalon',
                num_players: np,
                language: language,
                user_agent_id: userAgentId,
                mode: 'participate',
            }),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentAgentId = typeof userAgentId === 'number' ? userAgentId : parseInt(userAgentId, 10);
            console.log('Game started, setting currentAgentId:', userAgentId, '->', currentAgentId);
            setupTablePlayers(np);
            gameSetup.style.display = 'none';
            messagesContainer.style.display = 'flex';
            inputContainer.style.display = 'flex';
            gameStarted = true;
        } else {
            alert(`Error: ${result.detail || 'Failed to start game'}`);
            startGameBtn.disabled = false;
            startGameBtn.textContent = 'Start Game';
        }
    } catch (error) {
        console.error('Error starting game:', error);
        alert(`Error: ${error.message}`);
        startGameBtn.disabled = false;
        startGameBtn.textContent = 'Start Game';
    }
}

function updateBackExitButton(gameStatus) {
    let status = typeof gameStatus === 'boolean' ? (gameStatus ? 'running' : 'waiting') : gameStatus;
    
    const goHome = () => { window.location.href = '/'; };
    if (status === 'running') {
        backExitButton.textContent = '← Exit';
        backExitButton.title = 'Exit Game';
        backExitButton.href = '#';
        backExitButton.style.display = 'inline-block';
        backExitButton.onclick = async (e) => {
            e.preventDefault();
            try {
                await fetch('/api/stop-game', { method: 'POST' });
            } catch (error) {
                console.error('Error stopping game:', error);
            }
            goHome();
        };
    } else {
        backExitButton.textContent = '← Back';
        backExitButton.title = 'Back to Home';
        backExitButton.href = '/';
        backExitButton.style.display = 'inline-block';
        backExitButton.onclick = (e) => { e.preventDefault(); goHome(); };
    }
}

startGameBtn.addEventListener('click', startGame);

// Connect when page loads
wsClient.onConnect(() => {
    console.log('Connected to game server');
    gameStarted = false;
    messageCount = 0;
    hideInputRequest();
    
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
});

wsClient.onDisconnect(() => {
    console.log('Disconnected from game server');
    hideInputRequest();
});

// 初始化桌面并连接 WebSocket
function initializeTable() {
    // 初始化圆桌（数据已在脚本开头从 __EARLY_INIT__ 或 sessionStorage 加载）
    setupTablePlayers(numPlayers);
    
    // 连接 WebSocket
    wsClient.connect();
}

// 如果 DOM 已经加载完成，立即执行；否则等待 DOMContentLoaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeTable);
} else {
    // DOM 已经加载完成，立即执行
    initializeTable();
}

// Initialize button
updateBackExitButton(false);

// Handle window resize
window.addEventListener('resize', () => {
    setupTablePlayers(numPlayers);
});
