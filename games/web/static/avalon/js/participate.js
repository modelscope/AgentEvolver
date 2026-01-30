// Participate mode JavaScript - Pixel Town Style

window.addEventListener('beforeunload', () => {
    const keysToKeep = ['gameConfig', 'selectedPortraits', 'gameLanguage'];
    Object.keys(sessionStorage).forEach(key => {
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
const inputContainer = document.querySelector('.input-container');
const tablePlayers = document.getElementById('table-players');
const backExitButton = document.getElementById('back-exit-button');

let messageCount = 0;
let currentAgentId = null;
let currentAgentStringId = null;
let waitingForInput = false;
let gameStarted = false;
let numPlayers = 5;

const AVALON_SESSION_CLEAR_KEYS = new Set([
    'gameconfig',
    'selectedportraits',
    'gamelanguage',
    'gamerunning',
    'previewroles',
    'presetroles',
    'preview_roles',
    'preset_roles',
    'avalonpreviewroles',
    'avalonpresetroles',
    'avalonroleorder',
    'avalonrolepreview',
    'avalonroleselection',
]);

function clearAvalonSessionStorage() {
    Object.keys(sessionStorage).forEach((key) => {
        const lowerKey = key.toLowerCase();
        if (AVALON_SESSION_CLEAR_KEYS.has(lowerKey) || lowerKey.startsWith('avalon')) {
            sessionStorage.removeItem(key);
        }
    });
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
            clearAvalonSessionStorage();
            goHome();
        };
    } else {
        if (gameStatus === 'stopped' || gameStatus === 'finished' || gameStatus === 'waiting') {
            clearAvalonSessionStorage();
        }
        backExitButton.textContent = '← Back';
        backExitButton.title = 'Back to Home';
        backExitButton.href = '/';
        backExitButton.onclick = (e) => {
            e.preventDefault();
            goHome();
        };
    }
}

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

function getPortraitSrc(playerId) {
    const validId = (typeof playerId === 'number' && !isNaN(playerId)) 
        ? playerId 
        : (typeof playerId === 'string' ? parseInt(playerId, 10) : 0);
    
    const humanId = (currentAgentId !== null && currentAgentId !== undefined) 
        ? (typeof currentAgentId === 'number' ? currentAgentId : parseInt(currentAgentId, 10))
        : null;

    if (humanId !== null && !isNaN(humanId) && !isNaN(validId) && validId === humanId) {
        return `/static/portraits/portrait_human.png`;
    }
    
    if (selectedPortraits && selectedPortraits.length > 0) {
        let idx = validId;
        if (humanId !== null && !isNaN(humanId) && validId > humanId) {
            idx = validId - 1;
        }
        
        if (idx >= 0 && idx < selectedPortraits.length) {
            const portraitId = selectedPortraits[idx];
            return `/static/portraits/portrait_${portraitId}.png`;
        }
    }
    
    const id = (validId % 15) + 1;
    return `/static/portraits/portrait_${id}.png`;
}

function getModelName(playerId) {
    const validId = (typeof playerId === 'number' && !isNaN(playerId)) 
        ? playerId 
        : (typeof playerId === 'string' ? parseInt(playerId, 10) : 0);
    
    const humanId = (currentAgentId !== null && currentAgentId !== undefined) 
        ? (typeof currentAgentId === 'number' ? currentAgentId : parseInt(currentAgentId, 10))
        : null;
    
    if (humanId !== null && !isNaN(humanId) && !isNaN(validId) && validId === humanId) {
        return 'You';
    }
    
    let portraitId = null;
    if (selectedPortraits && selectedPortraits.length > 0) {
        let idx = validId;
        if (humanId !== null && !isNaN(humanId) && validId > humanId) {
            idx = validId - 1;
        }
        
        if (idx >= 0 && idx < selectedPortraits.length) {
            portraitId = selectedPortraits[idx];
        }
    }
    
    if (!portraitId) {
        portraitId = (validId % 15) + 1;
    }
    
    if (portraitId && agentConfigs) {
        const config = agentConfigs[portraitId] || agentConfigs[String(portraitId)];
        if (config && config.base_model) {
            return config.base_model;
        }
    }
    
    return 'Unknown';
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
        const baseRotation = (i % 2 ? 1 : -1) * 2;
        seat.style.setProperty('--base-rotation', `${baseRotation}deg`);
        seat.style.transform = `rotate(var(--base-rotation, 0deg))`;
        tablePlayers.appendChild(seat);
    }
}

function highlightSpeaker(playerId) {
    document.querySelectorAll('.seat').forEach(seat => {
        const seatPlayerId = seat.dataset.playerId;
        const isSpeaking = seatPlayerId === String(playerId);
        const wasSpeaking = seat.classList.contains('speaking');
        
        if (isSpeaking && !wasSpeaking) {
            const bubble = seat.querySelector('.speech-bubble');
            if (bubble) {
                seat.classList.remove('speaking');
                bubble.style.animation = 'none';
                bubble.style.opacity = '0';
                
                requestAnimationFrame(() => {
                    seat.classList.add('speaking');
                    bubble.offsetHeight; 
                    bubble.style.animation = 'bubble-pop 2s ease-out forwards';
                });
            } else {
                seat.classList.add('speaking');
            }
        } else if (!isSpeaking && wasSpeaking) {
            seat.classList.remove('speaking');
            const bubble = seat.querySelector('.speech-bubble');
            if (bubble) {
                bubble.style.animation = 'none';
                bubble.style.opacity = '0';
            }
        }
    });
}

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
        clearAllSpeaking();
    } else if (message.sender && message.sender.startsWith('Player')) {
        senderType = 'agent';
        const match = message.sender.match(/Player\s*(\d+)/);
        if (match) {
            playerId = parseInt(match[1], 10);
            console.log(`Parsed playerId from sender "${message.sender}": ${playerId}`);
            const portraitSrc = getPortraitSrc(playerId);
            console.log(`Using portrait for Player${playerId}: ${portraitSrc}`);
            avatarHtml = `<div class="chat-avatar"><img src="${portraitSrc}" alt="${senderName}"></div>`;
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
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateGameState(state) {
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
    
    if (state.num_players && state.num_players !== numPlayers) {
        setupTablePlayers(state.num_players);
    }
}

function showInputRequest(agentId, prompt) {
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
    
    addMessage({
        sender: 'You',
        content: content,
        role: 'user',
        timestamp: new Date().toISOString()
    });
    
    if (currentAgentId !== null && currentAgentId !== undefined) {
        highlightSpeaker(currentAgentId);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

sendButton.addEventListener('click', sendUserInput);

userInputElement.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendUserInput();
    }
});

wsClient.onMessage('message', (message) => {
    addMessage(message);
});

wsClient.onMessage('game_state', (state) => {
    updateGameState(state);
    updateBackExitButton(state.status);
    if (state.status === 'running' && !gameStarted) {
        gameSetup.style.display = 'none';
        messagesContainer.style.display = 'flex';
        inputContainer.style.display = 'flex';
        gameStarted = true;
    }
    if (state.status === 'stopped') {
        gameStarted = false;
        clearAvalonSessionStorage();
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
        inputContainer.style.display = 'none';
        hideInputRequest();
        messageCount = 0;
        messagesContainer.innerHTML = '<p style="text-align: center; color: var(--muted); padding: 20px; font-size: 9px;">Game stopped. You can start a new game.</p>';
    }
    if (state.status === 'finished') {
        gameStarted = false;
        clearAvalonSessionStorage();
    }
    if (state.status === 'waiting') {
        gameStarted = false;
        clearAvalonSessionStorage();
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
        inputContainer.style.display = 'none';
        hideInputRequest();
    }
});

wsClient.onMessage('user_input_request', (request) => {
    showInputRequest(request.agent_id, request.prompt);
});

wsClient.onMessage('mode_info', (info) => {
    console.log('Mode info:', info);
    if (info.mode !== null && info.mode !== undefined && info.mode !== 'participate') {
        console.warn('Expected participate mode, got:', info.mode);
    }
    if (info.user_agent_id !== undefined && currentAgentId === null) {
        currentAgentId = typeof info.user_agent_id === 'number'
            ? info.user_agent_id
            : parseInt(info.user_agent_id, 10);
        console.log('Setting currentAgentId from mode_info:', info.user_agent_id, '->', currentAgentId);
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

startGameBtn.addEventListener('click', startGame);

// Connect when page loads
wsClient.onConnect(() => {
    console.log('Connected to game server');
    gameStarted = false;
    messageCount = 0;
    hideInputRequest();
    
    if (window.__EARLY_INIT__ && window.__EARLY_INIT__.hasGameConfig && window.__EARLY_INIT__.config) {
        console.log('Found game config from early init, starting game automatically...');
        
        const config = window.__EARLY_INIT__.config;
        
        sessionStorage.removeItem('gameConfig');
        sessionStorage.setItem('gameRunning', 'true');
        window.__EARLY_INIT__.hasGameConfig = false;
        
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
    else if (window.__EARLY_INIT__ && window.__EARLY_INIT__.isGameRunning) {
        console.log('Game was running, reconnecting...');
        window.__EARLY_INIT__.isGameRunning = false;
    }
});

wsClient.onDisconnect(() => {
    console.log('Disconnected from game server');
    hideInputRequest();
});

function initializeTable() {
    setupTablePlayers(numPlayers);
    
    wsClient.connect();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeTable);
} else {
    initializeTable();
}

window.addEventListener('resize', () => {
    setupTablePlayers(numPlayers);
});
