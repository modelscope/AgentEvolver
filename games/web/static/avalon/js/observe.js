// Observe mode JavaScript - Pixel Town Style

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
const gameSetup = document.getElementById('game-setup');
const startGameBtn = document.getElementById('start-game-btn');
const numPlayersSelect = document.getElementById('num-players');
const languageSelect = document.getElementById('language');
const tablePlayers = document.getElementById('table-players');
const backExitButton = document.getElementById('back-exit-button');

let messageCount = 0;
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
            if (gameConfig.agent_configs) {
                agentConfigs = gameConfig.agent_configs;
            }
        }
    } catch (e) {}
}

function getPortraitSrc(playerId) {
    const validId = (typeof playerId === 'number' && !isNaN(playerId)) ? playerId : 0;
    
    console.log(`getPortraitSrc (observe): playerId=${playerId}, validId=${validId}, selectedPortraits=`, selectedPortraits);
    
    if (selectedPortraits && selectedPortraits.length > validId) {
        const portraitId = selectedPortraits[validId];
        console.log(`Player ${validId} -> selectedPortraits[${validId}] = ${portraitId}`);
        return `/static/portraits/portrait_${portraitId}.png`;
    }
    
    const id = (validId % 15) + 1;
    console.log(`Player ${validId} using default portrait ${id}`);
    return `/static/portraits/portrait_${id}.png`;
}

function getModelName(playerId) {
    const validId = (typeof playerId === 'number' && !isNaN(playerId)) ? playerId : 0;
    
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
}

function highlightSpeaker(playerId) {
    document.querySelectorAll('.seat').forEach(seat => {
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
        // Support formats like "Player0", "Player 0", "Player1"
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateRoleLabels(roles) {
    if (!roles || !Array.isArray(roles)) {
        return;
    }
    
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
});

wsClient.onMessage('game_state', (state) => {
    updateGameState(state);
    updateBackExitButton(state.status);
    
    if (state.roles && Array.isArray(state.roles)) {
        updateRoleLabels(state.roles);
    }
    
    if (state.status === 'running' && !gameStarted) {
        gameSetup.style.display = 'none';
        messagesContainer.style.display = 'flex';
        gameStarted = true;
    }
    if (state.status === 'stopped') {
        gameStarted = false;
        clearAvalonSessionStorage();
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
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
    }
});

wsClient.onMessage('mode_info', (info) => {
    console.log('Mode info:', info);
    if (info.mode !== null && info.mode !== undefined && info.mode !== 'observe') {
        console.warn('Expected observe mode, got:', info.mode);
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
    setupTablePlayers(parseInt(numPlayersSelect.value));
});

async function startGame() {
    const np = parseInt(numPlayersSelect.value);
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
                mode: 'observe',
            }),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            gameSetup.style.display = 'none';
            messagesContainer.style.display = 'flex';
            gameStarted = true;
        } else {
            alert(`Error: ${result.detail || 'Failed to start game'}`);
            startGameBtn.disabled = false;
            startGameBtn.textContent = 'Start Observing';
        }
    } catch (error) {
        console.error('Error starting game:', error);
        alert(`Error: ${error.message}`);
        startGameBtn.disabled = false;
        startGameBtn.textContent = 'Start Observing';
    }
}

startGameBtn.addEventListener('click', startGame);

wsClient.onConnect(() => {
    console.log('Connected to game server');
    gameStarted = false;
    messageCount = 0;
    
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
});

function initializeObserve() {
    setupTablePlayers(numPlayers);
    
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
