// Participate mode JavaScript
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

let messageCount = 0;
let currentAgentId = null;
let waitingForInput = false;
let gameStarted = false;

function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

function addMessage(message) {
    messageCount++;
    
    // Clear "waiting" message if this is the first message
    if (messageCount === 1) {
        messagesContainer.innerHTML = '';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    
    // Determine message class based on sender
    if (message.sender === 'Moderator') {
        messageDiv.classList.add('moderator');
    } else if (message.sender && message.sender.startsWith('Player')) {
        messageDiv.classList.add('agent');
    } else {
        messageDiv.classList.add('user');
    }
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-sender">${escapeHtml(message.sender || 'System')}</span>
            <span class="message-time">${formatTime(message.timestamp)}</span>
        </div>
        <div class="message-content">${escapeHtml(message.content || '')}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateGameState(state) {
    // Status bar similar to Diplomacy
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
}

function showInputRequest(agentId, prompt) {
    currentAgentId = agentId;
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
    if (!content || !currentAgentId) {
        return;
    }
    
    wsClient.sendUserInput(currentAgentId, content);
    hideInputRequest();
    
    // Show user's input in messages
    addMessage({
        sender: 'You',
        content: content,
        role: 'user',
        timestamp: new Date().toISOString()
    });
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
        messagesContainer.style.display = 'block';
        inputContainer.style.display = 'flex';
        gameStarted = true;
        // Change button to Exit (stops game)
        updateBackExitButton('running');
    }
    // Handle game stopped - reset state and show setup
    if (state.status === 'stopped') {
        gameStarted = false;
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
        inputContainer.style.display = 'none';
        hideInputRequest();
        updateBackExitButton('stopped');
        // Reset message count and clear messages
        messageCount = 0;
        messagesContainer.innerHTML = '<p style="text-align: center; color: #999; padding: 20px;">Game stopped. You can start a new game.</p>';
        // Don't redirect - allow user to start new game or go back manually
    }
    // Handle game finished - allow starting new game
    if (state.status === 'finished') {
        // Game finished normally, can start new game
        gameStarted = false;
        // Change button to Exit (goes back to home)
        updateBackExitButton('finished');
    }
    // Handle waiting state - show setup
    if (state.status === 'waiting') {
        gameStarted = false;
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
    if (info.user_agent_id) {
        currentAgentId = info.user_agent_id;
    }
});

wsClient.onMessage('error', (error) => {
    console.error('Error from server:', error);
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message';
    errorDiv.style.background = '#ffebee';
    errorDiv.style.borderLeftColor = '#f44336';
    errorDiv.innerHTML = `
        <div class="message-header">
            <span class="message-sender" style="color: #f44336;">Error</span>
        </div>
        <div class="message-content">${escapeHtml(error.message || 'Unknown error')}</div>
    `;
    messagesContainer.appendChild(errorDiv);
});

// Update user agent ID options based on num players
numPlayersSelect.addEventListener('change', () => {
    const numPlayers = parseInt(numPlayersSelect.value);
    userAgentIdSelect.innerHTML = '';
    for (let i = 0; i < numPlayers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = i;
        userAgentIdSelect.appendChild(option);
    }
});

async function startGame() {
    const numPlayers = parseInt(numPlayersSelect.value);
    const userAgentId = parseInt(userAgentIdSelect.value);
    const language = languageSelect.value;
    
    try {
        startGameBtn.disabled = true;
        startGameBtn.textContent = 'Starting...';
        
        const response = await fetch('/api/start-game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                game: 'avalon',
                num_players: numPlayers,
                language: language,
                user_agent_id: userAgentId,
                mode: 'participate',
            }),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Hide setup, show messages and input
            gameSetup.style.display = 'none';
            messagesContainer.style.display = 'block';
            inputContainer.style.display = 'flex';
            gameStatusElement.textContent = 'Game starting...';
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
    // gameStatus can be: 'running', 'finished', 'stopped', 'waiting', or boolean (for backward compatibility)
    let status = gameStatus;
    if (typeof gameStatus === 'boolean') {
        // Backward compatibility: convert boolean to status
        status = gameStatus ? 'running' : 'waiting';
    }
    
    const goHome = () => { window.location.href = '/'; };
    if (status === 'running') {
        backExitButton.textContent = 'Exit';
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
    // When reconnected, reset game state
    gameStarted = false;
    messageCount = 0;
    hideInputRequest();
    
    // Check if there's a game config from index.html
    const gameConfig = sessionStorage.getItem('gameConfig');
    if (gameConfig) {
        console.log('Found game config, starting game automatically...');
        sessionStorage.removeItem('gameConfig');
        
        // Parse and start game
        try {
            const config = JSON.parse(gameConfig);
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
        } catch (e) {
            console.error('Failed to parse game config:', e);
        }
    }
});

wsClient.onDisconnect(() => {
    console.log('Disconnected from game server');
    hideInputRequest();
});

// Initialize connection
wsClient.connect();

// Initialize button to show "Back to Home" on page load
updateBackExitButton(false);
