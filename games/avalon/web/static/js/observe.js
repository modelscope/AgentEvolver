// Observe mode JavaScript
const wsClient = new WebSocketClient();
const messagesContainer = document.getElementById('messages-container');
const gameStatusElement = document.getElementById('game-status');
const gameSetup = document.getElementById('game-setup');
const startGameBtn = document.getElementById('start-game-btn');
const numPlayersSelect = document.getElementById('num-players');
const languageSelect = document.getElementById('language');
const backExitButton = document.getElementById('back-exit-button');

let messageCount = 0;
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
    } else if (message.sender.startsWith('Player')) {
        messageDiv.classList.add('agent');
    } else {
        messageDiv.classList.add('user');
    }
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-sender">${escapeHtml(message.sender)}</span>
            <span class="message-time">${formatTime(message.timestamp)}</span>
        </div>
        <div class="message-content">${escapeHtml(message.content)}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateGameState(state) {
    if (state.phase !== null && state.phase !== undefined) {
        const phases = ['Team Selection', 'Team Voting', 'Quest Voting', 'Assassination'];
        const phaseName = phases[state.phase] || 'Unknown';
        let statusText = `Phase: ${phaseName}`;
        
        if (state.mission_id !== null) {
            statusText += ` | Mission: ${state.mission_id}`;
        }
        if (state.round_id !== null) {
            statusText += ` | Round: ${state.round_id}`;
        }
        if (state.leader !== null) {
            statusText += ` | Leader: Player ${state.leader}`;
        }
        
        gameStatusElement.textContent = statusText;
    } else {
        gameStatusElement.textContent = 'Waiting for game to start...';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

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
        gameStarted = true;
        // Change button to Exit
        updateBackExitButton(true);
    }
    // Handle game stopped - reset state and show setup
    if (state.status === 'stopped') {
        gameStarted = false;
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
        updateBackExitButton(false);
        // Reset message count and clear messages
        messageCount = 0;
        messagesContainer.innerHTML = '<p style="text-align: center; color: #999; padding: 20px;">Game stopped. You can start a new game.</p>';
        // Don't redirect - allow user to start new game or go back manually
    }
    // Handle game finished - allow starting new game
    if (state.status === 'finished') {
        // Game finished normally, can start new game
        gameStarted = false;
    }
    // Handle waiting state - show setup
    if (state.status === 'waiting') {
        gameStarted = false;
        gameSetup.style.display = 'block';
        messagesContainer.style.display = 'none';
        updateBackExitButton(false);
    }
});

wsClient.onMessage('mode_info', (info) => {
    console.log('Mode info:', info);
    if (info.mode !== 'observe') {
        console.warn('Expected observe mode, got:', info.mode);
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

async function startGame() {
    const numPlayers = parseInt(numPlayersSelect.value);
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
                num_players: numPlayers,
                language: language,
                mode: 'observe',
            }),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Hide setup, show messages
            gameSetup.style.display = 'none';
            messagesContainer.style.display = 'block';
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

function updateBackExitButton(isGameRunning) {
    if (isGameRunning) {
        // Game is running: show Exit button
        backExitButton.textContent = 'Exit';
        backExitButton.title = 'Exit Game';
        backExitButton.href = '#';
        backExitButton.onclick = async (e) => {
            e.preventDefault();
            if (confirm('Are you sure you want to exit the game?')) {
                try {
                    const response = await fetch('/api/stop-game', {
                        method: 'POST',
                    });
                    if (response.ok) {
                        // Wait for stopped state, then redirect
                        // The redirect will happen in game_state handler
                    } else {
                        alert('Failed to stop game');
                    }
                } catch (error) {
                    console.error('Error stopping game:', error);
                    alert('Error stopping game');
                }
            }
        };
    } else {
        // Game not running: show Back to Home button
        backExitButton.textContent = '← Back to Home';
        backExitButton.title = 'Back to Home';
        backExitButton.href = '/';
        backExitButton.onclick = null;
    }
}

startGameBtn.addEventListener('click', startGame);

// Connect when page loads
wsClient.onConnect(() => {
    console.log('Connected to game server');
    // When reconnected, reset game state
    gameStarted = false;
    messageCount = 0;
});

wsClient.onDisconnect(() => {
    console.log('Disconnected from game server');
});

// Initialize connection
wsClient.connect();

// Initialize button to show "Back to Home" on page load
updateBackExitButton(false);

