// WebSocket client for Avalon game
class WebSocketClient {
    constructor(url = null) {
        // Auto-detect WebSocket URL based on current page
        if (!url) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            url = `${protocol}//${host}/ws`;
        }
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.messageHandlers = new Map();
        this.onConnectCallbacks = [];
        this.onDisconnectCallbacks = [];
    }

    connect() {
        try {
            console.log(`Connecting to WebSocket: ${this.url}`);
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                this.onConnectCallbacks.forEach(callback => callback());
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                this.onDisconnectCallbacks.forEach(callback => callback());
                this.attemptReconnect();
            };
        } catch (error) {
            console.error('Error connecting WebSocket:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.updateConnectionStatus('connecting');
            setTimeout(() => {
                console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
                this.connect();
            }, this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus('disconnected');
        }
    }

    handleMessage(message) {
        const type = message.type;
        if (this.messageHandlers.has(type)) {
            this.messageHandlers.get(type).forEach(handler => handler(message));
        }
    }

    onMessage(type, handler) {
        if (!this.messageHandlers.has(type)) {
            this.messageHandlers.set(type, []);
        }
        this.messageHandlers.get(type).push(handler);
    }

    onConnect(callback) {
        this.onConnectCallbacks.push(callback);
    }

    onDisconnect(callback) {
        this.onDisconnectCallbacks.push(callback);
    }

    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket is not open');
        }
    }

    sendUserInput(agentId, content) {
        this.send({
            type: 'user_input',
            agent_id: agentId,
            content: content
        });
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.className = `connection-status ${status}`;
            statusElement.textContent = status === 'connected' ? '🟢 Connected' :
                                       status === 'connecting' ? '🟡 Connecting...' :
                                       '🔴 Disconnected';
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}
