# -*- coding: utf-8 -*-
"""Game state manager for web application."""
import asyncio
import queue
from typing import Dict, Optional, Any
import json
from datetime import datetime


class GameStateManager:
    """Manages game state, message queues, and WebSocket connections."""
    
    def __init__(self):
        """Initialize the game state manager."""
        # User input queues: {agent_id: queue.Queue}
        # Use thread-safe queue.Queue instead of asyncio.Queue because
        # game runs in a separate thread with different event loop
        self.input_queues: Dict[str, queue.Queue] = {}
        
        # Message broadcast queue
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # WebSocket connections: {connection_id: websocket}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Current game state
        self.game_state: Dict[str, Any] = {
            "phase": None,
            "mission_id": None,
            "round_id": None,
            "leader": None,
            "status": "waiting",  # waiting, running, finished
        }
        
        # Game mode: "observe" or "participate"
        self.mode: Optional[str] = None
        
        # User agent ID (for participate mode)
        self.user_agent_id: Optional[str] = None
        
        # Flag to stop the game
        self.should_stop: bool = False
        
        # Game thread reference (for stopping)
        self.game_thread: Optional[Any] = None
    
    def set_mode(self, mode: str, user_agent_id: Optional[str] = None):
        """Set the game mode.
        
        Args:
            mode: "observe" or "participate"
            user_agent_id: Agent ID for participate mode
        """
        self.mode = mode
        self.user_agent_id = user_agent_id
    
    def stop_game(self):
        """Stop the current game."""
        self.should_stop = True
        self.update_game_state(status="stopped")
    
    def reset(self):
        """Reset the game state manager."""
        self.should_stop = False
        self.game_thread = None
        self.game_state = {
            "phase": None,
            "mission_id": None,
            "round_id": None,
            "leader": None,
            "status": "waiting",
        }
    
    def set_game_thread(self, thread: Any):
        """Set the game thread reference.
        
        Args:
            thread: The game thread object
        """
        self.game_thread = thread
    
    async def put_user_input(self, agent_id: str, content: str):
        """Put user input into the queue for the specified agent.
        
        Args:
            agent_id: Agent identifier
            content: User input content
        """
        # Create queue if it doesn't exist
        if agent_id not in self.input_queues:
            self.input_queues[agent_id] = queue.Queue()
        
        print(f"[GameStateManager] Putting user input into queue: agent_id={agent_id}, content={content[:50]}...")
        # queue.Queue.put is synchronous, but we're in async context, so use run_in_executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.input_queues[agent_id].put, content)
        print(f"[GameStateManager] User input queued successfully for agent_id={agent_id}")
    
    async def get_user_input(self, agent_id: str, timeout: Optional[float] = None) -> str:
        """Get user input from the queue for the specified agent.
        
        Args:
            agent_id: Agent identifier
            timeout: Optional timeout in seconds
            
        Returns:
            User input content
        """
        print(f"[GameStateManager.get_user_input] Called with agent_id={agent_id}")
        print(f"[GameStateManager.get_user_input] Current queues: {list(self.input_queues.keys())}")
        
        # Create queue if it doesn't exist
        if agent_id not in self.input_queues:
            print(f"[GameStateManager.get_user_input] Creating new queue for agent_id={agent_id}")
            self.input_queues[agent_id] = queue.Queue()
        else:
            print(f"[GameStateManager.get_user_input] Queue exists, size={self.input_queues[agent_id].qsize()}")
        
        try:
            print(f"[GameStateManager.get_user_input] About to wait for input, queue_size={self.input_queues[agent_id].qsize()}")
            # Use run_in_executor to wait on thread-safe queue in async context
            loop = asyncio.get_event_loop()
            
            def get_from_queue():
                """Blocking function to get from queue."""
                if timeout:
                    try:
                        return self.input_queues[agent_id].get(timeout=timeout)
                    except queue.Empty:
                        raise TimeoutError(f"Timeout waiting for user input from agent {agent_id}")
                else:
                    return self.input_queues[agent_id].get()  # Block until item available
            
            print(f"[GameStateManager.get_user_input] Waiting indefinitely for queue: {agent_id}")
            result = await loop.run_in_executor(None, get_from_queue)
            print(f"[GameStateManager.get_user_input] Got result from queue")
            print(f"[GameStateManager.get_user_input] Got user input: agent_id={agent_id}, content={result[:50]}...")
            return result
        except TimeoutError:
            raise
        except Exception as e:
            print(f"[GameStateManager.get_user_input] Exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all WebSocket connections.
        
        Args:
            message: Message dictionary to broadcast
        """
        await self.message_queue.put(message)
        
        # Send to all connected WebSockets
        disconnected = []
        for conn_id, websocket in self.websocket_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error sending message to connection {conn_id}: {e}")
                disconnected.append(conn_id)
        
        # Remove disconnected connections
        for conn_id in disconnected:
            self.websocket_connections.pop(conn_id, None)
    
    def add_websocket_connection(self, connection_id: str, websocket: Any):
        """Add a WebSocket connection.
        
        Args:
            connection_id: Unique connection identifier
            websocket: WebSocket connection object
        """
        self.websocket_connections[connection_id] = websocket
    
    def remove_websocket_connection(self, connection_id: str):
        """Remove a WebSocket connection.
        
        Args:
            connection_id: Connection identifier to remove
        """
        self.websocket_connections.pop(connection_id, None)
    
    def update_game_state(self, **kwargs):
        """Update game state.
        
        Args:
            **kwargs: State fields to update
        """
        self.game_state.update(kwargs)
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state.
        
        Returns:
            Current game state dictionary
        """
        return self.game_state.copy()
    
    def format_message(self, sender: str, content: str, role: str = "assistant") -> Dict[str, Any]:
        """Format a message for WebSocket transmission.
        
        Args:
            sender: Message sender name
            content: Message content
            role: Message role (assistant/user)
            
        Returns:
            Formatted message dictionary
        """
        return {
            "type": "message",
            "sender": sender,
            "content": content,
            "role": role,
            "timestamp": datetime.now().isoformat(),
        }
    
    def format_game_state(self) -> Dict[str, Any]:
        """Format game state for WebSocket transmission.
        
        Returns:
            Formatted game state dictionary
        """
        return {
            "type": "game_state",
            **self.game_state,
        }
    
    def format_user_input_request(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """Format a user input request for WebSocket transmission.
        
        Args:
            agent_id: Agent identifier
            prompt: Prompt text for user input
            
        Returns:
            Formatted user input request dictionary
        """
        return {
            "type": "user_input_request",
            "agent_id": agent_id,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
        }

