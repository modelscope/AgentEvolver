# -*- coding: utf-8 -*-
"""Game state manager for web application."""
import asyncio
import queue
from typing import Dict, Optional, Any, List
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

        self.history: List[Dict[str, Any]] = [] # added mxj 新增历史消息存储
        
        # Current game state
        # Common fields: phase, round, status
        # Game specific fields go into 'meta'
        self.game_state: Dict[str, Any] = {
            "phase": "Initializing",
            "round": 0,
            "status": "waiting",  # waiting, running, finished, stopped
            "meta": {}            # Container for game-specific data (e.g. mission_id, map_svg)
        } # added mxj 新增meta
        
        # Game mode: "observe" or "participate"
        self.mode: Optional[str] = None
        
        # User agent ID (for participate mode)
        self.user_agent_id: Optional[str] = None
        
        # Flag to stop the game
        self.should_stop: bool = False
        
        # Game thread reference (for stopping)
        self.game_thread: Optional[Any] = None

        # added mxj：保存 WebSocket 所在的 event loop（uvicorn 那条）
        self.ws_loop: Optional[asyncio.AbstractEventLoop] = None


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
        self.input_queues = {}
        self.history = [] # added mxj 新增历史消息存储
        self.game_state = {
            "phase": "Initializing",
            "round": 0,
            "status": "waiting",
            "meta": {} # added mxj 新增meta
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
        
        if agent_id not in self.input_queues:
            self.input_queues[agent_id] = queue.Queue()
        
        # Optional: Broadcast a request for input (useful for frontend)
        # req_msg = self.format_user_input_request(agent_id, "Waiting for input...")
        # await self.broadcast_message(req_msg)

        try:
            loop = asyncio.get_event_loop()
            
            def get_from_queue():
                if timeout:
                    try:
                        return self.input_queues[agent_id].get(timeout=timeout)
                    except queue.Empty:
                        raise TimeoutError(f"Timeout waiting for user input from agent {agent_id}")
                else:
                    return self.input_queues[agent_id].get()
            
            result = await loop.run_in_executor(None, get_from_queue)
            return result
        except TimeoutError:
            raise
        except Exception as e:
            print(f"[GameStateManager.get_user_input] Exception: {e}")
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

        # added mxj 记录 ws 所在 event loop
        try:
            self.ws_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
    
    def remove_websocket_connection(self, connection_id: str):
        """Remove a WebSocket connection.
        
        Args:
            connection_id: Connection identifier to remove
        """
        self.websocket_connections.pop(connection_id, None)
    
    def update_game_state(self, **kwargs):
        """Update game state and trigger broadcast.
        
        Args:
            **kwargs: State fields to update. 
                      'phase', 'round', 'status' update base state.
                      Others update 'meta'.
        """
        updated = False
        for k, v in kwargs.items():
            if k in ["phase", "round", "status"]:
                if self.game_state.get(k) != v:
                    self.game_state[k] = v
                    updated = True
            else:
                # added mxj 保存历史快照
                snapshot = {
                    "phase": self.game_state.get("phase", "Unknown"),
                    "round": self.game_state.get("round", 0),
                    "meta": {}
                }
                # Handle game-specific fields in meta
                if k not in self.game_state["meta"] or self.game_state["meta"][k] != v:
                    self.game_state["meta"][k] = v
                    updated = True # added mxj 新增meta处理
                    snapshot["meta"][k] = v
                    
                # Avoid duplicates (same phase) or update existing
                if not self.history or self.history[-1]["phase"] != snapshot["phase"]:
                    self.history.append(snapshot)
                else:
                    self.history[-1]["phase"] = snapshot["phase"]
                    self.history[-1]["round"] = snapshot["round"]
                    # 合并 meta
                    self.history[-1].setdefault("meta", {})
                    self.history[-1]["meta"].update(snapshot["meta"])

        
        if updated:
            self.game_state.update(kwargs)
            msg = self.format_game_state()
            if self.ws_loop and self.ws_loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast_message(msg), self.ws_loop)
            else:
                pass

    def save_history_snapshot(self, kind: str = "state"): #added mxj 保存快照
        """保存历史快照"""
        from copy import deepcopy
        snap = {
            "phase": self.game_state.get("phase", "S1901M"),
            "round": self.game_state.get("round", 0),
            "kind": kind,  # "orders" / "result" / "init" ...
            "timestamp": datetime.now().isoformat(),
            "meta": deepcopy(self.game_state.get("meta", {})), 
        }
        self.history.append(snap)
    
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
            **self.game_state, # Flattens phase, round, status, meta
        }
    
    def format_user_input_request(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """Format a user input request for WebSocket transmission."""
        return {
            "type": "user_input_request",
            "agent_id": agent_id,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
        }
    

