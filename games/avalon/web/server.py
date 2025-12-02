# -*- coding: utf-8 -*-
"""Web server for Avalon game."""
import asyncio
import json
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
import uvicorn

from games.avalon.web.game_state_manager import GameStateManager


# Global state manager
state_manager = GameStateManager()

app = FastAPI(title="Avalon Game Web Interface")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory containing this file
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"

# Create static directory if it doesn't exist
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Root endpoint - redirect to index page."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("""
    <html>
        <head><title>Avalon Game</title></head>
        <body>
            <h1>Avalon Game Web Interface</h1>
            <p><a href="/observe">Observe Mode</a></p>
            <p><a href="/participate">Participate Mode</a></p>
        </body>
    </html>
    """)


@app.get("/observe")
async def observe_page():
    """Observe mode page."""
    observe_file = STATIC_DIR / "observe.html"
    if observe_file.exists():
        return FileResponse(str(observe_file))
    return HTMLResponse("<h1>Observe Mode</h1><p>Frontend not implemented yet.</p>")


@app.get("/participate")
async def participate_page():
    """Participate mode page."""
    participate_file = STATIC_DIR / "participate.html"
    if participate_file.exists():
        return FileResponse(str(participate_file))
    return HTMLResponse("<h1>Participate Mode</h1><p>Frontend not implemented yet.</p>")


async def _handle_websocket_connection(websocket: WebSocket, path: str = ""):
    """Common handler for WebSocket connections."""
    connection_id = str(uuid.uuid4())
    state_manager.add_websocket_connection(connection_id, websocket)
    print(f"WebSocket connection established: {connection_id}")
    
    try:
        # If game was stopped, reset to waiting state to allow new game
        current_status = state_manager.game_state.get("status")
        if current_status == "stopped":
            state_manager.reset()
            print(f"[WebSocket] Reset game state from 'stopped' to 'waiting' for new connection")
        
        # Send initial game state
        initial_state = state_manager.format_game_state()
        await websocket.send_json(initial_state)
        
        # Send mode information
        mode_info = {
            "type": "mode_info",
            "mode": state_manager.mode,
            "user_agent_id": state_manager.user_agent_id,
        }
        await websocket.send_json(mode_info)
        
        # Listen for messages from client
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle user input
                if message.get("type") == "user_input":
                    agent_id = message.get("agent_id")
                    content = message.get("content", "")
                    print(f"[WebSocket] Received user input: agent_id={agent_id}, content={content[:50]}...")
                    await state_manager.put_user_input(agent_id, content)
                
            except WebSocketDisconnect:
                # Client disconnected, break the loop
                print(f"WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError:
                # Only send error if connection is still open
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                except (WebSocketDisconnect, Exception):
                    # Connection closed, break the loop
                    break
            except Exception as e:
                # Only send error if connection is still open
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except (WebSocketDisconnect, Exception):
                    # Connection closed, break the loop
                    break
                
    except WebSocketDisconnect:
        # Normal disconnection, just log it
        print(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always remove the connection in finally block
        state_manager.remove_websocket_connection(connection_id)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    # Accept the WebSocket connection
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted from {websocket.client}")
    except Exception as e:
        print(f"Error accepting WebSocket connection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    await _handle_websocket_connection(websocket)


@app.websocket("/ws/{path:path}")
async def websocket_endpoint_with_path(websocket: WebSocket, path: str):
    """WebSocket endpoint for paths like /ws/game/... (for compatibility)."""
    # Log the path for debugging
    print(f"WebSocket connection attempt to /ws/{path}")
    
    # Accept and handle the same way as /ws
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted from {websocket.client} (path: /ws/{path})")
    except Exception as e:
        print(f"Error accepting WebSocket connection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    await _handle_websocket_connection(websocket, path)


@app.get("/api/game-state")
async def get_game_state():
    """Get current game state."""
    return state_manager.get_game_state()


@app.post("/api/set-mode")
async def set_mode(mode: str, user_agent_id: Optional[str] = None):
    """Set game mode.
    
    Args:
        mode: "observe" or "participate"
        user_agent_id: Agent ID for participate mode
    """
    if mode not in ["observe", "participate"]:
        raise HTTPException(status_code=400, detail="Mode must be 'observe' or 'participate'")
    
    state_manager.set_mode(mode, user_agent_id)
    return {"status": "ok", "mode": mode, "user_agent_id": user_agent_id}


class StartGameRequest(BaseModel):
    """Request model for starting a game."""
    num_players: int = 5
    language: str = "en"
    user_agent_id: int = 0
    mode: str = "observe"


@app.post("/api/start-game")
async def start_game(request: StartGameRequest):
    """Start the game.
    
    Request body:
        num_players: Number of players (default: 5)
        language: Language for prompts (default: "en")
        user_agent_id: Player ID to use UserAgent (for participate mode, default: 0)
        mode: "observe" or "participate" (default: "observe")
    """
    from games.avalon.web.run_web_game import start_game_thread
    
    num_players = request.num_players
    language = request.language
    user_agent_id = request.user_agent_id
    mode = request.mode
    
    print(f"[API] Starting game: mode={mode}, num_players={num_players}, language={language}, user_agent_id={user_agent_id}")
    
    # Check if game is already running
    current_status = state_manager.game_state.get("status")
    print(f"[API] Current game status: {current_status}")
    
    if current_status == "running":
        raise HTTPException(status_code=400, detail="Game is already running")
    
    # Validate mode
    if mode not in ["observe", "participate"]:
        raise HTTPException(status_code=400, detail="Mode must be 'observe' or 'participate'")
    
    # Reset state manager (this will reset status to "waiting" and clear should_stop flag)
    # This ensures we can start a new game even if previous game was stopped or finished
    state_manager.reset()
    print(f"[API] State manager reset. New status: {state_manager.game_state.get('status')}")
    
    # Set mode
    state_manager.set_mode(mode, str(user_agent_id) if mode == "participate" else None)
    
    # Start game thread
    start_game_thread(
        state_manager=state_manager,
        num_players=num_players,
        language=language,
        user_agent_id=user_agent_id,
        mode=mode,
    )
    
    return {
        "status": "ok",
        "message": "Game started",
        "num_players": num_players,
        "language": language,
        "user_agent_id": user_agent_id,
        "mode": mode,
    }


@app.post("/api/stop-game")
async def stop_game():
    """Stop the current game."""
    print("[API] Stopping game")
    
    if state_manager.game_state.get("status") != "running":
        raise HTTPException(status_code=400, detail="No game is currently running")
    
    # Stop the game
    state_manager.stop_game()
    
    # Broadcast stopped state
    await state_manager.broadcast_message(state_manager.format_game_state())
    
    stop_msg = state_manager.format_message(
        sender="System",
        content="Game stopped by user.",
        role="assistant",
    )
    await state_manager.broadcast_message(stop_msg)
    
    return {
        "status": "ok",
        "message": "Game stopped",
    }


def get_state_manager() -> GameStateManager:
    """Get the global state manager instance.
    
    Returns:
        GameStateManager instance
    """
    return state_manager


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

