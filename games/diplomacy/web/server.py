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
import os
from games.diplomacy.web.game_state_manager import GameStateManager


# Global state manager
state_manager = GameStateManager()

app = FastAPI(title="Diplomacy Game Web Interface")

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

@app.get("/api/history") # added mxj history
async def get_history():
    return [
        {
            "index": i,
            "phase": s.get("phase", "Unknown"),
            "round": s.get("round", 0),
            "kind": s.get("kind", "state"),
            # 可选：如果你有 timestamp 也可以带上
            # "timestamp": s.get("timestamp"),
        }
        for i, s in enumerate(state_manager.history)
    ]


@app.get("/api/history/{index}") # added mxj history
async def get_history_item(index: int):
    if not (0 <= index < len(state_manager.history)):
        raise HTTPException(status_code=404, detail="Index out of bounds")

    s = dict(state_manager.history[index]) 
    s.setdefault("kind", "state")
    s.setdefault("meta", {})
    s["index"] = index 
    return s


@app.get("/observe")
async def observe_page():
    """Observe mode page."""
    observe_file = STATIC_DIR / "index.html"
    if observe_file.exists():
        return FileResponse(str(observe_file))
    return HTMLResponse("<h1>Observe Mode</h1><p>Frontend not implemented yet.</p>")

@app.get("/api/options") # added mxj options api
async def get_options():
    cfg = DiplomacyBasicConfig.default()

    models = [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
    ]

    return {
        "powers": cfg.power_names,
        "models": models,
        "defaults": {
            "mode": "observe",
            "human_power": (cfg.power_names[0] if cfg.power_names else "ENGLAND"),
            "model_name": os.getenv("MODEL_NAME", "qwen-plus"),
            "max_phases": 20,
            "map_name": "standard",
            "negotiation_rounds": 3,
            "language": "en",
        },
    }

@app.get("/participate")
async def participate_page():
    """Participate mode page."""
    participate_file = STATIC_DIR / "index.html"
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


from pydantic import BaseModel

class StartGameRequest(BaseModel): # added mxj start game api
    mode: str = "observe"               # "observe" | "participate"
    human_power: str | None = None      # participate 时必填，例如 "ENGLAND"
    model_name: str = "qwen-plus"
    max_phases: int = 20
    map_name: str = "standard"
    negotiation_rounds: int = 3
    language: str = "en"

from games.diplomacy.engine import DiplomacyBasicConfig
from games.diplomacy.web.run_web_game import start_game_thread
from fastapi import HTTPException


@app.post("/api/start-game")
async def start_game(request: StartGameRequest):
    # 防止重复启动
    if state_manager.game_thread and state_manager.game_thread.is_alive():
        raise HTTPException(status_code=400, detail="Game already running")

    mode = request.mode
    if mode not in ("observe", "participate"):
        raise HTTPException(status_code=400, detail="mode must be observe or participate")

    if mode == "participate" and not request.human_power:
        raise HTTPException(status_code=400, detail="human_power is required in participate mode")

    # 1) 组装 DiplomacyBasicConfig（对齐 AvalonBasicConfig 用法）
    os.environ["MODEL_NAME"] = request.model_name
    config = DiplomacyBasicConfig.default()
    config.map_name = request.map_name
    config.max_phases = request.max_phases
    config.negotiation_rounds = request.negotiation_rounds
    config.language = request.language
    config.human_power = request.human_power if mode == "participate" else None

    # 2) 记录 mode/human_power 给前端显示输入框用（前端用 human_power 判断 participate）
    state_manager.set_mode(mode, config.human_power if mode == "participate" else None)
    state_manager.update_game_state(
        status="starting",
        human_power=config.human_power if mode == "participate" else None,
    )

    # 3) 启动线程：只传 config（对齐 Avalon 的 start_game_thread 形态）
    start_game_thread(
        state_manager=state_manager,
        config=config,
        mode=mode,
    )

    return {"ok": True, "mode": mode, "human_power": config.human_power}




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

