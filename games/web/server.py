# -*- coding: utf-8 -*-
"""Unified web server for Avalon + Diplomacy."""
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from games.web.game_state_manager import GameStateManager  #add gpt import unified gsm
from games.web.run_web_game import start_game_thread  #add gpt unified starter

# Global state manager
state_manager = GameStateManager()  #add gpt single gsm

app = FastAPI(title="Games Web Interface")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Root endpoint - serve unified index."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Games Web Interface</h1><p>index.html missing</p>")


def _page(path: str):
    f = STATIC_DIR / path
    if f.exists():
        return FileResponse(str(f))
    return HTMLResponse(f"<h1>Not found: {path}</h1>")


@app.get("/avalon/observe")
async def avalon_observe_page():
    return _page("avalon/observe.html")  #add gpt route avalon observe


@app.get("/avalon/participate")
async def avalon_participate_page():
    return _page("avalon/participate.html")  #add gpt route avalon participate


@app.get("/diplomacy/observe")
async def dip_observe_page():
    return _page("diplomacy/observe.html")  #add gpt route diplomacy observe


@app.get("/diplomacy/participate")
async def dip_participate_page():
    return _page("diplomacy/participate.html")  #add gpt route diplomacy participate


async def _handle_websocket_connection(websocket: WebSocket, path: str = ""):
    """Common handler for WebSocket connections."""
    connection_id = str(uuid.uuid4())
    state_manager.add_websocket_connection(connection_id, websocket)
    print(f"WebSocket connection established: {connection_id}")
    
    try:
        if state_manager.game_state.get("status") == "stopped":
            state_manager.reset()
            print(f"[WebSocket] Reset game state from 'stopped' to 'waiting' for new connection")
        
        await websocket.send_json(state_manager.format_game_state())
        await websocket.send_json({
            "type": "mode_info",
            "mode": state_manager.mode,
            "user_agent_id": state_manager.user_agent_id,
            "game": state_manager.game_state.get("game"),
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "user_input":
                    agent_id = message.get("agent_id")
                    content = message.get("content", "")
                    print(f"[WebSocket] Received user input: agent_id={agent_id}, content={content[:50]}...")
                    await state_manager.put_user_input(agent_id, content)
            except WebSocketDisconnect:
                print(f"WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError:
                try:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                except WebSocketDisconnect:
                    break
            except Exception as e:
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except WebSocketDisconnect:
                    break
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        state_manager.remove_websocket_connection(connection_id)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted from {websocket.client}")
    except Exception as e:
        print(f"Error accepting WebSocket connection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    await _handle_websocket_connection(websocket)


class StartGameRequest(BaseModel):
    game: str = "avalon"
    mode: str = "observe"
    language: str = "en"
    # avalon
    num_players: int = 5
    user_agent_id: int = 0
    # diplomacy
    human_power: Optional[str] = None
    max_phases: int = 20
    negotiation_rounds: int = 3
    power_models: Dict[str, str] | None = None


@app.post("/api/start-game")
async def start_game(request: StartGameRequest):
    """Start the game for avalon or diplomacy."""
    game = request.game
    mode = request.mode
    print(f"[API] Starting game: game={game}, mode={mode}")
    
    if game not in ["avalon", "diplomacy"]:
        raise HTTPException(status_code=400, detail="game must be 'avalon' or 'diplomacy'")
    if mode not in ["observe", "participate"]:
        raise HTTPException(status_code=400, detail="mode must be 'observe' or 'participate'")
    
    if state_manager.game_state.get("status") == "running":
        raise HTTPException(status_code=400, detail="Game is already running")
    
    state_manager.reset()
    state_manager.set_mode(mode, str(request.user_agent_id) if mode == "participate" else None, game=game)
    
    start_game_thread(
        state_manager=state_manager,
        game=game,
        mode=mode,
        language=request.language,
        num_players=request.num_players,
        user_agent_id=request.user_agent_id,
        human_power=request.human_power,
        max_phases=request.max_phases,
        negotiation_rounds=request.negotiation_rounds,
        power_models=request.power_models or {},
    )
    
    return {"status": "ok", "message": "Game started", "game": game, "mode": mode}


@app.post("/api/stop-game")
async def stop_game():
    """Stop the current game."""
    print("[API] Stopping game")
    
    if state_manager.game_state.get("status") != "running":
        raise HTTPException(status_code=400, detail="No game is currently running")
    
    state_manager.stop_game()
    await state_manager.broadcast_message(state_manager.format_game_state())
    stop_msg = state_manager.format_message(
        sender="System",
        content="Game stopped by user.",
        role="assistant",
    )
    await state_manager.broadcast_message(stop_msg)
    
    return {"status": "ok", "message": "Game stopped"}


@app.get("/api/history")
async def get_history():
    """History for diplomacy only."""
    if state_manager.game_state.get("game") != "diplomacy":
        raise HTTPException(status_code=404, detail="history only for diplomacy")
    history = state_manager.history
    return [
        {
            "index": i,
            "phase": (s.get("phase") or s.get("meta", {}).get("phase") or "Init"),
            "round": (s.get("round") if s.get("round") is not None else 0),
            "kind": s.get("kind", "state"),
        }
        for i, s in enumerate(history)
    ]


@app.get("/api/history/{index}")
async def get_history_item(index: int):
    """History item for diplomacy only."""
    if state_manager.game_state.get("game") != "diplomacy":
        raise HTTPException(status_code=404, detail="history only for diplomacy")
    if not (0 <= index < len(state_manager.history)):
        raise HTTPException(status_code=404, detail="Index out of bounds")
    s = dict(state_manager.history[index])
    s.setdefault("kind", "state")
    s.setdefault("meta", {})
    # Normalize phase/round to avoid null in frontend display
    s["phase"] = s.get("phase") or s["meta"].get("phase") or "Init"
    s["round"] = s.get("round") if s.get("round") is not None else 0
    s["index"] = index
    return s


@app.get("/api/options")
async def get_options(game: str = "diplomacy"):
    """Options for diplomacy (callable before start-game)."""
    if game != "diplomacy":
        raise HTTPException(status_code=404, detail="options only for diplomacy")
    from games.diplomacy.engine import DiplomacyConfig
    cfg = DiplomacyConfig.default()
    all_models = set()
    power_models = {}
    if cfg.models:
        for power in cfg.power_names:
            m = cfg.models.get(power) or cfg.models.get("default") or {}
            model_name = m.get("model_name", "qwen-plus")
            power_models[power] = model_name
            all_models.add(model_name)
        for v in cfg.models.values():
            if isinstance(v, dict) and v.get("model_name"):
                all_models.add(v["model_name"])
    else:
        for power in cfg.power_names:
            power_models[power] = "qwen-plus"
        all_models = {"qwen-turbo", "qwen-plus", "qwen-max"}
    return {
        "powers": cfg.power_names,
        "models": sorted(all_models),
        "power_models": power_models,
        "defaults": {
            "mode": "observe",
            "human_power": (cfg.power_names[0] if cfg.power_names else "ENGLAND"),
            "model_name": cfg.models.get("default", {}).get("model_name", "qwen-plus") if cfg.models else "qwen-plus",
            "max_phases": cfg.max_phases,
            "map_name": cfg.map_name,
            "negotiation_rounds": cfg.negotiation_rounds,
            "language": cfg.language,
        },
    }


def get_state_manager() -> GameStateManager:
    """Get the global state manager instance."""
    return state_manager


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

