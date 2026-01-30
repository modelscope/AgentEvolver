# -*- coding: utf-8 -*-
"""Unified web server for Avalon + Diplomacy."""
import asyncio
import copy
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

from games.web.game_state_manager import GameStateManager
from games.web.run_web_game import start_game_thread
from games.utils import load_config
from games.evaluation.leaderboard.leaderboard_db import LeaderboardDB
import os
state_manager = GameStateManager()

app = FastAPI(title="Games Web Interface")

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

@app.get("/favicon.ico")
async def favicon():
    favicon_png = STATIC_DIR / "favicon.png"
    if favicon_png.exists():
        return FileResponse(
            str(favicon_png),
            media_type="image/png"
        )

@app.get("/avalon/observe")
async def avalon_observe_page():
    """Avalon observe page"""
    return _page("avalon/observe.html")


@app.get("/avalon/participate")
async def avalon_participate_page():
    """Avalon participate page"""
    return _page("avalon/participate.html")


@app.get("/diplomacy/observe")
async def dip_observe_page():
    """Diplomacy observe page"""
    return _page("diplomacy/observe.html")


@app.get("/diplomacy/participate")
async def dip_participate_page():
    """Diplomacy participate page"""
    return _page("diplomacy/participate.html")


async def _handle_websocket_connection(websocket: WebSocket, path: str = ""):
    """WebSocket connection handler: receive user input, push game state and messages"""
    connection_id = str(uuid.uuid4())
    state_manager.add_websocket_connection(connection_id, websocket)
    
    try:
        if state_manager.game_state.get("status") == "stopped":
            state_manager.reset()
        
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
                    await state_manager.put_user_input(agent_id, content)
            except WebSocketDisconnect:
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
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        state_manager.remove_websocket_connection(connection_id)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    try:
        await websocket.accept()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
    
    await _handle_websocket_connection(websocket)


class StartGameRequest(BaseModel):
    """Start game request parameters"""
    game: str = "avalon"
    mode: str = "observe"
    language: str = "en"
    agent_configs: Dict[int, Dict[str, str]] | None = None  # Frontend agent config {portrait_id: {base_model, api_base, api_key}}
    # Avalon parameters
    num_players: int = 5
    user_agent_id: int = 0
    preset_roles: list[dict] | None = None  # Fixed role assignment from frontend preview
    selected_portrait_ids: list[int] | None = None  # Frontend selected portrait ids [1-15]
    # Diplomacy parameters
    human_power: Optional[str] = None
    max_phases: int = 20
    negotiation_rounds: int = 3
    power_names: list[str] | None = None  # Shuffled power order from frontend
    power_models: Dict[str, str] | None = None


@app.post("/api/start-game")
async def start_game(request: StartGameRequest):
    """Start game: run Avalon or Diplomacy in background thread"""
    game = request.game
    mode = request.mode
    
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
        preset_roles=request.preset_roles,
        selected_portrait_ids=request.selected_portrait_ids,
        agent_configs=request.agent_configs or {},
        human_power=request.human_power,
        max_phases=request.max_phases,
        negotiation_rounds=request.negotiation_rounds,
        power_names=request.power_names,
        power_models=request.power_models or {},
    )
    
    return {"status": "ok", "message": "Game started", "game": game, "mode": mode}


@app.post("/api/stop-game")
async def stop_game():
    """Stop current game"""
    if state_manager.game_state.get("status") != "running":
        raise HTTPException(status_code=400, detail="No game is currently running")
    
    state_manager.stop_game()
    
    if hasattr(state_manager, '_game_task') and state_manager._game_task:
        try:
            state_manager._game_task.cancel()
        except Exception:
            pass
    
    await state_manager.broadcast_message(state_manager.format_game_state())
    
    stop_msg = state_manager.format_message(
        sender="System",
        content="Game stopped by user.",
        role="assistant",
    )
    await state_manager.broadcast_message(stop_msg)
    
    import asyncio
    await asyncio.sleep(0.1)
    
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
    s["phase"] = s.get("phase") or s["meta"].get("phase") or "Init"
    s["round"] = s.get("round") if s.get("round") is not None else 0
    s["index"] = index
    return s


@app.get("/api/options")
async def get_options(game: str | None = None):
    """Get game configuration options for frontend pre-fill"""
    import os
    import yaml

    def _to_ui_lang(raw: str | None) -> str:
        lang = (raw or "").lower().strip()
        return "zh" if lang in {"zh", "zn", "cn", "zh-cn", "zh_cn", "chinese"} else "en"

    if not game:
        web_config_path = Path(__file__).parent / "web_config.yaml"
        result = {"portraits": {}, "default_model": {}}
        if web_config_path.exists():
            web_cfg = load_config(web_config_path)
            
            if isinstance(web_cfg, dict):
                portraits_cfg = web_cfg.get('portraits', {}) or {}
                sanitized_portraits = {}
                for pid, pdata in portraits_cfg.items():
                    if not isinstance(pdata, dict):
                        continue
                    portrait_copy = copy.deepcopy(pdata)
                    model_cfg = portrait_copy.get("model")
                    if isinstance(model_cfg, dict):
                        model_cfg["api_key"] = ""
                    sanitized_portraits[pid] = portrait_copy
                result["portraits"] = sanitized_portraits

                default_role = web_cfg.get("default_role", {})
                default_model = {}
                if isinstance(default_role, dict):
                    model_cfg = default_role.get("model", {}) or {}
                    agent_cfg = default_role.get("agent", {}) or {}
                    
                    default_model["model_name"] = model_cfg.get("model_name", "")
                    default_model["api_base"] = model_cfg.get("url", "") or model_cfg.get("api_base", "") or os.getenv("OPENAI_BASE_URL", "")
                    default_model["api_key"] = ""
                    default_model["temperature"] = model_cfg.get("temperature", 0.7)
                    default_model["max_tokens"] = model_cfg.get("max_tokens", 2048)
                    
                    if agent_cfg:
                        default_model["agent_class"] = agent_cfg.get("type", "")
                
                result["default_model"] = default_model
        return result

    if game == "diplomacy":
        yaml_path = os.environ.get("DIPLOMACY_CONFIG_YAML", "games/games/diplomacy/configs/default_config.yaml")
        diplomacy_cfg = load_config(yaml_path)['game']
        lang = _to_ui_lang(diplomacy_cfg['language'])

        return {
            "powers": diplomacy_cfg['power_names'],
            "defaults": {
                "mode": "observe",
                "human_power": (diplomacy_cfg['power_names'][0] if diplomacy_cfg['power_names'] else "ENGLAND"),
                "max_phases": diplomacy_cfg['max_phases'],
                "map_name": diplomacy_cfg['map_name'],
                "negotiation_rounds": diplomacy_cfg['negotiation_rounds'],
                "language": lang,
            },
        }

    if game == "avalon":
        yaml_path = os.environ.get("AVALON_CONFIG_YAML", "games/games/avalon/configs/default_config.yaml")
        avalon_cfg = load_config(yaml_path)['game']

        return {
            "roles": avalon_cfg.get("roles_name", []),
            "defaults": {
                "num_players": int(avalon_cfg.get("num_players", 5) or 5),
                "language": _to_ui_lang(str(avalon_cfg.get("language", "en"))),
            },
        }

    raise HTTPException(status_code=404, detail="options only for avalon/diplomacy")


@app.get("/api/leaderboard/{game}")
async def get_leaderboard(game: str):
    """Get leaderboard data for a specific game."""
    if game not in ["avalon", "diplomacy"]:
        raise HTTPException(status_code=400, detail="game must be 'avalon' or 'diplomacy'")
    
    try:
        # Load arena config to get models list
        if game == "avalon":
            yaml_path = os.environ.get("AVALON_CONFIG_YAML", "games/games/avalon/configs/arena_config.yaml")
        else:  # diplomacy
            yaml_path = os.environ.get("DIPLOMACY_CONFIG_YAML", "games/games/diplomacy/configs/arena_config.yaml")
        
        config_dict = load_config(yaml_path)
        arena_config = config_dict.get('arena', {})
        models = arena_config.get('models', [])
        
        # Load leaderboard database
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / f"games/evaluation/leaderboard/leaderboard_{game}.json"
        
        if not db_path.exists():
            # Return empty leaderboard if database doesn't exist
            # Get role names based on game type
            if game == "avalon":
                role_names = ['Merlin', 'Servant', 'Assassin', 'Minion']
            elif game == "diplomacy":
                role_names = []  # Will be populated when games are played
            else:
                role_names = []
            
            return {
                "game": game,
                "total_games": 0,
                "updated_at": "",
                "balance": {
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "std": 0,
                    "balance_ratio": 1.0
                },
                "role_names": role_names,
                "models": []
            }
        
        leaderboard_db = LeaderboardDB(str(db_path))
        leaderboard_data = leaderboard_db.get_leaderboard_data()
        
        # Get role names based on game type
        if game == "avalon":
            role_names = ['Merlin', 'Servant', 'Assassin', 'Minion']
        elif game == "diplomacy":
            # Extract power names from role_stats
            all_roles = set()
            for model_stat in leaderboard_data.get('models', {}).values():
                all_roles.update(model_stat.get('role_stats', {}).keys())
            role_names = sorted(list(all_roles)) if all_roles else []
        else:
            role_names = []
        
        # Sort models by Elo (using models list from config for ordering)
        model_stats = leaderboard_data.get('models', {})
        config_models = models if isinstance(models, list) else []
        all_models = list(dict.fromkeys(config_models + list(model_stats.keys())))
        sorted_models = sorted(all_models,
                      key=lambda m: model_stats.get(m, {}).get('elo', 0),
                      reverse=True)
        
        # Format response with model names included
        formatted_models = []
        for model in sorted_models:
            if model in model_stats:
                stats = model_stats[model]
                formatted_models.append({
                    "name": model,
                    "elo": stats.get('elo', 1500),
                    "win_rate": round(stats.get('win_rate', 0), 1),
                    "total_games": stats.get('total_games', 0),
                    "total_wins": stats.get('total_wins', 0),
                    "role_stats": stats.get('role_stats', {})
                })
        
        return {
            "game": game,
            "total_games": leaderboard_data.get('total_games', 0),
            "updated_at": leaderboard_data.get('updated_at', ''),
            "balance": leaderboard_data.get('balance', {}),
            "role_names": role_names,
            "models": formatted_models
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load leaderboard: {str(e)}")


@app.get("/leaderboard")
async def leaderboard_page():
    """Leaderboard page - default to avalon"""
    return _page("leaderboard.html")


@app.get("/leaderboard/{game}")
async def leaderboard_game_page(game: str):
    """Leaderboard page for specific game"""
    if game not in ["avalon", "diplomacy"]:
        raise HTTPException(status_code=404, detail="game must be 'avalon' or 'diplomacy'")
    return _page("leaderboard.html")


def get_state_manager() -> GameStateManager:
    """Get the global state manager instance."""
    return state_manager


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

