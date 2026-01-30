# -*- coding: utf-8 -*-
"""Unified web game launcher for Avalon + Diplomacy."""
import argparse
import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any
from games.utils import load_config
from games.evaluation.leaderboard.leaderboard_db import LeaderboardDB
# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agentscope
from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.tool import Toolkit

from games.web.game_state_manager import GameStateManager
from games.web.web_agent import WebUserAgent, ObserveAgent

# Avalon imports
from games.agents.thinking_react_agent import ThinkingReActAgent
from games.utils import load_agent_class
from games.games.avalon.game import avalon_game
from games.games.avalon.engine import AvalonBasicConfig
from games.games.avalon.workflows.eval_workflow import RoleManager
from games.games.avalon.engine import AvalonGameEnvironment


from games.games.diplomacy.engine import DiplomacyConfig
from games.games.diplomacy.game import diplomacy_game
from games.utils import (
    load_config,
    create_agent_from_config,
    create_model_from_config,
    deep_merge,
)

import copy

def _get_role_config(
    config_dict: Dict[str, Any],
    role_identifiers: list[str],
    frontend_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Get complete role configuration with priority-based lookup for multiple role identifiers.
    
    Args:
        config_dict: Configuration dict (contains default_role and roles)
        role_identifiers: List of role identifiers in priority order (e.g., ["Merlin_0", "Merlin"] or ["AUSTRIA"])
        frontend_cfg: Frontend config override (optional)
    
    Returns:
        Merged role configuration
    """
    default_role = config_dict.get('default_role', {})
    roles_config = config_dict.get('roles', {})
    
    if not isinstance(default_role, dict):
        default_role = {}
    if not isinstance(roles_config, dict):
        roles_config = {}
    
    # Start with default_role
    role_config = copy.deepcopy(default_role)
    
    # Find role-specific config by priority
    specific_role_config = None
    for identifier in role_identifiers:
        # Try exact match (case-sensitive)
        if identifier in roles_config:
            specific_role_config = roles_config[identifier]
            break
        # Try case-insensitive match
        identifier_upper = identifier.upper()
        for k, v in roles_config.items():
            if k.upper() == identifier_upper:
                specific_role_config = v
                break
        if specific_role_config:
            break
    
    # Merge role-specific config
    if specific_role_config and isinstance(specific_role_config, dict):
        role_config = deep_merge(role_config, specific_role_config)
    
    # Finally apply frontend config override if present
    if frontend_cfg and isinstance(frontend_cfg, dict):
        frontend_role_config = {}
        if frontend_cfg.get("base_model"):
            frontend_role_config["model"] = {
                "model_name": frontend_cfg.get("base_model"),
                "url": frontend_cfg.get("api_base") or os.getenv("OPENAI_BASE_URL", ""),
                "api_key": frontend_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", ""),
            }
        if frontend_cfg.get("agent_class"):
            frontend_role_config["agent"] = {
                "type": frontend_cfg.get("agent_class"),
                "kwargs": {}
            }
        
        if frontend_role_config:
            role_config = deep_merge(role_config, frontend_role_config)
    
    return role_config


def _update_avalon_leaderboard(
    assigned_roles: list[tuple[int, str, bool]] | None,
    agents: list,
    good_wins: bool | None,
    language: str,
):
    """Persist a single Avalon game result to the leaderboard database."""
    if good_wins is None:
        return

    try:
        base_dir = Path(__file__).parent.parent
        db_path = base_dir / "evaluation" / "leaderboard" / "leaderboard_avalon.json"
        leaderboard_db = LeaderboardDB(str(db_path))

        roles_payload = []
        for idx, role in enumerate(assigned_roles or []):
            try:
                role_id, role_name, is_good = role
            except Exception:
                role_id, role_name, is_good = None, f"player_{idx}", True

            agent = agents[idx] if idx < len(agents) else None
            model_name = None
            if agent is not None:
                model = getattr(agent, "model", None)
                if model and getattr(model, "model_name", None):
                    model_name = getattr(model, "model_name", None)
                if not model_name and getattr(agent, "name", None):
                    model_name = getattr(agent, "name", None)
            model_name = model_name or f"Player{idx}"

            winner = (bool(is_good) and bool(good_wins)) or (not bool(is_good) and not bool(good_wins))
            roles_payload.append({
                "role_name": str(role_name),
                "model_name": str(model_name),
                "score": 1 if winner else 0,
            })

        if roles_payload:
            leaderboard_db.update_from_game_results([
                {"roles": roles_payload, "language": language}
            ])
            print(f"[web] Updated Avalon leaderboard at {db_path} (models: {[r['model_name'] for r in roles_payload]})")
    except Exception as e:
        print(f"[web] Failed to update Avalon leaderboard: {e}")

async def run_avalon(
    state_manager: GameStateManager,
    num_players: int,
    language: str,
    user_agent_id: int,
    mode: str,
    preset_roles: list[tuple[int, str, bool]] | None = None,
    selected_portrait_ids: list[int] | None = None,
    agent_configs: Dict[int, Dict[str, str]] | None = None,
):
    """Run Avalon game"""
    config = AvalonBasicConfig.from_num_players(num_players)
    
    yaml_path = os.environ.get("AVALON_CONFIG_YAML", "games/games/avalon/configs/default_config.yaml")
    task_cfg = load_config(yaml_path)
    
    if not selected_portrait_ids:
        selected_portrait_ids = list(range(1, num_players + 1))

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)
    
    ai_portrait_index = 0  # AI player index

    # Use RoleManager to manage roles
    # If no preset_roles, use AvalonGameEnvironment to generate default roles
    if preset_roles is None:
        print("No preset roles, using AvalonGameEnvironment to generate default roles")
        env = AvalonGameEnvironment(config)
        assigned_roles = env.get_roles()
    else:
        assigned_roles = preset_roles
    
    role_manager = RoleManager(assigned_roles)

    for i in range(num_players):
        if mode == "participate" and i == user_agent_id:
            agent = WebUserAgent(name=f"Player{i}", state_manager=state_manager)
        else:
            if mode == "participate":
                if ai_portrait_index < len(selected_portrait_ids):
                    portrait_id = selected_portrait_ids[ai_portrait_index]
                    ai_portrait_index += 1
                else:
                    portrait_id = i + 1
            else:
                portrait_id = selected_portrait_ids[i] if i < len(selected_portrait_ids) else (i + 1)
            
            # Get frontend config
            frontend_cfg = None
            if agent_configs and portrait_id in agent_configs:
                frontend_cfg = agent_configs[portrait_id]
            
            # Use RoleManager to get role identifiers
            indexed_role = role_manager.get_indexed_role(i)
            base_role = role_manager.get_role_name(i)
            
            # Use unified _get_role_config function to get config
            role_config = _get_role_config(
                task_cfg,
                role_identifiers=[indexed_role, base_role],  # Priority: indexed_role first, then base_role
                frontend_cfg=frontend_cfg,
            )
            
            # Extract model and agent config
            model_config = role_config.get('model', {})
            agent_config = role_config.get('agent', {})
            
            # Use factory functions to create model and agent
            model = create_model_from_config(model_config)
            agent = create_agent_from_config(
                agent_config=agent_config,
                model=model,
                name=f"Player{i}",
                actor_rollout_ref=None,
            )
            
        agents.append(agent)

    state_manager.set_mode(mode, str(user_agent_id) if mode == "participate" else None, game="avalon")
    state_manager.update_game_state(status="running")
    await state_manager.broadcast_message(state_manager.format_game_state())

    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    good_wins = await avalon_game(
        agents=agents,
        config=config,
        log_dir=log_dir,
        language=language,
        web_mode=mode,
        web_observe_agent=observe_agent,
        state_manager=state_manager,
        preset_roles=assigned_roles,  # Use assigned_roles (may be generated or preset)
    )

    if good_wins is None or state_manager.should_stop:
        state_manager.update_game_state(status="stopped")
        await state_manager.broadcast_message(state_manager.format_game_state())
        return

    state_manager.update_game_state(status="finished", good_wins=good_wins)
    await state_manager.broadcast_message(state_manager.format_game_state())
    result_msg = state_manager.format_message(
        sender="System",
        content=f"Game finished! {'Good wins!' if good_wins else 'Evil wins!'}",
        role="assistant",
    )
    await state_manager.broadcast_message(result_msg)

    _update_avalon_leaderboard(assigned_roles, agents, good_wins, language)


async def run_diplomacy(
    state_manager: GameStateManager,
    config: DiplomacyConfig,
    mode: str,
    selected_portrait_ids: list[int] | None = None,
    agent_configs: Dict[int, Dict[str, str]] | None = None,
):
    """Run Diplomacy game."""
    agentscope.init()

    yaml_path = os.environ.get("DIPLOMACY_CONFIG_YAML", "games/games/diplomacy/configs/default_config.yaml")
    task_cfg = load_config(yaml_path)

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for power_idx, power in enumerate(config.power_names):
        if mode == "participate" and config.human_power and power == config.human_power:
            agent = WebUserAgent(name=power, state_manager=state_manager)
            state_manager.user_agent_id = agent.id
        else:
            portrait_id = None
            frontend_cfg = None
            if selected_portrait_ids and power_idx < len(selected_portrait_ids):
                portrait_id = selected_portrait_ids[power_idx]
                if portrait_id is not None and portrait_id != -1:
                    if agent_configs and portrait_id in agent_configs:
                        frontend_cfg = agent_configs[portrait_id]

            role_config = _get_role_config(
                task_cfg,
                role_identifiers=[power],
                frontend_cfg=frontend_cfg,
            )
            
            model_config = role_config.get('model', {})
            agent_config = role_config.get('agent', {})
            
            model = create_model_from_config(model_config)
            agent = create_agent_from_config(
                agent_config=agent_config,
                model=model,
                name=power,
                actor_rollout_ref=None,
            )
            agent.power_name = power
            agent.set_console_output_enabled(True)
        agents.append(agent)
    state_manager.set_mode(mode, config.human_power if mode == "participate" else None, game="diplomacy")
    state_manager.update_game_state(status="running", human_power=config.human_power if mode == "participate" else None)
    await state_manager.broadcast_message(state_manager.format_game_state())

    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    result = await diplomacy_game(
        agents=agents,
        config=config,
        state_manager=state_manager,
        log_dir=log_dir,
        observe_agent=observe_agent,
    )

    if result is None or state_manager.should_stop:
        state_manager.update_game_state(status="stopped")
        await state_manager.broadcast_message(state_manager.format_game_state())
        return

    state_manager.update_game_state(status="finished", result=result)
    await state_manager.broadcast_message(state_manager.format_game_state())
    end_msg = state_manager.format_message(sender="System", content=f"Diplomacy finished: {result}", role="assistant")
    await state_manager.broadcast_message(end_msg)


def start_game_thread(
    state_manager: GameStateManager,
    game: str,
    mode: str,
    language: str = "en",
    num_players: int = 5,
    user_agent_id: int = 0,
    preset_roles: list[dict] | None = None,
    selected_portrait_ids: list[int] | None = None,
    agent_configs: Dict[int, Dict[str, str]] | None = None,
    human_power: str | None = None,
    max_phases: int = 20,
    negotiation_rounds: int = 3,
    power_names: list[str] | None = None,
    power_models: Dict[str, str] | None = None,
):
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if game == "avalon":
                preset_roles_tuples: list[tuple[int, str, bool]] | None = None
                if preset_roles:
                    try:
                        from games.games.avalon.engine import AvalonBasicConfig
                        preset_roles_tuples = [
                            (AvalonBasicConfig.ROLES_REVERSE.get(str(x.get("role_name")), 0), str(x.get("role_name")), bool(x.get("is_good")))
                            for x in preset_roles
                            if isinstance(x, dict)
                        ]
                    except Exception:
                        pass
                
                portrait_ids = selected_portrait_ids if selected_portrait_ids else list(range(1, num_players + 1))
                task = loop.create_task(run_avalon(
                    state_manager=state_manager,
                    num_players=num_players,
                    language=language,
                    user_agent_id=user_agent_id,
                    mode=mode,
                    preset_roles=preset_roles_tuples,
                    selected_portrait_ids=portrait_ids,
                    agent_configs=agent_configs,
                ))
                state_manager._game_task = task
                
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass
                finally:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            else:
                cfg = DiplomacyConfig.default()
                cfg.max_phases = max_phases
                cfg.negotiation_rounds = negotiation_rounds
                cfg.language = language
                cfg.human_power = human_power
                if power_names:
                    cfg.power_names = list(power_names)
                
                task = loop.create_task(run_diplomacy(
                    state_manager=state_manager,
                    config=cfg,
                    mode=mode,
                    selected_portrait_ids=selected_portrait_ids,
                    agent_configs=agent_configs,
                ))
                state_manager._game_task = task
                
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass
                finally:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            try:
                loop.close()
            except Exception:
                pass
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    state_manager.set_game_thread(thread)
    return thread


def main():
    parser = argparse.ArgumentParser(description="Run web game (avalon|diplomacy)")
    parser.add_argument("--game", type=str, default="avalon", choices=["avalon", "diplomacy"])
    parser.add_argument("--mode", type=str, default="observe", choices=["observe", "participate"])
    parser.add_argument("--user-agent-id", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--human-power", type=str, default=None)
    parser.add_argument("--max-phases", type=int, default=20)
    parser.add_argument("--negotiation-rounds", type=int, default=3)
    args = parser.parse_args()

    state_manager = GameStateManager()
    start_game_thread(
        state_manager=state_manager,
        game=args.game,
        mode=args.mode,
        language=args.language,
        num_players=args.num_players,
        user_agent_id=args.user_agent_id,
        human_power=args.human_power,
        max_phases=args.max_phases,
        negotiation_rounds=args.negotiation_rounds,
        power_models={},
    )
    while True:
        asyncio.sleep(1)


if __name__ == "__main__":
    main()

