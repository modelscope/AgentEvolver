# -*- coding: utf-8 -*-
"""Unified web game launcher for Avalon + Diplomacy."""
import argparse
import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agentscope
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.tool import Toolkit

from games.web.game_state_manager import GameStateManager  #add gpt unified gsm
from games.web.web_agent import WebUserAgent, ObserveAgent  #add gpt unified agents

# Avalon imports
from games.agents.thinking_react_agent import ThinkingReActAgent
from games.avalon.game import avalon_game
from games.avalon.engine import AvalonBasicConfig


from games.diplomacy.engine import DiplomacyConfig
from games.diplomacy.game import diplomacy_game


async def run_avalon(
    state_manager: GameStateManager,
    num_players: int,
    language: str,
    user_agent_id: int,
    mode: str,
):
    """Run Avalon game."""
    config = AvalonBasicConfig.from_num_players(num_players)

    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    api_key = os.getenv("API_KEY", "sk-224e008372e144e496e06038077f65fc")

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for i in range(num_players):
        if mode == "participate" and i == user_agent_id:
            agent = WebUserAgent(name=f"Player{i}", state_manager=state_manager)
            print(f"Created {agent.name} (WebUserAgent - interactive)")
        else:
            model = DashScopeChatModel(model_name=model_name, api_key=api_key, stream=False)
            agent = ThinkingReActAgent(
                name=f"Player{i}",
                sys_prompt="",
                model=model,
                formatter=DashScopeMultiAgentFormatter(),
                memory=InMemoryMemory(),
                toolkit=Toolkit(),
            )
            print(f"Created {agent.name} (ThinkingReActAgent)")
        agents.append(agent)

    state_manager.set_mode(mode, str(user_agent_id) if mode == "participate" else None, game="avalon")  #add gpt set game avalon
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


async def run_diplomacy(
    state_manager: GameStateManager,
    config: DiplomacyConfig,
    mode: str,
):
    """Run Diplomacy game."""
    agentscope.init()

    model_name_default = os.getenv("MODEL_NAME", "qwen-plus")
    api_key_default = os.getenv("API_KEY", "sk-224e008372e144e496e06038077f65fc")

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for power in config.power_names:
        if mode == "participate" and config.human_power and power == config.human_power:
            agent = WebUserAgent(name=power, state_manager=state_manager)
            state_manager.user_agent_id = agent.id
            print(f"Created {agent.name} (WebUserAgent - interactive)")
        else:
            model_cfg = (config.models or {}).get(power, (config.models or {}).get("default", {}))
            model_name = model_cfg.get("model_name", model_name_default)
            api_key = model_cfg.get("api_key", api_key_default)
            if "gpt" in model_name:
                from agentscope.model import OpenAIChatModel
                model = OpenAIChatModel(model_name=model_name, api_key=api_key, stream=False)
            else:
                model = DashScopeChatModel(model_name=model_name, api_key=api_key, stream=False)
            agent = ThinkingReActAgent(
                name=power,
                sys_prompt="",
                model=model,
                formatter=DashScopeMultiAgentFormatter(),
                memory=InMemoryMemory(),
                toolkit=Toolkit(),
            )
            agent.power_name = power
            agent.set_console_output_enabled(True)
            print(f"Created {agent.name} (ThinkingReActAgent, model={model_name})")
        agents.append(agent)

    state_manager.set_mode(mode, config.human_power if mode == "participate" else None, game="diplomacy")  #add gpt set game diplomacy
    state_manager.update_game_state(status="running", human_power=config.human_power if mode == "participate" else None)
    await state_manager.broadcast_message(state_manager.format_game_state())

    log_dir = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
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
    human_power: str | None = None,
    max_phases: int = 20,
    negotiation_rounds: int = 3,
    power_models: Dict[str, str] | None = None,
):
    """Start game in a separate thread."""
    def run():
        if game == "avalon":
            asyncio.run(run_avalon(
                state_manager=state_manager,
                num_players=num_players,
                language=language,
                user_agent_id=user_agent_id,
                mode=mode,
            ))
        else:
            cfg = DiplomacyConfig.default()
            cfg.max_phases = max_phases
            cfg.negotiation_rounds = negotiation_rounds
            cfg.language = language
            cfg.human_power = human_power
            if power_models:
                cfg.models = cfg.models or {}
                for k, v in power_models.items():
                    cfg.models[k] = {"model_name": v}
            asyncio.run(run_diplomacy(
                state_manager=state_manager,
                config=cfg,
                mode=mode,
            ))
    
    thread = threading.Thread(target=run, daemon=True)  #add gpt run selected game in background
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
    # Keep main thread alive
    while True:
        asyncio.sleep(1)


if __name__ == "__main__":
    main()

