# -*- coding: utf-8 -*-
"""Web game launcher for Diplomacy (Avalon-style)."""

import argparse
import asyncio
import os
import sys
import threading
from pathlib import Path

# Add parent directory to path for imports (repo root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import agentscope
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.tool import Toolkit

from games.diplomacy.agents.thinking_react_agent import ThinkingReActAgent
from games.diplomacy.game import diplomacy_game
from games.diplomacy.engine import DiplomacyBasicConfig
from games.diplomacy.web.game_state_manager import GameStateManager
from games.diplomacy.web.web_agent import WebUserAgent, ObserveAgent


async def run_game_in_background(
    state_manager: GameStateManager,
    config: DiplomacyBasicConfig,
    mode: str = "observe",
):
    """
    Run Diplomacy game in background thread.

    Args:
        state_manager: GameStateManager instance
        config: DiplomacyBasicConfig
        mode: "observe" or "participate"
    """
    agentscope.init()

    # --- model configuration  ---
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    api_key = os.getenv("API_KEY", "sk-224e008372e144e496e06038077f65fc")  # added mxj
    os.environ["LANGUAGE"] = config.language

    # --- create agents ---
    agents = []
    observe_agent = None

    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for power in config.power_names:
        if mode == "participate" and config.human_power and power == config.human_power:
            agent = WebUserAgent(
                name=power,
                state_manager=state_manager,
            )
            state_manager.user_agent_id = agent.id
            print(f"Created {agent.name} (WebUserAgent - interactive)")
        else:
            # 每个 agent 一个 model
            model = DashScopeChatModel(
                model_name=model_name,
                api_key=api_key,
                stream=False,
            )
            agent = ThinkingReActAgent(
                name=power,
                sys_prompt="",  # System prompt 最好由 game.py 控制
                model=model,
                formatter=DashScopeMultiAgentFormatter(),
                memory=InMemoryMemory(),
                toolkit=Toolkit(),
            )

            agent.power_name = power
            agent.set_console_output_enabled(True)
            print(f"Created {agent.name} (ThinkingReActAgent)")

        agents.append(agent)

    # --- set mode in state manager ---
    # participate: human_power 写入；observe: None
    state_manager.set_mode(mode, config.human_power if mode == "participate" else None)

    # --- initial state update ---
    state_manager.update_game_state(
        status="running",
        human_power=config.human_power if mode == "participate" else None,
    )

    await state_manager.broadcast_message(state_manager.format_game_state())

    # --- run game ---
    log_dir = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
    os.makedirs(log_dir, exist_ok=True)

    try:
        result = await diplomacy_game(
            agents=agents,
            max_phases=config.max_phases,
            map_name=config.map_name,
            num_negotiation_rounds=config.negotiation_rounds,
            seed=config.seed,
            debug=False,
            state_manager=state_manager,
            log_dir=log_dir,
            observe_agent=observe_agent,
        )

        if result is None or state_manager.should_stop:
            print("\nGame stopped by user")
            state_manager.update_game_state(status="stopped")
            await state_manager.broadcast_message(state_manager.format_game_state())
            return None

        state_manager.update_game_state(status="finished")
        await state_manager.broadcast_message(state_manager.format_game_state())

        done_msg = state_manager.format_message(
            sender="System",
            content="Game finished!",
            role="assistant",
        )
        await state_manager.broadcast_message(done_msg)

    except Exception as e:
        import traceback
        traceback.print_exc()

        state_manager.update_game_state(status="error")
        await state_manager.broadcast_message(state_manager.format_game_state())

        err_msg = state_manager.format_message(
            sender="System",
            content=f"Game error: {e}",
            role="assistant",
        )
        await state_manager.broadcast_message(err_msg)
        raise


def start_game_thread(
    state_manager: GameStateManager,
    config: DiplomacyBasicConfig,
    mode: str = "observe",
):
    """Start game in a separate thread (align Avalon)."""

    def run():
        asyncio.run(
            run_game_in_background(
                state_manager=state_manager,
                config=config,
                mode=mode,
            )
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    state_manager.set_game_thread(thread)
    return thread


def main():
    """Main function (align Avalon CLI)."""
    parser = argparse.ArgumentParser(description="Run Diplomacy game with web interface")

    parser.add_argument(
        "--mode",
        type=str,
        default="observe",
        choices=["observe", "participate"],
        help="Game mode: observe or participate",
    )
    parser.add_argument(
        "--human-power",
        type=str,
        default="ENGLAND",
        help='Which power is controlled by human in participate mode (default: ENGLAND)',
    )
    parser.add_argument(
        "--max-phases",
        type=int,
        default=20,
        help="Max phases to run (default: 20)",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="standard",
        help='Map name (default: "standard")',
    )
    parser.add_argument(
        "--negotiation-rounds",
        type=int,
        default=3,
        help="Negotiation rounds per phase (default: 3)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        choices=["en", "zh", "cn", "chinese"],
        help='Language for prompts: "en" or "zh"/"cn"/"chinese" (default: en)',
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    args = parser.parse_args()

    # Build config (Avalon-style)
    config = DiplomacyBasicConfig.default()
    config.map_name = args.map_name
    config.max_phases = args.max_phases
    config.negotiation_rounds = args.negotiation_rounds
    config.language = args.language
    config.human_power = args.human_power if args.mode == "participate" else None

    print("=" * 60)
    print("Diplomacy Game Web Interface")
    print("=" * 60)
    print(f"Default Mode: {args.mode}")
    if args.mode == "participate":
        print(f"Default Human Power: {args.human_power}")
    print(f"Default Map: {args.map_name}")
    print(f"Default Max phases: {args.max_phases}")
    print(f"Default Negotiation rounds: {args.negotiation_rounds}")
    print(f"Default Language: {args.language}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print()
    print("Note: Game will start from the web interface, not automatically.")
    print()
    print(f"Web interface available at: http://localhost:{args.port}")
    print(f"  - Observe mode: http://localhost:{args.port}/observe")
    print(f"  - Participate mode: http://localhost:{args.port}/participate")
    print()
    print("Press Ctrl+C to stop the server.")
    print()


    import uvicorn
    uvicorn.run(
        "games.diplomacy.web.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
