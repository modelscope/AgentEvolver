# -*- coding: utf-8 -*-
"""Web game launcher for Avalon."""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from time import sleep

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter, DashScopeMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
from games.avalon.game import avalon_game
from games.avalon.engine import AvalonBasicConfig
from games.avalon.web.game_state_manager import GameStateManager
from games.avalon.web.web_agent import WebUserAgent, ObserveAgent
from games.avalon.web.server import get_state_manager, run_server
import threading


async def run_game_in_background(
    state_manager: GameStateManager,
    num_players: int = 5,
    language: str = "en",
    user_agent_id: int = 0,
    mode: str = "observe",
):
    """Run the game in the background.
    
    Args:
        state_manager: GameStateManager instance
        num_players: Number of players
        language: Language for prompts
        user_agent_id: Player ID to use UserAgent (for participate mode)
        mode: "observe" or "participate"
    """
    config = AvalonBasicConfig.from_num_players(num_players)
    
    # Model configuration
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    api_key = os.getenv("API_KEY", "sk-224e008372e144e496e06038077f65fc")
    
    # Create agents
    agents = []
    observe_agent = None
    
    if mode == "observe":
        # Create ObserveAgent for observing all messages
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)
    
    for i in range(num_players):
        if mode == "participate" and i == user_agent_id:
            # Create WebUserAgent for user participation
            agent = WebUserAgent(
                name=f"Player{i}",
                state_manager=state_manager,
            )
            print(f"Created {agent.name} (WebUserAgent - interactive)")
        else:
            # Create ThinkingReActAgent with model
            model = DashScopeChatModel(
                model_name=model_name,
                api_key=api_key,
                stream=True,
            )
            agent = ThinkingReActAgent(
                name=f"Player{i}",
                sys_prompt="",  # System prompt will be set in game.py
                model=model,
                formatter=DashScopeMultiAgentFormatter(),
                memory=InMemoryMemory(),
                toolkit=Toolkit(),
            )
            print(f"Created {agent.name} (ThinkingReActAgent)")
        agents.append(agent)
    
    # Set mode in state manager
    state_manager.set_mode(mode, str(user_agent_id) if mode == "participate" else None)
    
    # Update game state
    state_manager.update_game_state(status="running")
    await state_manager.broadcast_message(state_manager.format_game_state())
    
    try:
        # Run game with web mode support
        # We need to modify avalon_game to accept observe_agent
        # For now, we'll add it to all MsgHubs manually if needed
        # Enable logging for web mode
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        good_wins = await avalon_game(
            agents=agents,
            config=config,
            log_dir=log_dir,  # Enable logging for web mode
            language=language,
            web_mode=mode,
            web_observe_agent=observe_agent,
            state_manager=state_manager,
        )
        
        # Check if game was stopped (returns None when stopped)
        if good_wins is None or state_manager.should_stop:
            print("\nGame stopped by user")
            state_manager.update_game_state(status="stopped")
            await state_manager.broadcast_message(state_manager.format_game_state())
            return None
        
        # Update final state
        state_manager.update_game_state(
            status="finished",
            good_wins=good_wins,
        )
        await state_manager.broadcast_message(state_manager.format_game_state())
        
        result_msg = state_manager.format_message(
            sender="System",
            content=f"Game finished! {'Good wins!' if good_wins else 'Evil wins!'}",
            role="assistant",
        )
        await state_manager.broadcast_message(result_msg)
        
        print(f"\nGame finished. Result: {'Good wins!' if good_wins else 'Evil wins!'}")
        
    except Exception as e:
        print(f"Error during game: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_msg = state_manager.format_message(
            sender="System",
            content=f"Game error: {str(e)}",
            role="assistant",
        )
        await state_manager.broadcast_message(error_msg)
        
        state_manager.update_game_state(status="error")
        raise


def start_game_thread(
    state_manager: GameStateManager,
    num_players: int = 5,
    language: str = "en",
    user_agent_id: int = 0,
    mode: str = "observe",
):
    """Start game in a separate thread.
    
    Args:
        state_manager: GameStateManager instance
        num_players: Number of players
        language: Language for prompts
        user_agent_id: Player ID to use UserAgent
        mode: "observe" or "participate"
    """
    def run():
        asyncio.run(run_game_in_background(
            state_manager=state_manager,
            num_players=num_players,
            language=language,
            user_agent_id=user_agent_id,
            mode=mode,
        ))
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    # Save thread reference in state manager
    state_manager.set_game_thread(thread)
    return thread


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Avalon game with web interface")
    parser.add_argument(
        "--mode",
        type=str,
        default="observe",
        choices=["observe", "participate"],
        help="Game mode: observe or participate",
    )
    parser.add_argument(
        "--user-agent-id",
        type=int,
        default=0,
        help="Player ID to use UserAgent (for participate mode, default: 0)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=5,
        help="Number of players (default: 5)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        choices=["en", "zh", "cn", "chinese"],
        help='Language for prompts: "en" for English, "zh"/"cn"/"chinese" for Chinese (default: en)',
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
    
    state_manager = get_state_manager()
    
    print("=" * 60)
    print("Avalon Game Web Interface")
    print("=" * 60)
    print(f"Default Mode: {args.mode}")
    if args.mode == "participate":
        print(f"Default User Agent ID: {args.user_agent_id}")
    print(f"Default Number of players: {args.num_players}")
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
    
    # Run server
    import uvicorn
    uvicorn.run(
        "games.avalon.web.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

