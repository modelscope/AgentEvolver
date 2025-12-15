# -*- coding: utf-8 -*-
"""Example test script with role-specific model configuration and UserAgent support."""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add BeyondAgent directory to path for imports
ba_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ba_dir))

from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from games.games.avalon.game import AvalonGame
from games.games.avalon.engine import AvalonBasicConfig
from games.agents.thinking_react_agent import ThinkingReActAgent
from games.agents.terminal_user_agent import TerminalUserAgent


async def main(
    language: Optional[str] = None,
    use_user_agent: Optional[bool] = None,
    user_agent_id: Optional[int] = None,
    config_path: Optional[str] = None,
    num_players: Optional[int] = None,
):
    """Main function to run Avalon game with role-specific configurations.
    
    Args:
        language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
        use_user_agent: Whether to use UserAgent. If None, defaults to False.
        user_agent_id: Player ID to use UserAgent (0-indexed). If None and use_user_agent is True, defaults to 0.
        config_path: Path to config YAML file. If None, uses default config.yaml.
        num_players: Number of players. If None, defaults to 5.
    """
    # Default values
    if language is None:
        language = os.getenv("LANGUAGE", "en")
    if num_players is None:
        num_players = 5
    if use_user_agent is None:
        use_user_agent = False
    if use_user_agent and user_agent_id is None:
        user_agent_id = 0
    
    # Create game configuration
    config = AvalonBasicConfig.from_num_players(num_players)
    
    # Model configuration
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    # Create agents
    agents = []
    for i in range(num_players):
        if use_user_agent and i == user_agent_id:
            # Create TerminalUserAgent for user participation
            agent = TerminalUserAgent(
                name=f"Player{i}",
            )
            print(f"Created {agent.name} (TerminalUserAgent - interactive)")
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
    
    # Create game instance
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    game = AvalonGame(
        agents=agents,
        config=config,
        log_dir=log_dir,
        language=language,
    )
    
    try:
        good_wins = await game.run()
        return good_wins
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Avalon game with role-specific configurations")
    parser.add_argument("--config", "-c", type=str, help="Path to config YAML file (not used currently)")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=os.getenv("LANGUAGE", "en"),
        choices=["en", "zh", "cn", "chinese"],
        help='Language for prompts: "en" for English, "zh"/"cn"/"chinese" for Chinese (default: en)',
    )
    parser.add_argument(
        "--num-players",
        "-n",
        type=int,
        default=5,
        help="Number of players (default: 5)",
    )
    parser.add_argument("--use-user-agent", action="store_true", help="Enable UserAgent")
    parser.add_argument("--no-user-agent", action="store_true", help="Disable UserAgent")
    parser.add_argument("--user-agent-id", type=int, help="Player ID for UserAgent (0-indexed, default: 0 when --use-user-agent is set)")
    args = parser.parse_args()
    
    asyncio.run(
        main(
            language=args.language,
            use_user_agent=True if args.use_user_agent else (False if args.no_user_agent else None),
            user_agent_id=args.user_agent_id,
            config_path=args.config,
            num_players=args.num_players,
        )
    )
