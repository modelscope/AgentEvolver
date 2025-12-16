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

from games.web.game_state_manager import GameStateManager
from games.web.web_agent import WebUserAgent, ObserveAgent

# Avalon imports
from games.agents.thinking_react_agent import ThinkingReActAgent
from games.games.avalon.game import avalon_game
from games.games.avalon.engine import AvalonBasicConfig


from games.games.diplomacy.engine import DiplomacyConfig
from games.games.diplomacy.game import diplomacy_game


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
    """运行 Avalon 游戏"""
    config = AvalonBasicConfig.from_num_players(num_players)

    # 读取 web_config.yaml 中每个 portrait 的模型配置
    import yaml
    web_config_path = os.path.join(os.path.dirname(__file__), "web_config.yaml")
    web_config = {}
    try:
        if os.path.exists(web_config_path):
            with open(web_config_path, "r", encoding="utf-8") as f:
                web_config = yaml.safe_load(f) or {}
    except Exception:
        pass
    
    default_model = web_config.get("default_model", {})
    portraits = web_config.get("portraits", {})
    
    if not selected_portrait_ids:
        selected_portrait_ids = list(range(1, num_players + 1))

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for i in range(num_players):
        if mode == "participate" and i == user_agent_id:
            agent = WebUserAgent(name=f"Player{i}", state_manager=state_manager)
        else:
            # 优先使用前端传递的 agent 配置，否则从 web_config.yaml 读取
            portrait_id = selected_portrait_ids[i] if i < len(selected_portrait_ids) else (i + 1)
            
            # 先尝试使用前端传递的配置
            frontend_cfg = None
            if agent_configs and portrait_id in agent_configs:
                frontend_cfg = agent_configs[portrait_id]
            
            if frontend_cfg and frontend_cfg.get("base_model"):
                model_name = frontend_cfg.get("base_model", os.getenv("MODEL_NAME", "qwen-plus"))
                api_base = frontend_cfg.get("api_base", "")
                api_key = frontend_cfg.get("api_key", os.getenv("OPENAI_API_KEY", ""))
            else:
                portrait_cfg = portraits.get(str(portrait_id), {}) if isinstance(portraits, dict) else {}
                merged_cfg = dict(default_model)
                if isinstance(portrait_cfg, dict):
                    merged_cfg.update(portrait_cfg)
                
                api_key_raw = merged_cfg.get("api_key", "")
                if "${OPENAI_API_KEY}" in str(api_key_raw):
                    api_key_raw = os.getenv("OPENAI_API_KEY", "")
                
                model_name = merged_cfg.get("model_name", os.getenv("MODEL_NAME", "qwen-plus"))
                api_key = api_key_raw or os.getenv("OPENAI_API_KEY", "")
                api_base = merged_cfg.get("api_base", "")
            
            # 根据 api_base 决定使用哪个模型类
            if api_base and "openai" in api_base.lower():
                from agentscope.model import OpenAIChatModel
                model = OpenAIChatModel(model_name=model_name, api_key=api_key, api_base=api_base, stream=False)
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
        preset_roles=preset_roles,
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
    selected_portrait_ids: list[int] | None = None,
    agent_configs: Dict[int, Dict[str, str]] | None = None,
):
    """Run Diplomacy game."""
    agentscope.init()

    model_name_default = os.getenv("MODEL_NAME", "qwen-plus")
    api_key_default = os.getenv("OPENAI_API_KEY", "")

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for power_idx, power in enumerate(config.power_names):
        if mode == "participate" and config.human_power and power == config.human_power:
            agent = WebUserAgent(name=power, state_manager=state_manager)
            state_manager.user_agent_id = agent.id
        else:
            # 优先使用前端传递的 agent 配置
            portrait_id = None
            frontend_cfg = None
            if selected_portrait_ids and power_idx < len(selected_portrait_ids):
                portrait_id = selected_portrait_ids[power_idx]
                # 跳过占位符（-1 表示 human player 或空位）
                if portrait_id is not None and portrait_id != -1:
                    if agent_configs and portrait_id in agent_configs:
                        frontend_cfg = agent_configs[portrait_id]
                else:
                    portrait_id = None  # 将占位符转换为 None 以便调试输出
            
            if frontend_cfg and frontend_cfg.get("base_model"):
                model_name = frontend_cfg.get("base_model", model_name_default)
                api_base = frontend_cfg.get("api_base", "")
                api_key = frontend_cfg.get("api_key", api_key_default)
            else:
                model_cfg = (config.models or {}).get(power, (config.models or {}).get("default", {}))
                model_name = model_cfg.get("model_name", model_name_default)
                api_key = model_cfg.get("api_key", api_key_default)
                api_base = model_cfg.get("api_base", "")
            
            # 根据 api_base 和 model_name 决定使用哪个模型类
            if api_base and "openai" in api_base.lower():
                from agentscope.model import OpenAIChatModel
                model = OpenAIChatModel(model_name=model_name, api_key=api_key, api_base=api_base, stream=False)
            elif "gpt" in model_name.lower():
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
    """在后台线程中启动游戏"""
    def run():
        try:
            # 创建新的事件循环（每个线程需要自己的事件循环）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if game == "avalon":
                # 解析前端传入的角色分配
                preset_roles_tuples: list[tuple[int, str, bool]] | None = None
                if preset_roles:
                    try:
                        preset_roles_tuples = [
                            (int(x.get("role_id")), str(x.get("role_name")), bool(x.get("is_good")))
                            for x in preset_roles
                            if isinstance(x, dict)
                        ]
                    except Exception:
                        pass
                
                portrait_ids = selected_portrait_ids if selected_portrait_ids else list(range(1, num_players + 1))
                
                # 创建任务并保存引用，以便可以取消
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
                    # 清理未完成的任务
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    # 等待所有任务完成或取消
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
                if power_models:
                    cfg.models = cfg.models or {}
                    for k, v in power_models.items():
                        cfg.models[k] = {"model_name": v}
                
                # 创建任务并保存引用，以便可以取消
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
                    # 清理未完成的任务
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    # 等待所有任务完成或取消
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
    # Keep main thread alive
    while True:
        asyncio.sleep(1)


if __name__ == "__main__":
    main()

