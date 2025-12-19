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
    
    yaml_path = os.environ.get("AVALON_CONFIG_YAML", "games/games/avalon/configs/task_config.yaml")
    task_cfg = load_config(yaml_path)
    default_model = task_cfg.get("default_model", {})
    roles_config = task_cfg.get("roles", {})
    
    if not selected_portrait_ids:
        selected_portrait_ids = list(range(1, num_players + 1))

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)
    
    ai_portrait_index = 0 #ai玩家索引

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
            # 优先级：agent_configs > roles_config[role_name] > default_model > 环境变量
            
            # 1. 优先使用前端传递的 agent 配置
            frontend_cfg = None
            if agent_configs and portrait_id in agent_configs:
                frontend_cfg = agent_configs[portrait_id]
            
            # 获取 agent_class（优先级：frontend_cfg > roles_config > default_model）
            agent_class_path = None
            if frontend_cfg and frontend_cfg.get("base_model"):
                model_name = frontend_cfg.get("base_model", os.getenv("MODEL_NAME", "qwen-plus"))
                api_base = frontend_cfg.get("api_base") or os.getenv("OPENAI_API_BASE", "")
                api_key = frontend_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
                agent_class_path = frontend_cfg.get("agent_class", "")
            else:
                # 2. 从 task_config.yaml 的 roles 中查找（如果有 preset_roles）
                role_name = None
                if preset_roles:
                    # 从 preset_roles 中找到玩家 i 对应的角色名称
                    for role_id, name, _ in preset_roles:
                        if role_id == i:
                            role_name = name
                            break
                
                # 从 roles_config 中查找角色配置，如果没有则使用 default_model
                model_cfg = roles_config.get(role_name) if role_name and isinstance(roles_config, dict) and role_name in roles_config else {}
                
                # 合并 default_model 和 model_cfg（model_cfg 优先）
                merged_cfg = dict(default_model)
                if isinstance(model_cfg, dict):
                    merged_cfg.update(model_cfg)
                
                model_name = merged_cfg.get("model_name", os.getenv("MODEL_NAME", "qwen-plus"))
                api_key = merged_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
                api_base = merged_cfg.get("api_base") or merged_cfg.get("url") or os.getenv("OPENAI_API_BASE", "")
                agent_class_path = merged_cfg.get("agent_class", "")

            
            model_kwargs = {
                'model_name': model_name,
                'api_key': api_key,
                'stream': False,
            }
            if api_base:
                model_kwargs['client_args'] = {'base_url': api_base}
            model = OpenAIChatModel(**model_kwargs)

            # 使用 load_agent_class 加载 agent class
            AgentClass = load_agent_class(agent_class_path if agent_class_path else None)
            agent = AgentClass(
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

    # 从 task_config.yaml 读取默认模型配置作为保底
    yaml_path = os.environ.get("DIPLOMACY_CONFIG_YAML", "games/games/diplomacy/configs/task_config.yaml")
    task_cfg = load_config(yaml_path)
    default_model = task_cfg.get("default_model", {})
    roles_config = task_cfg.get("roles", {})

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for power_idx, power in enumerate(config.power_names):
        if mode == "participate" and config.human_power and power == config.human_power:
            agent = WebUserAgent(name=power, state_manager=state_manager)
            state_manager.user_agent_id = agent.id
        else:
            # 优先级：agent_configs > roles_config[power] > roles_config["default"] > default_model > 环境变量
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
            
            # 获取 agent_class（优先级：frontend_cfg > roles_config > default_model）
            agent_class_path = None
            if frontend_cfg and frontend_cfg.get("base_model"):
                # 1. 优先使用前端传递的 agent 配置
                model_name = frontend_cfg.get("base_model", os.getenv("MODEL_NAME", "qwen-plus"))
                api_base = frontend_cfg.get("api_base") or os.getenv("OPENAI_API_BASE", "")
                api_key = frontend_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
                agent_class_path = frontend_cfg.get("agent_class", "")
            else:
                # 2. 从 task_config.yaml 的 roles 中查找
                model_cfg = roles_config.get(power) or roles_config.get("default", {})
                
                # 合并 default_model 和 model_cfg（model_cfg 优先）
                merged_cfg = dict(default_model)
                if isinstance(model_cfg, dict):
                    merged_cfg.update(model_cfg)
                
                model_name = merged_cfg.get("model_name", os.getenv("MODEL_NAME", "qwen-plus"))
                api_key = merged_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
                api_base = merged_cfg.get("api_base") or merged_cfg.get("url") or os.getenv("OPENAI_API_BASE", "")
                agent_class_path = merged_cfg.get("agent_class", "")
            

            model_kwargs = {
                'model_name': model_name,
                'api_key': api_key,
                'stream': False,
            }
            if api_base:
                model_kwargs['client_args'] = {'base_url': api_base}
            model = OpenAIChatModel(**model_kwargs)
            
            # 使用 load_agent_class 加载 agent class
            AgentClass = load_agent_class(agent_class_path if agent_class_path else None)
            agent = AgentClass(
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

