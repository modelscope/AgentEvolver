# -*- coding: utf-8 -*-
"""DiplomacyWorkflow class for running Diplomacy game workflows."""
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Reward
from games.games.diplomacy.game import DiplomacyGame
from games.games.diplomacy.engine import DiplomacyConfig


class PowerManager:
    """Manages power indexing and identification."""

    DEFAULT_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

    def __init__(self, power_names: List[str]):
        self.power_names = power_names

    def get_power_name(self, index: int) -> str:
        return self.power_names[index]

    def get_power_index(self, power_name: str) -> int:
        return self.power_names.index(power_name.upper())

    def __len__(self) -> int:
        return len(self.power_names)


class DiplomacyWorkflow(BaseAgentscopeWorkflow):
    """Workflow class for Diplomacy game that runs games and returns Trajectory."""

    def __init__(
        self,
        task: Task,
        llm_chat_fn: Any,
        model_name: str,
        **kwargs
    ):
        """
        Initialize the Diplomacy workflow.

        Args:
            task (Task): The task containing diplomacy configuration in metadata.
            llm_chat_fn (Callable): The LLM chat function to use for agent creation.
            model_name (str): The name of the model for training roles.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(task, llm_chat_fn, model_name, **kwargs)
        self.config_dict = task.metadata.get('diplomacy_config', task.metadata)
        self.power_manager: Optional[PowerManager] = None
        self.training_indices: List[int] = []
        self._train_powers_cache: Optional[set] = None

    # ==================== Configuration Methods ====================

    def _get_train_powers(self) -> set:
        """Get set of training power names (cached)."""
        if self._train_powers_cache is None:
            powers_config = self.config_dict.get('powers', {})
            self._train_powers_cache = {p.upper() for p in powers_config.get('train', [])}
        return self._train_powers_cache

    def _get_game_config(self) -> Dict[str, Any]:
        """Get game configuration."""
        return self.config_dict.get('game', {})

    def _get_model_config(self, power_name: str) -> Dict[str, Any]:
        """Get model configuration for a non-training power."""
        default_model = self.config_dict.get('default_model', {})
        config = {
            'name': default_model.get('name', 'qwen-plus'),
            'api_key': default_model.get('api_key', os.getenv('OPENAI_API_KEY', '')),
            'stream': default_model.get('stream', True),
        }
        # Apply custom configs if any
        models_config = self.config_dict.get('models', {})
        power_key = power_name.upper()
        if power_key in models_config:
            config.update(models_config[power_key])
        return config

    # ==================== Agent Management Methods ====================

    def _is_training_power(self, power_name: str) -> bool:
        """Check if a power is a training power."""
        train_powers = self._get_train_powers()
        return power_name.upper() in train_powers

    def _create_agent(self, player_id: int, power_name: str):
        """Create an agent for a power."""
        from agentscope.model import DashScopeChatModel
        from agentscope.formatter import DashScopeMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from games.agents.thinking_react_agent import ThinkingReActAgent

        # Use training model if power is training, otherwise create default model
        if self._is_training_power(power_name):
            model = self.model
        else:
            model_config = self._get_model_config(power_name)
            model = DashScopeChatModel(
                model_name=model_config.get('model_name', model_config.get('name', 'qwen-plus')),
                api_key=model_config.get('api_key', os.getenv('OPENAI_API_KEY', '')),
                stream=model_config.get('stream', True),
            )

        return ThinkingReActAgent(
            name=f"Player{player_id}",
            sys_prompt="",
            model=model,
            formatter=DashScopeMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )

    def _create_agents(self, power_manager: PowerManager) -> List[Any]:
        """Create all agents for the game."""
        return [
            self._create_agent(i, power_manager.get_power_name(i))
            for i in range(len(power_manager))
        ]

    def _identify_training_agents(self, agents: List[Any], power_manager: PowerManager) -> List[int]:
        """Identify which agents are training agents."""
        train_powers = self._get_train_powers()
        training_indices = [
            i for i in range(len(agents))
            if power_manager.get_power_name(i).upper() in train_powers
        ]
        if not training_indices:
            raise ValueError(
                f"No training agents found. Train powers: {train_powers}, "
                f"Assigned powers: {power_manager.power_names}"
            )
        return training_indices

    # ==================== Trajectory Collection Methods ====================

    def _calculate_reward(self, game, power_name: str) -> float:
        """Calculate reward for a power based on game outcome."""
        # Get the power's supply center count
        power = game.powers.get(power_name.upper())
        if power is None:
            return 0.0

        # Reward based on supply centers and survival
        num_centers = len(power.centers)
        is_eliminated = power.is_eliminated()

        if is_eliminated:
            return 0.0
        
        # Normalize reward: 18 supply centers needed to win
        # Scale from 0 to 1 based on centers
        return min(num_centers / 18.0, 1.0)

    def _build_trajectory(
        self,
        agent: Any,
        agent_idx: int,
        power_name: str,
        game,
    ) -> Trajectory:
        """Build a single trajectory for a training agent."""
        game_config = self._get_game_config()
        model_call_history = getattr(agent, 'model_call_history', [])
        agent_reward = self._calculate_reward(game, power_name)

        return Trajectory(
            data_id=self.task.task_id,
            rollout_id=self.task.task_id,
            steps=[
                {
                    'role': 'assistant',
                    'content': call_record.get('response', ''),
                    'prompt': call_record.get('prompt', ''),
                }
                for call_record in model_call_history
            ],
            is_terminated=True,
            reward=Reward(
                outcome=agent_reward,
                success_rate=agent_reward,
            ),
            metadata={
                'game_config': {
                    'map_name': game_config.get('map_name', 'standard'),
                    'max_phases': game_config.get('max_phases', 20),
                    'language': game_config.get('language', 'en'),
                },
                'agent_index': agent_idx,
                'power_name': power_name,
                'supply_centers': len(game.powers[power_name.upper()].centers) if power_name.upper() in game.powers else 0,
                'is_eliminated': game.powers[power_name.upper()].is_eliminated() if power_name.upper() in game.powers else True,
                'game_outcome': game.outcome,
                'num_model_calls': len(model_call_history),
            }
        )

    def _collect_trajectories(
        self,
        agents: List[Any],
        training_indices: List[int],
        power_manager: PowerManager,
        game,
    ) -> List[Trajectory]:
        """Collect trajectories from training agents, one per agent."""
        return [
            self._build_trajectory(
                agents[idx], idx,
                power_manager.get_power_name(idx),
                game
            )
            for idx in training_indices
        ]

    # ==================== Game Execution Methods ====================

    async def _execute_async(self) -> Union[Trajectory, List[Trajectory]]:
        """Async execution of the game."""
        game_config = self._get_game_config()

        # Setup game configuration
        power_names = game_config.get('power_names', PowerManager.DEFAULT_POWERS)
        config = DiplomacyConfig(
            power_names=power_names,
            map_name=game_config.get('map_name', 'standard'),
            max_phases=game_config.get('max_phases', 20),
            negotiation_rounds=game_config.get('negotiation_rounds', 3),
            seed=game_config.get('seed', 42),
            language=game_config.get('language', 'en'),
        )
        self.power_manager = PowerManager(power_names)

        # Create agents and identify training agents
        self.agents = self._create_agents(self.power_manager)
        self.training_indices = self._identify_training_agents(self.agents, self.power_manager)

        # Run game
        diplomacy_game = DiplomacyGame(
            agents=self.agents,
            config=config,
            log_dir=game_config.get('log_dir', 'logs'),
        )
        game = await diplomacy_game.run()

        # Collect trajectories
        trajectories = self._collect_trajectories(
            self.agents,
            self.training_indices,
            self.power_manager,
            game,
        )

        # Return single Trajectory if only one training agent, otherwise return list
        return trajectories[0] if len(trajectories) == 1 else trajectories

    def execute(self) -> Union[Trajectory, List[Trajectory]]:
        """
        Execute the Diplomacy workflow and return Trajectory or List[Trajectory].

        Returns:
            Trajectory: If there is only one training agent.
            List[Trajectory]: If there are multiple training agents.
        """
        return asyncio.run(self._execute_async())
