# -*- coding: utf-8 -*-
"""DiplomacyWorkflow class for running Diplomacy game workflows."""
import asyncio
import os
import copy
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from games.utils import (
    cleanup_agent_llm_clients,
    create_agent_from_config,
    create_model_from_config,
    deep_merge,
)
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Reward
from games.games.diplomacy.game import DiplomacyGame
from games.games.diplomacy.engine import DiplomacyConfig
from games.agents.agentscope_cmt import AgentscopeCMT

# Import for formatter support
from agentscope.token import HuggingFaceTokenCounter
from games.agents.secure_multi_agent_formatter import SecureMultiAgentFormatter

# Lock for protecting HuggingFaceTokenCounter initialization from concurrent access
_tokenizer_lock = threading.Lock()


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
        config: Any,  # 添加
        tokenizer: Any,  # 添加
        data_id: str,  # 添加
        rollout_id: str,  # 添加
        **kwargs
    ):
        super().__init__(
            task, llm_chat_fn, model_name,
            config=config, tokenizer=tokenizer,
            data_id=data_id, rollout_id=rollout_id,
            **kwargs
        )
        self.config_dict = task.metadata.get('config', task.metadata)
        self.power_manager: Optional[PowerManager] = None
        self.training_indices: List[int] = []

    # ==================== Configuration Methods ====================

    def _get_game_config(self) -> Dict[str, Any]:
        """Get game configuration."""
        return self.config_dict.get('game', {})

    def _get_role_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """
        Get complete role configuration (including model, agent, trainable, act_by_user, etc.).
        Role-specific config overrides default_role config.
        """
        default_role = self.config_dict.get('default_role', {})
        roles_config = self.config_dict.get('roles', {})
        
        if not isinstance(default_role, dict):
            default_role = {}
        if not isinstance(roles_config, dict):
            roles_config = {}
        
        # Start with default_role config
        role_config = copy.deepcopy(default_role)
        
        # Find role-specific config (try indexed_role first, then base_role)
        specific_role_config = next(
            (v for k, v in roles_config.items() 
             if k.lower() in [indexed_role.lower(), base_role.lower()]),
            None
        )
        
        # Override with role-specific config if present
        if specific_role_config and isinstance(specific_role_config, dict):
            # Deep merge: recursively merge nested dicts
            role_config = deep_merge(role_config, specific_role_config)
        
        return role_config
    
    def _get_model_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """Get model configuration for a role."""
        role_config = self._get_role_config(indexed_role, base_role)
        return role_config.get('model', {})
    
    def _get_agent_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """Get agent configuration for a role."""
        role_config = self._get_role_config(indexed_role, base_role)
        return role_config.get('agent', {})

    # ==================== Agent Management Methods ====================
    
    def _is_training_power(self, indexed_role: str, base_role: str) -> bool:
        """Check if a power is a training power based on trainable flag in config."""
        role_config = self._get_role_config(indexed_role, base_role)
        # Check if trainable is explicitly set to True
        return role_config.get('trainable', False) is True

    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        """Create an agent for a power using create_agent_from_config."""
        model_config = self._get_model_config(indexed_role, base_role)
        agent_config = self._get_agent_config(indexed_role, base_role)
        
        # Create model (required for both modes)
        # Use training model if power is training, otherwise create from config
        if self._is_training_power(indexed_role, base_role):
            model = self.model
        else:
            model = create_model_from_config(model_config)
        
        # Validate agent_config
        if not agent_config:
            raise ValueError(
                f"agent config is required. Please specify it in default_role.agent or role-specific config for {indexed_role}."
            )
        
        return create_agent_from_config(
            agent_config=agent_config,
            model=model,
            name=f"Player{player_id}",
            actor_rollout_ref=self.config.actor_rollout_ref,
        )

    def _create_agents(self, power_manager: PowerManager) -> List[Any]:
        """Create all agents for the game."""
        return [
            self._create_agent(i, power_manager.get_power_name(i), power_manager.get_power_name(i))
            for i in range(len(power_manager))
        ]

    def _identify_training_agents(self) -> List[int]:
        """Identify which agents are training agents."""
        training_indices = []
        for i in range(len(self.agents)):
            power_name = self.power_manager.get_power_name(i)
            # For Diplomacy, indexed_role and base_role are both power_name
            if self._is_training_power(power_name, power_name):
                training_indices.append(i)
        
        if not training_indices:
            raise ValueError(
                f"No training agents found. "
                f"Assigned powers: {self.power_manager.power_names}"
            )
        return training_indices

    # ==================== Trajectory Collection Methods ====================

    def _calculate_reward(self, game, power_name: str) -> Reward:
        """Calculate reward for a power based on game outcome."""
        # Get the power's supply center count
        power = game.powers.get(power_name.upper())
        if power is None:
            return Reward(outcome=0.0, success_rate=0.0)

        # Reward based on supply centers and survival
        num_centers = len(power.centers)
        is_eliminated = power.is_eliminated()

        if is_eliminated:
            agent_reward = 0.0
        else:
            # Normalize reward: 18 supply centers needed to win
            # Scale from 0 to 1 based on centers
            agent_reward = min(num_centers / 18.0, 1.0)
        
        return Reward(
            outcome=agent_reward,
            success_rate=agent_reward,
        )

    # ==================== Game Execution Methods ====================

    async def _execute_async(self) -> Tuple[Any, List[int]]:
        """Execute the game asynchronously.
        
        Returns:
            Tuple[Any, List[int]]: (game, training_indices)
        """
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

        # Create agents
        self.agents = self._create_agents(self.power_manager)
        
        # Identify training agents
        self.training_indices = self._identify_training_agents()
        
        # Only enable console output for the first task (data_id="0" and rollout_id="0")
        # Disable console output for all other tasks to reduce log noise
        is_first_task = (self.data_id == "0" and self.rollout_id == "0")
        for i in range(len(self.agents)):
            if i in self.training_indices:
                self.agents[i].set_console_output_enabled(is_first_task)
            else:    
                self.agents[i].set_console_output_enabled(False)

        # Run game
        # Generate unique timestamp for parallel rollouts by including data_id and rollout_id
        # This prevents multiple parallel rollouts from overwriting each other's logs
        base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_timestamp = f"{base_timestamp}_d{self.data_id}_r{self.rollout_id}"
        
        # Get log_dir from game config and experiment_name from self.config
        log_dir = game_config.get('log_dir', 'logs')
        experiment_name = getattr(self.config.trainer, 'experiment_name', None) if hasattr(self.config, 'trainer') else None
        
        # If experiment_name is provided, append it to log_dir
        if experiment_name:
            # Sanitize experiment_name to avoid filesystem issues
            experiment_name = str(experiment_name).replace('/', '_').replace('\\', '_')
            log_dir = os.path.join(log_dir, experiment_name)
        
        if unique_timestamp:
            log_dir = os.path.join(log_dir, unique_timestamp)

        if self.data_id == "0" and self.rollout_id == "0":
            game_id = 0
        else:
            # 使用 data_id 和 rollout_id 组合生成唯一的非0 game_id
            try:
                data_id_int = int(self.data_id) if self.data_id.isdigit() else 9999
                rollout_id_int = int(self.rollout_id) if self.rollout_id.isdigit() else 9999
                game_id = data_id_int * 1000 + rollout_id_int + 1  # 确保非0
            except (ValueError, AttributeError):
                game_id = 9999
        
        diplomacy_game = DiplomacyGame(
            agents=self.agents,
            config=config,
            log_dir=log_dir,
            game_id=game_id,
        )
        game = await diplomacy_game.run()

        # Clean up httpx client resources in agent LLM clients
        await cleanup_agent_llm_clients(self.agents)

        return game, self.training_indices

    def execute(self) -> Trajectory:
        """
        Execute the Diplomacy rollout workflow and return a CMT object.
        
        Returns:
            Trajectory (AgentscopeCMT): A CMT object containing model_call_history
                from training agents, converted to training samples.
        """
        # Execute the game
        game, training_indices = asyncio.run(self._execute_async())
        
        # Collect model_call_history from all training agents
        # For now, we'll use the first training agent's history
        # TODO: Support multiple training agents (merge or return list of CMTs)
        if not training_indices:
            raise ValueError("No training agents found")
        
        # Use the first training agent's model_call_history
        training_agent_idx = training_indices[0]
        training_agent = self.agents[training_agent_idx]
        model_call_history = getattr(training_agent, 'model_call_history', [])
        
        if not model_call_history:
            logger.warning("No model_call_history found in training agent")
            power_name = self.power_manager.get_power_name(training_agent_idx)
            reward = self._calculate_reward(game, power_name)
            return Trajectory(
                data_id=self.task.task_id,
                rollout_id=self.task.task_id,
                steps=[],
                is_terminated=True,
                reward=reward,
                metadata={},
            )
        
        # Calculate reward for the training agent
        power_name = self.power_manager.get_power_name(training_agent_idx)
        reward = self._calculate_reward(game, power_name)
        
        # Create AgentscopeCMT from model_call_history
        cmt = AgentscopeCMT(
            config=self.config,
            tokenizer=self.tokenizer,
            model_call_history=model_call_history,
            reward=reward,
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task.task_id,
        )
        
        return cmt
