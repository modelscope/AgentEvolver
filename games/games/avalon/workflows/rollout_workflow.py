# -*- coding: utf-8 -*-
"""AvalonWorkflow class for running Avalon game workflows."""
import asyncio
import os
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

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
from games.games.avalon.game import AvalonGame
from games.games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
from games.games.avalon.utils import GameLogger
from games.agents.agentscope_cmt import AgentscopeCMT



class RoleManager:
    """Manages role indexing and identification."""
    
    def __init__(self, roles: List[tuple]):
        self.roles = roles
        role_counters = defaultdict(int)
        self.indexed_roles = []
        for _, role_name, _ in roles:
            self.indexed_roles.append(f"{role_name}_{role_counters[role_name]}")
            role_counters[role_name] += 1
    
    def get_indexed_role(self, index: int) -> str:
        return self.indexed_roles[index]
    
    def get_role_name(self, index: int) -> str:
        return self.roles[index][1]
    
    def is_good(self, index: int) -> bool:
        return self.roles[index][2]


class AvalonRolloutWorkflow(BaseAgentscopeWorkflow):
    """Workflow class for Avalon game training rollout.
    
    This workflow is designed for training scenarios where specific roles
    use the training model (via llm_chat_fn) while other roles use
    default models. Roles with trainable: true in config use self.model.
    
    Reference: Based on EvalAvalonWorkflow structure but adapted for training.
    """
    
    def __init__(
        self,
        task: Task,
        llm_chat_fn: Any,
        model_name: str,
        config: Any,
        tokenizer: Any,
        data_id: str,
        rollout_id: str,
        **kwargs
    ):
        """
        Initialize the Avalon rollout workflow.
        
        Args:
            task (Task): The task containing avalon configuration in metadata.
            llm_chat_fn (Callable): The LLM chat function to use for agent creation.
            model_name (str): The name of the model for training roles.
            config: Configuration object (required for CMT functionality).
            tokenizer: Tokenizer instance (required for CMT functionality).
            data_id (str): The ID of the data.
            rollout_id (str): The ID of the rollout.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            task, llm_chat_fn, model_name, 
            config=config, tokenizer=tokenizer,
            data_id=data_id, rollout_id=rollout_id,
            **kwargs
        )
        self.config_dict = task.metadata.get('config', task.metadata)
        self.role_manager: Optional[RoleManager] = None
        self.training_indices: List[int] = []
    
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
    
    def _is_training_role(self, indexed_role: str, base_role: str) -> bool:
        """Check if a role is a training role based on trainable flag in config."""
        role_config = self._get_role_config(indexed_role, base_role)
        # Check if trainable is explicitly set to True
        return role_config.get('trainable', False) is True
    
    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        """Create an agent for a player using create_agent_from_config."""
        model_config = self._get_model_config(indexed_role, base_role)
        agent_config = self._get_agent_config(indexed_role, base_role)
        
        # Create model (required for both modes)
        # Use training model if role is training, otherwise create from config
        if self._is_training_role(indexed_role, base_role):
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
    
    def _identify_training_agents(self) -> List[int]:
        """Identify which agents are training agents."""
        training_indices = []
        for i in range(len(self.agents)):
            indexed_role = self.role_manager.get_indexed_role(i)
            base_role = self.role_manager.get_role_name(i)
            if self._is_training_role(indexed_role, base_role):
                training_indices.append(i)
        
        if not training_indices:
            raise ValueError(
                f"No training agents found. "
                f"Assigned roles: {self.role_manager.indexed_roles}"
            )
        return training_indices
    
    def _calculate_reward(self, agent_idx: int, good_victory: bool) -> Reward:
        """Calculate reward for a training agent based on role and game outcome."""
        is_good = self.role_manager.is_good(agent_idx)
        # Reward is 1.0 if agent's team won, 0.0 otherwise
        agent_reward = 1.0 if (is_good == good_victory) else 0.0
        return Reward(
            outcome=agent_reward,
            success_rate=agent_reward,
        )
    
    async def _execute_async(self) -> Tuple[bool, List[int]]:
        """Execute the game asynchronously.
        
        Returns:
            Tuple[bool, List[int]]: (good_victory, training_indices)
        """
        game_config = self._get_game_config()
        
        # Setup environment and roles
        config = AvalonBasicConfig.from_num_players(game_config.get('num_players', 5))
        env = AvalonGameEnvironment(config)
        assigned_roles = env.get_roles()
        self.role_manager = RoleManager(assigned_roles)
        
        # Create agents
        self.agents = [
            self._create_agent(i, self.role_manager.get_indexed_role(i), 
                             self.role_manager.get_role_name(i))
            for i in range(len(assigned_roles))
        ]
        
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
        # Build log directory structure:
        # logs/{experiment_name}/{epoch}/data_id={data_id}rollout_id={rollout_id}/
        base_log_dir = game_config.get('log_dir', 'logs')
        experiment_name = getattr(self.config.trainer, 'experiment_name', None) if hasattr(self.config, 'trainer') else None
        epoch = None
        try:
            epoch = self.task.metadata.get("epoch") if isinstance(self.task.metadata, dict) else None
        except Exception:
            epoch = None
        epoch = str(epoch) if epoch is not None else "unknown_epoch"

        path_parts = [base_log_dir]
        if experiment_name:
            sanitized_name = str(experiment_name).replace('/', '_').replace('\\', '_')
            path_parts.append(sanitized_name)
        path_parts.append(epoch)
        rollout_dir = f"data_id={self.data_id}rollout_id={self.rollout_id}"
        path_parts.append(rollout_dir)
        log_dir = os.path.join(*path_parts)
        
        game = AvalonGame(
            agents=self.agents,
            config=config,
            log_dir=log_dir,
            language=game_config.get('language', 'en'),
            preset_roles=assigned_roles,
        )
        
        good_victory = await game.run() or False
        
        # Clean up httpx client resources in agent LLM clients
        await cleanup_agent_llm_clients(self.agents)
        
        return good_victory, self.training_indices
    
    def execute(self) -> Trajectory:
        """
        Execute the Avalon rollout workflow and return a CMT object.
        
        Returns:
            Trajectory (AgentscopeCMT): A CMT object containing model_call_history
                from training agents, converted to training samples.
        """
        # Execute the game
        good_victory, training_indices = asyncio.run(self._execute_async())
        
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
            return Trajectory(
                data_id=self.task.task_id,
                rollout_id=self.task.task_id,
                steps=[],
                is_terminated=True,
                reward=Reward(outcome=1.0 if good_victory else 0.0),
                metadata={},
            )
        
        # Calculate reward for the training agent
        reward = self._calculate_reward(training_agent_idx, good_victory)
        
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
