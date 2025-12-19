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
from games.utils import cleanup_agent_llm_clients, load_agent_class
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
    
    def _get_model_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """
        Get model configuration for a role.
        Role-specific config overrides default_model config.
        """
        default_model = self.config_dict.get('default_model', {})
        roles_config = self.config_dict.get('roles', {})
        
        # Start with default_model config
        config = copy.deepcopy({**default_model})
        
        # Find role-specific config (try indexed_role first, then base_role)
        role_config = next(
            (v for k, v in roles_config.items() 
             if k.lower() in [indexed_role.lower(), base_role.lower()]),
            None
        )
        
        # Override with role-specific config
        if role_config:
            config.update(role_config)
            # Handle model_name -> name mapping
            if 'model_name' in role_config:
                config['model_name'] = role_config['model_name']
        
        return config
    
    def _is_training_role(self, indexed_role: str, base_role: str) -> bool:
        """Check if a role is a training role based on trainable flag in config."""
        model_config = self._get_model_config(indexed_role, base_role)
        # Check if trainable is explicitly set to True
        return model_config.get('trainable', False) is True
    
    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        """Create an agent for a player."""
        from agentscope.model import OpenAIChatModel
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from agentscope.token import HuggingFaceTokenCounter
        from games.agents.secure_multi_agent_formatter import (
            SecureMultiAgentFormatter,
        )
        
        # Use training model if role is training, otherwise create default model
        if self._is_training_role(indexed_role, base_role):
            model = self.model
            # For training model, use model path from config
            model_name_for_tokenizer = self.config.actor_rollout_ref.model.path
        else:
            model_config = self._get_model_config(indexed_role, base_role)
            
            # Build model kwargs (aligned with eval_workflow.py)
            # Get base_url from config first, then from environment variable
            base_url = model_config.get('url') or os.environ.get('OPENAI_BASE_URL')
            if not base_url:
                raise ValueError(
                    "base_url is required. Please set it in config (url) or "
                    "environment variable (OPENAI_BASE_URL)."
                )
            
            model_kwargs = {
                'model_name': model_config['model_name'],
                'client_args': {'base_url': base_url},
            }
            
            # Add optional parameters
            # Get api_key from environment variable first, then from config
            api_key = model_config.get('api_key') or os.environ.get('OPENAI_API_KEY')
            

            if api_key:
                model_kwargs['api_key'] = api_key
            else:
                raise ValueError(
                    "api_key is required. Please set it in config (api_key) or "
                    "environment variable (OPENAI_API_KEY)."
                )
            if 'stream' in model_config:
                model_kwargs['stream'] = model_config['stream']
            
            # Build generate_kwargs
            generate_kwargs = {
                k: model_config[k] for k in ['temperature', 'max_tokens']
                if k in model_config
            }
            # turn off auto-thinking for qwen3
            generate_kwargs['extra_body'] = {
                    'enable_thinking': False,  # Required for non-streaming calls with DashScope
                }
            if generate_kwargs:
                model_kwargs['generate_kwargs'] = generate_kwargs
            
            model = OpenAIChatModel(**model_kwargs)
            model_name_for_tokenizer = self.config.actor_rollout_ref.model.path

        
        # Calculate max_tokens for formatter (leave room for response)
        max_model_len = self.config.actor_rollout_ref.rollout.max_model_len
        response_length = self.config.actor_rollout_ref.rollout.response_length
        max_tokens = max_model_len - response_length if max_model_len and response_length else None
        
        # Get preserved agent names from config (if available)
        # Default to preserving "主持人" if not specified
        preserved_agent_names = ["Moderator"]
        
        # Create formatter with truncation support
        formatter = SecureMultiAgentFormatter(
            token_counter=HuggingFaceTokenCounter(
                                pretrained_model_name_or_path=model_name_for_tokenizer,
                                use_mirror=True,
                          ),
            max_tokens=max_tokens,
            preserved_agent_names=preserved_agent_names,
        )
        
        # Load agent class from role config, default to ThinkingReActAgent
        model_config = self._get_model_config(indexed_role, base_role)
        agent_class_path = model_config.get('agent_class')
        AgentClass = load_agent_class(agent_class_path)
        
        return AgentClass(
            name=f"Player{player_id}",
            sys_prompt="",
            model=model,
            formatter=formatter,
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
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
