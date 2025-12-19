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
from games.utils import cleanup_agent_llm_clients, load_agent_class
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

    # ==================== Agent Management Methods ====================

    def _is_training_power(self, indexed_role: str, base_role: str) -> bool:
        """Check if a power is a training power based on trainable flag in config."""
        model_config = self._get_model_config(indexed_role, base_role)
        # Check if trainable is explicitly set to True
        return model_config.get('trainable', False) is True

    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        """Create an agent for a power."""
        from agentscope.model import OpenAIChatModel
        from agentscope.memory import InMemoryMemory
        from games.agents.memory import SlidingWindowMemory
        from agentscope.tool import Toolkit

        # Use training model if power is training, otherwise create default model
        if self._is_training_power(indexed_role, base_role):
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
            # Get api_key from config first, then from environment variable
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

        # token计数可能有问题，添加安全边界
        SAFETY_MARGIN = 100
        
        max_tokens = max_model_len - response_length - SAFETY_MARGIN if max_model_len and response_length else None


        # Ensure max_tokens is valid
        if max_tokens is not None and max_tokens <= 0:
            logger.warning(
                f"max_tokens calculated as {max_tokens}, setting to None. "
                f"max_model_len={max_model_len}, response_length={response_length}"
            )
            max_tokens = None
        # Get preserved agent names from config (if available)
        # Default to preserving "Moderator" if not specified
        preserved_agent_names = ["Moderator"]
        
        # Create formatter with truncation support
        # Use lock to protect HuggingFaceTokenCounter initialization from concurrent access
        with _tokenizer_lock:
            token_counter = HuggingFaceTokenCounter(
                pretrained_model_name_or_path=model_name_for_tokenizer,
                use_mirror=True,
            )
        
        formatter = SecureMultiAgentFormatter(
            token_counter=token_counter,
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
            memory=SlidingWindowMemory(),
            toolkit=Toolkit(),
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
