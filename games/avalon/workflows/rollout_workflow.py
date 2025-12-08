# -*- coding: utf-8 -*-
"""AvalonWorkflow class for running Avalon game workflows."""
import asyncio
import os
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Reward
from games.avalon.game import AvalonGame
from games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
from games.avalon.utils import GameLogger



# TODO: 传入一个配置，并行玩几局游戏，兼容训练和评测
# TODO： 如果训练的话，多传入Task，覆盖原先的task_config.yaml，主要包含要训练的模型&角色
# TODO：评测脚本，起llm-server

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


class AvalonWorkflow(BaseAgentscopeWorkflow):
    """Workflow class for Avalon game that runs games and returns Trajectory."""
    
    def __init__(
        self,
        task: Task,
        llm_chat_fn: Any,
        model_name: str,
        **kwargs
    ):
        """
        Initialize the Avalon workflow.
        
        Args:
            task (Task): The task containing avalon configuration in metadata.
            llm_chat_fn (Callable): The LLM chat function to use for agent creation.
            model_name (str): The name of the model for training roles.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(task, llm_chat_fn, model_name, **kwargs)
        self.config_dict = task.metadata.get('avalon_config', task.metadata)
        self.role_manager: Optional[RoleManager] = None
        self.training_indices: List[int] = []
        self._train_roles_cache: Optional[set] = None
    
    # ==================== Configuration Methods ====================
    
    def _get_train_roles(self) -> set:
        """Get set of training role names (cached)."""
        if self._train_roles_cache is None:
            roles_config = self.config_dict.get('roles', {})
            self._train_roles_cache = {r.lower() for r in roles_config.get('train', [])}
        return self._train_roles_cache
    
    def _get_game_config(self) -> Dict[str, Any]:
        """Get game configuration."""
        return self.config_dict.get('game', {})
    
    def _get_model_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """Get model configuration for a non-training role."""
        default_model = self.config_dict.get('default_model', {})
        config = {
            'name': default_model.get('name', 'qwen-plus'),
            'api_key': default_model.get('api_key', os.getenv('API_KEY', '')),
            'stream': default_model.get('stream', True),
        }
        # Apply custom configs if any
        roles_config = self.config_dict.get('roles', {})
        custom_configs = {k.lower(): v for k, v in roles_config.get('custom_configs', {}).items()}
        for role_key in [indexed_role.lower(), base_role.lower()]:
            if role_key in custom_configs:
                config.update(custom_configs[role_key])
                break
        return config
    
    # ==================== Agent Management Methods ====================
    
    def _is_training_role(self, indexed_role: str, base_role: str) -> bool:
        """Check if a role is a training role."""
        train_roles = self._get_train_roles()
        return indexed_role.lower() in train_roles or base_role.lower() in train_roles
    
    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        """Create an agent for a player."""
        from agentscope.model import DashScopeChatModel
        from agentscope.formatter import DashScopeMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
        

        # TODO: 检查DashScopeChatModel是否支持本地传入模型
        
        # Use training model if role is training, otherwise create default model
        if self._is_training_role(indexed_role, base_role):
            model = self.model
        else:
            model_config = self._get_model_config(indexed_role, base_role)
            model = DashScopeChatModel(
                model_name=model_config['name'],
                api_key=model_config['api_key'],
                stream=model_config['stream'],
            )
        
        return ThinkingReActAgent(
            name=f"Player{player_id}",
            sys_prompt="",
            model=model,
            formatter=DashScopeMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )
    
    def _create_agents(self, role_manager: RoleManager) -> List[Any]:
        """Create all agents for the game."""
        return [
            self._create_agent(i, role_manager.get_indexed_role(i), role_manager.get_role_name(i))
            for i in range(len(role_manager.roles))
        ]
    
    def _identify_training_agents(self, agents: List[Any], role_manager: RoleManager) -> List[int]:
        """Identify which agents are training agents."""
        train_roles = self._get_train_roles()
        training_indices = [
            i for i in range(len(agents))
            if (role_manager.get_indexed_role(i).lower() in train_roles or
                role_manager.get_role_name(i).lower() in train_roles)
        ]
        if not training_indices:
            raise ValueError(
                f"No training agents found. Train roles: {train_roles}, "
                f"Assigned roles: {role_manager.indexed_roles}"
            )
        return training_indices
    
    # ==================== Trajectory Collection Methods ====================
    
    def _build_trajectory(
        self,
        agent: Any,
        agent_idx: int,
        indexed_role: str,
        base_role: str,
        is_good: bool,
        good_victory: bool,
    ) -> Trajectory:
        """Build a single trajectory for a training agent."""
        game_config = self._get_game_config()
        model_call_history = getattr(agent, 'model_call_history', [])
        num_players = game_config.get('num_players', 5)
        is_good_serialized = GameLogger._convert_to_serializable(is_good)
        good_victory_serialized = GameLogger._convert_to_serializable(good_victory)
        agent_reward = 1.0 if (is_good == good_victory) else 0.0
        
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
                    'num_players': num_players,
                    'language': game_config.get('language', 'en'),
                },
                'agent_index': agent_idx,
                'indexed_role': indexed_role,
                'base_role': base_role,
                'is_good': is_good_serialized,
                'good_victory': good_victory_serialized,
                'num_model_calls': len(model_call_history),
            }
        )
    
    def _collect_trajectories(
        self,
        agents: List[Any],
        training_indices: List[int],
        role_manager: RoleManager,
        good_victory: bool,
    ) -> List[Trajectory]:
        """Collect trajectories from training agents, one per agent."""
        return [
            self._build_trajectory(
                agents[idx], idx,
                role_manager.get_indexed_role(idx),
                role_manager.get_role_name(idx),
                role_manager.is_good(idx),
                good_victory
            )
            for idx in training_indices
        ]
    
    # ==================== Game Execution Methods ====================
    
    async def _execute_async(self) -> Union[Trajectory, List[Trajectory]]:
        """Async execution of the game."""
        game_config = self._get_game_config()
        
        # Setup game environment
        config = AvalonBasicConfig.from_num_players(game_config.get('num_players', 5))
        env = AvalonGameEnvironment(config)
        assigned_roles = env.get_roles()
        self.role_manager = RoleManager(assigned_roles)
        
        # Create agents and identify training agents
        self.agents = self._create_agents(self.role_manager)
        self.training_indices = self._identify_training_agents(self.agents, self.role_manager)
        
        # Run game
        game = AvalonGame(
            agents=self.agents,
            config=config,
            log_dir=game_config.get('log_dir', 'logs'),
            language=game_config.get('language', 'en'),
            preset_roles=assigned_roles,
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
        )
        good_victory = await game.run() or False
        
        # Collect trajectories
        trajectories = self._collect_trajectories(
            self.agents,
            self.training_indices,
            self.role_manager,
            good_victory,
        )
        
        # Return single Trajectory if only one training agent, otherwise return list
        return trajectories[0] if len(trajectories) == 1 else trajectories
    
    def execute(self) -> Union[Trajectory, List[Trajectory]]:
        """
        Execute the Avalon workflow and return Trajectory or List[Trajectory].
        
        Returns:
            Trajectory: If there is only one training agent.
            List[Trajectory]: If there are multiple training agents.
        """
        return asyncio.run(self._execute_async())


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
        **kwargs
    ):
        """
        Initialize the Avalon rollout workflow.
        
        Args:
            task (Task): The task containing avalon configuration in metadata.
            llm_chat_fn (Callable): The LLM chat function to use for agent creation.
            model_name (str): The name of the model for training roles.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(task, llm_chat_fn, model_name, **kwargs)
        self.config_dict = task.metadata.get('avalon_config', task.metadata)
        self.role_manager: Optional[RoleManager] = None
    
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
        from agentscope.formatter import OpenAIMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
        
        # Use training model if role is training, otherwise create default model
        if self._is_training_role(indexed_role, base_role):
            model = self.model
        else:
            model_config = self._get_model_config(indexed_role, base_role)
            
            # Build model kwargs (aligned with eval_workflow.py)
            model_kwargs = {
                'model_name': model_config['model_name'],
                'client_args': {'base_url': model_config['url']},
            }
            
            # Add optional parameters
            # Get api_key from environment variable first, then from config
            api_key = os.environ.get('API_KEY') or model_config.get('api_key')
            if api_key:
                model_kwargs['api_key'] = api_key
            if 'stream' in model_config:
                model_kwargs['stream'] = model_config['stream']
            
            # Build generate_kwargs
            generate_kwargs = {
                k: model_config[k] for k in ['temperature', 'max_tokens']
                if k in model_config
            }
            if generate_kwargs:
                model_kwargs['generate_kwargs'] = generate_kwargs
            
            model = OpenAIChatModel(**model_kwargs)
        
        return ThinkingReActAgent(
            name=f"Player{player_id}",
            sys_prompt="",
            model=model,
            formatter=OpenAIMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )
    
    async def _execute_async(self) -> bool:
        """Execute the game asynchronously."""
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

        # Run game
        game = AvalonGame(
            agents=self.agents,
            config=config,
            log_dir=game_config.get('log_dir', 'logs'),
            language=game_config.get('language', 'en'),
            preset_roles=assigned_roles,
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
        )
        
        good_victory = await game.run() or False
        return good_victory
    
    def execute(self) -> Trajectory:
        """
        Execute the Avalon rollout workflow.
        
        Returns:
            Trajectory: The trajectory (to be implemented later).
        """
        result = asyncio.run(self._execute_async())
        # TODO: Collect trajectory from model_call_history
        # For now, return a placeholder Trajectory
        return Trajectory(
            data_id=self.task.task_id,
            rollout_id=self.task.task_id,
            steps=[],
            is_terminated=True,
            reward=None,
            metadata={},
        )
