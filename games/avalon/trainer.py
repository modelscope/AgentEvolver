# -*- coding: utf-8 -*-
"""AvalonTrainer class for training agents in Avalon game."""
import os
import yaml
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    name: str
    api_key: str
    temperature: float = 0.7
    stream: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'api_key': self.api_key,
            'temperature': self.temperature,
            'stream': self.stream,
        }


@dataclass
class TaskConfig:
    """Task configuration dataclass."""
    # Game configuration
    num_players: int = 5
    language: str = 'en'
    log_dir: str = 'logs'
    
    # Model configuration
    default_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name='qwen-plus',
        api_key=os.getenv('API_KEY', ''),
        temperature=0.7,
        stream=True,
    ))
    
    # Role configurations
    train_roles: set = field(default_factory=set)
    no_train_roles: set = field(default_factory=set)
    custom_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaskConfig':
        """Create TaskConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            TaskConfig instance.
        """
        # Parse game config
        game_config = config_dict.get('game', {})
        num_players = game_config.get('num_players', 5)
        language = game_config.get('language', 'en')
        log_dir = game_config.get('log_dir', 'logs')
        
        # Parse default model config
        default_model_dict = config_dict.get('default_model', {})
        default_model = ModelConfig(
            name=default_model_dict.get('name', 'qwen-plus'),
            api_key=default_model_dict.get('api_key', os.getenv('API_KEY', '')),
            temperature=default_model_dict.get('temperature', 0.7),
            stream=default_model_dict.get('stream', True),
        )
        
        # Parse role configurations
        roles_config = config_dict.get('roles', {})
        train_roles = {r.lower() for r in roles_config.get('train', [])}
        no_train_roles = {r.lower() for r in roles_config.get('no_train', [])}
        custom_configs = {
            k.lower(): v for k, v in roles_config.get('custom_configs', {}).items()
        }
        
        return cls(
            num_players=num_players,
            language=language,
            log_dir=log_dir,
            default_model=default_model,
            train_roles=train_roles,
            no_train_roles=no_train_roles,
            custom_configs=custom_configs,
        )


class AvalonTrainer:
    """Trainer class for Avalon game that runs games and collects training data."""
    
    def __init__(self, task_config: Union[str, Dict[str, Any]], model_path: str):
        """Initialize AvalonTrainer.
        
        Args:
            task_config: Path to task config YAML file or config dict.
            model_path: Path to the model to be trained (used for train roles).
        """
        config_dict = self._load_config(task_config)
        self.task_config = TaskConfig.from_dict(config_dict)
        self.model_path = model_path
    
    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dict.
        
        Args:
            config: Path to config file or config dict.
            
        Returns:
            Configuration dictionary.
        """
        if isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return config
    
    def _get_model_config(self, role_name: str, is_training: bool) -> ModelConfig:
        """Get model configuration for a role.
        
        Args:
            role_name: Role name.
            is_training: Whether this is a training role.
            
        Returns:
            ModelConfig instance.
        """
        # Start with default or training model
        config_dict = self.task_config.default_model.to_dict()
        if is_training:
            config_dict['name'] = self.model_path
        
        # Apply custom config if exists
        if role_name.lower() in self.task_config.custom_configs:
            config_dict.update(self.task_config.custom_configs[role_name.lower()])
        
        return ModelConfig(**config_dict)
    
    def _create_model(self, config: ModelConfig):
        """Create a DashScopeChatModel from config.
        
        Args:
            config: Model configuration.
            
        Returns:
            DashScopeChatModel instance.
        """
        from agentscope.model import DashScopeChatModel
        return DashScopeChatModel(
            model_name=config.name,
            api_key=config.api_key,
            stream=config.stream,
        )
    
    def _create_agent(self, player_id: int, role_name: str):
        """Create an agent for a player.
        
        Args:
            player_id: Player ID (0-indexed).
            role_name: Role name (e.g., 'merlin', 'assassin').
            
        Returns:
            ThinkingReActAgent instance.
        """
        from agentscope.formatter import DashScopeMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
        
        is_training = role_name.lower() in self.task_config.train_roles
        model_config = self._get_model_config(role_name, is_training)
        model = self._create_model(model_config)
        
        return ThinkingReActAgent(
            name=f"Player{player_id}",
            sys_prompt="",
            model=model,
            formatter=DashScopeMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )
    
    def _calculate_reward(
        self, 
        agent_index: int, 
        roles: List[Tuple[int, str, bool]], 
        good_victory: bool
    ) -> float:
        """Calculate reward for an agent based on game result.
        
        Args:
            agent_index: Index of the agent in the game.
            roles: List of role tuples (role_id, role_name, is_good).
            good_victory: Whether good side won.
            
        Returns:
            Reward value (1.0 if agent's side won, 0.0 otherwise).
        """
        if agent_index >= len(roles):
            return 0.0
        
        _, _, is_good = roles[agent_index]
        return 1.0 if is_good == good_victory else 0.0
    
    def _identify_training_agents(
        self, 
        agents: List, 
        roles: List[Tuple[int, str, bool]]
    ) -> List[int]:
        """Identify indices of training agents and verify their models.
        
        Args:
            agents: List of all agents.
            roles: List of role tuples (role_id, role_name, is_good).
            
        Returns:
            List of training agent indices.
            
        Raises:
            ValueError: If no training agents found or model mismatch detected.
        """
        training_indices = []
        
        for i, (_, role_name, _) in enumerate(roles):
            if role_name.lower() not in self.task_config.train_roles:
                continue
            
            agent = agents[i]
            if hasattr(agent, 'model') and hasattr(agent.model, 'model_name'):
                if agent.model.model_name != self.model_path:
                    raise ValueError(
                        f"Agent {i} with training role '{role_name}' has model "
                        f"'{agent.model.model_name}' but expected '{self.model_path}'"
                    )
            
            training_indices.append(i)
        
        if not training_indices:
            raise ValueError(
                f"No training agents found. Train roles: {self.task_config.train_roles}, "
                f"Assigned roles: {[r[1] for r in roles]}"
            )
        
        return training_indices
    
    def _collect_training_data(
        self,
        agents: List,
        training_indices: List[int],
        roles: List[Tuple[int, str, bool]],
        good_victory: bool,
    ) -> Dict[str, Any]:
        """Collect training data from training agents.
        
        Args:
            agents: List of all agents.
            training_indices: Indices of training agents.
            roles: List of role tuples.
            good_victory: Whether good side won.
            
        Returns:
            Dict with model_call_history and average reward.
        """
        all_histories = []
        all_rewards = []
        
        for idx in training_indices:
            agent = agents[idx]
            
            if hasattr(agent, 'model_call_history'):
                all_histories.extend(agent.model_call_history)
            
            reward = self._calculate_reward(idx, roles, good_victory)
            all_rewards.append(reward)
        
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        
        return {
            'model_call_history': all_histories,
            'reward': avg_reward,
        }
    
    async def train(self) -> Dict[str, Any]:
        """Run a training game and collect training data.
        
        Returns:
            Dict containing:
                - model_call_history: List of model call records for training agent(s)
                - reward: Average reward value for the training agent(s)
        """
        from games.avalon.game import AvalonGame
        from games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
        
        # Setup game
        config = AvalonBasicConfig.from_num_players(self.task_config.num_players)
        env = AvalonGameEnvironment(config)
        assigned_roles = env.get_roles()
        
        # Create agents
        agents = [
            self._create_agent(i, role_name) 
            for i, (_, role_name, _) in enumerate(assigned_roles)
        ]
        
        # Identify training agents
        training_indices = self._identify_training_agents(agents, assigned_roles)
        
        # Run game
        game = AvalonGame(
            agents=agents,
            config=config,
            log_dir=self.task_config.log_dir,
            language=self.task_config.language,
            preset_roles=assigned_roles,
        )
        good_victory = await game.run()
        
        # Collect and return training data
        return self._collect_training_data(
            agents, training_indices, game.roles, good_victory
        )
