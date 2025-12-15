# -*- coding: utf-8 -*-
"""AvalonWorkflow class for running Avalon game workflows."""
import asyncio
import os
import copy
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from openai import OpenAI

from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from games.games.avalon.game import AvalonGame
from games.games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment


class RoleManager:
    """Manages role indexing and identification."""
    
    def __init__(self, roles: List[tuple]):
        self.roles = roles
        self.indexed_roles = self._build_indexed_roles(roles)
    
    @staticmethod
    def _build_indexed_roles(roles: List[tuple]) -> List[str]:
        """Build indexed role names with counters."""
        role_counters = defaultdict(int)
        indexed_roles = []
        for _, role_name, _ in roles:
            indexed_roles.append(f"{role_name}_{role_counters[role_name]}")
            role_counters[role_name] += 1
        return indexed_roles
    
    def get_indexed_role(self, index: int) -> str:
        return self.indexed_roles[index]
    
    def get_role_name(self, index: int) -> str:
        return self.roles[index][1]


class EvalAvalonWorkflow:
    """Workflow class for Avalon game evaluation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize workflow with config dictionary.
        
        Args:
            config_dict: Configuration dictionary containing game settings,
                model configurations, etc.
        """
        self.config_dict = config_dict
        self.role_manager: Optional[RoleManager] = None
    
    def _get_model_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """
        Get model configuration for a role.
        Role-specific config overrides default_model config.
        """
        if self.config_dict is None:
            raise ValueError("config_dict is None. Please check your configuration file.")
        
        default_model = self.config_dict.get('default_model', {})
        roles_config = self.config_dict.get('roles', {})
        
        if not isinstance(default_model, dict):
            default_model = {}
        if not isinstance(roles_config, dict):
            roles_config = {}
        
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
    
    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        """Create an agent for a player."""
        from agentscope.model import OpenAIChatModel
        from agentscope.memory import InMemoryMemory
        from agentscope.formatter import OpenAIMultiAgentFormatter
        from agentscope.token import HuggingFaceTokenCounter
        from agentscope.tool import Toolkit
        from games.agents.thinking_react_agent import ThinkingReActAgent
        from games.agents.secure_multi_agent_formatter import SecureMultiAgentFormatter

        
        model_config = self._get_model_config(indexed_role, base_role)
        
        # Build model kwargs
        model_kwargs = {
            'model_name': model_config['model_name'],
            'client_args': {'base_url': model_config['url']},
        }
        
        # Add optional parameters
        # Get api_key from environment variable first, then from config
        api_key = os.environ.get('OPENAI_API_KEY') or model_config.get('api_key')
        if api_key:
            model_kwargs['api_key'] = api_key
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
        
        # FIXME: model_name_for_tokenizer defaults to HuggingFace Qwen3-4B
        model_name_for_tokenizer = "Qwen/Qwen3-4B"
        
        # Calculate max_tokens for formatter (leave room for response)
        # Follow the same logic as rollout_workflow.py
        formatter_config = self.config_dict.get('formatter', {}) if self.config_dict else {}
        max_model_len = formatter_config.get('max_model_len')
        response_length = formatter_config.get('response_length')
        max_tokens = max_model_len - response_length if max_model_len and response_length else None
        
        # Get preserved agent names from config (if available)
        # Default to preserving "Moderator" if not specified
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
        
        return ThinkingReActAgent(
            name=f"Player{player_id}",
            sys_prompt="",
            model=model,
            formatter=formatter,
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
            # thinking_sys_prompt=""
        )
    
    async def _execute_async(self) -> Dict[str, Any]:
        """Execute the game asynchronously.
        
        Returns:
            Dictionary containing game results with keys like:
            - good_victory: bool/int (1 for True, 0 for False)
            - quest_results: list of quest outcomes
            - num_quests: int (number of quests completed)
            - num_quest_successes: int (number of successful quests)
            - num_quest_failures: int (number of failed quests)
        """
        if self.config_dict is None:
            raise ValueError("config_dict is None. Please check your configuration file.")
        
        game_config = self.config_dict.get('game', {})
        if not isinstance(game_config, dict):
            game_config = {}
        
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
        # Generate unique timestamp for parallel games
        # This prevents multiple parallel games from overwriting each other's logs
        base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_timestamp = f"{base_timestamp}_{uuid.uuid4().hex[:8]}"

        # Get log_dir from game config and experiment_name from config
        log_dir = game_config.get('log_dir', 'logs')
        experiment_name = self.config_dict.get('experiment_name')
        
        # If experiment_name is provided, append it to log_dir
        if experiment_name:
            # Sanitize experiment_name to avoid filesystem issues
            experiment_name = str(experiment_name).replace('/', '_').replace('\\', '_')
            log_dir = os.path.join(log_dir, experiment_name)
        
        game = AvalonGame(
            agents=self.agents,
            config=config,
            log_dir=log_dir,
            language=game_config.get('language', 'en'),
            preset_roles=assigned_roles,
            timestamp=unique_timestamp,
        )
        
        good_victory = await game.run()
        
        # Build result dictionary
        if good_victory is None:
            # Game was stopped
            return {
                "good_victory": None,
            }
        
        return {
            "good_victory": 1 if good_victory else 0,  # Convert to int for averaging
        }
    
    def execute(self) -> Dict[str, Any]:
        """Execute the Avalon workflow.
        
        Returns:
            Dictionary containing game results.
        """
        return asyncio.run(self._execute_async())
