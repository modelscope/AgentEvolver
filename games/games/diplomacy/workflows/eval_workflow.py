# -*- coding: utf-8 -*-
"""EvalDiplomacyWorkflow class for running Diplomacy game evaluation."""
import asyncio
import os
import copy
import uuid
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from games.utils import cleanup_agent_llm_clients, load_agent_class
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from games.agents.memory import SlidingWindowMemory
from agentscope.tool import Toolkit

from games.games.diplomacy.game import DiplomacyGame
from games.games.diplomacy.engine import DiplomacyConfig
from agentscope.token import HuggingFaceTokenCounter
from games.agents.secure_multi_agent_formatter import SecureMultiAgentFormatter

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


class EvalDiplomacyWorkflow:
    """Workflow class for Diplomacy game evaluation."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize workflow with config dictionary.
        
        Args:
            config_dict: Configuration dictionary containing game settings,
                model configurations, etc.
        """
        self.config_dict = config_dict
        self.power_manager: Optional[PowerManager] = None

    def _get_model_config(self, power_name: str) -> Dict[str, Any]:
        """
        Get model configuration for a power.
        Power-specific config overrides default_model config.
        """
        default_model = self.config_dict.get('default_model', {})
        models_config = self.config_dict.get('roles', {})

        # Start with default_model config
        config = copy.deepcopy({**default_model})

        # Find power-specific config
        power_key = power_name.upper()
        if power_key in models_config:
            config.update(models_config[power_key])
        elif 'default' in models_config:
            config.update(models_config['default'])

        return config

    def _create_agent(self, player_id: int, power_name: str):
        """Create an agent for a power."""

        model_config = self._get_model_config(power_name)

        # Build model kwargs
        # Get base_url from config first, then from environment variable
        base_url = model_config.get('url') or os.environ.get('OPENAI_BASE_URL')
        if not base_url:
            raise ValueError(
                "base_url is required. Please set it in config (url) or "
                "environment variable (OPENAI_BASE_URL)."
            )
        
        model_kwargs = {
            'model_name': model_config.get('model_name', 'qwen-plus'),
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

        formatter_config = self.config_dict.get('formatter', {}) if self.config_dict else {}
        max_model_len = formatter_config.get('max_model_len')
        response_length = formatter_config.get('response_length')
        max_tokens = max_model_len - response_length if max_model_len and response_length else None
        
        model_name_for_tokenizer = "Qwen/Qwen3-4B"

        with _tokenizer_lock:
            token_counter = HuggingFaceTokenCounter(
                pretrained_model_name_or_path=model_name_for_tokenizer,
                use_mirror=True,
            )

        formatter = SecureMultiAgentFormatter(
            token_counter=token_counter,
            max_tokens=max_tokens,
            preserved_agent_names=["Moderator"],
        )

        # Load agent class from role config, default to ThinkingReActAgent
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


    async def _execute_async(self) -> Dict[str, Any]:
        """Execute the game asynchronously."""

        game_config = self.config_dict.get('game', {})
        base_log_dir = game_config.get('log_dir', 'logs')
        evaluation_timestamp = self.config_dict.get('evaluation_timestamp')
        game_id = self.config_dict.get('game_id', 0)
        experiment_name = self.config_dict.get('experiment_name')

        # Generate timestamp if not provided (backward compatibility)
        if not evaluation_timestamp:
            base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            evaluation_timestamp = f"{base_timestamp}_{uuid.uuid4().hex[:8]}"

        # Sanitize experiment_name and build directory path
        path_parts = [base_log_dir]
        if experiment_name:
            sanitized_name = str(experiment_name).replace('/', '_').replace('\\', '_')
            path_parts.append(sanitized_name)
        path_parts.append(evaluation_timestamp)
        game_timestamp = f"id={game_id:04d}"
        path_parts.append(game_timestamp)
        timestamp_dir = os.path.join(*path_parts)

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
        self.agents = [
            self._create_agent(i, self.power_manager.get_power_name(i))
            for i in range(len(power_names))
        ]

        for agent in self.agents:
            if game_id == 0:
                agent.set_console_output_enabled(True)
            else:
                agent.set_console_output_enabled(False)

        # Run game
        diplomacy_game = DiplomacyGame(
            agents=self.agents,
            config=config,
            log_dir=timestamp_dir,
            game_id=game_id,
        )

        game = await diplomacy_game.run()

        results = {
            'game_result': game.outcome,
            'roles': [],  # List of scores in power_names order
        }

        for power_name, power in game.powers.items():
            results['roles'].append({
                'role_name': power_name,
                'score': len(power.centers),
            })

        # Clean up httpx client resources in agent LLM clients
        await cleanup_agent_llm_clients(self.agents)
        
        return results

    def execute(self) -> Union[Trajectory, List[Trajectory], Dict[str, Any]]:
        """Execute the Diplomacy evaluation workflow."""
        return asyncio.run(self._execute_async())
