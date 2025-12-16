# -*- coding: utf-8 -*-
"""EvalDiplomacyWorkflow class for running Diplomacy game evaluation."""
import asyncio
import os
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
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


class EvalDiplomacyWorkflow:
    """Workflow class for Diplomacy game evaluation."""

    def __init__(self, task: Task):
        self.config_dict = task.metadata.get('diplomacy_config', task.metadata)
        self.power_manager: Optional[PowerManager] = None

    def _get_model_config(self, power_name: str) -> Dict[str, Any]:
        """
        Get model configuration for a power.
        Power-specific config overrides default_model config.
        """
        default_model = self.config_dict.get('default_model', {})
        models_config = self.config_dict.get('models', {})

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
        from agentscope.model import OpenAIChatModel
        from agentscope.formatter import OpenAIMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from games.agents.thinking_react_agent import ThinkingReActAgent

        model_config = self._get_model_config(power_name)

        # Build model kwargs
        model_kwargs = {
            'model_name': model_config.get('model_name', 'qwen-plus'),
        }

        # Add client_args if url is specified
        if 'url' in model_config:
            model_kwargs['client_args'] = {'base_url': model_config['url']}

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

    async def _execute_async(self) -> Dict[str, Any]:
        """Execute the game asynchronously."""
        game_config = self.config_dict.get('game', {})

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

        # Run game
        diplomacy_game = DiplomacyGame(
            agents=self.agents,
            config=config,
            log_dir=game_config.get('log_dir', 'logs'),
        )

        game = await diplomacy_game.run()

        # Return evaluation results
        results = {
            'outcome': game.outcome,
            'powers': {},
        }

        for power_name, power in game.powers.items():
            results['powers'][power_name] = {
                'supply_centers': len(power.centers),
                'is_eliminated': power.is_eliminated(),
                'units': len(power.units),
            }

        return results

    def execute(self) -> Union[Trajectory, List[Trajectory], Dict[str, Any]]:
        """Execute the Diplomacy evaluation workflow."""
        return asyncio.run(self._execute_async())
