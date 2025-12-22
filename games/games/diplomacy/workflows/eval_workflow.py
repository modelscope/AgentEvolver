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
from games.utils import (
    cleanup_agent_llm_clients,
    create_agent_from_config,
    create_model_from_config,
    deep_merge,
)
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

    def _get_role_config(self, power_name: str) -> Dict[str, Any]:
        """
        Get complete role configuration for a power (including model, agent, trainable, act_by_user, etc.).
        Power-specific config overrides default_role config.
        """
        default_role = self.config_dict.get('default_role', {})
        roles_config = self.config_dict.get('roles', {})

        if not isinstance(default_role, dict):
            default_role = {}
        if not isinstance(roles_config, dict):
            roles_config = {}

        # Start with default_role config
        role_config = copy.deepcopy(default_role)

        # Find power-specific config
        power_key = power_name.upper()
        if power_key in roles_config:
            specific_role_config = roles_config[power_key]
        else:
            specific_role_config = None

        # Override with power-specific config if present
        if specific_role_config and isinstance(specific_role_config, dict):
            # Deep merge: recursively merge nested dicts
            role_config = deep_merge(role_config, specific_role_config)

        return role_config
    
    def _get_model_config(self, power_name: str) -> Dict[str, Any]:
        """Get model configuration for a power."""
        role_config = self._get_role_config(power_name)
        return role_config.get('model', {})
    
    def _get_agent_config(self, power_name: str) -> Dict[str, Any]:
        """Get agent configuration for a power."""
        role_config = self._get_role_config(power_name)
        return role_config.get('agent', {})

    def _create_agent(self, player_id: int, power_name: str):
        """Create an agent for a power using create_agent_from_config."""
        model_config = self._get_model_config(power_name)
        agent_config = self._get_agent_config(power_name)

        # Create model using factory function
        model = create_model_from_config(model_config)
        
        # Validate agent_config
        if not agent_config:
            raise ValueError(
                f"agent config is required. Please specify it in default_role.agent or role-specific config for {power_name}."
            )
        
        return create_agent_from_config(
            agent_config=agent_config,
            model=model,
            name=f"Player{player_id}",
            actor_rollout_ref=None,  # eval workflow doesn't have actor_rollout_ref
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
