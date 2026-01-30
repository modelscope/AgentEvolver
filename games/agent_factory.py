# -*- coding: utf-8 -*-
"""Unified factory functions for creating agents, models, memories, and formatters from config.

This module provides a unified interface for creating agents and their components
from configuration dictionaries. This makes it easy for users to customize agents
by only modifying configuration files.

Example:
    ```yaml
    # In config file
    roles:
      merlin:
        agent_config:
          type: ThinkingReActAgent
          kwargs:
            sys_prompt: "You are Merlin..."
            memory:
              type: InMemoryMemory
              kwargs: {}
            formatter:
              type: SecureMultiAgentFormatter
              kwargs:
                max_tokens: 1000
                preserved_agent_names: ["Moderator"]
    ```

    ```python
    from games.agent_factory import create_agent_from_config
    
    agent = create_agent_from_config(
        agent_config=roles['merlin']['agent_config'],
        model=model_instance,
        name="Player0"
    )
    ```
"""
import os
import threading
import importlib
from typing import Any, Dict, Optional, Type, Union

from agentscope.model import ChatModelBase, OpenAIChatModel
from agentscope.memory import MemoryBase, InMemoryMemory
from agentscope.tool import Toolkit
from agentscope.token import HuggingFaceTokenCounter

from games.utils import load_agent_class
from games.agents.secure_multi_agent_formatter import SecureMultiAgentFormatter
# Note: SummarizedMemory and SlidingWindowMemory are imported lazily in create_memory_from_config
# to avoid circular import issues

# Lock for protecting HuggingFaceTokenCounter initialization from concurrent access
_tokenizer_lock = threading.Lock()


def create_model_from_config(
    model_config: Dict[str, Any],
    model_class: Optional[Type[ChatModelBase]] = None,
) -> ChatModelBase:
    """Create a model instance from configuration.
    
    Args:
        model_config: Model configuration dictionary. Should contain:
            - model_name: Name of the model
            - url: Base URL for the API (or use OPENAI_BASE_URL env var)
            - api_key: API key (or use OPENAI_API_KEY env var)
            - temperature: Optional temperature parameter
            - max_tokens: Optional max_tokens parameter
            - stream: Optional stream parameter
        model_class: Optional model class to use. If None, defaults to OpenAIChatModel.
    
    Returns:
        A ChatModelBase instance.
    
    Example:
        ```python
        model_config = {
            'model_name': 'qwen-plus',
            'url': 'https://api.example.com/v1',
            'api_key': 'your-key',
            'temperature': 0.7,
            'max_tokens': 2048,
        }
        model = create_model_from_config(model_config)
        ```
    """
    if model_class is None:
        model_class = OpenAIChatModel
    
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
    # Turn off auto-thinking for qwen3
    generate_kwargs['extra_body'] = {
        'enable_thinking': False,  # Required for non-streaming calls with DashScope
    }
    if generate_kwargs:
        model_kwargs['generate_kwargs'] = generate_kwargs
    
    return model_class(**model_kwargs)


def create_memory_from_config(
    memory_config: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    game_id: Optional[Union[str, int]] = None,
    log_dir: Optional[str] = None,
) -> MemoryBase:
    """Create a memory instance from configuration.
    
    Args:
        memory_config: Memory configuration dictionary. Should contain:
            - type: Memory class name or full path (e.g., "InMemoryMemory" or 
                    "games.agents.memory.SlidingWindowMemory")
            - kwargs: Optional keyword arguments for the memory constructor
        agent_id: Optional agent ID (usually the agent's name) to be passed to memory constructor.
        game_id: Optional game ID to be passed to memory constructor.
        log_dir: Optional log directory path to be passed to memory constructor.
    
    Returns:
        A MemoryBase instance. Defaults to InMemoryMemory if config is None.
    
    Example:
        ```python
        # Simple config
        memory_config = {
            'type': 'InMemoryMemory'
        }
        memory = create_memory_from_config(memory_config)
        
        # With custom memory
        memory_config = {
            'type': 'SlidingWindowMemory',
            'kwargs': {}
        }
        memory = create_memory_from_config(memory_config, agent_id="Player1", game_id="game_123")
        ```
    """
    if memory_config is None:
        return InMemoryMemory()
    
    memory_type = memory_config.get('type', 'InMemoryMemory')
    memory_kwargs = (memory_config.get('kwargs') or {}).copy()

    # Inject contextual parameters if provided
    # Only if they are not already in kwargs (to allow config override)
    if agent_id is not None and 'agent_id' not in memory_kwargs:
        memory_kwargs['agent_id'] = agent_id
    if game_id is not None and 'game_id' not in memory_kwargs:
        memory_kwargs['game_id'] = game_id
    if log_dir is not None and 'log_dir' not in memory_kwargs:
        memory_kwargs['log_dir'] = log_dir
    
    # Try to import from agentscope.memory first
    try:
        from agentscope.memory import __all__ as memory_exports
        if memory_type in memory_exports:
            module = importlib.import_module("agentscope.memory")
            MemoryClass = getattr(module, memory_type)
            return MemoryClass(**memory_kwargs)
    except (ImportError, AttributeError):
        pass
    
    # Try to import from games.agents.memory
    try:
        from games.agents.memory import __all__ as memory_exports
        if memory_type in memory_exports:
            module = importlib.import_module("games.agents.memory")
            MemoryClass = getattr(module, memory_type)
            return MemoryClass(**memory_kwargs)
    except (ImportError, AttributeError):
        pass
    
    # Try full module path
    if "." in memory_type:
        try:
            module_path, class_name = memory_type.rsplit(".", 1)
            module = importlib.import_module(module_path)
            MemoryClass = getattr(module, class_name)
            return MemoryClass(**memory_kwargs)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import memory class from '{memory_type}': {e}"
            ) from e
    
    # Default to InMemoryMemory
    return InMemoryMemory(**memory_kwargs)


def create_formatter_from_config(
    formatter_config: Optional[Dict[str, Any]] = None,
    actor_rollout_ref: Optional[Any] = None,
) -> Any:
    """Create a formatter instance from configuration.
    
    This function handles all formatter-related logic including:
    - Parsing formatter config from agent_config
    - Setting default values (type, preserved_agent_names)
    - Extracting tokenizer_path and formatter settings from actor_rollout_ref
    - Calculating max_tokens from actor_rollout_ref if not specified
    - Creating token_counter if needed
    
    Args:
        formatter_config: Formatter configuration dictionary from agent_config. Can be:
            - None: Use defaults
            - Dict with 'type' and 'kwargs': Use specified config
            - Dict with only 'kwargs': Use default type with specified kwargs
        actor_rollout_ref: Optional config object with actor_rollout_ref structure:
            - `actor_rollout_ref.model.path` -> tokenizer_path
            - `actor_rollout_ref.rollout.max_model_len` -> max_model_len
            - `actor_rollout_ref.rollout.response_length` -> response_length
            - None: Use defaults
    
    Returns:
        A formatter instance. Defaults to SecureMultiAgentFormatter if config is None.
    
    Example:
        ```python
        # From agent_config
        formatter_config = {
            'type': 'SecureMultiAgentFormatter',
            'kwargs': {
                'max_tokens': 1000,
                'preserved_agent_names': ['Moderator']
            }
        }
        # With actor_rollout_ref
        formatter = create_formatter_from_config(
            formatter_config=formatter_config,
            actor_rollout_ref=self.config.actor_rollout_ref
        )
        ```
    """
    # Parse formatter config
    if formatter_config is None:
        formatter_type = 'SecureMultiAgentFormatter'
        formatter_kwargs = {}
    elif isinstance(formatter_config, dict):
        formatter_type = formatter_config.get('type', 'SecureMultiAgentFormatter')
        formatter_kwargs = (formatter_config.get('kwargs') or {}).copy()
    else:
        # If formatter_config is not a dict, use defaults
        formatter_type = 'SecureMultiAgentFormatter'
        formatter_kwargs = {}
    
    # Extract tokenizer_path and formatter settings from actor_rollout_ref
    tokenizer_path = None
    max_model_len = None
    response_length = None
    
    if actor_rollout_ref is not None:
        try:
            tokenizer_path = actor_rollout_ref.model.path
            max_model_len = actor_rollout_ref.rollout.max_model_len
            response_length = actor_rollout_ref.rollout.response_length
        except AttributeError:
            pass
    
    # Calculate max_tokens if not specified and actor_rollout_ref provided
    if 'max_tokens' not in formatter_kwargs and max_model_len and response_length:
        formatter_kwargs['max_tokens'] = max_model_len - response_length
    
    # Set default preserved_agent_names if not specified
    if 'preserved_agent_names' not in formatter_kwargs:
        formatter_kwargs['preserved_agent_names'] = ["Moderator"]
    
    # Create token_counter if needed and tokenizer_path available
    if 'token_counter' not in formatter_kwargs and tokenizer_path:
        with _tokenizer_lock:
            token_counter = HuggingFaceTokenCounter(
                pretrained_model_name_or_path=tokenizer_path,
                use_mirror=True,
            )
        formatter_kwargs['token_counter'] = token_counter
    
    # Try to import from games.agents first
    try:
        if formatter_type == 'SecureMultiAgentFormatter':
            return SecureMultiAgentFormatter(**formatter_kwargs)
    except (ImportError, AttributeError):
        pass
    
    # Try full module path
    if "." in formatter_type:
        try:
            module_path, class_name = formatter_type.rsplit(".", 1)
            module = importlib.import_module(module_path)
            FormatterClass = getattr(module, class_name)
            return FormatterClass(**formatter_kwargs)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import formatter class from '{formatter_type}': {e}"
            ) from e
    
    # Default to SecureMultiAgentFormatter
    return SecureMultiAgentFormatter(**formatter_kwargs)


def create_agent_from_config(
    agent_config: Dict[str, Any],
    model: ChatModelBase,
    name: str,
    actor_rollout_ref: Optional[Any] = None,
    game_id: Optional[Union[str, int]] = None,
    log_dir: Optional[str] = None,
) -> Any:
    """Create an agent instance from configuration.
    
    This is the main entry point for creating agents. It handles:
    - Loading the agent class from agent_config['type']
    - Parsing and setting default values for all parameters
    - Creating memory, formatter, and toolkit from nested configs
    - Instantiating the agent with all components
    
    Args:
        agent_config: Agent configuration dictionary. Should contain:
            - type: Agent class name or full path (e.g., "ThinkingReActAgent")
            - kwargs: Keyword arguments for agent constructor, which can include:
                - sys_prompt: System prompt (defaults to "")
                - Note: 'name' should NOT be in kwargs, it will be set from the name parameter
                - memory: Optional memory config dict (see create_memory_from_config)
                - formatter: Optional formatter config dict (see create_formatter_from_config)
                - toolkit: Optional toolkit config dict (see create_toolkit_from_config)
                - Any other agent-specific parameters
        model: Pre-created model instance (required).
        name: Agent name (required).
        actor_rollout_ref: Optional config object with actor_rollout_ref structure.
            Will be passed to create_formatter_from_config to extract tokenizer_path
            and formatter settings. Can be:
            - `config.actor_rollout_ref` (for rollout workflow)
            - None: Use defaults (for eval workflow)
        game_id: Optional game ID to be passed to memory constructor.
        log_dir: Optional log directory path to be passed to memory constructor.
    
    Returns:
        An agent instance.
    
    Example:
        ```python
        agent_config = {
            'type': 'ThinkingReActAgent',
            'kwargs': {
                'sys_prompt': 'You are a helpful assistant.',
                'memory': {
                    'type': 'InMemoryMemory',
                    'kwargs': {}
                },
                'formatter': {
                    'type': 'SecureMultiAgentFormatter',
                    'kwargs': {
                        'max_tokens': 1000,
                        'preserved_agent_names': ['Moderator']
                    }
                }
            }
        }
        model = create_model_from_config(model_config)
        agent = create_agent_from_config(
            agent_config=agent_config,
            model=model,
            name="Player0",
            actor_rollout_ref=self.config.actor_rollout_ref
        )
        ```
    """
    # Validate agent_config structure
    if not isinstance(agent_config, dict):
        raise ValueError("agent_config must be a dictionary")
    
    # Ensure agent_config has required fields
    if 'type' not in agent_config:
        raise ValueError("agent_config must contain 'type' field")
    
    if 'kwargs' not in agent_config or agent_config['kwargs'] is None:
        agent_config['kwargs'] = {}
    
    # Load agent class
    agent_type = agent_config['type']
    AgentClass = load_agent_class(agent_type)
    
    # Get agent kwargs (make a copy to avoid modifying original)
    agent_kwargs = agent_config['kwargs'].copy()
    
    # Set model (required)
    agent_kwargs['model'] = model
    
    # Set name (name is controlled by workflow, not config)
    if not name:
        raise ValueError(
            "name is required. "
            "Please provide name when calling create_agent_from_config."
        )
    agent_kwargs['name'] = name
    
    # Set default sys_prompt if not provided
    if 'sys_prompt' not in agent_kwargs:
        agent_kwargs['sys_prompt'] = ""
    
    # Parse and create formatter (all logic is in create_formatter_from_config)
    formatter_config_dict = agent_kwargs.pop('formatter', None)
    agent_kwargs['formatter'] = create_formatter_from_config(
        formatter_config=formatter_config_dict,
        actor_rollout_ref=actor_rollout_ref,
    )
    
    # Parse and create memory
    if 'memory' in agent_kwargs:
        memory_config = agent_kwargs.pop('memory')
        agent_kwargs['memory'] = create_memory_from_config(memory_config, agent_id=name, game_id=game_id, log_dir=log_dir)
    else:
        # Default memory if not specified
        agent_kwargs['memory'] = InMemoryMemory()
    
    # Default toolkit if not specified
    agent_kwargs['toolkit'] = Toolkit()
    
    # Create and return agent instance
    return AgentClass(**agent_kwargs)

