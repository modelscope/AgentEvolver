# -*- coding: utf-8 -*-
"""Common utility functions shared by all games."""
import asyncio
import copy
import importlib
from pathlib import Path
from typing import Any, Dict, List, Type

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from loguru import logger


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file with Hydra inheritance support.
    
    Uses Hydra's compose API to load configuration with defaults inheritance.
    The config file should use Hydra's `defaults` list to specify base configs.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config_dir = config_path.parent.resolve()
    config_name = config_path.stem  # filename without extension
    
    # Initialize Hydra config directory and compose the config
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    
    # Convert OmegaConf DictConfig to regular Python dict
    return OmegaConf.to_container(cfg, resolve=True)


async def cleanup_agent_llm_clients(agents: List[Any]) -> None:
    """
    Clean up httpx client resources in all agents' LLM clients.
    
    This function iterates through all agents, finds their models, and closes
    the httpx clients within the models. This is important for async functions
    started with asyncio.run(), as httpx clients may not be properly closed
    when exiting.
    
    Args:
        agents: List of agent objects, each should have a model attribute.
    """
    for agent in agents:
        if not hasattr(agent, 'model'):
            continue
            
        model = agent.model
        
        try:
            # Try to access the httpx client from OpenAI client
            # OpenAIChatModel usually has a client attribute (OpenAI or AsyncOpenAI instance)
            if hasattr(model, 'client'):
                client = model.client
                
                # Check if it's an OpenAI client (sync or async)
                # OpenAI client has a _client attribute, which is httpx.Client or httpx.AsyncClient
                if hasattr(client, '_client'):
                    httpx_client = client._client
                    
                    # Close httpx.AsyncClient
                    if hasattr(httpx_client, 'aclose'):
                        try:
                            await httpx_client.aclose()
                            logger.debug(f"Closed httpx.AsyncClient for agent {getattr(agent, 'name', 'unknown')}")
                        except Exception as e:
                            logger.warning(f"Failed to close httpx.AsyncClient: {e}")
                    
                    # Close httpx.Client
                    elif hasattr(httpx_client, 'close'):
                        try:
                            httpx_client.close()
                            logger.debug(f"Closed httpx.Client for agent {getattr(agent, 'name', 'unknown')}")
                        except Exception as e:
                            logger.warning(f"Failed to close httpx.Client: {e}")
                
                # If client is itself an httpx client (direct httpx usage)
                elif hasattr(client, 'aclose'):
                    try:
                        await client.aclose()
                        logger.debug(f"Closed httpx.AsyncClient (direct) for agent {getattr(agent, 'name', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"Failed to close httpx.AsyncClient (direct): {e}")
                elif hasattr(client, 'close'):
                    try:
                        client.close()
                        logger.debug(f"Closed httpx.Client (direct) for agent {getattr(agent, 'name', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"Failed to close httpx.Client (direct): {e}")
        
        except Exception as e:
            # If unable to access or close client, log warning but don't raise exception
            logger.warning(
                f"Failed to cleanup httpx client for agent {getattr(agent, 'name', 'unknown')}: {e}"
            )


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries recursively.
    
    Values from override will override values in base, but nested dicts will be merged
    recursively instead of being replaced entirely.
    
    Args:
        base: Base dictionary to merge into
        override: Dictionary with values to override/merge into base
        
    Returns:
        A new dictionary with merged values
        
    Example:
        >>> base = {'model': {'name': 'qwen', 'temp': 0.7}, 'trainable': False}
        >>> override = {'model': {'name': 'qwen-max'}}
        >>> deep_merge(base, override)
        {'model': {'name': 'qwen-max', 'temp': 0.7}, 'trainable': False}
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Direct assignment for non-dict values or when base doesn't have this key
            result[key] = copy.deepcopy(value)
    return result


def load_agent_class(agent_class_path: str | None = None) -> Type[Any]:
    """
    Dynamically load an agent class from a module path string.
    
    The agent_class_path can be:
    1. A full module path: "games.agents.thinking_react_agent.ThinkingReActAgent"
    2. A short name from games.agents: "ThinkingReActAgent" (will import from games.agents)
    3. None: returns the default ThinkingReActAgent
    
    Args:
        agent_class_path: Path to the agent class. If None, returns default ThinkingReActAgent.
        
    Returns:
        The agent class (not an instance).
        
    Examples:
        >>> AgentClass = load_agent_class("games.agents.thinking_react_agent.ThinkingReActAgent")
        >>> AgentClass = load_agent_class("ThinkingReActAgent")  # Short form
        >>> AgentClass = load_agent_class()  # Default
    """
    # Default to ThinkingReActAgent
    if agent_class_path is None or agent_class_path == "":
        from games.agents.thinking_react_agent import ThinkingReActAgent
        return ThinkingReActAgent
    
    # Try to import from games.agents first (short form)
    try:
        from games.agents import __all__ as agent_exports
        if agent_class_path in agent_exports:
            module = importlib.import_module("games.agents")
            return getattr(module, agent_class_path)
    except (ImportError, AttributeError):
        pass
    
    # Try full module path (e.g., "games.agents.thinking_react_agent.ThinkingReActAgent")
    if "." in agent_class_path:
        try:
            module_path, class_name = agent_class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import agent class from '{agent_class_path}': {e}"
            ) from e
    
    # If it's just a class name, try to import from games.agents
    try:
        module = importlib.import_module("games.agents")
        return getattr(module, agent_class_path)
    except AttributeError as e:
        raise ImportError(
            f"Agent class '{agent_class_path}' not found. "
            f"Please provide a full module path like 'games.agents.thinking_react_agent.ThinkingReActAgent' "
            f"or a class name exported from games.agents."
        ) from e


# Import factory functions for backward compatibility and convenience
from games.agent_factory import (
    create_agent_from_config,
    create_model_from_config,
    create_memory_from_config,
    create_formatter_from_config,
    create_toolkit_from_config,
)

__all__ = [
    "load_config",
    "cleanup_agent_llm_clients",
    "load_agent_class",
    "deep_merge",
    "create_agent_from_config",
    "create_model_from_config",
    "create_memory_from_config",
    "create_formatter_from_config",
    "create_toolkit_from_config",
]

