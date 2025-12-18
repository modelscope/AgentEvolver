# -*- coding: utf-8 -*-
"""Common utility functions shared by all games."""
import asyncio
from pathlib import Path
from typing import Any, Dict, List

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


__all__ = ["load_config", "cleanup_agent_llm_clients"]

