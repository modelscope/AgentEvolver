# -*- coding: utf-8 -*-
# registry.py
"""
This module contains the registry for training boxes.
"""

from typing import Dict, Type
from .base import BaseEnv


class Registry:
    """
    Registry for registering and managing environment classes.
    Stores all BaseEnv subclasses with their names.
    """

    _envs: Dict[str, Type[BaseEnv]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Registry for registering
        """

        def _register(env_class):
            if not hasattr(env_class, "__bases__") or not any(
                "BaseEnv" in str(base) for base in env_class.__bases__
            ):
                raise TypeError(
                    f"{env_class.__name__} must inherit from BaseEnv",
                )
            cls._envs[name] = env_class
            return env_class

        return _register

    @classmethod
    def get(cls, name: str) -> Type[BaseEnv]:
        """
        Get the Environment subclass by name.
        """
        if name not in cls._envs:
            raise KeyError(
                f"Environment '{name}' not found. "
                f"Available: {list(cls._envs.keys())}",
            )
        return cls._envs[name]

    @classmethod
    def list(cls) -> list:
        """
        Get the environment classes registered.
        """
        return list(cls._envs.keys())
