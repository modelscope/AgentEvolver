# -*- coding: utf-8 -*-
"""Module for the BaseEnv class."""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEnv(ABC):
    """
    Environment Base class.

    This is an abstract base class that defines the interface for
    creating and interacting with environments in the training sandbox.
    """

    @abstractmethod
    def __init__(self, task_id: str = None, instance_id: str = None):
        """
        Initialize the BaseEnv.

        Args:
            task_id (str): The ID of the task associated with this environment.
            instance_id (str): The ID of the instance of the environment.
        """

    @abstractmethod
    def get_init_state(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the initial state of the environment.

        Args:
            params (Dict[str, Any]):
                    Additional parameters for initializing the environment.

        Returns:
            Dict[str, Any]: The initial state of the environment.
        """

    @abstractmethod
    def step(
        self,
        action: Dict[str, Any],
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a step in the environment.

        Args:
            action (Dict[str, Any]): The action
                        to be executed in the environment.
            params (Dict[str, Any]): Additional parameters
                        for the step.

        Returns:
            Dict[str, Any]: The result of the step, including next state,
                        reward, and information.
        """

    @abstractmethod
    def evaluate(
        self,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ) -> float:
        """
        Evaluate the performance of the environment.

        Args:
            messages (Dict[str, Any]): Additional messages
                        for the evaluation.
            params (Dict[str, Any]): Additional parameters
                        for the evaluation.

        Returns:
            float: The evaluation score.
        """

    @abstractmethod
    def close(self):
        """
        Close the environment and release any resources.
        """

    @abstractmethod
    def get_info(
        self,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ):
        """
        Get information about the environment.

        Args:
            messages (Dict[str, Any]): Additional messages
                        for the information request.
            params (Dict[str, Any]): Additional parameters
                        for the information request.

        Returns:
            Dict[str, Any]: The requested information
                        about the environment.
        """

    @staticmethod
    @abstractmethod
    def get_query_list(
        split: str = "train",
        params: Dict[str, Any] = None,
    ):
        """
        Get a list of queries for the specified split.

        Args:
            split (str): The split to get
                        the queries for (e.g., "train", "val", "test").
            params (Dict[str, Any]): Additional
                        parameters for the query list.

        Returns:
            List: A list of queries.
        """
