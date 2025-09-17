from typing import Any, Callable

from omegaconf import DictConfig

from beyondagent.client.env_client import EnvClient
from beyondagent.schema.trajectory import Trajectory


class BaseAgentFlow(object):

    def __init__(self,
                 llm_chat_fn: Callable,
                 tokenizer: Any,
                 config: DictConfig = None,
                 **kwargs):
        """
        Initializes the BaseAgentFlow with the necessary components.

        Args:
            llm_chat_fn (Callable): A callable function for LLM chat.
            tokenizer (Any): The tokenizer used for tokenizing text.
            config (DictConfig, optional): Configuration settings. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # super.__init__(**kwargs)
        self.llm_chat_fn: Callable = llm_chat_fn  # ⭐ Store the LLM chat function
        self.tokenizer = tokenizer  # ⭐ Store the tokenizer
        self.config: DictConfig = config  # ⭐ Store the configuration
        self.max_steps: int = self.config.actor_rollout_ref.rollout.multi_turn.max_steps  # ⭐ Set the maximum steps
        self.max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len  # ⭐ Set the maximum model length
        self.max_env_len: int = self.config.actor_rollout_ref.rollout.max_env_len  # ⭐ Set the maximum environment length

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        """
        Abstract method to execute a trajectory in the given environment.

        Args:
            trajectory (Trajectory): The trajectory to be executed.
            env (EnvClient): The environment client.
            instance_id (str): The ID of the instance.
            **kwargs: Additional keyword arguments.

        Returns:
            Trajectory: The updated trajectory after execution.
        """
        raise NotImplementedError
