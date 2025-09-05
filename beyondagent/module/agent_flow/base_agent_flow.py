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
        # super.__init__(**kwargs)
        self.llm_chat_fn: Callable = llm_chat_fn
        self.tokenizer = tokenizer
        self.config: DictConfig = config
        self.max_steps: int = self.config.actor_rollout_ref.rollout.multi_turn.max_steps
        self.max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len
        self.max_env_len: int = self.config.actor_rollout_ref.rollout.max_env_len

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        raise NotImplementedError
