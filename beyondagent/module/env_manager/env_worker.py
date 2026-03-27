import uuid
import random

from omegaconf import DictConfig

from beyondagent.client.env_client import EnvClient
from beyondagent.client.env_client_ng import EnvClient as EnvClientNg
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from context_manager_templates.cmt_linear import CMTLinear
from loguru import logger
from typing import List

class EnvWorker(object):

    def __init__(self, env_type: str, task_id: str, instance_id: str = None, thread_index: int = None, config: DictConfig = None):
        if config.actor_rollout_ref.rollout.env_array:
            random_int = random.randint(8501, 8505)
            url = 'http://localhost:'+str(random_int)
        else:
            url = config.env_service.env_url

        if env_type == "appworld":
            self.env = EnvClient(base_url=url)
            self.env_params = None
        elif env_type == "webshop":
            self.env = EnvClientNg(base_url=url)
            self.env_params = {
                'base_url': 'http://127.0.0.1:1907',
                'human_goals': False,
            }
        elif env_type == "crafters":
            self.env = EnvClientNg(base_url=url)
            self.env_params = {}
        else:
            self.env = EnvClientNg(base_url=url)
            self.env_params = None

        self.env_type: str = env_type
        self.task_id: str = task_id
        self.instance_id: str = uuid.uuid4().hex
        self.thread_index: int = thread_index


    def execute(self, data_id: str, rollout_id: str, agent_flow: AgentFlow, tmux: List[int], stop: List[bool], **kwargs) -> CMTLinear:

        # >>>>>>>>>>>>>> create
        try:
            init_response = self.env.create_instance(env_type=self.env_type,
                                                    task_id=self.task_id,
                                                    instance_id=self.instance_id,
                                                    params=self.env_params)
        except Exception as e:
            logger.bind(exception=True).exception(f"encounter exception in env_worker.create_instance~ error={e.args}")
            self.env.release_instance(self.instance_id)
            raise e

        # =============== simulate
        try:
            state_message: dict = init_response["state"] if "state" in init_response else init_response['data']["state"]
            query, init_messages = self.get_init_messages(state_message)
            cmt = agent_flow.execute(init_messages=init_messages, env=self.env, instance_id=self.instance_id, tmux=tmux, stop=stop, thread_index=self.thread_index, task_id=self.task_id, data_id=data_id, rollout_id=rollout_id, query=query, **kwargs)
            cmt.data_id = data_id
            cmt.rollout_id = rollout_id
            cmt.task_id = self.task_id
        except Exception as e:
            logger.bind(exception=True).exception(f"encounter exception in env_worker.agent_flow~ error={e.args}")
            self.env.release_instance(self.instance_id)
            raise e

        # <<<<<<<<<<<<<< destory
        try:
            self.env.release_instance(self.instance_id)
        except Exception as e:
            logger.bind(exception=True).exception(f"encounter exception in env_worker.release_instance~ error={e.args}")
            raise e

        return cmt

    def get_init_messages(self, state_message) -> tuple:
        """
        Process state_message to extract query and init_messages.

        Args:
            state_message (Union[dict, list]): The state message to process

        Returns:
            tuple: (query, init_messages) where query is a string and init_messages is a list

        Raises:
            ValueError: If state_message is neither dict nor list
        """
        if isinstance(state_message, dict):
            query = state_message["content"]
            init_messages = [state_message]
        elif isinstance(state_message, list):
            assert isinstance(state_message[0], dict)
            query = state_message[-1]["content"]
            init_messages = state_message
        else:
            raise ValueError(f"state_message should be dict or list, but got {type(state_message)}")

        return query, init_messages
