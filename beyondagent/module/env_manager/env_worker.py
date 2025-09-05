from typing import Optional
import uuid

from omegaconf import DictConfig
from loguru import logger
from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory
from beyondagent.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
# from beyondagent.module.context_manager.cmt_memory import MemoryCMT, GroupedSteps
from beyondagent.module.context_manager.cmt_linear_think import LinearThinkCMT
from beyondagent.module.context_manager.cmt_context_clip import SelfContextClipCMT
from typing import List, Dict, Any, Optional


class EnvWorker(object):

    def __init__(self, task: Task, instance_id: str = None, thread_index: int = None, tokenizer=None,
                 config: DictConfig = None):
        self.config = config
        self.env = EnvClient(base_url=config.env_service.env_url)
        self.task = task
        self.env_type: str = task.env_type
        self.task_id: str = task.task_id
        self.instance_id: str = instance_id if instance_id is not None else uuid.uuid4().hex
        self.thread_index: int = thread_index
        self.tokenizer = tokenizer

    def execute(self, data_id: str, rollout_id: str, add_exp: bool, task_train_exp_mode: str,agent_flow: BaseAgentFlow, tmux:dict,stop:list[bool], system_prompt: Optional[str] = None, **kwargs) -> Trajectory:    # add add_exp & task_train_exp_mode by ANNI
        trajectory: Trajectory = Trajectory(data_id=data_id, rollout_id=rollout_id, steps=[], query="")
        try:
            init_response = self.env.create_instance(env_type=self.env_type,
                                                    task_id=self.task_id,
                                                    instance_id=self.instance_id)

            init_messages: list[dict] = init_response["state"]
            assert isinstance(init_messages, list) and len(init_messages)==2, "init_messages must be list and its length must be 2"
            # replace query if new query is in task
            if self.task.query is not None:
                assert init_messages[-1]["role"] == "user", "the latest message from environment must be user query"
                init_messages[-1]["content"] = self.task.query
            else:
                self.task.query = init_messages[-1]["content"]
            
            # insert custom system prompt
            if system_prompt is not None:
                init_messages.insert(1, {"role": "user", "content": system_prompt})

            if self.config.actor_rollout_ref.rollout.context_template == "linear":
                traj_cmt: Linear_CMT = Linear_CMT(self.config, self.tokenizer)
            elif self.config.actor_rollout_ref.rollout.context_template == "linear_think":
                traj_cmt: LinearThinkCMT = LinearThinkCMT(self.config, self.tokenizer)
            elif self.config.actor_rollout_ref.rollout.context_template == "context_selfclip":
                traj_cmt: SelfContextClipCMT = SelfContextClipCMT(self.config, self.tokenizer, self.llm_chat_fn)
            else:
                raise ValueError(f"Unsupported context template: {self.config.actor_rollout_ref.rollout.context_template}")

            traj_cmt.data_id = data_id
            traj_cmt.rollout_id = rollout_id
            traj_cmt.task_id = self.task_id
            traj_cmt.instance_id = self.instance_id
            traj_cmt.task_train_exp_mode = self.task.metadata.get("task_train_exp_mode")
            traj_cmt.metadata["task_train_exp_mode"] = task_train_exp_mode
            traj_cmt.query = init_messages[-1]["content"]

            traj_cmt: Trajectory = agent_flow.execute(
                context_manager=traj_cmt,
                init_messages=init_messages,
                env=self.env,
                instance_id=self.instance_id,
                tmux=tmux,
                stop=stop,
                thread_index=self.thread_index,
                task_id=self.task_id,
                data_id=data_id,
                rollout_id=rollout_id,
                query=self.task.query,
                add_exp=add_exp,
                **kwargs
            )
            self.env.release_instance(self.instance_id)

        except Exception as e:
            self.env.release_instance(self.instance_id)
            raise RuntimeError(f"env.create_instance failed! error={e.args}") from e

        return traj_cmt
