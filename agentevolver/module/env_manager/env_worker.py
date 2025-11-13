from typing import Optional
import uuid

from omegaconf import DictConfig
from loguru import logger
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from agentevolver.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
from agentevolver.module.context_manager.cmt_linear_think import LinearThinkCMT
from agentevolver.module.context_manager.cmt_context_clip import SelfContextClipCMT
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from typing import List, Dict, Any, Optional


class EnvWorker(object):

    def __init__(self, task: Task, instance_id: str = None, thread_index: int = None, tokenizer=None,
                 config: DictConfig = None):
        """
        Initializes the EnvWorker with the provided task, configuration, and other optional parameters.

        Args:
            task (Task): The task object that contains the details of the task to be executed.
            instance_id (str, optional): A unique identifier for the instance. If not provided, a new UUID will be generated.
            thread_index (int, optional): The index of the thread if this worker is part of a multi-threaded setup.
            tokenizer (optional): The tokenizer to be used for processing text.
            config (DictConfig, optional): The configuration settings for the environment and other components.
        """
        self.config = config  # Store the provided configuration
        self.env = EnvClient(base_url=config.env_service.env_url)  # Initialize the environment client
        self.is_open_query = task.open_query # open query has no clear stop conditions, so we allow agent to decide when to stop.
        self.task = task  # Store the task object
        self.env_type: str = task.env_type  # Set the environment type based on the task
        self.task_id: str = task.task_id  # Set the task ID
        self.instance_id: str = instance_id if instance_id is not None else uuid.uuid4().hex  # Set or generate the instance ID
        self.thread_index: int = thread_index  # Set the thread index
        self.tokenizer = tokenizer  # Store the tokenizer

    def execute(self, data_id: str, rollout_id: str, traj_exp_config: TrajExpConfig, agent_flow: BaseAgentFlow, tmux:dict,stop:list[bool], system_prompt: Optional[str] = None, **kwargs) -> Trajectory:
        """
        Executes the task in the environment, generates a trajectory, and returns it.

        Args:
            data_id (str): The unique identifier for the data.
            rollout_id (str): The unique identifier for the rollout.
            traj_exp_config (TrajExpConfig): Experience Configuration for the trajectory.
            agent_flow (BaseAgentFlow): The agent flow to execute the task.
            tmux (dict): TMUX configuration.
            stop (list[bool]): List of flags to indicate stopping conditions.
            system_prompt (Optional[str]): Custom system prompt to be inserted.
            **kwargs: Additional keyword arguments.

        Returns:
            Trajectory: The generated trajectory from the task execution.
        """

        try:
            init_response = self.env.create_instance(env_type=self.env_type,
                                                    task_id=self.task_id,
                                                    instance_id=self.instance_id,
                                                    params={'is_open_query': self.is_open_query})

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
                # FIXME quick fix for test
                assert self.task.query is not None
                system_prompt=system_prompt.replace('[USER_QUESTION]',self.task.query)
                init_messages.insert(1, {"role": "user", "content": system_prompt})
                init_messages.pop() # remove the last original query

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
            # traj_cmt.task_train_exp_mode = self.task.metadata.get("task_train_exp_mode")
            # traj_cmt.metadata["task_train_exp_mode"] = task_train_exp_mode
            assert self.task.query is not None
            traj_cmt.query = self.task.query

            # traj_exp_config.query=self.task.query
            # init_messages, traj_exp_config = self.exp_worker.manage_rollout_context(
            #     init_messages=init_messages,
            #     traj_exp_config=traj_exp_config
            # )
            # traj_cmt.metadata["task_train_exp_mode"] = traj_exp_config.train_mode
            # traj_cmt.metadata["add_exp"] = traj_exp_config.add_exp
            # traj_cmt.metadata["experience_list"] = traj_exp_config.experience_list

            traj_cmt: Trajectory = agent_flow.execute(
                context_manager=traj_cmt,
                init_messages=init_messages,
                env=self.env,
                instance_id=self.instance_id,
                tmux=tmux,
                stop=stop,
                thread_index=self.thread_index,
                task_id=self.task_id,
                traj_exp_config=traj_exp_config,
                data_id=data_id,
                rollout_id=rollout_id,
                query=self.task.query,
                **kwargs
            )  # ⭐ Execute the task and generate the trajectory
            self.env.release_instance(self.instance_id)

        except Exception as e:
            self.env.release_instance(self.instance_id)
            raise RuntimeError(f"env.create_instance failed! error={e.args}") from e

        return traj_cmt
