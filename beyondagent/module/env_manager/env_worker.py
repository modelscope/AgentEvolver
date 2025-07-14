import uuid

from omegaconf import DictConfig

from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory
from loguru import logger

class EnvWorker(object):

    def __init__(self, task: Task, instance_id: str = None, thread_index: int = None,
                 config: DictConfig = None):
        self.env = EnvClient(base_url=config.env_service.env_url)
        self.task = task
        self.env_type: str = task.env_type
        self.task_id: str = task.task_id
        self.instance_id: str = instance_id if instance_id is not None else uuid.uuid4().hex
        self.thread_index: int = thread_index

    def execute(self, data_id: str, rollout_id: str, agent_flow: BaseAgentFlow, **kwargs) -> Trajectory:
        trajectory: Trajectory = Trajectory(data_id=data_id, rollout_id=rollout_id, steps=[], query="")

        try:
            # cc: this is just a temporary solution
            if not self.task.is_query_empty:
                params={"query":self.task.first_query}
            else:
                params=None
            init_response = self.env.create_instance(env_type=self.env_type,
                                                    task_id=self.task_id,
                                                    instance_id=self.instance_id,
                                                    params=params) # type: ignore
        except Exception as e:
            logger.exception(f"encounter exception in env_worker.create_instance~ error={e.args}")
            return trajectory

        try:
            state_message: dict = init_response["state"]
            trajectory: Trajectory = Trajectory(data_id=data_id,
                                                rollout_id=rollout_id,
                                                steps=[state_message],
                                                query=state_message["content"])
            trajectory: Trajectory = agent_flow.execute(trajectory=trajectory, env=self.env, instance_id=self.instance_id,
                                                        **kwargs)
        except Exception as e:
            logger.exception(f"encounter exception in env_worker.agent_flow~ error={e.args}")

        try:
            self.env.release_instance(self.instance_id)
        except Exception as e:
            logger.exception(f"encounter exception in env_worker.release_instance~ error={e.args}")

        return trajectory
