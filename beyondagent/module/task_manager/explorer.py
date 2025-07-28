from typing import Optional
import uuid

from omegaconf import DictConfig

from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.schema.trajectory import Trajectory
from loguru import logger


# TODO replace this with new env worker
class EnvWorkerWithPrompt(object):

    def __init__(
        self,
        env_type: str,
        task_id: str,
        instance_id: Optional[str],
        env_service_url: str,
    ):
        """
        init a environment worker and interact with env service.

        Args:
            env_type: environment type
            task_id: task id
            instance_id: instance id
        """
        self.env = EnvClient(base_url=env_service_url)
        self.env_type: str = env_type
        self.task_id: str = task_id
        self.instance_id: str = (
            instance_id if instance_id is not None else uuid.uuid4().hex
        )

    def execute(
        self,
        data_id: str,
        rollout_id: str,
        agent_flow: BaseAgentFlow,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Trajectory:
        try:
            init_response = self.env.create_instance(
                env_type=self.env_type,
                task_id=self.task_id,
                instance_id=self.instance_id,
            )
        except Exception as e:
            logger.exception(
                f"encounter exception in env_worker.create_instance~ error={e.args}"
            )
            return Trajectory(
                data_id=data_id, rollout_id=rollout_id, steps=[], query="unknown"
            )

        try:
            state_message: list[dict] = init_response["state"]
            assert isinstance(state_message,list) and len(state_message)==2, "state_message must be list and its length must be 2"
            step:list[dict] = []
            step.extend(state_message)
            if system_prompt is not None:
                step.insert(1, {"role": "system", "content": system_prompt})
            # Example of state_message after rearrangement
            # [
            #     {
            #         "role": "system",
            #         "content": "<system prompt that describes the environments and others>"
            #     },
            #     {
            #         "role": "system",
            #         "content": "<system prompt inserted by task manager to control the exploration process>"
            #     },
            #     {
            #         "role": "user", 
            #         "content": "<the query>"
            #     }
            # ]
            step[-1]["content"] = f"Former query: {state_message[-1]['content']}. Ignore the query, now do your exploration!"
            
            trajectory: Trajectory = Trajectory(
                data_id=data_id, rollout_id=rollout_id, steps=step, query=state_message[-1]['content']
            )
            trajectory: Trajectory = agent_flow.execute(
                trajectory=trajectory,
                env=self.env,
                instance_id=self.instance_id,
                **kwargs,
            )

            self.env.release_instance(self.instance_id)
            return trajectory

        except Exception as e:
            logger.exception(
                f"encounter exception in env_worker.agent_flow~ error={e.args}"
            )
            return Trajectory(
                data_id=data_id, rollout_id=rollout_id, steps=[], query="unknown"
            )
