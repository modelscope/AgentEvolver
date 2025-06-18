import uuid

from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.schema.trajectory import Trajectory
from omegaconf import DictConfig

class EnvWorker(object):

    def __init__(self, env_type: str, task_id: str, instance_id: str = None, thread_index: int = None, config: DictConfig = None,
                 **kwargs):
        # super.__init__(**kwargs)
        self.env = EnvClient(base_url=config.beyond_agent.env_url)
        self.env_type: str = env_type
        self.task_id: str = task_id
        self.instance_id: str = instance_id if instance_id is not None else uuid.uuid4().hex
        self.thread_index: int = thread_index

    def execute(self, data_id: str, rollout_id: str, agent_flow: BaseAgentFlow, **kwargs) -> Trajectory:
        
        try:
            init_response = self.env.create_instance(env_type=self.env_type,
                                                    task_id=self.task_id,
                                                    instance_id=self.instance_id)
            state_message: dict = init_response["state"]
            trajectory: Trajectory = Trajectory(data_id=data_id,
                                                rollout_id=rollout_id,
                                                steps=[state_message],
                                                query=state_message["content"])
            trajectory: Trajectory = agent_flow.execute(trajectory=trajectory, env=self.env, instance_id=self.instance_id, **kwargs)

            self.env.release_instance(self.instance_id)
        
        except Exception as e:
            print("Error in EnvWorker: ", e)
            trajectory = Trajectory(data_id=data_id, rollout_id=rollout_id, steps=[], query="")
            try:
                self.env.release_instance(self.instance_id)
            except Exception as e:
                print(f"Env instance has been released: {self.instance_id}; Error: {e}")

        return trajectory
