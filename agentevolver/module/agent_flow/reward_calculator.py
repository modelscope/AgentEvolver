from abc import ABC, abstractmethod
from typing import TypedDict

from agentevolver.client.env_client import EnvClient
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

class GraderResult(TypedDict):
    score: float
    reason: str | None

class RewardCalculator(ABC):
    def __init__(self,task: Task):
        self._task=task
    
    @property
    def task(self):
        return self._task
        
    @abstractmethod
    def calculate_reward(self, trajectory: Trajectory, env:EnvClient, instance_id:str) -> GraderResult:
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        pass