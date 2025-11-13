from typing import cast
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager

@grader_manager.reg("env")
class EnvGrader(RewardCalculator):
    def __init__(self, task:Task):
        super().__init__(task)
        pass
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        """
        Evaluates the provided trajectory in the specified environment and calculates a reward.

        Args:
            trajectory (Trajectory): The trajectory to be evaluated.
            env (EnvClient): The environment client used for evaluation.
            instance_id (str): The ID of the instance being evaluated.

        Returns:
            GraderResult: A dictionary containing the score and the reason for the result.
        """
        score = env.evaluate(instance_id, params={"sparse": True})  # ⭐ Evaluate the trajectory and get the score
        return {
            "score": score,
            "reason": "Env grader gives no reason."
        }