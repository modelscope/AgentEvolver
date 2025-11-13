import re
import threading
from typing import Any, Optional, Type, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.module.task_manager.rewards.binary_judge_gt import LlmAsJudgeBinaryRewardCalculatorWithGT
from agentevolver.module.task_manager.rewards.reward import LlmAsJudgeRewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from . import grader_manager


class AvgJudge(RewardCalculator):
    def __init__(self, task: Task):
        super().__init__(task)
        self._judges: list[RewardCalculator] = []

    def add_judge(self, x: RewardCalculator):
        """
        Adds a new judge to the list of judges.

        Args:
            x (RewardCalculator): The judge to be added.
        """
        self._judges.append(x)

    def calculate_reward(
        self, trajectory: Trajectory, env: EnvClient, instance_id: str, max_workers: int = 4
    ) -> GraderResult:
        """
        Calculates the average reward from all added judges by running them in parallel and averaging their scores.

        Args:
            trajectory (Trajectory): The trajectory for which the reward is calculated.
            env (EnvClient): The environment client.
            instance_id (str): The instance ID.
            max_workers (int, optional): The maximum number of worker threads. Defaults to 4.

        Returns:
            GraderResult: The average score and reason.
        """
        rewards: list[float] = []

        def worker(judge: RewardCalculator):
            try:
                result = judge.calculate_reward(trajectory, env, instance_id)  # ⭐ Calculate the reward for the given trajectory
                return result["score"]
            except Exception as e:
                logger.error(f"Judge failed: {e}")
                return 0.0

        # run judges in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, j) for j in self._judges]
            for f in as_completed(futures):
                rewards.append(f.result())

        if not rewards:
            return {"score": 0.0, "reason": "No valid rewards"}

        return {
            "score": sum(rewards) / len(rewards),  # ⭐ Calculate the average score from all judges
            "reason": "AvgJudge (threaded)"
        }


@grader_manager.reg("avg-llm-binary-gt")
class AvgBinaryGTJudge(AvgJudge):
    def __init__(self, task: Task, n: int = 3):
        """
        Initializes the judge with a given task and a specified number of judges.

        Args:
            task (Task): The task for which the judges will calculate rewards.
            n (int, optional): The number of judges to add. Defaults to 3.
        """
        super().__init__(task)
        for i in range(n):
            self.add_judge(
                LlmAsJudgeBinaryRewardCalculatorWithGT(
                    task, model_name="qwq-plus", use_mean_constraint=True
                )
            )  # ⭐ Adds a judge with a binary reward calculator and a mean constraint


@grader_manager.reg("avg-llm")
class AvgLlmJudge(AvgJudge):
    def __init__(self, task: Task, n: int = 3):
        """
        Initializes the judge with a given task and a specified number of judges.

        Args:
            task (Task): The task for which the judges will calculate rewards.
            n (int, optional): The number of judges to add. Defaults to 3.
        """
        super().__init__(task)
        for i in range(n):
            self.add_judge(
                LlmAsJudgeRewardCalculator(
                    task, model_name="qwq-plus"
                )
            )  # ⭐ Adds a judge with a standard reward calculator