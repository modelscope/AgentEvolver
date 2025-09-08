import time
from typing import List

from loguru import logger
from pydantic import Field

from beyondagent.schema.trajectory import Trajectory, Reward
from beyondagent.utils.http_client import HttpClient


class EMClient(HttpClient):
    base_url: str = Field(default="http://localhost:8001")
    timeout: int = Field(default=1200 , description="request timeout, second")


    def call_context_generator(self, trajectory: Trajectory, retrieve_top_k: int = 1, workspace_id: str = "default",
                               **kwargs) -> str:
        start_time = time.time()
        self.url = self.base_url + "/retriever"
        json_data = {
            "query": trajectory.query,
            "retrieve_top_k": retrieve_top_k,
            "workspace_id": workspace_id,
            "metadata": kwargs
        }
        response = self.request(json_data=json_data, headers={"Content-Type": "application/json"})
        if response is None:
            logger.warning("error call_context_generator")
            return ""

        # TODO return raw experience instead of context @jinli
        trajectory.metadata["context_time_cost"] = time.time() - start_time
        return response["experience_merged"]

    def call_summarizer(self, trajectories: List[Trajectory], workspace_id: str = "default", **kwargs):
        start_time = time.time()

        self.url = self.base_url + "/summarizer"
        json_data = {
            "traj_list": [{"messages": x.steps, "score": x.reward.outcome} for x in trajectories],
            "workspace_id": workspace_id,
            "metadata": kwargs
        }
        response = self.request(json_data=json_data, headers={"Content-Type": "application/json"})
        if response is None:
            logger.warning("error call_context_generator")
            return "", time.time() - start_time

        return response["experience_list"], time.time() - start_time


def main():
    client = EMClient()
    traj = Trajectory(
        steps=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            },
            {
                "role": "assistant",
                "content": "Paris"
            }
        ],
        query="What is the capital of France?",
        reward=Reward(outcome=1.0)
    )
    workspace_id = "w_agent_enhanced2"

    print(client.call_summarizer(trajectories=[traj], workspace_id=workspace_id))
    print(client.call_context_generator(traj, retrieve_top_k=3, workspace_id=workspace_id))

if __name__ == "__main__":
    main()
