import asyncio
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
        self.url = self.base_url + "/context_generator"
        json_data = {
            "trajectory": trajectory.model_dump(),
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
        return response["context_msg"]["content"]

    async def async_call_context_generator(self, executor=None, **kwargs):
        loop = asyncio.get_event_loop()

        def func():
            return self.call_context_generator(**kwargs)

        return await loop.run_in_executor(executor=executor, func=func)

    def call_summarizer(self, trajectories: List[Trajectory], workspace_id: str = "default", **kwargs):
        start_time = time.time()

        self.url = self.base_url + "/summarizer"
        json_data = {
            "trajectories": [x.model_dump() for x in trajectories],
            "workspace_id": workspace_id,
            "metadata": kwargs
        }
        response = self.request(json_data=json_data, headers={"Content-Type": "application/json"})
        if response is None:
            logger.warning("error call_context_generator")
            return "", time.time() - start_time

        return response["experiences"], time.time() - start_time

    async def async_call_summarizer(self, executor=None, **kwargs):
        loop = asyncio.get_event_loop()

        def func():
            return self.call_summarizer(**kwargs)

        return await loop.run_in_executor(executor=executor, func=func)


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
    workspace_id = "w_agent_enhanced"

    print(client.call_summarizer(trajectories=[traj], workspace_id=workspace_id))
    print(client.call_context_generator(traj, retrieve_top_k=3, workspace_id=workspace_id))

if __name__ == "__main__":
    main()
