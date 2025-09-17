import time
from typing import List

from loguru import logger
from pydantic import Field

from beyondagent.schema.trajectory import Trajectory, Reward
from beyondagent.utils.http_client import HttpClient


class EMClient(HttpClient):
    base_url: str = Field(default="http://localhost:8001")
    timeout: int = Field(default=1200, description="request timeout, second")

    def call_context_generator(self, trajectory: Trajectory, retrieve_top_k: int = 1, workspace_id: str = "default",
                               **kwargs) -> str:
        """
        Sends a request to the server to generate context based on the provided trajectory.

        Args:
            trajectory (Trajectory): The trajectory object containing the query and other metadata.
            retrieve_top_k (int, optional): The number of top results to retrieve. Defaults to 1.
            workspace_id (str, optional): The ID of the workspace. Defaults to "default".
            **kwargs: Additional metadata to be included in the request.

        Returns:
            str: The merged experience string from the server's response.
        """
        start_time = time.time()
        self.url = self.base_url + "/retriever"  # ⭐ Set the URL for the request
        json_data = {
            "query": trajectory.query,
            "retrieve_top_k": retrieve_top_k,
            "workspace_id": workspace_id,
            "metadata": kwargs
        }
        response = self.request(json_data=json_data, headers={"Content-Type": "application/json"})  # ⭐ Send the request to the server
        if response is None:
            logger.warning("error call_context_generator")
            return ""

        # TODO return raw experience instead of context @jinli
        trajectory.metadata["context_time_cost"] = time.time() - start_time  # ⭐ Log the time taken for the operation
        return response["experience_merged"]  # ⭐ Return the merged experience from the response

    def call_summarizer(self, trajectories: List[Trajectory], workspace_id: str = "default", **kwargs):
        """
        Sends a request to the summarizer endpoint with a list of trajectories and additional metadata.

        Args:
            trajectories (List[Trajectory]): A list of trajectory objects to be summarized.
            workspace_id (str, optional): The ID of the workspace. Defaults to "default".
            **kwargs: Additional metadata to be included in the request.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the list of summarized experiences and the time taken for the operation.
        """
        start_time = time.time()

        self.url = self.base_url + "/summarizer"  # ⭐ Set the URL for the summarizer endpoint
        json_data = {
            "traj_list": [{"messages": x.steps, "score": x.reward.outcome} for x in trajectories],
            "workspace_id": workspace_id,
            "metadata": kwargs
        }
        response = self.request(json_data=json_data, headers={"Content-Type": "application/json"})  # ⭐ Send the request to the server
        if response is None:
            logger.warning("error call_context_generator")
            return "", time.time() - start_time

        return response["experience_list"], time.time() - start_time


def main():
    """
    Demonstrates the use of EMClient to call summarizer and context generator with a predefined trajectory.

    The function creates an instance of EMClient, defines a trajectory, and uses it to call the summarizer and context generator methods of the client. It prints the results of these calls.

    Args:
        None

    Returns:
        None
    """
    client = EMClient()  # ⭐ Initialize the EMClient
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
    )  # ⭐ Define a sample trajectory
    workspace_id = "w_agent_enhanced2"

    print(client.call_summarizer(trajectories=[traj], workspace_id=workspace_id))  # ⭐ Call the summarizer with the defined trajectory
    print(client.call_context_generator(traj, retrieve_top_k=3, workspace_id=workspace_id))  # ⭐ Call the context generator with the defined trajectory

if __name__ == "__main__":
    main()
