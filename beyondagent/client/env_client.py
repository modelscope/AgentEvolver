# env_client.py
from typing import Dict, List, Any

import requests
from loguru import logger


class EnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 300.0

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str = None,
        instance_id: str = None,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ) -> Dict:
        """
        Handles making a POST request to the specified API endpoint.

        Args:
            endpoint (str): The API endpoint to send the request to.
            env_type (str, optional): The type of environment. Defaults to "default".
            task_id (str, optional): The task ID. Defaults to None.
            instance_id (str, optional): The instance ID. Defaults to None.
            messages (Dict[str, Any], optional): Messages to be sent. Defaults to None.
            params (Dict[str, Any], optional): Additional parameters. Defaults to None.

        Returns:
            Dict: The JSON response from the API.
        """
        url = f"{self.base_url}/{endpoint}"  # ⭐ Constructs the full URL for the request
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)  # ⭐ Sends the POST request
            response.raise_for_status()
            return response.json()  # ⭐ Parses and returns the JSON response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}, data: {data}")
            raise

    def get_env_profile(
        self, env_type: str, split: str = "train", params: dict | None = None
    ) -> List[str]:
        """
        Retrieves a list of task IDs based on the specified environment type, split, and optional parameters.

        Args:
            env_type (str): The type of the environment.
            split (str, optional): The data split to use. Defaults to "train".
            params (dict | None, optional): Additional parameters for the request. Defaults to None.

        Returns:
            List[str]: A list of task IDs.
        """
        payload: dict = {"env_type": env_type}  # ⭐ Initialize the payload with the environment type
        if params:
            payload["params"] = params  # ⭐ Add additional parameters to the payload if provided
        response = self._make_request(
            endpoint="/get_env_profile", env_type=env_type, params={"split": split}
        )  # ⭐ Make the request to the API endpoint
        return response["data"]  # ⭐ Return the list of task IDs from the response

    def get_tools_info(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """
        Retrieves information about the tools in a specific environment instance.

        Args:
            instance_id (str): The ID of the environment instance.
            messages (Dict, optional): Additional messages to be sent with the request. Defaults to {}.
            params (Dict, optional): Additional parameters to be sent with the request. Defaults to {}.

        Returns:
            float: The data from the API response.
        """
        response = self._make_request(
            endpoint="get_info",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )  # ⭐ Make the API request to get tools information
        return response["data"]

    def create_instance(
        self, env_type: str, task_id: str, instance_id: str = None, params: Dict = None
    ) -> dict:
        """
        Creates an environment instance by sending a request to the API.

        Args:
            env_type (str): The type of the environment to be created.
            task_id (str): The unique identifier for the task.
            instance_id (str, optional): The unique identifier for the instance. Defaults to None.
            params (Dict, optional): Additional parameters for the environment creation. Defaults to None.

        Returns:
            dict: The data part of the API response containing information about the created instance.
        """
        response = self._make_request(  # ⭐ Sends the request to the API to create the environment instance
            endpoint="create",
            env_type=env_type,
            task_id=task_id,
            instance_id=instance_id,
            params=params,
        )
        return response["data"]

    def step(self, instance_id: str, action: Dict = {}, params: Dict = {}) -> dict:
        """
        Sends a request to the environment API to execute a step in the specified instance.

        Args:
            instance_id (str): The ID of the environment instance.
            action (Dict, optional): The action to be performed. Defaults to {}.
            params (Dict, optional): Additional parameters for the action. Defaults to {}.

        Returns:
            dict: The data returned from the environment API after executing the step.
        """
        response = self._make_request(
            endpoint="step", instance_id=instance_id, messages=action, params=params
        )  # ⭐ Sends the request to the environment API
        return response["data"]

    def evaluate(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """
        Sends a request to evaluate the specified environment instance and returns the evaluation result.

        Args:
            instance_id (str): The ID of the environment instance to be evaluated.
            messages (Dict, optional): A dictionary containing messages for the evaluation. Defaults to {}.
            params (Dict, optional): A dictionary containing additional parameters for the evaluation. Defaults to {}.

        Returns:
            float: The evaluation result.
        """
        response = self._make_request(  # ⭐ Sends the evaluation request to the API
            endpoint="evaluate",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def release_instance(self, instance_id: str) -> bool:
        """
        Sends a request to release the specified environment instance.

        Args:
            instance_id (str): The ID of the environment instance to be released.

        Returns:
            bool: True if the release operation was successful, False otherwise.
        """
        response = self._make_request(endpoint="release", instance_id=instance_id)  # ⭐ Send the release request
        return response["success"]
def main():
    """
    Demonstrates the use of EnvClient by performing a sequence of operations:
    - Fetching available tasks for a given environment type
    - Creating an instance based on one of the fetched tasks
    - Stepping through the created instance with a specified action
    - Evaluating the instance
    - Releasing the instance

    This function is intended to be run as a standalone script to test the functionality of the EnvClient.
    """
    client = EnvClient()

    env_type = "appworld"
    # 获取任务列表
    task_ids = client.get_env_profile(env_type)  # ⭐ Retrieve the list of available tasks for the specified environment type
    print(f"Available tasks: {task_ids}")

    # 创建实例
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)  # ⭐ Create an instance using the first available task
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # 执行动作
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)  # ⭐ Execute an action within the created instance
    print(f"Step result: {result}")

    # 评估
    score = client.evaluate(instance_id)  # ⭐ Evaluate the current state of the instance
    print(f"Evaluation score: {score}")

    # 释放实例
    success = client.release_instance(instance_id)  # ⭐ Release the instance, freeing up resources
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()
