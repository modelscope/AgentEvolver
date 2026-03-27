# env_client.py
from typing import Dict, List, Any

import requests


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
        """Unified HTTP POST helper for env endpoints."""
        url = f"{self.base_url}/{endpoint}"
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}, data: {data}")

    def get_task_ids(
        self, env_type: str, split: str = "train", params: dict | None = None
    ) -> List[str]:
        """Return task id list for the given env type and split."""
        payload: dict = {"env_type": env_type}
        if params:
            payload["params"] = params
        response = self._make_request(
            endpoint="get_task_ids", env_type=env_type, params={"split": split}
        )
        return response["data"]

    def get_tools_info(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """Return tool / observation metadata for the instance."""
        response = self._make_request(
            endpoint="get_info",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def create_instance(
        self, env_type: str, task_id: str, instance_id: str = None, params: Dict = None
    ) -> dict:
        """Create a remote environment instance."""
        response = self._make_request(
            endpoint="create",
            env_type=env_type,
            task_id=task_id,
            instance_id=instance_id,
            params=params,
        )
        return response["data"]

    def step(self, instance_id: str, action: Dict = {}, params: Dict = {}) -> dict:
        """Apply one action (step) on the instance."""
        response = self._make_request(
            endpoint="step", instance_id=instance_id, messages=action, params=params
        )
        return response["data"]

    def evaluate(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """Run final evaluation and return a scalar score."""
        response = self._make_request(
            endpoint="evaluate",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def release_instance(self, instance_id: str) -> bool:
        """Release server-side resources for the instance."""
        response = self._make_request(endpoint="release", instance_id=instance_id)
        return response["success"]


# Example usage
def main():
    client = EnvClient()

    env_type = "appworld"
    # fetch task ids
    task_ids = client.get_task_ids(env_type)
    print(f"Available tasks: {task_ids}")

    # create instance
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # step the environment
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    # evaluate
    score = client.evaluate(instance_id)
    print(f"Evaluation score: {score}")

    # release instance
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()
