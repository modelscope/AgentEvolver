import copy
import time
from typing import Callable, NotRequired, Optional, Sequence, TypedDict, Unpack

from loguru import logger

from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.task_manager.strategies.common.prompts.prompt_explore import get_agent_interaction_system_prompt
from agentevolver.module.task_manager.strategies.common.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory

from .embedding import StateRecorder
from .controlled_agent_flow import ControlledAgentFlow


class LlmDedupExploreStrategyProps(TypedDict):
    max_explore_step: int
    max_llm_retries: int
    env_url: str
    exploration_llm_temperature: NotRequired[float]
    exploration_llm_top_p: NotRequired[float]
    exploration_llm_top_k: NotRequired[int]
    task_summary_history_length: NotRequired[int]
    
    temp_db_path: NotRequired[str]
    state_similarity_threshold: float
    
    

class LlmDedupSamplingExploreStrategy(TaskExploreStrategy):
    def __init__(self, * , tokenizer, config,**kwargs: Unpack[LlmDedupExploreStrategyProps]):
        self._tokenizer = tokenizer
        self._config = config
        
        self._max_llm_retries = kwargs.get("max_llm_retries", 3)
        self._max_explore_step = kwargs.get("max_explore_step", 10)
        self._env_service_url = kwargs.get("env_url")
        
        self._exploration_llm_temperature=kwargs.get("exploration_llm_temperature", 1.0)
        self._exploration_llm_top_p=kwargs.get("exploration_llm_top_p", 1.0)
        self._exploration_llm_top_k=kwargs.get("exploration_llm_top_k", 1)
        self._task_summary_history_length=kwargs.get("task_summary_history_length", self._max_explore_step)
        
        self._temp_db_path=kwargs.get("temp_db_path", "./.temp_vec_db")
        self._state_similarity_threshold=kwargs.get("state_similarity_threshold")
        
        self._state_recorder=StateRecorder(similarity_threshold=self._state_similarity_threshold,chroma_db_path=self._temp_db_path)
        
    
    def explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        env_worker = EnvWorkerWithPrompt(
            env_type=task.env_type,
            task_id=task.task_id,
            instance_id=None,
            env_service_url=self._env_service_url,
        )
        llm_chat_fn = self._get_llm_chat_fn(
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )
        agent_flow: BaseAgentFlow = ControlledAgentFlow(
            state_recorder=self._state_recorder,
            llm_chat_fn=llm_chat_fn,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        agent_flow.max_steps = self._max_explore_step  # this is ugly
        agent_flow.max_model_len=102400 # TODO max len

        traj = env_worker.execute(
            data_id=data_id,
            rollout_id=rollout_id,
            system_prompt=get_agent_interaction_system_prompt(task),
            agent_flow=agent_flow,
        )

        return [traj]
    
    def summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        llm_fn = self._get_llm_chat_fn()
        old_objectives = self._old_retrival.retrieve_objectives(task)
        system_prompt, user_prompt = get_task_summarize_prompt(
            [trajectory], old_objectives, len_history=self._task_summary_history_length
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm_output = llm_fn(messages=messages)["content"]
        task=task.copy()
        task.evaluator='synthetic'
        tasks = parse_tasks_from_response(task, llm_output)
        return tasks
    
    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            res = None
            for i in range(self._max_llm_retries):
                try:
                    res = self.llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)

            assert res is not None, f"LLM client failed to chat"
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat