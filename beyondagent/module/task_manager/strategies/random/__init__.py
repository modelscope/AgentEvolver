import copy
import os
import time
from typing import Callable, NotRequired, Optional, Sequence, TypedDict, Unpack
import uuid

from loguru import logger

from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager.agent_flow import ModifiedAgentFlow
from beyondagent.module.task_manager.base import LlmClient
from beyondagent.module.env_manager.env_worker import EnvWorker
from beyondagent.module.task_manager.strategies.common.prompts.prompt_explore import get_agent_interaction_system_prompt
from beyondagent.module.task_manager.strategies.common.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from beyondagent.module.task_manager.strategies import TaskExploreStrategy
from beyondagent.module.task_manager.prelude_profiles import bfcl, appworld
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory



class LlmRandomSamplingExploreStrategyProps(TypedDict):
    max_explore_step: int
    max_llm_retries: int
    env_url: str
    exploration_llm_temperature: NotRequired[float]
    exploration_llm_top_p: NotRequired[float]
    exploration_llm_top_k: NotRequired[int]
    
    

class LlmRandomSamplingExploreStrategy(TaskExploreStrategy):
    def __init__(self, * , tokenizer, config,**kwargs: Unpack[LlmRandomSamplingExploreStrategyProps]):
        self._tokenizer = tokenizer
        self._config = config # 这东西仅仅用于给其他需要它的 class 使用
        
        self._max_llm_retries = kwargs.get("max_llm_retries", 3)
        self._max_explore_step = kwargs.get("max_explore_step", 10)
        self._env_service_url = kwargs.get("env_url")
        
        self._exploration_llm_temperature=kwargs.get("exploration_llm_temperature", 1.0)
        self._exploration_llm_top_p=kwargs.get("exploration_llm_top_p", 1.0)
        self._exploration_llm_top_k=kwargs.get("exploration_llm_top_k", 1)
        
    
    def explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        task.query="Now do your exploration!"
        env_worker = EnvWorker(
            task=task,
            config=self._config, # FIXME 不是，你既然一定需要这3个东西就不要设置成 = None 啊 
            thread_index=0,
            tokenizer=self._tokenizer
        )
        llm_chat_fn = self._get_llm_chat_fn(self.llm_client_explore,
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )
        agent_flow: BaseAgentFlow = ModifiedAgentFlow(
            enable_context_generator=False,
            llm_chat_fn=llm_chat_fn,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        agent_flow.max_steps = self._max_explore_step  # this is ugly
        agent_flow.max_model_len=102400 # TODO max len

        traj = env_worker.execute(
            data_id=data_id,
            rollout_id=rollout_id,
            add_exp=False,
            task_train_exp_mode='woexp',
            agent_flow=agent_flow,
            tmux={
                'step':[0],
                'token':[0],
            },
            stop=[False], # 这俩玩意有没有什么办法能封装一下
            system_prompt=get_agent_interaction_system_prompt(appworld.user_profile), # FIXME debug profile
        )

        return [traj]
    
    def summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        llm_fn = self._get_llm_chat_fn(self.llm_client_summarize)
        old_objectives = self._old_retrival.retrieve_objectives(task)
        system_prompt, user_prompt = get_task_summarize_prompt(
            [trajectory], old_objectives, appworld.user_profile # FIXME debug profile
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
    
    def _get_llm_chat_fn(self, llm_client:LlmClient, sampling_params: Optional[dict] = None) -> Callable:
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
                    res = llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    if res is not None and res!="":
                        break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(2**i)

            assert res is not None and res!="", f"LLM client failed to chat"
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat