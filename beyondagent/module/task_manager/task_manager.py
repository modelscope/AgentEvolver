from concurrent.futures import ThreadPoolExecutor
import copy
import functools
import json
import os
import random
import threading
import time
from typing import (
    Callable,
    Iterable,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
    Unpack,
)

import hydra
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import IterableDataset,Dataset
from tqdm import tqdm
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager import adapter
from beyondagent.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from beyondagent.module.task_manager.explorer import Explorer
from beyondagent.module.task_manager.filters import TaskPostFilter
from beyondagent.module.task_manager.prompts.prompt_explore import (
    get_agent_interaction_system_prompt,
)
from beyondagent.module.task_manager.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from beyondagent.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset

class TaskManagerProps(TypedDict):
    max_llm_retries: int
    max_explore_step: int
    num_explore_threads: int
    n: int
    
    exploration_llm_temperature: NotRequired[float]
    exploration_llm_top_p: NotRequired[float]
    exploration_llm_top_k: NotRequired[int]
    
    task_summary_history_length: NotRequired[int]
    
    use_original_tasks: NotRequired[bool]


# TODO: 针对不同环境的统一接口，message-in message-out？那可能不需要这个
# TODO: 能够替换的 exploration & extraction (summary) strategy


class TaskManager(object):

    def __init__(
        self,
        config: DictConfig,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        tokenizer,
        env_service_url: str,
        **kwargs: Unpack[TaskManagerProps],
    ):
        self._config = config
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._env_service_url = env_service_url
        self._tokenizer = tokenizer  # cc: 这玩意似乎不该在这
        self._max_llm_retries = kwargs["max_llm_retries"] or 3
        self._max_explore_step = kwargs["max_explore_step"] or 10
        self._num_exploration_threads = kwargs["num_explore_threads"] or 10
        self._n = kwargs["n"]
        self._exploration_llm_temperature = kwargs.get(
            "exploration_llm_temperature", 1.0
        )
        self._exploration_llm_top_p = kwargs.get("exploration_llm_top_p", 1.0)
        self._exploration_llm_top_k = kwargs.get("exploration_llm_top_k", 1)
        self._task_summary_history_length = kwargs.get("task_summary_history_length", self._max_explore_step)

        # 混合原有数据和生成数据
        # TODO: a better mixture strategy is possible
        self._use_original_tasks = kwargs.get("use_original_tasks", False)

        self._filters: list[TaskPostFilter] = []
        
        self._tasks: list[Task]=[]
    
    @property
    def seed_tasks(self):
        return self._tasks
        
    def load_tasks_from_dataset(self, dataset: RLHFDataset,*, env_type:str):
        self._tasks.extend(adapter.convert_to_tasks(dataset,env_type=env_type))
        logger.info(f"loaded tasks from dataset, current # of tasks={len(self._tasks)}")
    
    def load_tasks_from_environment(self,env:EnvClient,*,env_type:str,split:str,params:Optional[dict]=None):
        response=env.get_task_ids(env_type,split,params)
        # FIXME: find the query and evaluator type
        self._tasks.extend([Task(task_id=x,env_type=env_type) for x in response])
        logger.info(f"loaded tasks from environment, current # of tasks={len(self._tasks)}")

    def register_filter(self, filter: TaskPostFilter):
        self._filters.append(filter)

    def get_onthefly_dataset(self, bs: int, tokenizer, config,processor):
        """
        Get dataset on the fly.

        Args:
            tasks: Iterable[Task]
            bs: int. 该 batch size 决定一次读取的 task 数量。每次生成的 dataset 大小为 bs * self._n。
            tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
            config: DictConfig. Only for RLHFDataset.
        """

        return AutoReloadDataset(self,iter(self._tasks),bs,self._use_original_tasks,tokenizer=tokenizer,config=config,processor=processor)
    
    def get_or_load_full_dataset(self,filepath:Optional[str],*,config,tokenizer,processor)->"FullDataset":
        """Get the full dataset, or load from file.
        """     
        dataset=FullDataset(self,self._tasks,self._use_original_tasks,tokenizer=tokenizer,config=config,processor=processor)
        if filepath is not None and os.path.exists(filepath):
            logger.info(f"loading full dataset from {filepath}")
            dataset.load_from_file(filepath)
        else:
            dataset.reload()
            if filepath is not None:
                dataset.save_to_file(filepath)
        
        return dataset
    
    def debug_get_original_seed_dataset(self,*,tokenizer,config,processor)->Dataset:
        """THIS IS A DEBUG FUNCTION, WILL BE REMOVED IN FUTURE.
        """
        return OriginalDataset(self._tasks,tokenizer=tokenizer,config=config,processor=processor)
    
    
    def generate_task(self, tasks: Sequence[Task],*,show_progress=False) -> list[TaskObjective]:
        task_q = list(copy.copy(tasks)) * self._n
        res = []
        
        # 每次最多探索所有不同任务，或者最大线程个任务，防止同批次中生成相同任务
        parallel_num = min(self._num_exploration_threads, len(tasks))
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            for i in tqdm(range(0, len(task_q), parallel_num), disable=not show_progress):
                futures = [
                    pool.submit(self._exlore_and_summarize, task, data_id, rollout_id)
                    for task, data_id, rollout_id in zip(
                        task_q[i : i + parallel_num],
                        ["unknown"] * parallel_num,
                        ["unknown"] * parallel_num,
                    )
                ]
                task_objectives = sum([future.result() for future in futures], [])
                res.extend(task_objectives)

        # post filter
        res = functools.reduce(lambda x, f: f.filter(x), self._filters, res)
        
        random.shuffle(res) # shuffle

        return res

    
    def _exlore_and_summarize(self,task:Task,data_id:str,rollout_id:str):
        trajectory=self._step_explore(task,data_id,rollout_id)
        task_objectives=self._step_summarize(task,trajectory)
        return task_objectives


    def _step_explore(self, task: Task, data_id: str, rollout_id: str):
        """
        Step 1: explore the environment to find out possible actions and their results.
        """
        # reset env every time
        env_worker = Explorer(
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
        agent_flow: BaseAgentFlow = AgentFlow(
            enable_context_generator=False,
            llm_chat_fn=llm_chat_fn,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        agent_flow.max_steps = self._max_explore_step  # TODO(cc): this is ugly

        old_objectives = self._old_retrival.retrieve_objectives(task)

        traj = env_worker.execute(
            data_id=data_id,
            rollout_id=rollout_id,
            system_prompt=get_agent_interaction_system_prompt(task, old_objectives),
            agent_flow=agent_flow,
        )

        return traj


    def _step_summarize(
        self, task: Task, trajectory: Trajectory
    ) -> list[TaskObjective]:
        """
        Step 2: summarize the results of the exploration to generate the TASK (query and gt).

        Args:
            task: Task
            trajectories: Trajectory.
        """
        # 这个方法从现在看基本上是固定的
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
                    res = self._llm_client.chat(
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


class OriginalDataset(Dataset):
    def __init__(self,tasks:Sequence[Task],*,tokenizer,config, processor):
        self._tasks=list(tasks)
        self._tokenizer = tokenizer
        self._config=config
        self._processor=processor
    
        self._objectives=[TaskObjective(task=x,ground_truth="[env]",confidence=1.0,reward=None) for x in self._tasks]
        logger.info("mixed original tasks")

        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config,self._processor)
        
        logger.warning(f"loaded original dataset: #task={len(self._tasks)} #rlhf={len(self._dataset)}")
        exit(0)
    
    
    def __getitem__(self, index):
        return self._dataset[index]
    
    def __len__(self):
        return len(self._dataset)


class FullDataset(Dataset):
    
    """FullDataset
    """
    
    def __init__(self,manager:TaskManager, tasks:Sequence[Task],mix_origins:bool=False,*,tokenizer,config, processor):
        self._manager=manager
        self._tasks=list(tasks)
        self._mix_origins=mix_origins
        self._tokenizer = tokenizer
        self._config=config
        self._processor=processor
    
    def save_to_file(self,filepath:str):
        with open(filepath,"w") as f:
            f.writelines([ob.json()+"\n" for ob in self._objectives])
    
    def load_from_file(self,filepath:str):
        with open(filepath,"r") as f:
            self._objectives=[TaskObjective.parse_raw(line) for line in filter(lambda x: x.strip()!="", f.readlines())]
        # mix
        if self._mix_origins:
            self._objectives.extend([TaskObjective(task=x,ground_truth="[env]",confidence=1.0,reward=None) for x in self._tasks])
            logger.info("mixed original tasks")
        random.shuffle(self._objectives)

        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config,self._processor)
    
    def reload(self):
        self._objectives=self._manager.generate_task(self._tasks,show_progress=True)
        # mix
        if self._mix_origins:
            self._objectives.extend([TaskObjective(task=x,ground_truth="[env]",confidence=1.0,reward=None) for x in self._tasks])
            logger.info("mixed original tasks")
        random.shuffle(self._objectives)
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config,self._processor)
    
    def __getitem__(self, index):
        return self._dataset[index]
    
    def __len__(self):
        return len(self._dataset)


# wrapper for data auto-reloading
class AutoReloadDataset(IterableDataset):
    """AytoReloadDataset
    
    the number of workers of DataLoader must be 1.
    """
    def __init__(self,manager:TaskManager, tasks:Iterable[Task], bs: int, mix_origins:bool=False, *, tokenizer, config, processor):
        self._manager=manager
        self._tasks=tasks
        self._bs = bs
        self._mix_origins=mix_origins
        assert self._mix_origins==False, "mix_origins is not supported yet"
        self._tokenizer = tokenizer
        self._config=config
        self._processor = processor
        
        self._dataset = OnflyRlDataset(release_used_dataset=True)
    
    def reload(self):
        delta = []
        for task in self._tasks:
            delta.append(task)
            if len(delta) == self._bs:
                break

        ls = self._manager.generate_task(delta)
        while len(ls) < self._bs * self._manager._n:
            logger.debug("failed to generate enough tasks, retrying")
            ls = self._manager.generate_task(delta)

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config,self._processor))
        return self._dataset.num_rest_data

    def __iter__(self):
        return self

    def __next__(self):
        if self._dataset.num_rest_data == 0:
            logger.debug("no data left")
            if self.reload() == 0:
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)