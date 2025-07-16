from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import functools
import json
import os
import pickle
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
from beyondagent.module.task_manager.data_mixture import MixtureStrategy, OriginalOnlyStrategy
from beyondagent.module.task_manager.strategies import TaskExploreStrategy
from beyondagent.module.task_manager.explorer import Explorer
from beyondagent.module.task_manager.filters import TaskPostFilter
from beyondagent.module.task_manager.prompts.prompt_explore import (
    get_agent_interaction_system_prompt,
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
    
    mix_original_tasks: NotRequired[bool]


# TODO: 能够替换的 exploration & extraction (summary) strategy

class TaskManager(object):

    def __init__(
        self,
        config: DictConfig,
        exploration_strategy: TaskExploreStrategy,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        mixture_strategy: MixtureStrategy,
        tokenizer,
        env_service_url: str,
        **kwargs: Unpack[TaskManagerProps],
    ):
        self._config = config
        self._exploration_strategy=exploration_strategy
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._mixture_strategy = mixture_strategy
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

        self._filters: list[TaskPostFilter] = []
        
        self._tasks: list[Task]=[]
        self._exploration_strategy._inject_deps(self._old_retrival,self._llm_client)
    
    @property
    def seed_tasks(self):
        return self._tasks
    
    def load_tasks(self,tasks:Sequence[Task]):
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")
        
    def load_tasks_from_dataset(self, dataset: RLHFDataset,*, env_type:str):
        self._tasks.extend(adapter.convert_to_tasks(dataset,env_type=env_type))
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")
    
    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        try:
            response = env.get_env_profile(env_type, split, params)
            self._tasks.extend([Task(task_id=str(x),env_type=env_type,evaluator='env') for x in response])
            assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
        except:
            logger.error(f"failed to load tasks from environment")
            return 0
        return len(response)

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
        # autoreloaddataset 没适配 mixture
        raise NotImplementedError("get_onthefly_dataset is not implemented")
        # return AutoReloadDataset(self,iter(self._tasks),bs,self._mix_original_tasks,tokenizer=tokenizer,config=config,processor=processor)
    
    def get_or_load_full_dataset(self,filepath:Optional[str],*,config,tokenizer,processor)->"FullDataset":
        """Get the full dataset, or load from file.
        """
        seed_tasks=[TaskObjective(task=task,ground_truth='[env]',confidence=1.0,reward=None) for task in self._tasks]
        dataset=FullDataset(self,seed_tasks,self._mixture_strategy,tokenizer=tokenizer,config=config,processor=processor)
        
        if filepath is not None and os.path.exists(filepath):
            logger.info(f"loading full dataset from {filepath}")
            dataset.load_from_file(filepath)
        else:
            dataset.reload()
            if filepath is not None:
                dataset.save_to_file(filepath)
        
        return dataset
    
    def get_original_dataset(self,*,tokenizer,config,processor)->"FullDataset":
        """Get the original dataset.
        """
        seed_tasks=[TaskObjective(task=task,ground_truth='[env]',confidence=1.0,reward=None) for task in self._tasks]
        dataset = FullDataset(self,seed_tasks,OriginalOnlyStrategy(),tokenizer=tokenizer,config=config,processor=processor)
        dataset.load_from_file('[unknown]')
        return dataset
    
    
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

    
    def _exlore_and_summarize(self,task:Task,data_id:str,rollout_id:str)->list[TaskObjective]:
        trajectories=self._step_explore(task,data_id,rollout_id)
        task_objectives=sum([self._step_summarize(task,trajectory) for trajectory in trajectories],[])
        return task_objectives


    def _step_explore(self, task: Task, data_id: str, rollout_id: str)->list[Trajectory]:
        """
        Step 1: explore the environment to find out possible actions and their results.
        """
        return self._exploration_strategy.explore(task,data_id,rollout_id)


    def _step_summarize(
        self, task: Task, trajectory: Trajectory
    ) -> list[TaskObjective]:
        """
        Step 2: summarize the results of the exploration to generate the TASK (query and gt).

        Args:
            task: Task
            trajectories: Trajectory.
        """
        return self._exploration_strategy.summarize(task, trajectory)


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


class FullDataset(Dataset):
    """FullDataset with MixtureStrategy support"""
    
    def __init__(self, 
                 manager: TaskManager, 
                 tasks: Sequence[TaskObjective],
                 mixture_strategy: MixtureStrategy,
                 *, 
                 tokenizer, 
                 config, 
                 processor):
        """
        Args:
            manager: TaskManager实例
            tasks: 原始任务列表
            mixture_strategy: 数据混合策略，如果为None则使用默认的NoMixtureStrategy
            tokenizer: tokenizer
            config: 配置
            processor: processor
        """
        self._manager = manager
        self._tasks = list(tasks)
        self._mixture_strategy = mixture_strategy
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        self._objectives = []
        self._dataset = None
    
    def set_mixture_strategy(self, strategy: MixtureStrategy):
        """设置新的混合策略"""
        self._mixture_strategy = strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")
    
    def save_to_file(self, filepath: str):
        """保存objectives到文件"""
        with open(filepath, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._objectives])
        logger.info(f"Saved {len(self._objectives)} objectives to {filepath}")
    
    def load_from_file(self, filepath: str):
        """从文件加载objectives"""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                synthetic_objectives = [
                    TaskObjective.parse_raw(line) 
                    for line in filter(lambda x: x.strip() != "", f.readlines())
                ]
        else:
            logger.warning(f"failed to load objectives from {filepath}, file not found.")
            synthetic_objectives = []
        
        # 使用混合策略处理数据
        self._objectives = self._mixture_strategy.mix_data(synthetic_objectives, self._tasks)
        
        # 转换为RL dataset
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)
        logger.info(f"Loaded and mixed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")
    
    def reload(self):
        """重新生成数据"""
        # 生成合成数据
        synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        
        # 使用混合策略处理数据
        self._objectives = self._mixture_strategy.mix_data(synthetic_objectives, self._tasks)
        
        # 转换为RL dataset
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)
        logger.info(f"Reloaded and mixed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        if not self._objectives:
            return {
                "total": 0, 
                "synthetic": 0, 
                "original": 0,
                "synthetic_ratio": 0.0,
                "strategy_info": str(self._mixture_strategy)
            }
        
        synthetic_count = sum(1 for obj in self._objectives if obj.task.evaluator != "env")
        original_count = len(self._objectives) - synthetic_count
        
        return {
            "total": len(self._objectives),
            "synthetic": synthetic_count,
            "original": original_count,
            "synthetic_ratio": synthetic_count / len(self._objectives) if len(self._objectives) > 0 else 0,
            "strategy_info": str(self._mixture_strategy)
        }
    
    def __getitem__(self, index):
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")
        return self._dataset[index]
    
    def __len__(self):
        if self._dataset is None:
            return 0
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