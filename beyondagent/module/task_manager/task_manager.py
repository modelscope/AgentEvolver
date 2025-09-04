from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import functools
import hashlib
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
import requests
from torch.utils.data import IterableDataset,Dataset
from tqdm import tqdm
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager import adapter
from beyondagent.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from beyondagent.module.task_manager.data_mixture import MixtureStrategy, OriginalOnlyStrategy
from beyondagent.module.task_manager.filters.llm_filter import LlmFilter
from beyondagent.module.task_manager.strategies import TaskExploreStrategy
from beyondagent.module.task_manager.explorer import EnvWorkerWithPrompt
from beyondagent.module.task_manager.filters.filters import NaiveTaskPostFilter, TaskPostFilter

from beyondagent.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from beyondagent.module.task_manager.strategies.deduplication import LlmDedupSamplingExploreStrategy
from beyondagent.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from beyondagent.module.task_manager.user_profiles import UserProfile
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset

class TaskManagerProps(TypedDict):
    num_explore_threads: int
    n: int # 重复探索的控制必须放在这里，task manager 要规划 task 执行顺序，避免在同时探索相同任务导致潜在的 query 重复

class RewardProps(TypedDict):
    original_grader:str
    synthetic_grader:str

class TaskManager(object):

    def __init__(
        self,
        config: DictConfig,
        exploration_strategy: str,
        user_profile:UserProfile,
        exploration_strategy_args,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        mixture_strategy: MixtureStrategy,
        reward_config: RewardProps,
        tokenizer,
        env_service_url: str,
        **kwargs: Unpack[TaskManagerProps],
    ):
        self._config = config
        self._exploration_strategy=get_exploration_strategy(exploration_strategy,exploration_strategy_args,tokenizer=tokenizer,config=config)
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._env_service_url = env_service_url
        self._tokenizer = tokenizer  # cc: 这玩意似乎不该在这
        self._num_exploration_threads = kwargs["num_explore_threads"] or 10
        self._n = kwargs["n"]

        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        self._post_filter: list[TaskPostFilter] = [LlmFilter(env_service_url,llm_client,self._num_exploration_threads,tokenizer=tokenizer,config=config)]
        
        self._tasks: list[Task]=[]
        self._exploration_strategy._inject_deps(self._old_retrival,self._llm_client,DashScopeClient(model_name='qwen3-235b-a22b-instruct-2507',max_tokens=8192),user_profile=user_profile)
    
    @property
    def seed_tasks(self):
        return self._tasks
    
    def load_tasks(self,tasks:Sequence[Task]):
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")
        
    def load_tasks_from_dataset(self, dataset: RLHFDataset,*, env_type:str):
        self._tasks.extend(adapter.convert_to_tasks(dataset,env_type=env_type,grader=self._reward_config["original_grader"]))
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")
    
    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        try:
            response = env.get_env_profile(env_type, split, params)
            self._tasks.extend([Task(task_id=str(x),env_type=env_type,evaluator=self._reward_config["original_grader"]) for x in response])
            assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to load tasks from environment: {e}")
            return 0
        return len(response)

    def register_filter(self, filter: TaskPostFilter):
        self._realtime_filters.append(filter)

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
        seed_tasks=[TaskObjective(task=task,confidence=1.0,reward=None) for task in self._tasks]
        dataset=FullDataset(self,seed_tasks,self._mixture_strategy,self._reward_config,tokenizer=tokenizer,config=config,processor=processor)
        
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
        seed_tasks=[TaskObjective(task=task,confidence=1.0,reward=None) for task in self._tasks]
        dataset = FullDataset(self,seed_tasks,OriginalOnlyStrategy(),self._reward_config,tokenizer=tokenizer,config=config,processor=processor)
        dataset.load_from_file('[unknown]')
        return dataset
    
    
    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        """Compute hash of tasks to verify consistency during resume."""
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        return hashlib.md5(combined_str.encode()).hexdigest()
    
    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        if resume_file is None:
            resume_file = '.generate_task.checkpoint.json'
        
        # Compute hash of current tasks
        current_tasks_hash = self._compute_tasks_hash(tasks)
        # Load from checkpoint if resume_file exists
        res = []
        processed_indices = set()
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    checkpoint = json.load(f)
                    # Check if tasks hash matches
                    if checkpoint['tasks_hash'] != current_tasks_hash:
                        logger.warning(f"Tasks hash mismatch. Expected: {current_tasks_hash}, got: {checkpoint['tasks_hash']}. Removing checkpoint.")
                        os.remove(resume_file)
                    else:
                        res = [TaskObjective.parse_raw(json.dumps(obj)) for obj in checkpoint.get('results', [])]
                        processed_indices = {int(i) for i in checkpoint.get('processed_indices', [])}
                        logger.info(f"Resumed from checkpoint: {len(res)} results loaded, {len(processed_indices)} batches processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")
        
        # 每个任务都 roll n 次
        task_q = list(copy.copy(tasks)) * self._n
        
        # 每次最多探索所有不同任务，或者最大线程个任务，防止同批次中生成相同任务
        parallel_num = min(self._num_exploration_threads, len(tasks))
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            batch_indices = list(range(0, len(task_q), parallel_num))
            for idx, i in enumerate(tqdm(batch_indices, desc="generating tasks", disable=not show_progress)):
                # Skip already processed batches when resuming
                if idx in processed_indices:
                    continue
                    
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
                # realtime filter
                res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
                self._old_retrival.reset()
                for j in res:
                    self._old_retrival.add_objective(j)
                    
                # Mark this batch as processed
                processed_indices.add(idx)
                
                # Save checkpoint
                if resume_file:
                    try:
                        checkpoint_data = {
                            'results': [obj.dict() for obj in res],
                            'processed_indices': list(processed_indices),
                            'total_batches': len(batch_indices),
                            'tasks_hash': current_tasks_hash,
                            'timestamp': time.time()
                        }
                        with open(resume_file, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")
                
                
                    
        res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
        # post filter
        logger.info("running post filter on generated tasks")
        cnt_before_filter=len(res)
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)
        cnt_after_filter=len(res)
        logger.info(f"finish post filter: #before={cnt_before_filter}, #after={cnt_after_filter}")
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


def get_exploration_strategy(name:str, strategy_args, *, tokenizer, config)->TaskExploreStrategy:
    """Get exploration strategy by name."""
    logger.info(f"loading exploration strategy {name}")
    if name=="random":
        return LlmRandomSamplingExploreStrategy(tokenizer=tokenizer,config=config,**strategy_args)
    elif name == "deduplication":
        return LlmDedupSamplingExploreStrategy(tokenizer=tokenizer,config=config,**strategy_args)
    else:
        raise NotImplementedError(f"exploration strategy {name} not implemented")





class FullDataset(Dataset):
    """FullDataset with MixtureStrategy support and auto-refresh after one DataLoader epoch"""
    
    def __init__(self, 
                 manager: TaskManager, 
                 tasks: Sequence[TaskObjective],
                 mixture_strategy: MixtureStrategy,
                 reward_config:RewardProps,
                 *, 
                 tokenizer, 
                 config, 
                 processor):
        self._manager = manager
        self._tasks = list(tasks)
        assert all([x.task.evaluator==reward_config["original_grader"] for x in tasks]), "task evaluator must be set as the config"
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        self._objectives = []
        self._dataset = None
        self._synthetic_objectives = []
        
        # 标记是否需要在下一轮迭代开始前刷新
        self._refresh_after_epoch = False

    def _rebuild_dataset(self):
        """使用混合策略重新生成 dataset"""
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)
        logger.info(f"Auto-refreshed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")

    def update(self):
        """手动触发一次数据集重建"""
        if not self._synthetic_objectives:
            logger.warning("No synthetic objectives available, did you call load_from_file() or reload() first?")
        self._rebuild_dataset()
        logger.info("Dataset updated manually via update().")

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        self._mixture_strategy = strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")
    
    def save_to_file(self, filepath: str):
        with open(filepath, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])
        logger.info(f"Saved {len(self._objectives)} objectives to {filepath}")
    
    def load_from_file(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self._synthetic_objectives = []
                for line in filter(lambda x: x.strip() != "", f.readlines()):
                    tmp=TaskObjective.parse_raw(line)
                    # patch old data
                    if tmp.ground_truth is None:
                        tmp.ground_truth = json.loads(line)['ground_truth']
                    self._synthetic_objectives.append(tmp)
        else:
            logger.warning(f"failed to load objectives from {filepath}, file not found.")
            self._synthetic_objectives = []
        
        for item in self._synthetic_objectives:
            assert item.ground_truth is not None
        
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]
        
        self._rebuild_dataset()
    
    def reload(self):
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]
        self._rebuild_dataset()
    
    def get_statistics(self) -> dict:
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