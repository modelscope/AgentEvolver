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
from agentevolver.client.env_client import EnvClient
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.task_manager import adapter
from agentevolver.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from agentevolver.module.task_manager.data_mixture import MixtureStrategy, OriginalOnlyStrategy
from agentevolver.module.task_manager.filters.llm_filter import LlmFilter
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.module.task_manager.filters.filters import NaiveTaskPostFilter, TaskPostFilter

from agentevolver.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from agentevolver.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from agentevolver.module.task_manager.env_profiles import EnvProfile
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset

class TaskManagerProps(TypedDict):
    num_explore_threads: int
    n: int # n must be placed here. The task manager needs to plan the task execution order to avoid potential duplicate queries resulting from simultaneously exploring the same task.

class RewardProps(TypedDict):
    original_grader:str
    synthetic_grader:str

class TaskManager(object):

    def __init__(
        self,
        config: DictConfig,
        exploration_strategy: str,
        env_profile:EnvProfile,
        exploration_strategy_args,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        mixture_strategy: MixtureStrategy,
        reward_config: RewardProps,
        tokenizer,
        env_service_url: str,
        **kwargs: Unpack[TaskManagerProps],
    ):
        """
        Initializes the TaskManager with the given configuration and dependencies.

        Args:
            config (DictConfig): Configuration for the task manager.
            exploration_strategy (str): Name of the exploration strategy to use.
            user_profile (UserProfile): User profile for the task manager.
            exploration_strategy_args: Arguments for the exploration strategy.
            llm_client (LlmClient): Client for interacting with the language model.
            old_retrival (TaskObjectiveRetrieval): Retrieval mechanism for old tasks.
            mixture_strategy (MixtureStrategy): Strategy for mixing tasks.
            reward_config (RewardProps): Configuration for rewards.
            tokenizer: Tokenizer for the language model.
            env_service_url (str): URL for the environment service.
            **kwargs (Unpack[TaskManagerProps]): Additional properties for the task manager.
        """
        self._config = config
        self._exploration_strategy=get_exploration_strategy(exploration_strategy,exploration_strategy_args,tokenizer=tokenizer,config=config)  # ⭐ Initialize the exploration strategy
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._env_service_url = env_service_url
        self._tokenizer = tokenizer  # why is tokenizer here?
        self._num_exploration_threads = kwargs["num_explore_threads"] or 10
        self._n = kwargs["n"]

        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        self._post_filter: list[TaskPostFilter] = [LlmFilter(env_service_url,llm_client,self._num_exploration_threads,tokenizer=tokenizer,config=config)]  # ⭐ Initialize the post filter

        self._tasks: list[Task]=[]
        self._exploration_strategy._inject_deps(self._old_retrival,self._llm_client,DashScopeClient(model_name='qwen3-235b-a22b-instruct-2507',max_tokens=8192),env_profile=env_profile)  # ⭐ Inject dependencies into the exploration strategy

    @property
    def seed_tasks(self):
        """
        Returns the list of seed tasks.

        Returns:
            list[Task]: The list of seed tasks.
        """
        return self._tasks
    
    @property
    def seed_task_objectives(self):
        return [TaskObjective(task=task,confidence=1.0,reward=None) for task in self.seed_tasks]

    def load_tasks(self,tasks:Sequence[Task]):
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")

    def load_tasks_from_dataset(self, dataset: RLHFDataset,*, env_type:str):
        """
        Loads tasks from a given dataset and appends them to the internal task list.

        Args:
            dataset (RLHFDataset): The dataset from which to load tasks.
            env_type (str): The type of environment for the tasks.

        Returns:
            None
        """
        self._tasks.extend(adapter.convert_to_tasks(dataset,env_type=env_type,grader=self._reward_config["original_grader"]))  # ⭐ Convert dataset to tasks and add to the task list
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")

    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        """
        Loads tasks from a given environment and appends them to the internal task list.

        Args:
            env (EnvClient): The environment client from which to load tasks.
            env_type (str): The type of environment for the tasks.
            split (str): The split of the data to load (e.g., 'train', 'test').
            params (Optional[dict]): Additional parameters for the environment request.

        Returns:
            int: The number of tasks loaded from the environment, or 0 if the request failed.
        """
        try:
            response = env.get_env_profile(env_type, split, params)
            self._tasks.extend([Task(task_id=str(x),env_type=env_type,open_query=False,evaluator=self._reward_config["original_grader"]) for x in response])  # ⭐ Create Task objects from the response and add to the task list
            assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to load tasks from environment: {e}")
            raise
        return len(response)

    def register_filter(self, filter: TaskPostFilter):
        self._realtime_filters.append(filter)

    def _get_onthefly_dataset(self, bs: int, tokenizer, config,processor):
        """
        Get dataset on the fly.

        Args:
            tasks: Iterable[Task]
            bs: int. This batch size determines the number of tasks read at one time. The size of the dataset generated each time is bs * self._n.
            tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
            config: DictConfig. Only for RLHFDataset.
        """
        # autoreloaddataset does not support mixture
        raise NotImplementedError("get_onthefly_dataset is not implemented")
        # return AutoReloadDataset(self,iter(self._tasks),bs,self._mix_original_tasks,tokenizer=tokenizer,config=config,processor=processor)


    def _compute_tasks_hash(self, tasks: Sequence[Task]) -> str:
        """
        Computes a hash of the given tasks to verify consistency during resume.

        Args:
            tasks (Sequence[Task]): A sequence of Task objects.

        Returns:
            str: The MD5 hash of the combined string representation of the tasks.
        """
        task_strs = [f"{task.task_id}:{task.env_type}" for task in tasks]
        combined_str = "|".join(task_strs)
        return hashlib.md5(combined_str.encode()).hexdigest()  # ⭐ Compute the MD5 hash of the combined string

    def generate_task(self, tasks: Sequence[Task], *, show_progress=False, resume_file: Optional[str] = None) -> list[TaskObjective]:
        """
        Generates task objectives by exploring and summarizing tasks, with support for resuming from a checkpoint and applying filters.

        Args:
            tasks (Sequence[Task]): A sequence of Task objects.
            show_progress (bool): Whether to show a progress bar.
            resume_file (Optional[str]): The path to the resume file. If not provided, a default file is used.

        Returns:
            list[TaskObjective]: A list of generated TaskObjective objects.
        """
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

        # we roll n times for each task
        task_q = list(copy.copy(tasks)) * self._n

        # in each batch, we explore all different tasks or max_threads tasks, in order to avoid generating same task.
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
                task_objectives = sum([future.result() for future in futures], [])  # ⭐ Collect results from all futures
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
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)  # ⭐ Apply post filters to the results
        cnt_after_filter=len(res)
        logger.info(f"finish post filter: #before={cnt_before_filter}, #after={cnt_after_filter}")
        random.shuffle(res)  # ⭐ Shuffle the final list of task objectives

        return res


    def _exlore_and_summarize(self,task:Task,data_id:str,rollout_id:str)->list[TaskObjective]:
        """
        Explores the environment based on the provided task and then summarizes the results to generate a list of TaskObjective objects.

        Args:
            task (Task): The task to be explored.
            data_id (str): The ID of the data.
            rollout_id (str): The ID of the rollout.

        Returns:
            list[TaskObjective]: A list of TaskObjective objects generated from the exploration and summarization.
        """
        trajectories=self._step_explore(task,data_id,rollout_id)  # ⭐ Explore the environment
        task_objectives=sum([self._step_summarize(task,trajectory) for trajectory in trajectories],[])  # ⭐ Summarize the exploration results
        # check open query
        assert all([x.task.open_query==True for x in task_objectives]), "all synthetic tasks must have open query"
        return task_objectives


    def _step_explore(self, task: Task, data_id: str, rollout_id: str)->list[Trajectory]:
        """
        Step 1: explore the environment to find out possible actions and their results.

        Args:
            task (Task): The task to be explored.
            data_id (str): The ID of the data.
            rollout_id (str): The ID of the rollout.

        Returns:
            list[Trajectory]: A list of Trajectory objects representing the possible actions and their results.
        """
        return self._exploration_strategy.explore(task,data_id,rollout_id)  # ⭐ Execute the exploration strategy


    def _step_summarize(
        self, task: Task, trajectory: Trajectory
    ) -> list[TaskObjective]:
        """
        Step 2: summarize the results of the exploration to generate the TASK (query and gt).

        Args:
            task (Task): The task that was explored.
            trajectory (Trajectory): The trajectory resulting from the exploration.

        Returns:
            list[TaskObjective]: A list of TaskObjective objects generated from the summarization.
        """
        return self._exploration_strategy.summarize(task, trajectory)  # ⭐ Execute the summarization strategy


def get_exploration_strategy(name:str, strategy_args, *, tokenizer, config)->TaskExploreStrategy:
    """Get exploration strategy by name."""
    logger.info(f"loading exploration strategy {name}")
    if name=="random":
        return LlmRandomSamplingExploreStrategy(tokenizer=tokenizer,config=config,**strategy_args)
    else:
        raise NotImplementedError(f"exploration strategy {name} not implemented")





class FullDataset(Dataset):
    """FullDataset with MixtureStrategy support and auto-refresh after one DataLoader epoch"""

    def __init__(self,
                 manager: TaskManager,
                 mixture_strategy: MixtureStrategy,
                 reward_config:RewardProps,
                 cache_path: Optional[str] = None,
                 *,
                 tokenizer,
                 config,
                 processor):
        self._manager = manager
        self._tasks = self._manager.seed_task_objectives
        assert all([x.task.evaluator==reward_config["original_grader"] for x in self._tasks]), "task evaluator must be set as the config"
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._cache_path = cache_path
        
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        
        self._objectives = []
        self._dataset = None
        self._synthetic_objectives = []

        # tag, used to mark whether the dataset needs to be refreshed
        self._refresh_after_epoch = False
        
        # prepare the synthetic dataset if needed
        if self._mixture_strategy.need_synthetic:
            logger.info("preparing synthetic tasks")
            if self._cache_path is not None and os.path.exists(self._cache_path):
                logger.info(f"loading synthetic tasks from file {self._cache_path}")
                self.load_from_file() # load synthetic data
            else:
                self.reload_new_task() # generate synthetic data
                if self._cache_path is not None:
                    logger.debug("saving synthetic tasks to cache file")
                    self.save_to_file()
        else:
            logger.info(f"the mixture strategy need no synthetic data ({self._mixture_strategy}), skipping synthetic data...")
        
        # build hybrid dataset
        self._rebuild_dataset()
        

    def _rebuild_dataset(self):
        """
        Regenerates the dataset using a mixture strategy.

        This method mixes synthetic objectives with the current tasks, then converts the mixed data into a reinforcement learning (RL) dataset. It also logs the number of objectives and RLHF (Reinforcement Learning from Human Feedback) items in the new dataset.

        Args:
            None

        Returns:
            None
        """
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)  # ⭐ Mixes synthetic objectives with current tasks
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)  # ⭐ Converts the mixed data into an RL dataset
        logger.info(f"Auto-refreshed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")  # ⭐ Logs the number of objectives and RLHF items

    def update(self):
        """
        Manually triggers the rebuilding of the dataset.

        This method first checks if there are any synthetic objectives available. If not, it logs a warning suggesting
        that `load_from_file()` or `reload()` should be called first. It then rebuilds the dataset and logs an
        informational message upon completion.

        Returns:
            None
        """
        if not self._synthetic_objectives:
            logger.warning("No synthetic objectives available, did you call load_from_file() or reload() first?")
        self._rebuild_dataset()  # ⭐ Rebuilds the dataset
        logger.info("Dataset updated manually via update().")

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        """
        Sets the mixture strategy for the TaskManager and logs the update.

        Args:
            strategy (MixtureStrategy): The new mixture strategy to be set.
        """
        self._mixture_strategy = strategy  # ⭐ Update the mixture strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")

    def save_to_file(self):
        """
        Saves the JSON representation of each synthetic objective to a specified file.

        Args:
            filepath (str): The path to the file where the objectives will be saved.

        Returns:
            None
        """
        assert self._cache_path is not None
        with open(self._cache_path, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])  # ⭐ Writes each objective's JSON to the file
        logger.info(f"Saved {len(self._objectives)} objectives to {self._cache_path}")  # ⭐ Logs the number of objectives saved

    def load_from_file(self):
        """
        Loads objectives from a specified file. This function is currently incomplete.

        Args:
            filepath (str): The path to the file from which the objectives will be loaded.

        Returns:
            None
        """
        if self._cache_path is None:
            logger.error("trying to load synthetic objectives from file, but cache_path is not set")
            return
        
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                self._synthetic_objectives = []
                for line in filter(lambda x: x.strip() != "", f.readlines()):
                    # patch old data: open query
                    t=json.loads(line)
                    assert 'task' in t
                    if 'open_query' not in t['task']:
                        t['task']['open_query'] = True # all synthetic data is open query
                    
                    # patch old data: ground_truth
                    tmp=TaskObjective.parse_obj(t)
                    if tmp.ground_truth is None:
                        tmp.ground_truth = json.loads(line)['ground_truth']
                    self._synthetic_objectives.append(tmp)
        else:
            raise FileNotFoundError(f"failed to load synthetic objectives from file {self._cache_path}, file not found")
        
        # check if all synthetic objectives have ground_truth
        for item in self._synthetic_objectives:
            assert item.ground_truth is not None

        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]  # ⭐ Update the evaluator for each task


    def reload_new_task(self):
        """
        Regenerates the synthetic objectives, updates their evaluators, and rebuilds the dataset.

        This method is used to refresh the task objectives and ensure they are up-to-date with the current configuration.
        """
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]  # ⭐ Update the evaluator for each task
        

    def get_statistics(self) -> dict:
        """
        Computes and returns a dictionary containing statistics about the tasks, such as the total number of tasks,
        the number of synthetic and original tasks, the ratio of synthetic tasks, and the strategy information.

        Returns:
            dict: A dictionary with keys 'total', 'synthetic', 'original', 'synthetic_ratio', and 'strategy_info'.
        """
        if not self._objectives:
            return {
                "total": 0,
                "synthetic": 0,
                "original": 0,
                "synthetic_ratio": 0.0,
                "strategy_info": str(self._mixture_strategy)
            }

        synthetic_count = sum(1 for obj in self._objectives if obj.task.evaluator != "env")  # ⭐ Count the number of synthetic tasks
        original_count = len(self._objectives) - synthetic_count  # ⭐ Calculate the number of original tasks

        return {
            "total": len(self._objectives),
            "synthetic": synthetic_count,
            "original": original_count,
            "synthetic_ratio": synthetic_count / len(self._objectives) if len(self._objectives) > 0 else 0,
            "strategy_info": str(self._mixture_strategy)
        }

    def __getitem__(self, index):
        """
        Allows indexing of the TaskManager instance to access items in the underlying dataset.

        Args:
            index (int): The index of the item to retrieve from the dataset.

        Returns:
            The item at the specified index in the dataset.

        Raises:
            RuntimeError: If the dataset has not been loaded.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")  # ⭐ Ensures the dataset is loaded before accessing
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
        """
        Returns the iterator object itself, allowing the TaskManager instance to be used as an iterator.

        Returns:
            TaskManager: The iterator object (self).
        """
        return self

    def __next__(self):
        """
        Fetches the next task from the dataset. If no tasks are left, it tries to reload the dataset.
        If reloading does not provide any new tasks, it raises a StopIteration exception.

        Returns:
            Any: The next task from the dataset.

        Raises:
            StopIteration: If there are no more tasks left after attempting to reload the dataset.
        """
        if self._dataset.num_rest_data == 0:  # ⭐ Check if there are any remaining tasks
            logger.debug("no data left")
            if self.reload() == 0:  # ⭐ Attempt to reload the dataset
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)  # ⭐ Get the next task from the dataset