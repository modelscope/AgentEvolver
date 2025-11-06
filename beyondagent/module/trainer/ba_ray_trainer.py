# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications copyright 2025 Alibaba Tongyi EconML Lab. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from pprint import pprint
from typing import List, Optional, Any
import warnings

from loguru import logger
import numpy as np
import ray
import torch
import random
import json
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from torch.utils.data import SequentialSampler,IterableDataset,Dataset,Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from beyondagent.client.env_client import EnvClient
from beyondagent.module.task_manager.task_manager import AutoReloadDataset, FullDataset
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from beyondagent.utils.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer, ResourcePoolManager, WorkerType,
                                          _timer, apply_kl_penalty,
                                          compute_response_mask, Role)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.metric import reduce_metrics

from beyondagent.client.llm_client import DashScopeClient
from beyondagent.client.em_client import EMClient
from beyondagent.module.env_manager.env_manager import ParallelEnvManager
from beyondagent.module.task_manager import adapter as task_adapter
from beyondagent.module.task_manager import TaskManager,NaiveTaskObjectiveRetrieval
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory

from beyondagent.utils.tracking import ValidationGenerationsLogger

from beyondagent.module.adv_processor.adca_grpo_pipeline import apply_adca_grpo

from beyondagent.module.exp_manager.exp_manager import ExperienceManager


def parse_reward_from_dataproto(data: DataProto, return_dict=False) -> dict | torch.Tensor:
    """
    Compute reward for a batch of data.

    Args:
        data: DataProto object containing the input data.
        return_dict: Whether to return a dictionary or just the reward tensor.

    Returns:
        Tensor of shape (bs, response_len) if return_dict is False,
        or a dict with 'reward_tensor' and 'reward_extra_info'.
    """
    # Within DataFlow, world.execute() will pass a float score, which will be contained in the DataProto.non_tensor_batch('reward_scores')

    # Initialize reward tensor
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, reslen)  # ⭐ Initialize the reward tensor
    reward_extra_info = defaultdict(list)

    # Batch-level processing
    prompt_ids_batch = data.batch["prompts"]  # (bs, prompt_len)
    prompt_lengths = prompt_ids_batch.shape[-1]

    # Get attention masks for all items
    attention_masks = data.batch["attention_mask"]  # (bs, total_len)
    response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs, )

    # Get reward scores
    reward_scores_list = [item["outcome"] for item in data.non_tensor_batch["reward_scores"]]
    reward_scores = torch.tensor(reward_scores_list, device=reward_tensor.device, dtype=torch.float32)  # (bs, )  # ⭐ Convert reward scores to a tensor

    # Use advanced indexing to assign rewards
    reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
    else:
        return reward_tensor


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler

def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto):
    """
    Merges the `gen_batch_output` with the `batch` based on the `task_id`.

    Args:
        tasks (list): A list of task objects, each containing a `task_id`.
        batch (DataProto): The original batch of data.
        gen_batch_output (DataProto): The generated batch output that needs to be merged.

    Returns:
        DataProto: The final merged batch.
    """
    map_task_id_to_index = {t.task_id:i for i, t in enumerate(tasks)}  # ⭐ Create a mapping from task_id to its index in tasks
    gen_task_task_ids = gen_batch_output.non_tensor_batch['task_ids']
    indices = [map_task_id_to_index[tid] for tid in gen_task_task_ids]
    batch_extend = batch.select_idxs(indices)
    batch_final = batch_extend.union(gen_batch_output)  # ⭐ Merge the selected part of the batch with the gen_batch_output
    return batch_final


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    if scores.dim()!=1:
        logger.warning("scores.dim()!=1")

    with torch.no_grad():
        bsz = scores.shape[0]
        
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
                # no std
                # if llm judge output similar rewards for undistinguishable samples, we may want to reduce its weight according to the batch std
                # scores[i] = scores[i] / (batch_std + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores



def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """
    Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )  # ⭐ Compute advantages and returns for GRPO
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)  # ⭐ Compute advantages and returns for other estimators
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class BeyondAgentRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        train_task_manager:TaskManager,
        val_task_manager:TaskManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # type: ignore
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        collate_fn=None,
        shuffle_trainset:bool=False,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.

        Args:
            config: Configuration object containing various settings.
            tokenizer: Tokenizer used for processing text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping of roles to worker types.
            resource_pool_manager (ResourcePoolManager): Manager for resource pools.
            train_task_manager (TaskManager): Task manager for training tasks.
            val_task_manager (TaskManager): Task manager for validation tasks.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor (optional): Processor for additional data processing.
            reward_fn (optional): Function to compute rewards.
            val_reward_fn (optional): Function to compute validation rewards.
            collate_fn (optional): Function to collate data.
            shuffle_trainset (bool, optional): Whether to shuffle the training dataset. Defaults to False.
            device_name (str, optional): Name of the device to use. Defaults to "cuda".
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"  # ⭐ Ensure the hybrid engine is supported

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"  # ⭐ Ensure ActorRollout role is present in the mapping

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()

        self.em_client: EMClient | None = None
        self.env_manager: ParallelEnvManager | None = None
        self.thread_pool: ThreadPoolExecutor | None = None

        self.train_task_manager=train_task_manager
        self.val_task_manager=val_task_manager
        self._collate_fn=collate_fn

        self._create_dataloader_from_manager(collate_fn, shuffle_trainset)  # ⭐ Create dataloader from the provided manager


    def init_workers(self):
        """
        Initializes distributed training workers using the Ray backend.

        This function creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)

        Args:
            None

        Returns:
            None
        """
        self.resource_pool_manager.create_resource_pool()  # ⭐ Initialize the resource pools

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls,
                                                device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()  # ⭐ Initialize the critic model

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()  # ⭐ Initialize the reference policy model

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()  # ⭐ Initialize the reward model

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()  # ⭐ Initialize the actor rollout model

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from beyondagent.module.trainer.ba_async_llm_server_manager import BaAsyncLLMServerManager
            self.async_rollout_mode = True
            self.async_rollout_manager = BaAsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg)  # ⭐ Create the asynchronous rollout manager

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto

        self.em_client = EMClient(base_url=self.config.experience_maker.base_url)
        self.env_manager = ParallelEnvManager(config=self.config, async_rollout_manager=self.async_rollout_manager, max_parallel=self.config.actor_rollout_ref.rollout.max_env_worker)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)
        self.exp_manager = ExperienceManager(config=self.config)


    def _create_dataloader_from_manager(self, collate_fn, shuffle_trainset: bool = True):
        """
        Creates the train and validation dataloaders.

        1. Check the existence of train and val files and load local tasks from them. If no files given, load tasks from environment (train and val/dev splits).
        2. Use task manager to generate synthetic tasks for trainset, and load the original val dataset.
        3. Use task manager to mix tasks from different sources.
        4. Adapt datasets and create dataloaders used in the trainer.

        Args:
            collate_fn (callable): The function to use for collating data into batches.
            shuffle_trainset (bool, optional): Whether to shuffle the training set. Defaults to True.

        Returns:
            None
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn


        from verl.trainer.main_ppo import create_rl_dataset
        # load train dataset from files or environment
        env_client=EnvClient(self.config.env_service.env_url)
        if self.config.data.train_files is not None:
            train_seed_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
            assert isinstance(train_seed_dataset,RLHFDataset), "train_dataset must be RLHFDataset"
            self.train_task_manager.load_tasks_from_dataset(train_seed_dataset,env_type=self.config.env_service.env_type)
        else:
            self.train_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split="train")
        # load val dataset
        
        if self.config.data.val_files is not None:
            val_seed_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
            assert isinstance(val_seed_dataset,RLHFDataset), "train_dataset must be RLHFDataset"
            self.val_task_manager.load_tasks_from_dataset(val_seed_dataset,env_type=self.config.env_service.env_type)
        else:
            num_loaded_val_tasks = 0
            if 'val_on_test' in os.environ.get("DEBUG_ARG",'') or (self.config.data.val_type == 'test_normal' and self.config.env_service.env_type == "appworld"):
                logger.warning("using test_normal as val dataset")
                num_loaded_val_tasks += self.val_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split="test_normal")
            else:
                for split in ['val','dev']:
                    try:
                        num_loaded_val_tasks += self.val_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split=split)
                    except:
                        logger.warning(f"failed to load val dataset from environment, split={split}. this may be *normal* if your dataset is split into train/dev")    
            
            assert num_loaded_val_tasks > 0, "failed to load val/dev dataset from environment"
        
        self.train_dataset = FullDataset(
            self.train_task_manager,
            self.train_task_manager._mixture_strategy,
            self.train_task_manager._reward_config,
            self.config.task_manager.train_data_path,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
        )
        self.val_dataset = FullDataset(
            self.val_task_manager,
            self.val_task_manager._mixture_strategy,
            self.val_task_manager._reward_config,
            cache_path=None,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
        )

        assert not isinstance(self.train_dataset,AutoReloadDataset), "please disable multiple workers for AutoReloadDataset"
        assert not isinstance(self.val_dataset,AutoReloadDataset), "please disable multiple workers for AutoReloadDataset"
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data,self.train_dataset),
        )  # ⭐ Create the train dataloader with specified parameters

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset) # type: ignore

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )  # ⭐ Create the validation dataloader with specified parameters

        # train dataloader is on-the-fly, so we don't need to check the size
        # assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        if not isinstance(self.train_dataset,IterableDataset):
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        else:
            # FIXME: need a elegant way to set total_training_steps
            total_training_steps = len(self.train_task_manager.seed_tasks)*self.config.trainer.total_epochs
            print(f"Size of train dataloader: unknown, Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")


    def _get_attribution_config(self):
        """
        Retrieves and validates the configuration for attribution-driven credit assignment, including the setup for API retry attempts.

        Returns:
            dict: The validated and possibly updated configuration dictionary.

        Raises:
            ValueError: If the required 'attribution_driven_credit_assignment' block is missing from the configuration.
        """
        if not hasattr(self.config, 'attribution_driven_credit_assignment'):
            raise ValueError("attribution_driven_credit_assignment configuration block is required")

        config = self.config.attribution_driven_credit_assignment

        # set the default api_max_retries
        if not hasattr(config, 'api_max_retries'):
            config.api_max_retries = 200  # ⭐ Set the default number of API retries to 200
            print(f"[attribution_config] Using default api_max_retries: {config.api_max_retries}")

        return config


    def _validate_config(self):
        """
        Validates the configuration settings to ensure they are consistent and meet the necessary requirements for the training process.

        This function checks:
        - The total number of GPUs and their allocation.
        - The total batch size and its divisibility by the minimal possible batch size.
        - Mutual exclusivity of certain micro-batch size parameters.
        - Consistency in actor, critic, and reward model configurations.
        - Other critical settings such as loss aggregation mode and sequence parallelism.

        Raises:
            AssertionError: If any of the configuration settings do not meet the required conditions.
            ValueError: If mutually exclusive parameters are both set or neither is set.
        """
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size  # ⭐ Ensure train_batch_size is at least as large as ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0  # ⭐ Ensure ppo_mini_batch_size is divisible by ppo_micro_batch_size
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus  # ⭐ Ensure sufficient GPU allocation for micro-batch size and sequence parallelism

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size  # ⭐ Ensure train_batch_size is at least as large as ppo_mini_batch_size for critic
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0  # ⭐ Ensure ppo_mini_batch_size is divisible by ppo_micro_batch_size for critic
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus  # ⭐ Ensure sufficient GPU allocation for micro-batch size and sequence parallelism for critic

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            # 0623 yunpeng comment: no need this tool_config_path
            # assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None, "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    ##################
    # ANNI
    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """
        Dumps rollout/validation samples as JSONL.

        Args:
            inputs (list): List of input data.
            outputs (list): List of output data.
            scores (list): List of score data.
            reward_extra_infos_dict (dict): Dictionary containing additional reward information.
            dump_path (str): Path to the directory where the JSONL file will be saved.

        Returns:
            None
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")  # ⭐ Create the filename for the JSONL file

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")  # ⭐ Write the data to the JSONL file

        print(f"Dumped generations to {filename}")


    def _validate(self):
        """
        Validates the model by generating sequences, collecting samples, and storing the results.

        This function processes each batch of validation data, generates outputs, and collects
        input, output, and experience information for further analysis.

        Args:
            None

        Returns:
            None
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for i, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "extras" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extras")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                raise NotImplementedError

            else:
                self.async_rollout_manager.wake_up()
                tasks = [Task(
                            task_id=test_gen_batch.non_tensor_batch["extras"][i]["task_id"],
                            query=test_gen_batch.non_tensor_batch["extras"][i]['new_query'],
                            env_type=self.config.env_service.env_type,
                            open_query=test_gen_batch.non_tensor_batch["extras"][i]['open_query'],
                            # evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'], # avoid potential bugs
                         ) for i in range(len(test_gen_batch))]
                task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="validate")
                print("=" * 10 + "start validate rollout" + "=" * 10)
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}")  # ⭐ Execute the rollout to generate trajectories
                print("=" * 10 + "end validate rollout" + "=" * 10)
                test_output_gen_batch = self.env_manager.to_dataproto(trajectories)
                # test_output_gen_batch_padded = self.explorer_manager.rollout(test_gen_batch_padded)
                # test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            ##################
            # ANNI
            # summary in batch: summary for experience of experience_maker, updating candidate context
            # if self.config.experience_maker.enable_summarizer and self.config.experience_maker.val_summarizer_save:
            #     self.summarize_trajectories_in_batches(trajectories)
            ##################

            # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store original inputs
            input_ids = test_output_gen_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            ##################
            # ANNI
            # # Store extracted experiences
            # experience_infos_dict = test_output_gen_batch.non_tensor_batch["extras"]
            # sample_experiences_dict.extend(experience_infos_dict)
            # sample_experiences_dict:[{'add_exp':bool, 'experience':str}, ...]
            ##################

            # repeat test batch
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)  # ⭐ Evaluate the test batch using the reward function
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # val_data_dir = "experiments/validation_log"
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)  # ⭐ Process the validation metrics for different data sources
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict
    
    # [anni1015] TODO: initialize_exp_pool() 初始化经验池函数

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from beyondagent.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        # 确保训练前正常load参数，避免valid_before_train的效果过差
        self.async_rollout_manager.wake_up()
        self.async_rollout_manager.sleep()
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()  # ⭐ Perform initial validation and get the validation metrics
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # [anni1015] TODO: 调用initialize_exp_pool()函数

        # [0616] qingxu: add `RAY_DEBUG_POST_MORTEM` env var to activate breakpoint debugging
        # vscode_conditional_breakpoint()
        # breakpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        
        for epoch in range(self.config.trainer.total_epochs):
            for i, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "extras" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("extras")
                    batch_extras = deepcopy(batch.non_tensor_batch["extras"])
                else:
                    batch_extras = None
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        trajectories: List[Trajectory] = []
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            # gen_batch_output = self.explorer_manager.rollout(gen_batch)

                            tasks = [Task(
                                        task_id=gen_batch.non_tensor_batch["extras"][i]["task_id"],
                                        query=gen_batch.non_tensor_batch["extras"][i]['new_query'],
                                        env_type=self.config.env_service.env_type,
                                        open_query=gen_batch.non_tensor_batch["extras"][i]['open_query'],
                                        evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'],
                                        ground_truth=gen_batch.non_tensor_batch['extras'][i]['ground_truth']
                                    ) for i in range(len(gen_batch))
                            ]
                            task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="sample")
                            assert len(task_exp_configs)==len(tasks), "{len(task_exp_configs)=}, {len(gen_batch)=}"

                            # TODO enable tracing by jinli 0619
                            print("=" * 10 + "start fit rollout" + "=" * 10)
                            trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}")  # ⭐ Generate trajectories using the environment manager
                            assert len(trajectories)>0, "{len(trajectories)=}?"
                            print("=" * 10 + "end fit rollout" + "=" * 10)

                            gen_batch_output = self.env_manager.to_dataproto(trajectories)
                            ##########
                            # ANNI1027
                            exp_mask_ratio = gen_batch_output.batch["exp_mask"].float().mean()
                            metrics.update({"exp_mask_ratio": exp_mask_ratio.detach().item()})
                            ##########
                            context_time_cost = [x.metadata["context_time_cost"] for x in trajectories if "context_time_cost" in x.metadata]
                            if context_time_cost:
                                metrics.update({
                                  "experience_maker/context_cost_avg":   np.mean(context_time_cost),
                                  "experience_maker/context_cost_max":   np.max(context_time_cost),
                                  "experience_maker/context_cost_min":   np.min(context_time_cost),
                                })

                            print(f"gen_batch_output.info batch.keys={gen_batch_output.batch.keys()}")
                            num_term_traj = sum([traj.is_terminated  for traj in trajectories])
                            num_not_none_traj = sum([len(traj.steps)>0  for traj in trajectories])

                            # gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)  # ⭐ Generate baseline sequences for advantage estimation

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor  # ⭐ Add reward baselines to the batch

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)  # ⭐ Generate unique UIDs for each item in the batch

                    # in the new code, the rollout process generates new extras, which should be merged with the original extra.
                    # by now, they are stored seperately.
                    # assert len(gen_batch_output.non_tensor_batch["extras"].keys()&batch_extras.keys())==0, "extra of extra should not overlap with existing extra...how funny..."
                    batch.non_tensor_batch['original_extras']=batch_extras  # ⭐ Store original extras before scaling
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output)  # ⭐ Merge generated batch with the current batch

                    batch.batch["response_mask"] = compute_response_mask(batch)  # ⭐ Compute and add response mask to the batch

                    ##################
                    # [anni1015] TODO: 保留异步功能，拆分成两个函数
                    # summary for experience of experience_maker, updating candidate context
                    experience_metrics = self.exp_manager.schedule_exp_update(self.global_steps, trajectories)
                    metrics.update(experience_metrics)
                    ##################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)  # ⭐ Balance the batch to distribute valid tokens evenly

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()  # ⭐ Compute and store the global token numbers

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)  # ⭐ Compute reward scores using the reward model
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)  # ⭐ Compute rewards and extra information

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  # ⭐ Compute old log probabilities
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # ⭐ Compute reference log probabilities
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)  # ⭐ Compute values using the critic model
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)  # ⭐ Get the reward tensor and extra info from the async call
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)  # ⭐ Apply KL divergence penalty
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        if os.environ.get("DEBUG_ARG","").find("disable_adv_std")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change norm_adv_by_std_in_grpo from True to False, using batch std!")
                            norm_adv_by_std_in_grpo = False

                        # call the original compute_advantage for compatibility
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )
                        # shuchang
                        # ==================== Begin ADCA GRPO  ====================
                        attribution_cfg = self._get_attribution_config()
                        if getattr(attribution_cfg, 'enable', False):
                            batch, adca_metrics = apply_adca_grpo(
                                batch=batch,
                                attribution_cfg=attribution_cfg,
                                tokenizer=self.tokenizer,
                                global_steps=self.global_steps,
                                epoch=epoch,
                                i=i,
                            )
                            metrics.update(adca_metrics)
                        # ==================== End ADCA GRPO ====================
                        # Apply decay factor of 0.5 to non_tensor_batch['extras'][i]['evaluator'] != 'env'
                        if os.environ.get("DEBUG_ARG","").find("synth_decay")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                            assert 'extras' in batch.non_tensor_batch
                            if 'extras' in batch.non_tensor_batch:
                                for i in range(len(batch.non_tensor_batch['extras'])):
                                    assert 'evaluator' in batch.non_tensor_batch['extras'][i]
                                    evaluator = batch.non_tensor_batch['extras'][i]['evaluator']
                                    if evaluator != 'env':
                                        batch.batch["advantages"][i] *= 0.5  # ⭐ Apply decay factor to synthetic data

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)  # ⭐ Update the critic model
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)  # ⭐ Update the actor with the new batch
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )  # ⭐ Dump the generated experiences and trajectories

                            # save original trajectory
                            filename = os.path.join(rollout_data_dir, f"traj_{self.global_steps}.jsonl")
                            with open(filename, "w") as f:
                                for traj in trajectories:
                                    f.write(traj.json() + "\n")
                            # save tasks
                            filename = os.path.join(rollout_data_dir, f"task_{self.global_steps}.jsonl")
                            with open(filename,"w") as f:
                                for task in tasks: # this must be bounded # type: ignore
                                    f.write(task.json() + "\n")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()  # ⭐ Validate the model and collect validation metrics
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()  # ⭐ Save the current state of the model as a checkpoint

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/num_not_none_traj": num_not_none_traj,
                        "training/num_term_traj": num_term_traj
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)  # ⭐ Log the collected metrics

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

            # we expect the train dataset is fully explored at the beginning, no reload needed.
            # if isinstance(self.train_dataset, FullDataset):
            #     self.train_dataset.reload()
            if os.environ.get("DEBUG_ARG",'').find("ratio_decay")!=-1:
                from beyondagent.module.task_manager.data_mixture import UnifiedMixtureStrategy
                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                assert isinstance(self.train_dataset._mixture_strategy,UnifiedMixtureStrategy)
                self.train_dataset._mixture_strategy._synthetic_ratio-=1/5 # initial 1, 0 at about epoch 5 (about step 30)
            self.train_dataset.update()  # ⭐ Update the training dataset for the next iteration


