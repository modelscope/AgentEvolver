# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import time
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


from beyondagent.client.em_client import EMClient
from beyondagent.module.env_manager.env_manager import ParallelEnvManager
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory
from beyondagent.utils.message import send_train_message
from beyondagent.utils.vscode_breakpoint import vscode_conditional_breakpoint
from best_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem
from context_manager_templates.cmt_linear import CMTLinear
import os
import json
from typing import List, Dict, Union, Type
from loguru import logger

WorkerType = type[Worker]


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
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, reslen)
    reward_extra_info = defaultdict(list)

    # Batch-level processing
    prompt_ids_batch = data.batch["prompts"]  # (bs, prompt_len)
    prompt_lengths = prompt_ids_batch.shape[-1]

    # Get attention masks for all items
    attention_masks = data.batch["attention_mask"]  # (bs, total_len)
    response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs, )

    # Get reward scores
    reward_scores_list = [item["outcome"] for item in data.non_tensor_batch["reward_scores"]]
    reward_scores = torch.tensor(reward_scores_list, device=reward_tensor.device, dtype=torch.float32)  # (bs, )

    # Use advanced indexing to assign rewards
    reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
    else:
        return reward_tensor

def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto):
    """
    Union the gen_batch_output with the batch based on task_id.

    batch :
        - task_1
        - task_1
        - task_2
        - task_2
        - task_3
        - task_3

    gen_batch_output
        - task_1
        - task_1
        - task_1
        - task_1
        - task_3
        - task_3

    """
    map_task_id_to_index = {t.task_id:i for i, t in enumerate(tasks)}
    gen_task_task_ids = gen_batch_output.non_tensor_batch['task_ids']
    indices = [map_task_id_to_index[tid] for tid in gen_task_task_ids]
    batch_extend = batch.select_idxs(indices)
    batch_final = batch_extend.union(gen_batch_output)
    return batch_final


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    special_norm: bool = True,
    config: Optional[AlgoConfig] = None,
    ratios: torch.Tensor = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
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
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        try:
            from vsdb import bp
        except ImportError:
            def bp(*args, **kwargs): pass
        bp('mask')
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # If multi-turn, replace the mask with the relevant part of loss_mask
        # Get length from the initial response mask
        response_length = grpo_calculation_mask.size(1)
        # This mask is the one intended for GRPO
        grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        # advantages, returns, std_ge1_freq = core_algos.compute_grpo_outcome_advantage(
        #     token_level_rewards=data.batch["token_level_rewards"],
        #     response_mask=grpo_calculation_mask,
        #     index=data.non_tensor_batch["uid"],
        #     norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        # )
        advantages, returns = compute_grpo_outcome_advantage_local(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            special_norm=special_norm,
        )
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
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # This part is for SeeUPO-ALGs
    if ratios is not None:
        ratios_expanded = ratios.unsqueeze(-1)  # [worker] -> [worker, 1]
        data.batch["advantages"] = data.batch["advantages"] * ratios_expanded.detach()
    return data

def compute_grpo_outcome_advantage_local(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    special_norm: bool = False,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
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

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        if special_norm:
            # pass
            assert not norm_adv_by_std_in_grpo
            if torch.std(scores) > 1e-6:
                scores = (scores - torch.mean(scores)) / (torch.std(scores))
            else:
                scores = (scores - torch.mean(scores))
            scores = torch.clamp(scores, -3, 3)
        scores = scores.unsqueeze(-1) * response_mask


    return scores, scores


def update_ratios(
    ratios: torch.Tensor,
    last_agent_log_prob: torch.Tensor,
    now_agent_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
):
    """
    Compute the ratios for SeeUPO sequential multi-turn training.
    ratios are sequence-level ratios
    """

    assert config is not None
    clip_ratio_low = config.adv_clip_ratio_low if config.adv_clip_ratio_low is not None else config.adv_clip_ratio
    clip_ratio_high = config.adv_clip_ratio_high if config.adv_clip_ratio_high is not None else config.adv_clip_ratio

    negative_approx_kl = now_agent_log_prob - last_agent_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level(original version for GSPO):
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = negative_approx_kl_seq.detach()
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, min=-10.0, max=10.0)  # clamp for numerical stability
    
    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)
    seq_importance_ratio = torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)


    updated_ratio = ratios * seq_importance_ratio if ratios is not None else seq_importance_ratio
    # Clip the accumulated sequence-level ratio again for numerical and training stability
    updated_ratio = torch.clamp(updated_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)


    return updated_ratio

def get_agent_update_order(
    max_agents: int,
    mode: Optional[str] = None,
    custom_agent_order: Optional[list[int]] = None,
) -> list[int]:
    """Get the update order for agents.

    Args:
        max_agents: total number of agents (turns)
        mode: update mode ('sequential', 'reverse', 'random', 'custom'). None or unknown values fall back to sequential.
        custom_agent_order: when mode is 'custom', permutation of agent ids; length must equal max_agents.

    Returns:
        The update order (list of agent indices) for agents
    """
    if max_agents <= 0:
        return []

    if mode is None or mode == "sequential":
        return list(range(max_agents))
    if mode == "reverse":
        return list(range(max_agents - 1, -1, -1))
    if mode == "random":
        order = list(range(max_agents))
        random.shuffle(order)
        return order
    if mode == "custom":
        if custom_agent_order:
            co = list(custom_agent_order)
            if len(co) == max_agents:
                return co
        return list(range(max_agents))

    return list(range(max_agents))

class BeyondAgentRayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"


        if self.hybrid_engine:
            vscode_conditional_breakpoint('init')  # uses local beyondagent.utils.vscode_breakpoint
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
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

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

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

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
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
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            # sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
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
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            self.async_rollout_manager.wake_up()
            main_val_dataset, test_normal_dataset, test_chanllenge_dataset = self.get_eval_dataset()

            print("=" * 10 + "start validate rollout" + "=" * 10)
            trajectories, tasks, val_metrics = self.eval_dataset(
                target_dataset=main_val_dataset,
                target_dataset_name="main_val_dataset",
                mode="validate",
                epoch=f"test.1"
            )
            print("=" * 10 + "end validate rollout" + "=" * 10)
            # TODO 
            test_output_gen_batch = self.env_manager.to_dataproto(trajectories)
            self.async_rollout_manager.sleep()
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            tasks = tasks[:len(main_val_dataset)]
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch)
            # test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            break   # hack to escape the loop after one batch

        sample_inputs = [m['messages'][0]['content'] for m in test_batch.non_tensor_batch['messages']]
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        metric_dict = val_metrics

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
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
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
            from verl.experimental.agent_loop.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            agent_loop_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )
            self.async_server_list = agent_loop_manager.async_llm_servers
            self.async_rollout_manager = AsyncLLMServerManager(self.config, self.async_server_list)

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto
        from concurrent.futures import ThreadPoolExecutor
        self.em_client = EMClient(base_url=self.config.experience_maker.base_url)
        self.env_manager = ParallelEnvManager(config=self.config, async_rollout_manager=self.async_rollout_manager, max_parallel=self.config.actor_rollout_ref.rollout.max_env_worker, tokenizer=self.tokenizer)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # breakpoint()

        

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.tracking_logger = logger
        self.global_steps = 0


        # load checkpoint before doing anything
        self._load_checkpoint()

        # wake and sleep to enforce param sync
        self.async_rollout_manager.wake_up()
        self.async_rollout_manager.sleep()
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return


        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        # breakpoint()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                init_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                init_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(init_batch.batch))], dtype=object
                )

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in init_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in init_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in init_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in init_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in init_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in init_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                if "extras" in init_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("extras")

                gen_batch = init_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps

                # breakpoint()

                with marked_timer("step", timing_raw):
                    # generate a batch
                    print("=== + rollout step begin ===")
                    with marked_timer("gen", timing_raw, color="red"):
                        assert self.async_rollout_mode
                        print("=== wake up begin ===")
                        self.async_rollout_manager.wake_up()
                        print("=== wake up end ===")
                        tasks = [
                            Task(
                                task_id=str(gen_batch.non_tensor_batch["extras"][i]["task_id"]),
                                query=gen_batch.non_tensor_batch["raw_prompt"][i],
                                env_type=self.config.env_service.env_type
                            ) for i in range(len(gen_batch))
                        ]
                        print([gen_batch.non_tensor_batch["extras"][i]["task_id"] for i in range(len(gen_batch))])
                        print("=" * 10 + "start fit rollout" + "=" * 10)
                        trajectories: List[CMTLinear] = self.env_manager.rollout(tasks, mode="sample", epoch=f"train.{epoch}")
                        print("=" * 10 + "end fit rollout" + "=" * 10)
                        print("begin to convert trajectories to dataproto")
                        if self.config.algorithm.sequential_update:
                            gen_batch_output = self.env_manager.to_dataproto_list(trajectories)
                        else:
                            gen_batch_output = self.env_manager.to_dataproto(trajectories)
                        print("end convertion")
                        context_time_cost = [traj.context_time_cost for traj in trajectories]
                        if context_time_cost:
                            metrics.update({
                                "context_cost_avg":   np.mean(context_time_cost),
                                "context_cost_max":   np.max(context_time_cost),
                                "context_cost_min":   np.min(context_time_cost),
                            })
                        success_rate = [traj.reward.success_rate for traj in trajectories]
                        madness_rate = [traj.reward.madness for traj in trajectories]
                        metrics.update({
                            "critic/madness_rate": np.mean(madness_rate),
                            "critic/success_rate": np.mean(success_rate),
                            "critic/real_success_rate": np.mean(trajectories[0].current_batch_success_rate),
                        })
                        if self.config.algorithm.sequential_update:
                            print(f"gen_batch_output.info init_batch.keys={gen_batch_output[-1].batch.keys()}")
                        else:
                            print(f"gen_batch_output.info init_batch.keys={gen_batch_output.batch.keys()}")
                        num_term_traj = sum([traj.is_terminated  for traj in trajectories])
                        # num_not_none_traj = sum([len(traj.steps)>0  for traj in trajectories])

                        # gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        self.async_rollout_manager.sleep()
                    print("=== - rollout step end ===")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError("REMAX is not supported in GRPO yet.")
                    
                    if self.config.algorithm.sequential_update:
                        print("===  begin sequential_update mode===")
                        max_agent_num = int(self.config.actor_rollout_ref.rollout.multi_turn.max_steps)
                        update_mode = self.config.algorithm.update_order if self.config.algorithm.update_order else None
                        custom_agent_order = getattr(
                            self.config.algorithm, "custom_agent_order", None
                        )
                        update_order = get_agent_update_order(
                            max_agents=max_agent_num,
                            mode=update_mode,
                            custom_agent_order=custom_agent_order,
                        )
                    else: 
                        print("===  begin normal_update mode===")
                        update_order = [0]
                    
                    seeupo_ratios = None

                    import copy
                    total_std_freq = 0
                    for agent_id in update_order:
                        if agent_id > 0:
                            print("=== Agent {} - update step begin ===".format(agent_id))

                        batch = copy.deepcopy(init_batch)

                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(init_batch.batch))], dtype=object)
                        # repeat to align with repeated responses in rollout
                        # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # batch = batch.union(gen_batch_output)
                        # --->
                        batch = union_gen_batch_via_task_id(tasks, init_batch, gen_batch_output[agent_id]) if self.config.algorithm.sequential_update else union_gen_batch_via_task_id(tasks, batch, gen_batch_output)
                        batch.batch["response_mask"] = compute_response_mask(batch)

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                raise NotImplementedError("launch_reward_fn_async is not supported in GRPO yet.")
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            vscode_conditional_breakpoint(tag='MAJOR_LOOP')
                            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            assert not torch.isnan(entropy_loss).item(), "Entropy loss should not be NaN, something must have gone terribly wrong."
                            old_log_prob_metrics = {"actor/entropy": entropy_loss.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process

                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor
                            special_norm = self.config.algorithm.get(
                                "special_norm", False
                            )  


                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                special_norm=special_norm,
                                config=self.config.algorithm,
                                ratios=seeupo_ratios,
                            )

                            


                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup and update actor
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)


                        # update ratios
                        if self.config.algorithm.sequential_update:
                            if self.config.algorithm.adv_updator == "seeupo":
                                # recalculate new_log_prob
                                new_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                # calculate ratios
                                seeupo_ratios = update_ratios(
                                            ratios=seeupo_ratios,
                                            last_agent_log_prob=batch.batch['old_log_probs'],
                                            now_agent_log_prob=new_log_prob.batch['old_log_probs'],
                                            response_mask=batch.batch["response_mask"],
                                            config=self.config.algorithm,
                                        )
                                pass

                        
                    # --- ---------------------------------------------------------------------------------------------
                    

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


    def eval_dataset(self, target_dataset, target_dataset_name, mode, epoch):
        """
        Evaluate a dataset by running rollouts and computing task completion metrics.

        Args:
            target_dataset: The dataset to evaluate
            target_dataset_name: Name for logging purposes
            mode: Evaluation mode ("sample" or "validate")
            epoch: Current epoch for logging

        Returns:
            Tuple of (cmts, tasks) containing trajectory results and task definitions
        """
        pass_n = self.config.trainer.eval_pass_n
        # if pass_n == 1:
        #     return self.eval_dataset_legacy(target_dataset, target_dataset_name, mode, epoch)

        tasks = []
        for _ in range(pass_n):
            tasks += [
                Task(
                    task_id=str(dat['extras']['task_id']),
                    query=dat['raw_prompt'],
                    env_type=self.config.env_service.env_type
                ) for dat in target_dataset
            ]

        cmts = self.env_manager.rollout(tasks=tasks, mode=mode, epoch=epoch) # "sample" or "validate"
        task_results = {}
        for _cmt in cmts:
            reward = _cmt.reward.outcome
            task_id = _cmt.task_id
            if task_id not in task_results:
                task_results[task_id] = {}
                task_results[task_id]['reward_arr'] = []
                task_results[task_id]['tag_arr'] = []
            if reward >= 1:
                _cmt.tag = "success"
            elif reward == 0:
                _cmt.tag = "failure"
            else:
                _cmt.tag = "half_success"
            task_results[task_id]['tag_arr'] += [_cmt.tag]
            task_results[task_id]['reward_arr'] += [_cmt.reward.outcome]
            task_results[task_id]['scenario'] = task_id.split('_')[0]

        task_scenario = [task_id.split('_')[0] for task_id in task_results.keys()]
        set_scenarios = set(task_scenario)
        num_scenarios = len(set_scenarios)

        repeated_success_tasks = 0
        num_all_success_tasks = 0   # tasks with success on all n runs
        num_pass_n_tasks = 0    # tasks with at least one success in n runs

        # reward-related aggregates
        all_task_rewards_mean = []  # per-task mean reward across n runs
        all_task_rewards_first = []  # per-task reward from the first run

        for task_id, task_outcomes in task_results.items():
            # update num_all_success_tasks / num_pass_n_tasks from tag_arr
            assert len(task_outcomes['tag_arr']) == pass_n
            assert len(task_outcomes['reward_arr']) == pass_n

            if all(tag == "success" for tag in task_outcomes['tag_arr']):
                num_all_success_tasks += 1
            if any(tag == "success" for tag in task_outcomes['tag_arr']):
                num_pass_n_tasks += 1
            repeated_success_tasks += task_outcomes['tag_arr'].count("success")

            # collect per-task reward stats
            task_reward_mean = sum(task_outcomes['reward_arr']) / len(task_outcomes['reward_arr'])
            task_reward_first = task_outcomes['reward_arr'][0]
            all_task_rewards_mean.append(task_reward_mean)
            all_task_rewards_first.append(task_reward_first)

        num_all_success_scenarios = 0  # scenarios where every task succeeds on all n runs
        num_pass_n_scenarios = 0  # scenarios where every task has at least one success in n runs
        repeated_num_pass_1_scenarios = 0   # count run indices x where all tasks in a scenario succeed on run x
        for scenario in set_scenarios:
            scenario_task_results = {task_id: task_outcomes for task_id, task_outcomes in task_results.items() if task_outcomes['scenario'] == scenario}
            # num_all_success_scenarios
            if all(all(tag == "success" for tag in task_outcomes['tag_arr']) for task_outcomes in scenario_task_results.values()):
                num_all_success_scenarios += 1
            # num_pass_n_scenarios
            if all(any(tag == "success" for tag in task_outcomes['tag_arr']) for task_outcomes in scenario_task_results.values()):
                num_pass_n_scenarios += 1
            # num_pass_1_scenarios
            for x in range(pass_n):
                if all(task_outcomes['tag_arr'][x]=='success' for task_outcomes in scenario_task_results.values()):
                    repeated_num_pass_1_scenarios += 1

        # logging
        nested_items_print_buffer = {}
        # rebuild task_scenario to align with cmts length
        task_scenario_for_cmts = [_cmt.task_id.split('_')[0] for _cmt in cmts]
        for _cmt, scenario in zip(cmts, task_scenario_for_cmts):
            task_outcome = _cmt.tag
            selectors = [scenario, _cmt.task_id, task_outcome]
            cmt_content_render = _cmt.group_render_token_log()
            reward = _cmt.reward.outcome
            nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
                item_id=f"item",
                outcome=task_outcome,
                reward=f"{float(reward):.3f}",
                content=SeqItem(
                    text = cmt_content_render['text_arr'],  # display text
                    title = cmt_content_render['text_arr'], # hover title
                    count = cmt_content_render['input_id_arr'], # highlighted spans
                    color = cmt_content_render['loss_mask_color_arr']   # colors
                )
            )

        print_nested(nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"Evaluation: {target_dataset_name}",
            mod="evaluation",
            narrow=False,
            attach="copy this message to clipboard"
        )

        rewards = [ _cmt.reward.outcome for _cmt in cmts ]
        num_tasks = len(task_results)
        assert num_tasks == len(cmts) // pass_n

        # mean@n reward metrics
        reward_mean_at_1 = sum(all_task_rewards_first) / len(all_task_rewards_first) if all_task_rewards_first else 0
        reward_mean_at_n = sum(all_task_rewards_mean) / len(all_task_rewards_mean) if all_task_rewards_mean else 0

        val_metrics = {
            "target dataset name": target_dataset_name,
            "pass_n": pass_n,

            "total_tasks": len(task_results),
            "num_all_success_tasks": num_all_success_tasks,
            f"num_pass_n_tasks(pass@{pass_n})": num_pass_n_tasks,

            "num_scenarios": num_scenarios,
            "num_all_success_scenarios": num_all_success_scenarios,
            f"num_pass_n_scenarios(pass@{pass_n})": num_pass_n_scenarios,

            "TGC@1":                           repeated_success_tasks / (num_tasks * pass_n),
            f"TGC@{pass_n}":                         num_pass_n_tasks / num_tasks,
            f"TGC@{pass_n}-all-pass":           num_all_success_tasks / num_tasks,
            f"SGC@1":                   repeated_num_pass_1_scenarios / (num_scenarios * pass_n),
            f"SGC@{pass_n}":                     num_pass_n_scenarios / num_scenarios,
            f"SGC@{pass_n}-all-pass":       num_all_success_scenarios / num_scenarios,

            # legacy mean reward over all rollouts
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,

            # mean@1 and mean@n over per-task aggregates
            "reward/mean@1": reward_mean_at_1,
            f"reward/mean@{pass_n}": reward_mean_at_n,

        }

        print_dict(val_metrics, narrow=True, header=target_dataset_name, mod="evaluation")

        self.tracking_logger.log(data=val_metrics, step=self.global_steps)

        return cmts, tasks, val_metrics


    def get_eval_dataset(self):
        from beyondagent.main_ppo import create_rl_dataset
        if self.config.env_service.env_type == "appworld":
            if hasattr(self, 'main_val_dataset'):
                return self.main_val_dataset, None, None
            else:
                config = self.config
                appworld_dataset_base = os.path.dirname(config.data.val_files)
                dev_dataset_path = os.path.join(appworld_dataset_base, "dev.parquet")
                test_normal_dataset_path = os.path.join(appworld_dataset_base, "test_normal.parquet")
                test_chanllenge_dataset_path = os.path.join(appworld_dataset_base, "test_challenge.parquet")
                test_dev_dataset = create_rl_dataset(dev_dataset_path, config.data, tokenizer=self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                test_normal_dataset = create_rl_dataset(test_normal_dataset_path, config.data, tokenizer=self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                # test_chanllenge_dataset = create_rl_dataset(test_chanllenge_dataset_path, config.data, tokenizer=self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                if config.data.fast_eval:  # use a small eval subset
                    self.main_val_dataset = test_dev_dataset
                    return self.main_val_dataset, None, None
                else:
                    self.main_val_dataset = test_normal_dataset
                    return self.main_val_dataset, None, None

        elif self.config.env_service.env_type == "webshop":
            if hasattr(self, 'main_val_dataset'):
                return self.main_val_dataset, None, None
            else:
                config = self.config
                self.main_val_dataset = create_rl_dataset(config.data.val_files, config.data, self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                # self.test_normal_dataset = create_rl_dataset(config.data.val_files, config.data, self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                if config.data.fast_eval:  # use a small eval subset
                    self.main_val_dataset.dataframe = self.main_val_dataset.dataframe.shuffle(seed=42).select(range(100))   # limit to 100 samples
                    return self.main_val_dataset, None, None
                else:
                    self.main_val_dataset.dataframe = self.main_val_dataset.dataframe.shuffle(seed=42).select(range(500))   # limit to 100 samples
                    return self.main_val_dataset, None, None
        
        elif self.config.env_service.env_type == "bfcl":
            if hasattr(self, 'main_val_dataset'):
                return self.main_val_dataset, None, None
            else:
                config = self.config
                self.main_val_dataset = create_rl_dataset(config.data.val_files, config.data, self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                # self.test_normal_dataset = create_rl_dataset(config.data.val_files, config.data, self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                if config.data.fast_eval:  # use a small eval subset
                    self.main_val_dataset.dataframe = self.main_val_dataset.dataframe.shuffle(seed=42).select(range(100))   # limit to 100 samples
                    return self.main_val_dataset, None, None
                else:
                    self.main_val_dataset.dataframe = self.main_val_dataset.dataframe.shuffle(seed=42).select(range(500))   # limit to 100 samples
                    return self.main_val_dataset, None, None

        elif self.config.env_service.env_type == "crafters":
            if hasattr(self, 'main_val_dataset'):
                return self.main_val_dataset, None, None
            else:
                config = self.config
                self.main_val_dataset = create_rl_dataset(config.data.val_files, config.data, self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                # self.test_normal_dataset = create_rl_dataset(config.data.val_files, config.data, self.tokenizer, processor=None, is_train=False, env_config=config.env_service)
                if config.data.fast_eval:  # use a small eval subset
                    self.main_val_dataset.dataframe = self.main_val_dataset.dataframe.shuffle(seed=42).select(range(100))   # limit to 100 samples
                    return self.main_val_dataset, None, None
                else:
                    self.main_val_dataset.dataframe = self.main_val_dataset.dataframe.shuffle(seed=42).select(range(500))   # limit to 100 samples
                    return self.main_val_dataset, None, None

        else:
            raise NotImplementedError