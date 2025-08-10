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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from pprint import pprint
from typing import List

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer,
                                          _timer, apply_kl_penalty,
                                          compute_advantage,
                                          compute_response_mask, Role)
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics

from beyondagent.client.em_client import EMClient
from beyondagent.module.env_manager.env_manager import ParallelEnvManager
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory
from beyondagent.utils.plot_entropy import log_token_entropy
from beyondagent.module.advantage_assignment.token_level_assignment import _add_entropy_mask as _add_advantage_mask
from beyondagent.module.advantage_assignment.semantic_level_assignment import (
    evaluate_step_flags, apply_step_mask
)
from beyondagent.utils.advantage_structure_checker import (
    debug_advantage_structure, 
    validate_grpo_advantage_structure
)
from beyondagent.utils.step_parser import verify_step_alignment, verify_step_content
# from beyondagent.module.credit_assign import AssignmentConfig, CreditAssigner

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

class BeyondAgentRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.em_client: EMClient | None = None
        self.env_manager: ParallelEnvManager | None = None
        self.thread_pool: ThreadPoolExecutor | None = None

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
            from beyondagent.module.trainer.ba_async_llm_server_manager import BaAsyncLLMServerManager
            self.async_rollout_mode = True
            self.async_rollout_manager = BaAsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg)

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto

        self.em_client = EMClient(base_url=self.config.experience_maker.base_url)
        self.env_manager = ParallelEnvManager(config=self.config, async_rollout_manager=self.async_rollout_manager, max_parallel=self.config.actor_rollout_ref.rollout.max_env_worker)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)

    def _get_semantic_config(self):
        """
        获取语义评估配置 - 支持API重试配置
        """
        if not hasattr(self.config, 'semantic_advantage'):
            raise ValueError("semantic_advantage configuration block is required")
        
        config = self.config.semantic_advantage
        
        # 设置默认的API重试次数
        if not hasattr(config, 'api_max_retries'):
            config.api_max_retries = 200  # 默认200次重试
            print(f"[semantic_config] Using default api_max_retries: {config.api_max_retries}")
        
        return config

    def _validate_config(self):
        # 0623 yunpeng add. keep the same as the original func except for the param of tool_config_path
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
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

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
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

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

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for i, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

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
                            query=test_gen_batch.non_tensor_batch["raw_prompt"][i],
                            env_type=self.config.env_service.env_type
                         ) for i in range(len(test_gen_batch))]
                print("=" * 10 + "start validate rollout" + "=" * 10)
                trajectories = self.env_manager.rollout(tasks, mode="validate", epoch=f"test.1.{i}")
                print("=" * 10 + "end validate rollout" + "=" * 10)
                test_output_gen_batch = self.env_manager.to_dataproto(trajectories)
                # test_output_gen_batch_padded = self.explorer_manager.rollout(test_gen_batch_padded)
                # test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

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

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
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

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
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

    def _init_credit_assigner(self):
        raw = self._get_semantic_config()               # 原先解析 config 的方法
        cfg = AssignmentConfig(backend=raw.evaluation_type,
                              model=raw.model,
                              concurrent=raw.concurrent,
                              log_dir=pathlib.Path(self.config.trainer.llm_evaluation_log_dir))
        self.credit_assigner = CreditAssigner(cfg)


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # [0616] qingxu: add `RAY_DEBUG_POST_MORTEM` env var to activate breakpoint debugging
        # vscode_conditional_breakpoint()
        # breakpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        # # TODO shuchang 尝试在fit函数中添加让manager
        # self.async_rollout_manager.wake_up()
        # self.async_rollout_manager.sleep()
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
                                        query=gen_batch.non_tensor_batch["raw_prompt"][i],
                                        env_type=self.config.env_service.env_type
                                    ) for i in range(len(gen_batch))]

                            # TODO enable tracing by jinli 0619
                            print("=" * 10 + "start fit rollout" + "=" * 10)
                            trajectories = self.env_manager.rollout(tasks, mode="sample", epoch=f"train.{epoch}.{i}")
                            print("=" * 10 + "end fit rollout" + "=" * 10)

                            gen_batch_output = self.env_manager.to_dataproto(trajectories)
                            context_time_cost = [x.metadata["context_time_cost"] for x in trajectories if "context_time_cost" in x.metadata]
                            if context_time_cost:
                                metrics.update({
                                  "context_cost_avg":   np.mean(context_time_cost),
                                  "context_cost_max":   np.max(context_time_cost),
                                  "context_cost_min":   np.min(context_time_cost),
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
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # breakpoint()

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # summary for experience of experience_maker, updating candidate context
                    summary_task = None
                    if self.config.experience_maker.enable_summarizer:
                        summary_task = self.thread_pool.submit(self.em_client.call_summarizer,
                                                               trajectories=trajectories,
                                                               workspace_id=self.config.experience_maker.workspace_id)
                        print("async submit summary_task~")
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # old_log_prob.batch.pop("entropys")
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
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

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

                        # # ===========  0714 shuchang: add semantic mask  =========== 
                        # print("^^^^^^^^^^^^^^^^^ start evaluate_step_flags")
                        # step_flags = evaluate_step_flags(
                        #     tokenizer  = self.tokenizer,
                        #     batch      = batch,
                        # )     # list[list[bool]]
                        # print("^^^^^^^^^^^^^^^^^ end evaluate_step_flags", f"shape: {len(step_flags)} samples")
                        # apply_step_mask(
                        #     batch        = batch,
                        #     step_flags   = step_flags,
                        #     consistent_scale   = semantic_config.consistent_scale,
                        #     pos_unconsistent_scale    = semantic_config.pos_unconsistent_scale,
                        #     neg_unconsistent_scale = semantic_config.neg_unconsistent_scale,
                        # )                  # breakpoint()
                        # print("$$$$$$$$$$$$$$$$$$$$ ")
                        
                        # # ===========  0804 shuchang: add semantic mask  =========== 
                        # prompts    = self.tokenizer.batch_decode(batch.batch["prompts"],    skip_special_tokens=True)
                        # references = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        # mask       = batch.batch["response_mask"].float()

                        # adv, stats = self.credit_assigner.step_level(prompts, references, mask, self.global_steps)

                        # # broadcast sample-level scores到 token 维度并乘回去
                        # batch.batch["advantages"] = batch.batch["advantages"] * adv.unsqueeze(1)

                        # # 你若想监控统计量可加：
                        # metrics.update({f"sem_credit/{k}": v for k, v in stats.items()})
                        
                        # 🔧 统一的语义评估配置访问
                        semantic_config = self._get_semantic_config()
                        enable_semantic_eval = semantic_config.enable
                        
                        if enable_semantic_eval:
                            if not verify_step_alignment(batch, self.tokenizer, self.global_steps):
                                raise RuntimeError("Step alignment check failed!")
                            for sample_idx in range(min(3, len(batch.batch["prompts"]))):
                                verify_step_content(batch, self.tokenizer, sample_idx)
                            print("^^^^^^^^^^^^^^^^^ start parallel semantic processing")
                            print(f"[Config] Semantic evaluation config: {semantic_config}")
                            
                            from beyondagent.module.advantage_assignment.parallel_semantic_assignment import (
                                ParallelSemanticProcessor
                            )

                            # 创建处理器（可以作为类成员变量避免重复创建）
                            if not hasattr(self, '_semantic_processor'):
                                self._semantic_processor = ParallelSemanticProcessor(
                                    max_concurrent=semantic_config.concurrent,
                                    model_name=semantic_config.model,
                                    evaluation_type=semantic_config.evaluation_type,  # 🔧 新增：支持评估类型
                                    api_max_retries=getattr(semantic_config, 'api_max_retries', 200)  # 支持配置重试次数
                                )
                            
                            # 根据配置选择mask类型
                            response_length = batch.batch["responses"].size(1)
                            mask_type = semantic_config.mask_type
                            
                            print(f"[semantic_eval] Using mask type: {mask_type}")
                            print(f"[semantic_eval] Using evaluation type: {semantic_config.evaluation_type}")
                            
                            if mask_type == "response_mask":
                                selected_mask = batch.batch["response_mask"]
                                print(f"[semantic_eval] response_mask shape: {selected_mask.shape}")
                            elif mask_type == "loss_mask":
                                selected_mask = batch.batch["loss_mask"][:, -response_length:]
                                print(f"[semantic_eval] loss_mask shape: {selected_mask.shape}")
                            else:
                                supported_types = ["response_mask", "loss_mask"]
                                raise ValueError(f"Unsupported mask_type: '{mask_type}'. Supported types: {supported_types}")
                            
                            # 🆕 新增：构造保存目录和相关信息
                            llm_eval_save_dir = None
                            if hasattr(self.config.trainer, 'llm_evaluation_log_dir') and self.config.trainer.llm_evaluation_log_dir:
                                # 从配置文件获取保存目录
                                base_save_dir = self.config.trainer.llm_evaluation_log_dir
                                # 创建按global_step分组的子目录
                                llm_eval_save_dir = f"{base_save_dir}/step_{self.global_steps:06d}"
                                print(f"[semantic_eval] LLM evaluation records will be saved to: {llm_eval_save_dir}")
                            else:
                                print(f"[semantic_eval] No llm_evaluation_log_dir configured, evaluation records will not be saved")
                            
                            # 构造epoch信息
                            epoch_info = f"train.{epoch}.{i}"
                            
                            # 使用统一的配置获取缩放参数
                            semantic_stats = self._semantic_processor.process_batch_sync(
                                tokenizer=self.tokenizer,
                                batch=batch,
                                consistent_scale=semantic_config.consistent_scale,
                                pos_unconsistent_scale=semantic_config.pos_unconsistent_scale,
                                neg_unconsistent_scale=semantic_config.neg_unconsistent_scale,
                                mask_tensor=selected_mask,
                                save_dir=llm_eval_save_dir,  # 🆕 传递保存目录
                                global_step=self.global_steps,  # 🆕 传递全局步数
                                epoch=epoch_info,  # 🆕 传递epoch信息
                            )

                            # 添加到metrics中
                            metrics.update({
                                "semantic_eval/total_time": semantic_stats["total_processing_time"],
                                "semantic_eval/api_time": semantic_stats["evaluation_time"],
                                "semantic_eval/mask_time": semantic_stats["mask_application_time"],
                                "semantic_eval/good_steps": semantic_stats["mask_stats"]["good_steps"],
                                "semantic_eval/bad_steps": semantic_stats["mask_stats"]["bad_steps"],
                            })
                            ms = semantic_stats["mask_stats"]
                            metrics.update({
                                # 四象限 step 统计
                                "semantic_eval/pos_good_steps": ms["pos_good_steps"],
                                "semantic_eval/pos_bad_steps":  ms["pos_bad_steps"],
                                "semantic_eval/neg_good_steps": ms["neg_good_steps"],
                                "semantic_eval/neg_bad_steps":  ms["neg_bad_steps"],

                                # token / seq 级别
                                "semantic_eval/pos_tokens":     ms["pos_tokens"],
                                "semantic_eval/neg_tokens":     ms["neg_tokens"],
                                "semantic_eval/pos_tokens_raw": ms["pos_tokens_raw"],   # 原始adv符号
                                "semantic_eval/neg_tokens_raw": ms["neg_tokens_raw"],
                                "semantic_eval/pos_sequences":  ms["positive_samples"],
                                "semantic_eval/neg_sequences":  ms["negative_samples"],

                                # 简单比例看负样本是否 dominate
                                "semantic_eval/neg_token_ratio": ms["neg_tokens_raw"] / max(1, ms["pos_tokens_raw"] + ms["neg_tokens_raw"]),
                            })
                            
                            # 🆕 新增：记录保存相关的metrics
                            if llm_eval_save_dir:
                                metrics.update({
                                    "semantic_eval/save_dir_configured": 1,
                                    "semantic_eval/records_saved": semantic_stats["evaluation_stats"].get("total_tasks", 0),
                                })
                            else:
                                metrics.update({
                                    "semantic_eval/save_dir_configured": 0,
                                    "semantic_eval/records_saved": 0,
                                })
                            # TODO：advantage norm
                            # === Advantage normalization (under semantic_advantage.adv_norm) ===
                            # === Advantage normalization (semantic_advantage.adv_norm) ==========
                            norm_root = getattr(self.config, "semantic_advantage", None)
                            adv_norm_cfg = getattr(norm_root, "adv_norm", None) if norm_root else None

                            if adv_norm_cfg and getattr(adv_norm_cfg, "enable", True):
                                level = getattr(adv_norm_cfg, "level", "batch")          # "batch" | "group"
                                group_size = getattr(adv_norm_cfg, "group_size", None)   # group 模式专用
                                use_mask_type = getattr(norm_root, "mask_type", "loss_mask")

                                with torch.no_grad():
                                    adv = batch.batch["advantages"]                      # (bs, L)
                                    bs, resp_len = adv.shape

                                    # ------- 选 mask -------
                                    if use_mask_type == "loss_mask" and "loss_mask" in batch.batch:
                                        mask_all = batch.batch["loss_mask"][:, -resp_len:].bool()
                                    else:
                                        mask_all = batch.batch["response_mask"].bool()

                                    nonzero_mask_all = mask_all & (adv != 0)             # 仅对 adv≠0 的 token 归一化
                                    norm_adv = adv.clone()

                                    # ------- BATCH 模式 -------
                                    if level == "batch":
                                        nz_adv = adv[nonzero_mask_all]
                                        if nz_adv.numel() > 0:
                                            med = torch.median(nz_adv)
                                            std = nz_adv.std(unbiased=False).clamp_min(1e-8)
                                            norm_adv[nonzero_mask_all] = (adv[nonzero_mask_all] - med) / std
                                        else:
                                            med, std = torch.tensor(0.0, device=adv.device), torch.tensor(1.0, device=adv.device)

                                        group_cnt = 1
                                        tokens_normed = int(nonzero_mask_all.sum().item())
                                        med_mean, std_mean = float(med), float(std)
                                        zero_groups = 0  # batch 模式无意义

                                    # ------- GROUP 模式 -------
                                    elif level == "group":
                                        if group_size is None:
                                            group_size = self.config.actor_rollout_ref.rollout.n
                                        device = adv.device
                                        group_ids = torch.arange(bs, device=device) // group_size

                                        tokens_normed = 0
                                        med_list, std_list = [], []
                                        zero_groups = 0

                                        for gid in group_ids.unique():
                                            g_sample_mask = (group_ids == gid).unsqueeze(1)
                                            g_mask = g_sample_mask & nonzero_mask_all
                                            if not g_mask.any():
                                                continue

                                            g_adv = adv[g_mask]
                                            med = torch.median(g_adv)
                                            std = g_adv.std(unbiased=False)
                                            if std <= 1e-8:
                                                zero_groups += 1
                                                continue
                                            norm_style = getattr(adv_norm_cfg, "norm_style", None)  
                                            if norm_style == "nostd":
                                                norm_adv[g_mask] = (adv[g_mask] - med)
                                            elif  norm_style == "nomedian":
                                                norm_adv[g_mask] = (adv[g_mask] ) / std
                                            else:
                                                norm_adv[g_mask] = (adv[g_mask] - med) / std
                                            #  # TODO: 尝试不减 median
                                            # norm_adv[g_mask] = (adv[g_mask]) / std
                                            med_list.append(med)
                                            std_list.append(std)
                                            tokens_normed += int(g_adv.numel())

                                        group_cnt = int(group_ids.unique().numel())
                                        med_mean = torch.stack(med_list).mean().item() if med_list else 0.0
                                        std_mean = torch.stack(std_list).mean().item() if std_list else 1.0

                                    else:
                                        raise ValueError(f"Unknown adv_norm.level: {level}")

                                    # 写回
                                    batch.batch["advantages"] = norm_adv

                                    # -------- Stats after normalization --------
                                    pos_tok = int((norm_adv[mask_all] > 0).sum().item())
                                    neg_tok = int((norm_adv[mask_all] < 0).sum().item())
                                    zero_tok = int((norm_adv[mask_all] == 0).sum().item())

                                    seq_sum = (norm_adv * mask_all).sum(dim=1)
                                    pos_seq = int((seq_sum > 0).sum().item())
                                    neg_seq = int((seq_sum < 0).sum().item())
                                    zero_seq = int((seq_sum == 0).sum().item())

                                # --------- Metrics ---------
                                metrics.update({
                                    "semantic_advantage/adv_norm/level": level,
                                    "semantic_advantage/adv_norm/groups": group_cnt,
                                    "semantic_advantage/adv_norm/tokens_normed": tokens_normed,
                                    "semantic_advantage/adv_norm/zero_groups": zero_groups,
                                    "semantic_advantage/adv_norm/median_mean": float(med_mean),
                                    "semantic_advantage/adv_norm/std_mean":   float(std_mean),

                                    "semantic_advantage/adv_norm/pos_tokens": pos_tok,
                                    "semantic_advantage/adv_norm/neg_tokens": neg_tok,
                                    "semantic_advantage/adv_norm/zero_tokens": zero_tok,
                                    "semantic_advantage/adv_norm/pos_sequences": pos_seq,
                                    "semantic_advantage/adv_norm/neg_sequences": neg_seq,
                                    "semantic_advantage/adv_norm/zero_sequences": zero_seq,
                                    "semantic_advantage/adv_norm/neg_token_ratio": neg_tok / max(1, pos_tok + neg_tok),
                                })

                            
                            print("^^^^^^^^^^^^^^^^^ end parallel semantic processing")
                            
                        # # ===========  0714 shuchang: add semantic mask  =========== 
                        # # ---------------- token-entropy logging ----------------
                        # responses       = batch.batch["responses"]         # (bs, resp_len)
                        # response_masks  = batch.batch["response_mask"]     # True 表示有效 token
                        # log_token_entropy(
                        #     tokenizer      = self.tokenizer,
                        #     responses      = responses.cpu(),
                        #     entropys       = entropys.cpu(),
                        #     masks          = response_masks.cpu(),
                        #     global_step    = self.global_steps,
                        #     save_dir       = self.config.save_dir,               # 也可写成 config 里的路径
                        #     print_first_n  = 1,                            # 只想打印更多可调这个值
                        # )

                        # # =========== add entropy mask ===========
                        # # 0710 shuchang: add entropy mask for actor rollout
                        # rho  = self.config.actor_rollout_ref.actor.entropy_mask.rho       # 例如 0.2
                        # mode = self.config.actor_rollout_ref.actor.entropy_mask.mode       # "pos-high-neg-high" / "pos-high-neg-low" / None

                        # entropy_mask, tau, (ent_pos, ent_neg) = _add_advantage_mask(
                        #     entropy       = batch.batch["entropys"],          # (bs, resp_len)
                        #     rho           = rho,
                        #     response_mask = batch.batch["response_mask"],     # 初始 mask（padding / multi_turn）
                        #     adv           = batch.batch["advantages"],        # (bs, 1) or (bs, resp_len)
                        #     mode          = mode,
                        # )

                        # batch.batch["response_mask"] = entropy_mask           # 覆盖为"高熵"版本
                        # metrics.update({                                       # 记录诊断指标（可选）
                        #     "actor/tau_pos": tau[0] if tau else 0.0,
                        #     "actor/tau_neg": tau[1] if tau else 0.0,
                        #     "actor/entropy_pos": ent_pos,
                        #     "actor/entropy_neg": ent_neg,
                        # })

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if summary_task is not None:
                        experience_list, time_cost = summary_task.result()
                        metrics.update({
                            "experience_maker/summary": time_cost,
                        })
                        print("wait for summary_task complete~")
                        if experience_list:
                            for i, experience in enumerate(experience_list):
                                print(f"index={i} experience={experience}")
                    
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
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

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
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
