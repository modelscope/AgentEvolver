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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
# from best_logger import register_logger
import torch
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.task_manager.base import NaiveTaskObjectiveRetrieval
from beyondagent.module.task_manager.data_mixture import OriginalOnlyStrategy, UnifiedMixtureStrategy
from beyondagent.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from beyondagent.module.task_manager.task_manager import TaskManager

# non_console_mods = ["appworld_io"]
# register_logger(non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path="logs/beyondagent", debug=True)

import os
import hydra
import ray

from beyondagent.module.task_manager.env_profiles import EnvProfile
from verl.trainer.ppo.reward import load_reward_manager

from beyondagent.module.trainer.ba_ray_trainer import BeyondAgentRayPPOTrainer

from verl.trainer.ppo import core_algos
if "kl_control" in os.environ.get("DEBUG_ARG",""):
    print("monkeypatching kl loss")
    def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
        """Compute KL divergence given logprob and ref_logprob.
        Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
        See more description in http://joschu.net/blog/kl-approx.html

        Args:
            logprob:
            ref_logprob:

        Returns:

        """
        if kl_penalty in ("kl", "k1"):
            res = logprob - ref_logprob
        elif kl_penalty == "abs":
            res = (logprob - ref_logprob).abs()
        elif kl_penalty in ("mse", "k2"):
            res = 0.5 * (logprob - ref_logprob).square()
        elif kl_penalty in ("low_var_kl", "k3"):
            kl = ref_logprob - logprob
            # For numerical stability
            kl = torch.clamp(kl, min=-20, max=20)
            ratio = torch.exp(kl)
            kld = (ratio - kl - 1).contiguous()
            res = torch.clamp(kld, min=-10, max=10)
        elif kl_penalty == "full":
            # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        assert res.dim() == 2, f"kl_penalty should be 2-dim tensor, but got {res.dim()}-dim tensor"
        # control the kl penalty by the length. frontier tokens have higher penalty.
        import beyondagent.utils.utils as ut
        lengths=(res!=0).int().sum(dim=-1)
        weights=ut.get_batched_exponential_decay_weights_vectorized(lengths.tolist())
        res=res*weights.unsqueeze(-1)
        return res
    
    core_algos.kl_penalty = kl_penalty
    print("patched")
        
        



def get_custom_reward_fn(config):
    """
    Dynamically loads a custom reward function from a specified Python file.

    Args:
        config (dict): Configuration dictionary containing the path and name of the reward function.

    Returns:
        function: The loaded and wrapped reward function, or None if no custom function is specified.
    """
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)  # ⭐ Executes the module to load the custom reward function
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="../config", config_name="beyond_agent_dataflow", version_base=None)
def main(config):
    """
    Entry point for the PPO training process.

    This function initializes and runs the PPO training using the provided configuration.

    Args:
        config (DictConfig): The configuration object that contains all the necessary parameters for setting up and running the PPO training.

    Returns:
        None
    """
    run_ppo(config)  # ⭐ Executes the PPO training process with the given configuration


def run_ppo(config) -> None:
    """
    Initializes the Ray cluster, sets up the environment, and starts the PPO training process.

    Args:
        config (dict): Configuration dictionary containing settings for the PPO training process.

    Returns:
        None
    """
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN",
             "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "VLLM_USE_V1": "1", "WANDB_API_KEY": "local-e93291bd40698a593a1fcc5b99da6a71a753a383",
             "WANDB_BASE_URL": "http://22.6.186.25:8080"}},
            num_cpus=config.ray_init.num_cpus,
        )  # ⭐ Initialize the Ray cluster with the specified runtime environment and number of CPUs

    max_model_len: int = config.actor_rollout_ref.rollout.max_model_len
    assert config.data.max_prompt_length + config.data.max_response_length <= max_model_len, f"max_prompt_length {config.data.max_prompt_length} + max_response_length {config.data.max_response_length} should be <= max_model_len {max_model_len}"  # ⭐ Ensure the sum of max prompt and response lengths does not exceed the max model length

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))  # ⭐ Start the PPO training process by calling the run method on the TaskRunner actor


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """
    A Ray actor class responsible for running the PPO tasks.

    This class is decorated with @ray.remote to ensure it is scheduled on a worker node and not the head node.
    """
    def run(self, config):
        """
        Initializes and runs the PPO training process, including downloading necessary models, defining worker classes,
        and setting up the resource pool and reward functions.

        Args:
            config (dict): The configuration dictionary containing all the necessary parameters for the PPO setup.
        """
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))  # ⭐ Download the model checkpoint to a local path

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)  # ⭐ Initialize the tokenizer
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get('lora_rank', 0) > 0:
                if not is_version_ge(pkg='vllm', minver='0.7.3'):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import (ActorRolloutRefWorker,
                                                   AsyncActorRolloutRefWorker,
                                                   CriticWorker)
            ####################
            # ANNI
            from beyondagent.module.exp_manager.het_fsdp_worker import HETAsyncActorRolloutRefWorker, HETActorRolloutRefWorker
            actor_rollout_cls = HETAsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else HETActorRolloutRefWorker  # ⭐ Define the actor rollout class based on the mode
            # actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ####################
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import \
                NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (ActorRolloutRefWorker,
                                                       CriticWorker)

            actor_rollout_cls = ActorRolloutRefWorker  # ⭐ Define the actor rollout class
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)  # ⭐ Add the reward model worker to the mapping
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)  # ⭐ Add the reference policy worker to the mapping
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))  # ⭐ Load the reward manager for training
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))  # ⭐ Load the reward manager for validation
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)  # ⭐ Initialize the resource pool manager

        from verl.utils.dataset.rl_dataset import collate_fn

        # init task manager
        llm_client=DashScopeClient(model_name=config.task_manager.llm_client)
        train_task_manager=TaskManager(
            config=config,
            exploration_strategy=config.task_manager.strategy,
            env_profile=EnvProfile.load_from_json(config.task_manager.env_profile),
            exploration_strategy_args=config.task_manager.strategy_args,
            llm_client=llm_client, # or use policy model
            old_retrival=NaiveTaskObjectiveRetrieval(),
            mixture_strategy=UnifiedMixtureStrategy(
                use_original=config.task_manager.mixture.use_original_tasks,
                synthetic_ratio=config.task_manager.mixture.synthetic_data_ratio,
                shuffle=config.task_manager.mixture.shuffle,
                seed=42,
                ),
            reward_config=config.task_manager.grader,
            tokenizer=tokenizer,
            env_service_url=config.env_service.env_url,
            num_explore_threads=config.task_manager.num_explore_threads,
            n=config.task_manager.n,
        )  # ⭐ Initialize the training task manager with specified configurations
        val_task_manager=TaskManager(
            config=config,
            exploration_strategy=config.task_manager.strategy,
            env_profile=EnvProfile.load_from_json(config.task_manager.env_profile),
            exploration_strategy_args=config.task_manager.strategy_args,
            llm_client=llm_client, # or use policy model
            old_retrival=NaiveTaskObjectiveRetrieval(),
            mixture_strategy=OriginalOnlyStrategy(),
            reward_config=config.task_manager.grader,
            tokenizer=tokenizer,
            env_service_url=config.env_service.env_url,
            num_explore_threads=config.task_manager.num_explore_threads,
            n=config.task_manager.n,
        )  # ⭐ Initialize the validation task manager with specified configurations
        trainer = BeyondAgentRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_task_manager=train_task_manager,
            val_task_manager=val_task_manager,
            collate_fn=collate_fn,
            device_name=config.trainer.device,
        )  # ⭐ Initialize the PPO trainer with the given parameters
        trainer.init_workers()  # ⭐ Initialize the workers for the trainer
        trainer.fit()  # ⭐ Start the training process

def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """
    Create a dataset for reinforcement learning.

    Args:
        data_paths (list of str): Paths to the data files.
        data_config (dict): Configuration for the dataset.
        tokenizer (Tokenizer): The tokenizer to be used.
        processor (Processor): The processor to be used.

    Returns:
        Dataset: The created dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )  # ⭐ Instantiate the dataset with the provided data paths, tokenizer, processor, and configuration

    return dataset

if __name__ == "__main__":
    import shutil
    # this will break ray's initialization
    # def _safe_guard(name:str):
    #     import traceback
    #     traceback.print_stack()
    #     print(f"{name} is overwritten")
    # shutil.rmtree=lambda *args, **kwargs: _safe_guard("rmtree")
    main()
