## 概述

> AgentEvolver 旨在打造一个能够自主迭代演进的智能体学习系统，组成其核心能力的重要一环就是自主学习能力，即对给定环境能自举式探索，自动生成与环境功能适配、且与符合模型能力的学习任务。

TaskManager 及其组件实现了环境探索 + 数据生成部分的系统设计，能够进行环境自主探索、任务自动生成和数据的自定义混合，并扩充和系统化设计奖励计算策略。详情参考[语雀文档](https://aliyuque.antfin.com/bayotg/wgzss4/mor93govlqawbb43)。

引入 TaskManager 的 AgentEvolver 运行过程如下：

1. **从 EnvService 拉取 Task 或本地读取 原始Task**。当train_files val_files不为 None 时，优先自本地（train_files val_files）加载。
2. **自举探索和生成 Task**（目前）在开始训练前，加载环境探索算法，从已有数据的环境描述为起点，**全量探索用于训练的合成 Task 并缓存**。在提供缓存路径的情况下，优先使用缓存的合成 Task。
3. **混合原始数据和合成数据** 使用配置的数据混合策略，按比例混合原始数据与合成数据，并转换为 RLHFDataset。
4. **构建 dataloader** 初始化 dataloader，继续后续训练流程。

## 参数

需要配置的参数主要为
- 数据的读取方式（本地？EnvService？）
- 环境探索算法
- 数据混合策略（纯原始数据？合成数据？）

在 `config/script_config.yaml` 中有如下配置，其作用可见对应 comment。

```

data
  # 当指定为 null 时，自 EnvService 加载数据
  train_files: null
  # 当指定为 null 时，自 EnvService 加载数据
  val_files: null

task_manager:
  # 指定合成训练数据的保存位置。一旦设置，就会仅探索一次并在此后一直使用相同的探索合成数据。
  train_data_path: tasks_explored.train.json 
  # 此参数暂无实际用处
  val_data_path: tasks_explored.val.json
  # 用于探索 & 总结的 llm。
  llm_client: qwen-plus
  # 每个任务最多探索的次数。0 时停止探索合成。
  n: 0
  # 此参数暂无实际用处
  bs: ${data.train_batch_size}
  # 探索时的最大线程数
  num_explore_threads: ${thread_pool.max_workers}

  # 数据混合策略
  mixture:
    # 是否使用原始数据
    use_original_tasks: True
    # 使用合成数据的倍数（n 倍于原始数据数量）。0 时仅使用原始数据，99999999 时尽可能使用全部合成数据。
    synthetic_data_ratio: 0.0
    # 是否对数据进行 shuffle
    shuffle: True

  # 探索策略。目前仅有 random。
  strategy: random
  # 探索策略参数
  strategy_args:
    # 最大探索步数
    max_explore_step: 15
    # 最大 LLM 请求尝试次数
    max_llm_retries: 3
    # EnvService 地址
    env_url: ${env_service.env_url}
    # llm 参数
    exploration_llm_temperature: 1.0
    exploration_llm_top_p: 1.0
    exploration_llm_top_k: 1
    # 任务总结时的可见历史长度
    task_summary_history_length: 10 # the size of sliding windows used in task summarization
```

## 测试 Script

请先启动 EnvService（[add_live_span_and_dynamic_load](https://code.alibaba-inc.com/EconML/EnvService/tree/add_live_span_and_dynamic_load/) branch），然后执行以下脚本

```
#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
# export RAY_DEBUG_POST_MORTEM=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
# completion_callback=none
env_url=http://localhost:8000
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="dlc_log_${current_time}.log"

python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    env_service.env_url=$env_url \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2 \
    task_manager.bs=2 \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=/mnt/data_cpfs/xielipeng.xlp/models/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.use_qwen3=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='ba-taskmanager' \
    trainer.experiment_name='qwen25_3b-taskmanager-debug' \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="experiments/exp_${current_time}/validation_log" \
    trainer.rollout_data_dir="experiments/exp_${current_time}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    critic.ppo_max_token_len_per_gpu=20480 \
    critic.forward_max_token_len_per_gpu=20480 \
    data.train_files=null \
    data.val_files=null \
    env_service.env_type=appworld \
    env_service.env_url="http://127.0.0.1:8000" \
    experience_maker.enable_summarizer=False \
    experience_maker.enable_context_generator=False \
    experience_maker.workspace_id="w1_qwen25_v2_${current_time}" \
    $@
```