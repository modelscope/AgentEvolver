#!/bin/bash

source /mnt/data/taoshuchang.tsc/anaconda3/etc/profile.d/conda.sh

ulimit -n 1048576
ulimit -s 16384

export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export ES_HOSTS=http://11.160.132.46:8200
export HF_ENDPOINT=https://hf-mirror.com
export APPWORLD_ROOT=/mnt/data/fuqingxu/EnvService/EnvService/appworld_root


export ENV_PATH="/mnt/data/taoshuchang.tsc/beyondagent/EnvService"
export PROJECT_PATH="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent"


# ================= Begin Training Setup ==================
suffix="adca_grpo"
current_time=$(date "+%Y%m%d_%H%M%S")
log_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/logs/origindata_1node"
log_file="${log_dir}/${suffix}_${current_time}.log"
mkdir -p "$log_dir"
env_log_dir="/mnt/data/taoshuchang.tsc/beyondagent/EnvService/logs/origindata_1node/"
mkdir -p "$env_log_dir"
llm_evaluation_log_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/experiments/origindata_1node/exp_${suffix}_${current_time}/llm_evaluation_logs"
mkdir -p "/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/experiments/origindata_1node"


CONFIG_PATH="$PROJECT_PATH/config"
cd $PROJECT_PATH
# ================== End Training Setup ==================

# ================== Begin Env_service Appworld ==================
#NOTE: 注意需要修改成自己的APPWORLD_ROOT
conda activate appworld
env_url=http://localhost:8080

export RAY_ENV_NAME=appworld
echo "APPWORLD_ROOT: $APPWORLD_ROOT"
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 导航到项目根目录 (agent_workbench)
PROJECT_ROOT="$SCRIPT_DIR/../"
cd "$PROJECT_ROOT"
# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$ENV_PATH:$PYTHONPATH"
# 打印当前工作目录和 PYTHONPATH 以进行调试
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
# 运行 Python 命令
python -m env_sandbox.env_service --env appworld --portal 0.0.0.0 --port 8080 &> "${env_log_dir}${suffix}.log" &
# ================== End Env_service Appworld ==================



# ================== Begin Training ==================
conda activate verl
swanlab login --api-key xSxgnzpo2HEXkIzoxD2Ua
cd $PROJECT_PATH
set -xeu
export HYDRA_FULL_ERROR=1
# export RAY_DEBUG_POST_MORTEM=1
export PYTHONFAULTHANDLER=1

# 直接执行训练命令，而不是使用 ray job submit
python -m beyondagent.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='beyond_agent_dataflow' \
    env_service.env_url=$env_url \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    algorithm.adv_estimator=grpo \
    attribution_driven_credit_assignment.enable=true \
    attribution_driven_credit_assignment.adca_grpo.enable_adca_metric=true \
    attribution_driven_credit_assignment.adca_grpo.prm_scheme='decouple' \
    attribution_driven_credit_assignment.llm_evaluation_log_dir="$llm_evaluation_log_dir" \
    attribution_driven_credit_assignment.adca_grpo.alpha=0.1 \
    attribution_driven_credit_assignment.adca_grpo.skip_type='none' \
    attribution_driven_credit_assignment.adca_grpo.equal_trajectory_weight=false \
    attribution_driven_credit_assignment.evaluation_type='api' \
    attribution_driven_credit_assignment.consistent_scale=1.0 \
    attribution_driven_credit_assignment.pos_unconsistent_scale=0.2 \
    attribution_driven_credit_assignment.neg_unconsistent_scale=0.2 \
    attribution_driven_credit_assignment.model='qwen-plus' \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.use_qwen3=False \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.max_model_len=25600 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=/mnt/data_aisys_cpfs/xielipeng.xlp/models/Qwen2.5-14B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    trainer.n_gpus_per_node=8 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name='ba_w_1node' \
    trainer.experiment_name="${suffix}" \
    trainer.nnodes=1 \
    trainer.default_local_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/checkpoints/origindata_1node/${suffix}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=40 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/experiments/origindata_1node/exp_${suffix}_${current_time}/validation_log" \
    trainer.rollout_data_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/experiments/origindata_1node/exp_${suffix}_${current_time}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25600 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25600 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25600 \
    critic.ppo_max_token_len_per_gpu=25600 \
    critic.forward_max_token_len_per_gpu=25600 \
    data.train_files=null \
    data.val_files=null \
    experience_maker.enable_summarizer=False \
    experience_maker.enable_context_generator=False \
    experience_maker.workspace_id="w1_qwen25_api_turbo_${current_time}" \
    2>&1 | tee "$log_file"