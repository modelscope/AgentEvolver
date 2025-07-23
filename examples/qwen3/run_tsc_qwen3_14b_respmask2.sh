# run on 4xH100 with local vLLM semantic evaluation
# make sure your current working directory is the root of the project
# 126 tsc_no
set -x
export HYDRA_FULL_ERROR=1
# export RAY_POST_MORTEM=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
# completion_callback=none
env_url=http://localhost:8000
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="logs/qwen3/qwen3_14b_sparse_sem_turbo_bad0.2_negbad-0.2_respmask2_${current_time}.log"
EN_SAVE_DIR="./save_dir/save_entropy"

swanlab login --api-key xSxgnzpo2HEXkIzoxD2Ua

python3 -m beyondagent.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='beyond_agent_dataflow' \
    env_service.env_url=$env_url \
    save_dir=$EN_SAVE_DIR \
    algorithm.adv_estimator=grpo \
    semantic_advantage.enable=true \
    semantic_advantage.evaluation_type='api' \
    semantic_advantage.mask_type='response_mask' \
    semantic_advantage.mode='semantic' \
    semantic_advantage.consistent_scale=1.0 \
    semantic_advantage.pos_unconsistent_scale=0.2 \
    semantic_advantage.neg_unconsistent_scale=-0.2 \
    semantic_advantage.api_max_retries=200 \
    semantic_advantage.concurrent=5 \
    semantic_advantage.model='qwen-turbo' \
    env_sparse=true \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.use_qwen3=True \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=/mnt/data/yunpeng.zyp/models/Qwen3-14B \
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
    trainer.logger=['console','swanlab'] \
    trainer.project_name='beyondagent' \
    trainer.experiment_name="qwen3_14b_sparse_sem_turbo_bad0.2_negbad-0.2_respmask2" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=20 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="experiments/exp_qwen3_14b_sparse_sem_turbo_bad0.2_negbad-0.2_respmask2_${current_time}/validation_log" \
    trainer.rollout_data_dir="experiments/exp_qwen3_14b_sparse_sem_turbo_bad0.2_negbad-0.2_respmask2_${current_time}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    critic.ppo_max_token_len_per_gpu=20480 \
    critic.forward_max_token_len_per_gpu=20480 \
    data.train_files=/mnt/data/yunpeng.zyp/data/appworld_verl/train.parquet \
    data.val_files=/mnt/data/yunpeng.zyp/data/appworld_verl/dev.parquet \
    experience_maker.enable_summarizer=False \
    experience_maker.enable_context_generator=False \
    experience_maker.workspace_id="w1_qwen3_api_turbo_${current_time}" \
    2>&1 | tee "$log_file" 