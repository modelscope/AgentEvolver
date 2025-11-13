# ---- Start Environment Service ----
# conda activate appworld
# bash env_service/launch_script/appworld.sh


# ---- Start ReMe Service ----
# conda activate reme
# cd external/reme
# reme \
#   config=default \
#   backend=http \
#   thread_pool_max_workers=256 \
#   http.host="127.0.0.1" \
#   http.port=8001 \
#   http.limit_concurrency=256 \
#   llm.default.model_name=qwen-max-2025-01-25 \
#   embedding_model.default.model_name=text-embedding-v4 \
#   vector_store.default.backend=local \
#   op.rerank_memory_op.params.enable_llm_rerank=false


# ---- Start Training ----

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
env_url=http://localhost:8080
em_url=http://localhost:8001
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="log_${current_time}.log"



python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    env_service.env_url=$env_url \
    exp_manager.val_rollout_mode="woexp" \
    exp_manager.train_rollout_mode="mixed" \
    exp_manager.rollout_ratio=0.5 \
    exp_manager.train_sample_mode="alldiscard" \
    exp_manager.train_sample_keepratio=0.0 \
    exp_manager.init_exp_before_training=True \
    exp_manager.init_exp_only=False \
    exp_manager.reme.base_url=${em_url} \
    exp_manager.reme.workspace_id="appworld_7b_syndata" \
    exp_manager.reme.enable_summarizer=True \
    exp_manager.reme.enable_context_generator=True \
    exp_manager.reme.updated_freq=0 \
    actor_rollout_ref.actor.off_cliprange_high=0.6 \
    attribution_driven_credit_assignment.enable=true \
    attribution_driven_credit_assignment.adca_grpo.enable_adca_metric=true \
    attribution_driven_credit_assignment.adca_grpo.prm_scheme='decouple' \
    attribution_driven_credit_assignment.llm_evaluation_log_dir="experiments/tech_synthetic/${experiment_name}/llm_evaluation_logs" \
    attribution_driven_credit_assignment.adca_grpo.alpha=0.2 \
    attribution_driven_credit_assignment.adca_grpo.skip_type='none' \
    attribution_driven_credit_assignment.adca_grpo.equal_trajectory_weight=true \
    attribution_driven_credit_assignment.evaluation_type='api' \
    attribution_driven_credit_assignment.consistent_scale=1.0 \
    attribution_driven_credit_assignment.pos_unconsistent_scale=0.2 \
    attribution_driven_credit_assignment.neg_unconsistent_scale=0.2 \
    attribution_driven_credit_assignment.model='qwen-max' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=4000 \
    data.max_response_length=21580 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.use_qwen3=False \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.max_model_len=25580 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    trainer.logger="['console']" \
    trainer.project_name="appworld_qwen25-7b" \
    trainer.experiment_name="appworld_qwen25-7b_agentevolver" \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=10 \
    trainer.total_epochs=40 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="experiments/tech_synthetic/${experiment_name}/validation_log" \
    trainer.rollout_data_dir="experiments/tech_synthetic/${experiment_name}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25580 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25580 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25580 \
    critic.ppo_max_token_len_per_gpu=25580 \
    critic.forward_max_token_len_per_gpu=25580 \
    data.train_files=null \
    data.val_files=null \
    env_service.env_type=appworld \
    task_manager.n=8 \
    task_manager.mixture.synthetic_data_ratio=999999.0 \
    task_manager.mixture.use_original_tasks=False \
    task_manager.grader.synthetic_grader=llm-binary-gt-no_constraint \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    2>&1 | tee "$log_file" \