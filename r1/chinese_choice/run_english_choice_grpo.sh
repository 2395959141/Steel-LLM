set -x
DATA_ROOT_DIR=/xxx/data/r1/openbookqa
CKPT_ROOT_DIR=/xxx/model/llm/r1_model
PROJECT_NAME=verl_grpo_wanjuan
EXPERIMENT_NAME=openbookqa_rollout8_8k
BASE_MODEL=/xxx/model/qwen-25-1_5b
wandb login xxx
export VLLM_ATTENTION_BACKEND=XFORMERS
# https://github.com/Unakar/Logic-RL/blob/main/main_grpo.sh
export CUDA_VISIBLE_DEVICES=6,7
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_ROOT_DIR/openbookqa_train.parquet \
    data.val_files=$DATA_ROOT_DIR/openbookqa_test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=384 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 $@