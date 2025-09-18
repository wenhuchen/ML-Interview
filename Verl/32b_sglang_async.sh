set -x

NOW=$(date +%Y%m%d-%H)

export WANDB_DIR=wandb/gsm8k-grpo-qwen3-30ba3b-${NOW}
export WANDB_PROJECT=gsm8k-grpo-qwen3-30ba3b
export WANDB_EXP=$(basename ${WANDB_DIR})
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

gsm8k_train_path=data/gsm8k/train.parquet
gsm8k_test_path=data/gsm8k/test.parquet
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"
rollout_mode="async"

model_path=Qwen3-30B-A3B-Thinking-2507

ngpu_per_node=8
# Use NNODES environment variable if set, otherwise default to 1
nnodes=${NNODES:-1}
GLOBAL_BATCH_SIZE=$((128 * nnodes))
MINI_BATCH_SIZE=$((128 * nnodes))
MAX_LENGTH=12288
SEQ_PARALLEL_SIZE=1
FSDP_SIZE=-1
MICRO_BATCH_SIZE=32

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${GLOBAL_BATCH_SIZE} \
    data.val_batch_size=${GLOBAL_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=${MAX_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path  \
    actor_rollout_ref.model.use_shm=False  \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SEQ_PARALLEL_SIZE} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=${ngpu_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=20 \
    trainer.test_freq=15 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 $@