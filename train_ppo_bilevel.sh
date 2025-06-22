set -e

export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
export WANDB_ENTITY="rl_agent"

# export WANDB_MODE=offline  # Save offline log

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

################################################################################################################################################
# qwen2.5 1.5b webshop


# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-1-lllam-1 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=1 algorithm.lam=1 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 &

# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-1-lllam-1 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=1 algorithm.lam=1 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \

# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-1-lllam-0.95 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=1 algorithm.lam=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 &

# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-1-lllam-0.95 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=1 algorithm.lam=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \

# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-0.95-lllam-0.95 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=0.95 algorithm.lam=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 &

# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-0.95-lllam-0.95 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=0.95 algorithm.lam=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \

# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-0.95-lllam-1 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=0.95 algorithm.lam=1 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 &

# nohup env \
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae-hlllam-0.95-lllam-1 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=True algorithm.high_level_lam=0.95 algorithm.lam=1 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
