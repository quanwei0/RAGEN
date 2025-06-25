set -e

export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
export WANDB_ENTITY="rl_agent"

# export WANDB_MODE=offline  # Save offline log

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# qwen2.5 1.5b webshop

################################################################################################################################################
# # verl ppo
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=qw-webshop-1.5b-ppo $USE_PPO $USE_BASE \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \

# ################################################################################################################################################
# # RAGEN bilevel gae
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=qw-webshop-1.5b-ppo-bilevel-gae $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=True algorithm.high_level_lam=1 algorithm.high_level_gamma=1 algorithm.lam=1 algorithm.gamma=1 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \

# ################################################################################################################################################
# # multi-turn GAE with skipping env tokens
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=qw-webshop-1.5b-ppo-critic_mask_True_MTGAE_True_v2 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     critic.mask_obs=True \
#     algorithm.multi_turn_gae=True \

# ################################################################################################################################################
# weighted bilevel gae
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=qw-webshop-1.5b-ppo-reweighted-bilevel-gae-weight0.1 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=False algorithm.high_level_lam=1 algorithm.high_level_gamma=1 algorithm.lam=1 algorithm.gamma=1 \
    algorithm.weighted_bi_level_gae=True algorithm.turn_level_weight=0.1 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
