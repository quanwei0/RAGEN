set -e
# export JAVA_HOME=/usr/lib/jvm/java-1.21.0-openjdk-amd64
# export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

### 
MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=lcl-webshop-1.5b-ppo-critic_mask_False_MTGAE_True_rolloutfilter0.25 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=64 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
    +critic.mask_obs=False \
    +algorithm.multi_turn_gae=True &

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=lcl-webshop-1.5b-ppo-critic_mask_False_MTGAE_True_rolloutfilter0.50 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=64 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.50 \
    +critic.mask_obs=False \
    +algorithm.multi_turn_gae=True 

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=lcl-webshop-1.5b-ppo-critic_mask_False_MTGAE_True_rolloutfilter0.75 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=64 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.75 \
    +critic.mask_obs=False \
    +algorithm.multi_turn_gae=True 

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=lcl-webshop-1.5b-ppo-critic_mask_False_MTGAE_True_rolloutfilter1 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=64 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.rollout_filter_ratio=1 \
    +critic.mask_obs=False \
    +algorithm.multi_turn_gae=True 
