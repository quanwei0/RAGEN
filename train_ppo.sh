set -e
# export JAVA_HOME=/usr/lib/jvm/java-1.21.0-openjdk-amd64
# export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export WANDB_API_KEY=c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9
# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_DAPO="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" trainer.n_gpus_per_node=8 \
#     trainer.experiment_name=webshop-3b-ppo $USE_PPO $USE_BASE \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 \
#     trainer.experiment_name=webshop-1.5b-ppo-multi-turn-300 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=True algorithm.high_level_gamma=0.0 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 \
#     trainer.experiment_name=webshop-1.5b-ppo-300 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1



# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1\" trainer.n_gpus_per_node=2 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_nonmask $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2


### 
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_mask_False_MTGAE_True $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=True &

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_mask_False_MTGAE_True $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=True \

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_mask_False_MTGAE_False $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=False &

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_mask_False_MTGAE_False $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=False \


# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_mask_False_MTGAE_True $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=True &

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_mask_False_MTGAE_True $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=True \

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-hierarchical_0.7_avg $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=True \
#     +algorithm.claude_alg=hierarchical \
#     +algorithm.alpha=0.7 \
#     +algorithm.turn_level_method=average \

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-hierarchical_0.7_avg $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=True \
#     +algorithm.claude_alg=hierarchical \
#     +algorithm.alpha=0.7 \
#     +algorithm.turn_level_method=average \

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-hierarchical_0.7_gae $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=True \
#     +algorithm.claude_alg=hierarchical \
#     +algorithm.alpha=0.7 \
#     +alrotihm.turn_level_method=gae \

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=zxn-webshop-1.5b-ppo-hierarchical_0.9_gae_v2 $USE_PPO $USE_BASE \
    algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
    es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    +critic.mask_obs=True \
    +algorithm.multi_turn_gae=False \
    +algorithm.use_claude=True \
    +algorithm.claude_alg=hierarchical \
    +algorithm.alpha=0.9 \
    +alrotihm.turn_level_method=gae \

#### Bilevel GAE Experiments
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-bilevel $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=True algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=False \
#     +algorithm.claude_alg=hierarchical \
#     +algorithm.alpha=0.7 \
#     +alrotihm.turn_level_method=gae \

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-bilevel-gamma1 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=True algorithm.high_level_gamma=1.00 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=False \
#     +algorithm.claude_alg=hierarchical \
#     +algorithm.alpha=0.7 \
#     +algorithm.turn_level_method=gae \
#     +algorithm.multi_turn_gae_v2=False &

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-bilevel-gamma1 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=True algorithm.high_level_gamma=1.00 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=False \
#     +algorithm.claude_alg=hierarchical \
#     +algorithm.alpha=0.7 \
#     +algorithm.turn_level_method=gae \


# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-momentum_v2 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=True \
#     +algorithm.claude_alg=momentum \
    

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-critic_mask_False_MTGAE_True_v2 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=False \
#     +algorithm.claude_alg=momentum \
#     +algorithm.multi_turn_gae_v2=True

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-cliphigh-critic_mask_True_MTGAE_True_v2_new $USE_PPO $USE_DAPO \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=False \
#     +algorithm.claude_alg=momentum \
#     +algorithm.multi_turn_gae_v2=True




# 3b
# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" trainer.n_gpus_per_node=8 \
#     trainer.experiment_name=zxn-webshop-3b-ppo-critic_mask_True_MTGAE_True $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=False \
#     +algorithm.claude_alg=momentum \
#     +algorithm.multi_turn_gae_v2=True

# wait 

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_3b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" trainer.n_gpus_per_node=8 \
#     trainer.experiment_name=zxn-webshop-3b-ppo-critic_mask_False_MTGAE_True $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=False algorithm.high_level_gamma=0.95 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=True 



# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-webshop-1.5b-ppo-bilevel-gamma1 $USE_PPO $USE_BASE \
#     algorithm.bi_level_gae=True algorithm.high_level_gamma=1.00 \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=True \
#     +algorithm.multi_turn_gae=False \
#     +algorithm.use_claude=False \
#     +algorithm.claude_alg=hierarchical \
#     +algorithm.alpha=0.7 \
#     +algorithm.turn_level_method=gae \

# MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name webshop_1.5b_train system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=lcl-debug $USE_PPO $USE_BASE \
#     es_manager.train.env_groups=2 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[2] \
#     trainer.nnodes=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=True \