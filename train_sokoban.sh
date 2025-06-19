USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"



# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" \
#     trainer.n_gpus_per_node=4 \
#     trainer.experiment_name=zxn-sokoban-ppo-1.5b-critic_mask_False_MTGAE_True \
#     model_path=Qwen/Qwen2.5-1.5B-Instruct \
#     $USE_BASE $USE_PPO \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     +critic.mask_obs=False \
#     +algorithm.multi_turn_gae=True 

python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"4,5,6,7\" \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=zxn-sokoban-ppo-1.5b-critic_mask_False_MTGAE_False \
    model_path=Qwen/Qwen2.5-1.5B-Instruct \
    $USE_BASE $USE_PPO \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    +critic.mask_obs=False \
    +algorithm.multi_turn_gae=False \
    +algorithm.use_claude=False \
    +algorithm.claude_alg=hierarchical \
    +algorithm.alpha=0.9 \

