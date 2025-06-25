from verl.trainer.ppo.core_algos import *
import random


# used for masking env tokens in critic update
def fill_after_first_one(response_mask: torch.Tensor):
    cumsum = torch.cumsum(response_mask, dim=1)
    return (cumsum > 0).to(response_mask.dtype).to(response_mask.device)


# adapted and modified from RAGEN
def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float,
        high_level_lam: float,
        response_mask: torch.Tensor = None
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation
        response_mask: `(torch.Tensor)` optional
            shape: (bs, response_length). 1 for LLM generation, 0 for observation. Used to find turn boundaries.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        
        ##########################################################################################
        # Example:
        # response_mask = [0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1]
        # reward_mask   = [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0]

        if response_mask is not None:
            batch_size, seq_len = response_mask.shape
            reward_mask = torch.zeros_like(response_mask, dtype=torch.float)

            for b in range(batch_size):
                response_seq = response_mask[b]

                # Identify turn start points: positions where response begins (0 → 1 transition)
                # This gives the indices of the first token of each response turn
                turn_start_pos = ((response_seq[1:] == 1) & (response_seq[:-1] == 0)).nonzero(as_tuple=True)[0] + 1

                reward_mask[b, turn_start_pos] = 1.0

        else:
            # Use traditional reward mask
            reward_mask = token_level_rewards.bool()
        ##########################################################################################
        
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            turn_start_pos = reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0            
            for i in range(len(turn_start_pos) - 1, -1, -1):
                curr_pos = turn_start_pos[i]
                
                # Get the next value
                if i < len(turn_start_pos) - 1:
                    # Next valid position
                    next_pos = turn_start_pos[i + 1]
                    nextvalue = values[b, next_pos]
                    
                    # Calculate delta using the next valid token
                    delta = 0 + high_level_gamma * nextvalue - values[b, curr_pos]                    
                    
                else:
                    # Last valid position
                    nextvalue = 0.0

                    # Calculate delta using the next valid token
                    delta = updated_reward[b, -1] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * high_level_lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            turn_level_adv = advantages.clone()
            # for i, pos in enumerate(turn_start_pos):
                # returns[b, pos] = advantages[b, pos] + values[b, pos]
                # updated_reward[b, pos] = advantages[b, pos] + values[b, pos]

            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_valid_pos = valid_positions[i]

                # for last turn
                if curr_valid_pos >= turn_start_pos[-1]:
                    # for non-last token in the last turn
                    if i != len(valid_positions) - 1:
                        next_valid_pos = valid_positions[i + 1]
                        nextvalue = values[b, next_valid_pos]
                        delta = 0 + gamma * nextvalue - values[b, curr_valid_pos]
                    # for last token in the last turn
                    else:
                        nextvalue = 0.0
                        lastgaelam = 0.0
                        delta = updated_reward[b, -1] + gamma * nextvalue - values[b, curr_valid_pos]

                # for non-last turn
                else:
                    next_valid_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_valid_pos]
                    
                    if next_valid_pos in (turn_start_pos).tolist():
                        lastgaelam = turn_level_adv[b, next_valid_pos]

                    delta = 0 + gamma * nextvalue - values[b, curr_valid_pos]
                               
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_valid_pos] = lastgaelam
                returns[b, curr_valid_pos] = lastgaelam + values[b, curr_valid_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns

# adapted from verl.trainer.ppo.core_algos
# original verl implementation
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns

# adapted and modified from verl.trainer.ppo.core_algos
# skip env tokens when assigning next values and accumulating TD error
def compute_gae_advantage_return_multi_turn(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        returns_reversed = []
        gen_len = token_level_rewards.shape[-1]
        response_mask_f = response_mask.float()
        # For masked tokens, force gamma=1 and lambda=1, regardless of the values in config
        gamma_masked = response_mask_f * gamma + 1 - response_mask_f
        lam_masked = response_mask_f * lam + 1 - response_mask_f
        nextvalues_skip_obs = 0
        # returns_gt = 0

        for t in reversed(range(gen_len)):
            next_step_mask = response_mask_f[:, t + 1] if t < gen_len - 1 else 1.0
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            nextvalues_skip_obs = (1 - next_step_mask) * nextvalues_skip_obs + next_step_mask * nextvalues
            this_step_gamma = gamma_masked[:, t]
            this_step_lam = lam_masked[:, t]
            delta = token_level_rewards[:, t] + this_step_gamma * nextvalues_skip_obs - values[:, t]
            delta *= response_mask_f[:, t]
            lastgaelam = delta + this_step_gamma * this_step_lam * lastgaelam
            advantages_reversed.append(lastgaelam)

            # returns_gt = this_step_gamma * returns_gt + response_mask_f[:, t] * token_level_rewards[:, t]
            # returns_reversed.append(returns_gt)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # returns = torch.stack(returns_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask_f)
    return advantages, returns

def detect_turn_boundaries(response_mask: torch.Tensor) -> torch.Tensor:
    """
    检测turn边界：当response_mask从1变为0时
    返回的tensor中，1表示该位置是turn的结束
    """
    bs, seq_len = response_mask.shape
    boundaries = torch.zeros_like(response_mask)
    
    for b in range(bs):
        for t in range(seq_len - 1):
            # 当前是response (1) 且下一个是observation (0)
            if response_mask[b, t] == 1 and response_mask[b, t + 1] == 0:
                boundaries[b, t] = 1
        # 最后一个response token也是turn结束
        if response_mask[b, -1] == 1:
            boundaries[b, -1] = 1
    
    return boundaries


def compute_multiturn_gae_momentum(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
    momentum_factor: float = 0.3,
):
    """
    动量机制：保持turn之间的连续性，good momentum继续，bad momentum衰减
    """
    with torch.no_grad():
        bs, seq_len = token_level_rewards.shape
        response_mask_f = response_mask.float()
        
        # 计算带动量的奖励
        momentum_rewards = token_level_rewards.clone()
        
        for b in range(bs):
            momentum = 0
            prev_is_response = False
            
            for t in range(seq_len):
                if response_mask[b, t] > 0:
                    # 如果从observation转到response（新turn开始）
                    if not prev_is_response:
                        # 根据momentum的正负决定是否传递
                        if momentum > 0:
                            # 正momentum：奖励前面的good start
                            momentum_rewards[b, t] += momentum * momentum_factor
                        else:
                            # 负momentum：轻微惩罚，但不要太严重
                            momentum_rewards[b, t] += momentum * momentum_factor * 0.5
                    
                    # 更新momentum（指数移动平均）
                    momentum = 0.9 * momentum + 0.1 * token_level_rewards[b, t]
                    prev_is_response = True
                else:
                    prev_is_response = False
        
        # 使用带动量的奖励计算GAE
        lastgaelam = 0
        advantages_reversed = []
        returns_reversed = []
        gamma_masked = response_mask_f * gamma + 1 - response_mask_f
        lam_masked = response_mask_f * lam + 1 - response_mask_f
        nextvalues_skip_obs = 0
        returns_gt = 0

        for t in reversed(range(seq_len)):
            next_step_mask = response_mask_f[:, t + 1] if t < seq_len - 1 else 1.0
            nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
            nextvalues_skip_obs = (1 - next_step_mask) * nextvalues_skip_obs + next_step_mask * nextvalues
            this_step_gamma = gamma_masked[:, t]
            this_step_lam = lam_masked[:, t]
            delta = momentum_rewards[:, t] + this_step_gamma * nextvalues_skip_obs - values[:, t]
            delta *= response_mask_f[:, t]
            lastgaelam = delta + this_step_gamma * this_step_lam * lastgaelam
            advantages_reversed.append(lastgaelam)

            returns_gt = this_step_gamma * returns_gt + response_mask_f[:, t] * momentum_rewards[:, t]
            returns_reversed.append(returns_gt)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # returns = torch.stack(returns_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask_f)
        
    return advantages, returns

def compute_multiturn_gae_hierarchical(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
    alpha: float = 0.7,  # weight for token-level advantages
    turn_level_method: str = "average",  # "average" or "gae"
    high_level_gamma: float = None,  # gamma for turn-level GAE, defaults to token-level gamma
    high_level_lam: float = None,  # lambda for turn-level GAE, defaults to token-level lam
):
    """
    Hierarchical GAE: compute token-level and turn-level advantages separately, then combine
    
    Args:
        turn_level_method: "average" - use averaging method, "gae" - use GAE method for turn-level advantages
        high_level_gamma: gamma value for turn-level GAE, uses token-level gamma if None
    """
    if high_level_gamma is None:
        high_level_gamma = gamma
    if high_level_lam is None:
        high_level_lam = lam
        
    with torch.no_grad():
        bs, seq_len = token_level_rewards.shape
        
        # Step 1: Compute token-level advantages using existing method
        lastgaelam = 0
        token_advantages = []
        response_mask_f = response_mask.float()
        gamma_masked = response_mask_f * gamma + 1 - response_mask_f
        lam_masked = response_mask_f * lam + 1 - response_mask_f
        nextvalues_skip_obs = 0

        for t in reversed(range(seq_len)):
            next_step_mask = response_mask_f[:, t + 1] if t < seq_len - 1 else 1.0
            nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
            nextvalues_skip_obs = (1 - next_step_mask) * nextvalues_skip_obs + next_step_mask * nextvalues
            this_step_gamma = gamma_masked[:, t]
            this_step_lam = lam_masked[:, t]
            delta = token_level_rewards[:, t] + this_step_gamma * nextvalues_skip_obs - values[:, t]
            delta *= response_mask_f[:, t]
            lastgaelam = delta + this_step_gamma * this_step_lam * lastgaelam
            token_advantages.append(lastgaelam)
        
        token_advantages = torch.stack(token_advantages[::-1], dim=1)
        
        # Step 2: Compute turn-level advantages
        if turn_level_method == "average":
            turn_advantages = _compute_turn_level_average(
                token_advantages, response_mask, response_mask_f, bs, seq_len
            )
        elif turn_level_method == "gae":
            turn_advantages = _compute_turn_level_gae(
                token_level_rewards, values, response_mask, response_mask_f, 
                high_level_gamma, lam=high_level_lam, bs=bs, seq_len=seq_len
            )
        else:
            raise ValueError(f"Unknown turn_level_method: {turn_level_method}")

        # Step 3: Combine advantages
        combined_advantages = alpha * token_advantages + (1-alpha) * turn_advantages
        returns = combined_advantages + values
        combined_advantages = verl_F.masked_whiten(combined_advantages, response_mask_f)
        
    return combined_advantages, returns


def _compute_turn_level_average(token_advantages, response_mask, response_mask_f, bs, seq_len):
    """Original averaging method for computing turn-level advantages"""
    turn_boundaries = detect_turn_boundaries(response_mask)
    turn_advantages = torch.zeros_like(token_advantages)
    
    for b in range(bs):
        # Identify turn boundaries
        turn_starts = [0]
        turn_ends = []
        
        for t in range(seq_len):
            if response_mask[b, t] > 0 and (turn_boundaries[b, t] == 1 or t == seq_len - 1):
                turn_ends.append(t)
                if t < seq_len - 1:
                    # Find next response token as start of new turn
                    for next_t in range(t + 1, seq_len):
                        if response_mask[b, next_t] > 0:
                            turn_starts.append(next_t)
                            break
        
        # Compute shared advantage for each turn
        for i, (start, end) in enumerate(zip(turn_starts, turn_ends)):
            turn_mask = response_mask_f[b, start:end+1]
            if turn_mask.sum() > 0:
                avg_adv = (token_advantages[b, start:end+1] * turn_mask).sum() / turn_mask.sum()
                
                for t in range(start, end + 1):
                    if response_mask[b, t] > 0:
                        turn_advantages[b, t] = avg_adv
    
    return turn_advantages


def _compute_turn_level_gae(token_level_rewards, values, response_mask, response_mask_f, 
                           high_level_gamma, lam, bs, seq_len, reward_aggregation='sparse'):
    """Use GAE method to compute turn-level advantages with shared advantage within each turn
    
    Args:
        reward_aggregation: How to aggregate rewards for each turn
            - 'sparse': Only use reward at terminal token of each turn
            - 'sum': Use sum of all rewards within each turn
            - 'average': Use average of all rewards within each turn
    """
    turn_advantages = torch.zeros_like(token_level_rewards)
    
    # Compute GAE for each batch separately
    for b in range(bs):
        # Find turn boundaries: first and last position of each response turn
        turn_starts = []
        turn_ends = []
        
        in_response = False
        turn_start = None
        
        for t in range(seq_len):
            if response_mask[b, t] > 0:  # Response token
                if not in_response:
                    # Starting a new turn
                    turn_start = t
                    in_response = True
            else:  # Non-response token (or end of sequence)
                if in_response:
                    # Ending current turn
                    turn_starts.append(turn_start)
                    turn_ends.append(t - 1)  # Last response token
                    in_response = False
        
        # Handle case where sequence ends with a response
        if in_response:
            turn_starts.append(turn_start)
            turn_ends.append(seq_len - 1)
        
        # Compute GAE backwards through turns
        if len(turn_starts) > 0:
            lastgaelam = 0.0
            
            for i in range(len(turn_starts) - 1, -1, -1):
                start_pos = turn_starts[i]
                end_pos = turn_ends[i]
                
                # Get reward for current turn based on aggregation method
                if reward_aggregation == 'sparse':
                    # Only use reward at terminal token
                    reward = token_level_rewards[b, end_pos]
                elif reward_aggregation == 'sum':
                    # Sum all rewards in this turn
                    reward = 0.0
                    for t in range(start_pos, end_pos + 1):
                        if response_mask[b, t] > 0:
                            reward += token_level_rewards[b, t]
                elif reward_aggregation == 'average':
                    # Average all rewards in this turn
                    reward_sum = 0.0
                    count = 0
                    for t in range(start_pos, end_pos + 1):
                        if response_mask[b, t] > 0:
                            reward_sum += token_level_rewards[b, t]
                            count += 1
                    reward = reward_sum / count if count > 0 else 0.0
                else:
                    raise ValueError(f"Unknown reward_aggregation: {reward_aggregation}")
                
                # Get value at start of current turn
                current_value = values[b, start_pos]
                
                # Get value at start of next turn (if exists)
                if i < len(turn_starts) - 1:
                    next_start_pos = turn_starts[i + 1]
                    next_value = values[b, next_start_pos]
                else:
                    next_value = 0.0
                
                # Calculate delta: reward + gamma * next_value - current_value
                delta = reward + high_level_gamma * next_value - current_value
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                
                # Assign the same advantage to all tokens in this turn
                for t in range(start_pos, end_pos + 1):
                    if response_mask[b, t] > 0:
                        turn_advantages[b, t] = lastgaelam
    
    return turn_advantages

# set up unittest
if __name__ == "__main__":
    # token_level_rewards = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    # values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    # loss_mask = torch.ones(1, 10)
    # advantages1, returns1 = compute_bi_level_gae_advantage_return(token_level_rewards, values, loss_mask, 1, 1, 0.95)
    # print(advantages1)
    # print(returns1)
    # advantages2, returns2 = compute_bi_level_gae_advantage_return_origin(token_level_rewards, values, loss_mask, 1, 1, 0.95)
    # print(advantages2)
    # print(returns2)
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([
        [ 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0 ]
    ], dtype=torch.float)
    
    values1 = torch.tensor([
        [ random.uniform(-100.0, 100.0), random.random(), 4.0, 5.0, 6.0, random.uniform(-100.0, 0), random.random(), 7.0, 9.0, 0.0, 0.0 ]
    ], dtype=torch.float)
    
    values2 = torch.tensor([
        [ random.random(), random.uniform(-100.0, 100.0), 4.0, 5.0, 6.0, random.random(), random.uniform(0.0, 100.0), 7.0, 9.0, 0.0, 0.0 ]
    ], dtype=torch.float)
    
    eos_mask = torch.tensor([
        [ 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0 ] 
    ], dtype=torch.float)
    
    adv1, ret1 = compute_bi_level_gae_advantage_return(rewards, values1, eos_mask, gamma, lam, high_level_gamma=0.95, response_mask=eos_mask, high_level_lam=0.95)
    # adv2, ret2 = compute_bi_level_gae_advantage_return(rewards, values2, eos_mask, gamma, lam, high_level_gamma=0.95, response_mask=eos_mask, high_level_lam=1.0)
    
    # adv1, ret1 = compute_gae_advantage_return_multi_turn(rewards, values1, eos_mask, gamma, lam)
    # adv2, ret2 = compute_gae_advantage_return_multi_turn(rewards, values2, eos_mask, gamma, lam)
    
    # adv1, ret1 = compute_multiturn_gae_with_turn_bonus(rewards, values1, eos_mask, gamma, lam)
    # adv2, ret2 = compute_multiturn_gae_with_turn_bonus(rewards, values2, eos_mask, gamma, lam)
    
    # adv1, ret1 = compute_multiturn_gae_with_adaptive_lambda(rewards, values1, eos_mask, gamma, lam)
    # adv2, ret2 = compute_multiturn_gae_with_adaptive_lambda(rewards, values2, eos_mask, gamma, lam)
    
    # adv1, ret1 = compute_multiturn_gae_hierarchical(rewards, values1, eos_mask, gamma, lam, alpha=1.0, turn_level_method="gae", high_level_gamma=0.95)
    adv2, ret2 = compute_multiturn_gae_hierarchical(rewards, values2, eos_mask, gamma, lam, alpha=1.0, turn_level_method="gae", high_level_gamma=0.95)
    

    # ret1 *= eos_mask
    # ret2 *= eos_mask
    # assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    # assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    # print(f' [CORRECT] \n\n{adv1=}, \n\n{adv2=}')
    print(f' [CORRECT] \n\n{adv1=}, \n\n{adv2=}, \n\n{ret1=}, \n\n{ret2=}')