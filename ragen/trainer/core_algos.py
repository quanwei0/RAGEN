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
        high_level_lam: `(float)`
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


# adapted and modified from RAGEN
def compute_weighted_cross_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float,
        high_level_lam: float,
        turn_level_weight: float,
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
        high_level_lam: `(float)`
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
            
            # compute weighted advantages, token-level-advantage + turn-level-advantage
            turn_index = len(turn_start_pos) - 1
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_valid_pos = valid_positions[i]
                if curr_valid_pos >= turn_start_pos[turn_index]:
                    advantages[b, curr_valid_pos] = (1 - turn_level_weight) * advantages[b, curr_valid_pos] + turn_level_weight * turn_level_adv[b, turn_start_pos[turn_index]]
                else:
                    turn_index -= 1
                    advantages[b, curr_valid_pos] = (1 - turn_level_weight) * advantages[b, curr_valid_pos] + turn_level_weight * turn_level_adv[b, turn_start_pos[turn_index]]

        advantages = verl_F.masked_whiten(advantages, loss_mask)

    return advantages, returns

# supported by Kangrui Wang
# adapted from RAGEN
def compute_bi_level_gae_advantage_return_original(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float
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

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]
            
            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

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
        
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        batch_size, gen_len = token_level_rewards.shape

        for b in range(batch_size):
            lastgaelam = 0
            valid_positions = response_mask[b].nonzero(as_tuple=True)[0]

            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                
                if i != len(valid_positions) - 1:
                    next_pos = valid_positions[i + 1]
                    nextvalues = values[b, next_pos]
                else:
                    nextvalues = 0.0
                
                delta = token_level_rewards[b, curr_pos] + gamma * nextvalues - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = advantages[b, curr_pos] + values[b, curr_pos]
            
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# adapted and modified from verl.trainer.ppo.core_algos
# skip env tokens when assigning next values and accumulating TD error
def compute_gae_advantage_return_multi_turn_old(
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
    
    adv1, ret1 = compute_bi_level_gae_advantage_return(rewards, values1, eos_mask, gamma, lam, high_level_gamma=0.95, response_mask=eos_mask)
    adv2, ret2 = compute_bi_level_gae_advantage_return(rewards, values2, eos_mask, gamma, lam, high_level_gamma=0.95, response_mask=eos_mask)
    # adv1, ret1 = compute_gae_advantage_return(rewards, values1, eos_mask, gamma, lam)
    # adv2, ret2 = compute_gae_advantage_return_multi_turn(rewards, values2, eos_mask, gamma, lam)

    ret1 *= eos_mask
    ret2 *= eos_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f' [CORRECT] \n\n{adv1=}, \n\n{adv2=}, \n\n{ret1=}, \n\n{ret2=}')