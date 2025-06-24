def demonstrate_correct_indexing():
    """演示正确的索引方式"""
    
    print("\n" + "="*60)
    print("=== 正确的Turn-level GAE实现 ===")
    print()
    
    # 模拟数据
    bs = 1
    seq_len = 10
    
    # 示例：prompt + response1 + obs + response2 + obs
    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 1, 1, 0, 0]])  # 1=response token
    token_rewards = torch.tensor([[0, 0, 0, 0, 2.0, 0, 0, 1.5, 0, 0]])  # reward在response结束位置
    values = torch.tensor([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]])
    
    print("示例数据：")
    print(f"Response mask: {response_mask.tolist()[0]}")
    print(f"Token rewards: {token_rewards.tolist()[0]}")  
    print(f"Values:        {values.tolist()[0]}")
    print()
    
    # 找到response结束位置
    response_ends = []
    for t in range(seq_len):
        if response_mask[0, t] == 1:
            # 检查是否为response结束
            if t == seq_len - 1 or response_mask[0, t + 1] == 0:
                response_ends.append(t)
    
    print(f"Response结束位置: {response_ends}")
    print()
    
    # 正确的GAE计算
    print("正确的GAE计算过程：")
    
    # 定义状态位置
    state_positions = [-1] + response_ends  # s_0在prompt结束，s_1, s_2...在各response结束
    print(f"状态位置: s_0@{state_positions[0]}, s_1@{state_positions[1]}, s_2@{state_positions[2]}")
    print()
    
    # 计算每个response的GAE
    advantages = {}
    lastgaelam = 0.0
    gamma = 0.99
    lam = 0.95
    
    # 从最后一个response开始反向计算
    for i in range(len(response_ends) - 1, -1, -1):
        response_end_pos = response_ends[i]
        
        print(f"--- 计算Response {i+1}的advantage ---")
        
        # 当前状态：response i结束时
        curr_state_pos = response_end_pos
        curr_value = values[0, curr_state_pos]
        
        # 前一个状态：response i开始前
        if i == 0:
            # 第一个response，前状态是prompt结束
            prev_state_pos = state_positions[0]  # 这里需要特殊处理prompt
            prev_value = values[0, 2]  # 假设位置2是prompt结束后的位置
        else:
            prev_state_pos = response_ends[i-1] 
            prev_value = values[0, prev_state_pos]
        
        # 下一个状态：response i+1结束时
        if i < len(response_ends) - 1:
            next_state_pos = response_ends[i+1]
            next_value = values[0, next_state_pos]
        else:
            next_value = 0.0
        
        # 获取该response的总reward
        response_reward = token_rewards[0, response_end_pos]
        
        print(f"  前状态值 V(s_{i}): {prev_value:.3f} @pos{prev_state_pos}")
        print(f"  当前状态值 V(s_{i+1}): {curr_value:.3f} @pos{curr_state_pos}")
        print(f"  下一状态值 V(s_{i+2}): {next_value:.3f}")
        print(f"  Response reward: {response_reward:.3f}")
        
        # 正确的delta计算
        delta = response_reward + gamma * next_value - curr_value
        lastgaelam = delta + gamma * lam * lastgaelam
        
        print(f"  Delta: {response_reward:.3f} + {gamma:.2f} * {next_value:.3f} - {curr_value:.3f} = {delta:.3f}")
        print(f"  Advantage: {lastgaelam:.3f}")
        print()
        
        advantages[i] = lastgaelam
    
    return advantages

def correct_compute_turn_level_gae(token_level_rewards, values, response_mask, 
                                 high_level_gamma, lam, bs, seq_len):
    """正确的turn-level GAE实现"""
    turn_advantages = torch.zeros_like(token_level_rewards)
    
    for b in range(bs):
        # 找到所有response结束位置
        response_ends = []
        for t in range(seq_len):
            if response_mask[b, t] == 1:
                if t == seq_len - 1 or response_mask[b, t + 1] == 0:
                    response_ends.append(t)
        
        if len(response_ends) == 0:
            continue
            
        # 从最后一个response开始反向计算GAE
        lastgaelam = 0.0
        
        for i in range(len(response_ends) - 1, -1, -1):
            curr_end_pos = response_ends[i]
            
            # 当前response结束时的状态值
            curr_value = values[b, curr_end_pos]
            
            # 下一个response结束时的状态值（作为nextvalue）
            if i < len(response_ends) - 1:
                next_end_pos = response_ends[i + 1] 
                next_value = values[b, next_end_pos]
            else:
                next_value = 0.0  # 最后一个response
            
            # 该response的reward
            response_reward = token_level_rewards[b, curr_end_pos]
            
            # 计算delta和advantage
            delta = response_reward + high_level_gamma * next_value - curr_value
            lastgaelam = delta + high_level_gamma * lam * lastgaelam
            
            # 将advantage分配给该response的所有token
            # 找到该response的开始位置
            if i == 0:
                # 第一个response，从第一个response token开始
                start_pos = 0
                for t in range(seq_len):
                    if response_mask[b, t] == 1:
                        start_pos = t
                        break
            else:
                # 后续response，从前一个response结束后开始
                start_pos = response_ends[i-1] + 1
                for t in range(start_pos, seq_len):
                    if response_mask[b, t] == 1:
                        start_pos = t
                        break
            
            # 分配advantage到该response的所有token
            for t in range(start_pos, curr_end_pos + 1):
                if response_mask[b, t] == 1:
                    turn_advantages[b, t] = lastgaelam
    
    return turn_advantages