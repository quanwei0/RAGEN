import torch

# 创建一个简单的1D tensor
reward_mask = torch.tensor([0, 1, 0, 1, 0])
print('reward_mask:', reward_mask)

# 不使用as_tuple=True
result_default = reward_mask.nonzero()
print('nonzero() 默认返回:', result_default)
print('类型:', type(result_default))

# 使用as_tuple=True
result_tuple = reward_mask.nonzero(as_tuple=True)
print('nonzero(as_tuple=True) 返回:', result_tuple)
print('类型:', type(result_tuple))
print('tuple长度:', len(result_tuple))
print('第一个元素:', result_tuple[0])
print('第一个元素类型:', type(result_tuple[0]))

print('\n--- 对于2D tensor的情况 ---')
# 2D tensor的例子
reward_mask_2d = torch.tensor([[0, 1, 0], [1, 0, 1]])
print('reward_mask_2d:', reward_mask_2d)

result_2d_tuple = reward_mask_2d.nonzero(as_tuple=True)
print('2D nonzero(as_tuple=True) 返回:', result_2d_tuple)
print('tuple长度:', len(result_2d_tuple))
print('第一个元素 (行索引):', result_2d_tuple[0])
print('第二个元素 (列索引):', result_2d_tuple[1]) 