import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

def test_llm_input_output_relationship():
    """测试LLM中input_ids和output.logits之间的关系"""
    
    # 1. 加载小型模型和分词器
    model_name = "gpt2"  # 使用GPT-2作为示例，比较小
    print(f"加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 添加pad_token（GPT-2默认没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 准备测试文本
    texts = [
        "Hello, how are you?",
        "The weather is nice today.",
        "Machine learning is fascinating!"
    ]
    
    print("=" * 60)
    print("测试文本:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
    
    # 3. 分词并创建batch
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=20)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("=" * 60)
    print("输入维度:")
    print(f"input_ids.shape: {input_ids.shape}")  # [batch_size, seq_len]
    print(f"attention_mask.shape: {attention_mask.shape}")
    
    # 4. 模型前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    print("=" * 60)
    print("输出维度:")
    print(f"output.logits.shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
    print(f"词汇表大小: {logits.shape[-1]}")
    
    # 5. 分析每个位置的预测
    batch_size, seq_len, vocab_size = logits.shape
    
    print("=" * 60)
    print("详细分析第一个样本:")
    print(f"原文: '{texts[0]}'")
    
    # 显示tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("Tokens:", tokens)
    print("Token IDs:", input_ids[0].tolist())
    
    print("\n每个位置的预测分析:")
    for pos in range(seq_len):
        if attention_mask[0, pos] == 1:  # 只分析有效位置
            current_token = tokens[pos]
            current_id = input_ids[0, pos].item()
            
            # 获取该位置的logits
            position_logits = logits[0, pos]  # [vocab_size]
            
            # 找到概率最高的top-3预测
            probs = F.softmax(position_logits, dim=-1)
            top_values, top_indices = torch.topk(probs, k=3)
            
            predicted_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
            
            print(f"位置 {pos}: 当前token='{current_token}' (ID:{current_id})")
            print(f"  预测下一个token的top-3:")
            for i, (token, prob) in enumerate(zip(predicted_tokens, top_values)):
                print(f"    {i+1}. '{token}' (概率: {prob:.4f})")
            print()

def test_critic_style_output():
    """模拟Critic网络的输出（每个位置输出一个价值）"""
    print("=" * 60)
    print("模拟Critic网络输出:")
    
    # 模拟critic网络的输出头：vocab_size -> 1
    batch_size, seq_len, vocab_size = 2, 10, 1000
    
    # 模拟原始LLM输出
    mock_logits = torch.randn(batch_size, seq_len, vocab_size)
    print(f"原始LLM logits: {mock_logits.shape}")
    
    # 模拟critic head: 线性层将vocab_size维度压缩到1
    critic_head = torch.nn.Linear(vocab_size, 1)
    
    # Critic输出：每个位置一个价值
    critic_values = critic_head(mock_logits)  # [batch_size, seq_len, 1]
    critic_values = critic_values.squeeze(-1)  # [batch_size, seq_len]
    
    print(f"Critic价值估计: {critic_values.shape}")
    print(f"每个位置的价值: {critic_values[0]}")
    
    # 模拟response部分截取
    response_length = 5
    response_values = critic_values[:, -response_length-1:-1]  # [batch_size, response_length]
    print(f"Response部分价值: {response_values.shape}")
    print(f"Response价值: {response_values[0]}")

if __name__ == "__main__":
    print("🚀 测试LLM输入输出关系")
    
    try:
        # 测试标准语言模型
        test_llm_input_output_relationship()
        
        # 测试critic风格的输出
        test_critic_style_output()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        print("💡 请确保已安装transformers库: pip install transformers torch") 