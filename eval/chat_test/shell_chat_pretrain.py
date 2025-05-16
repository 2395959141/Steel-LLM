import torch
import sys
from pathlib import Path
import os

# 添加必要的路径，以便导入模型
parent_dir = Path(__file__).parent.parent.parent.resolve()
steel_model_dir = parent_dir / "pretrain_modify_from_TinyLlama" / "model" / "steel_modify_from_qwen_1_5"

# 添加tokenizer路径
tokenizer_dir = "/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct"

sys.path.append(str(parent_dir))
sys.path.append(str(steel_model_dir))  # 添加模型目录到路径中

# 导入SteelLLM模型和tokenizer
from pretrain_modify_from_TinyLlama.model.steel_modify_from_qwen_1_5.modeling_steel import SteelForCausalLM
from transformers import AutoConfig, AutoTokenizer

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 1. 加载 state.pth 文件
checkpoint_path = "/DATA/disk2/yuhang/.cache/ckpt/steel_llm/step-050000-iter-400000-ckpt/state.pth"
state = torch.load(checkpoint_path, map_location=device)

# 2. 获取模型配置
config = AutoConfig.from_pretrained(
    str(steel_model_dir),
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    use_custom_rmsnorm=True
)

# 设置block_size和flash_attn
block_size = 200
config.block_size = block_size
config.use_flash_attn = torch.cuda.is_available()

# 3. 初始化模型
model = SteelForCausalLM(config)

# 4. 将state中的权重加载到模型中
model.load_state_dict(state["model"], strict=True)
model.to(device)
model.eval()

# 5. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    str(tokenizer_dir),
    trust_remote_code=True
)

# 6. 构建对话模板
def build_prompt(history):
    prompt = ""
    for i, (query, response) in enumerate(history):
        prompt += f"用户: {query}\n"
        if response:
            prompt += f"助手: {response}\n"
    
    # 最后一轮对话如果没有助手回复，添加助手提示符
    if history and not history[-1][1]:
        prompt += f"助手: "
    
    # 将<|im_end|>添加到输入文本的末尾
    prompt += "<|im_end|>"
    
    return prompt.strip()

# 7. 使用transformers的generate函数进行推理
@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    # 确保prompt不包含<|im_end|>，然后手动添加
    if not prompt.endswith("<|im_end|>"):
        prompt = prompt + "<|im_end|>"
        
    # 对输入进行编码
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 如果输入过长，截断到适合模型处理的长度
    if input_ids.size(1) > config.block_size:
        input_ids = input_ids[:, -config.block_size:]
    
    # 使用transformers的generate函数
    streamer = None
    try:
        from transformers import TextStreamer
        streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    except:
        print("无法导入TextStreamer，不进行流式输出")
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.05,
        streamer=streamer,
        use_cache=True
    )
    
    # 获取新生成的token
    generated_ids = output_ids[0, input_ids.size(1):]
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# 交互式对话
def interactive_chat():
    history = []
    clear_screen()
    print("=" * 50)
    print("欢迎使用 Steel-LLM 交互式对话系统")
    print("输入'退出'或'exit'结束对话")
    print("输入'清空'或'clear'清除对话历史")
    print("输入'设置'或'settings'调整生成参数")
    print("所有输入文本将自动添加<|im_end|>标记以测试预训练模型能力")
    print("=" * 50)
    
    # 可调整的参数
    generation_params = {
        "max_new_tokens": 200,
        "temperature": 0.7,  # 调整为更合适的温度
        "top_p": 0.9
    }
    
    while True:
        user_input = input("\n用户: ")
        
        # 检查退出命令
        if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
            print("谢谢使用，再见！")
            break
            
        # 检查清空历史命令
        if user_input.lower() in ['清空', 'clear']:
            history = []
            clear_screen()
            print("对话历史已清空")
            continue
            
        # 检查设置命令
        if user_input.lower() in ['设置', 'settings']:
            try:
                temp = float(input("请输入温度参数(0.1-1.0，当前值: {:.1f}): ".format(generation_params["temperature"])))
                if 0.1 <= temp <= 1.0:
                    generation_params["temperature"] = temp
                    
                tokens = int(input("请输入最大生成长度(10-500，当前值: {}): ".format(generation_params["max_new_tokens"])))
                if 10 <= tokens <= 500:
                    generation_params["max_new_tokens"] = tokens
                    
                print(f"参数已更新: 温度={generation_params['temperature']}, 最大长度={generation_params['max_new_tokens']}")
            except ValueError:
                print("输入无效，保持原有设置")
            continue
        
        # 添加用户输入到历史
        history.append((user_input, ""))
        
        # 构建完整提示
        prompt = build_prompt(history)
        
        # 显示实际输入到模型的内容
        print(f"\n实际输入到模型: {prompt}")
        print("\n助手: ", end="", flush=True)
        
        # 生成回复
        response = generate(
            prompt, 
            max_new_tokens=generation_params["max_new_tokens"], 
            temperature=generation_params["temperature"], 
            top_p=generation_params["top_p"]
        )
        
        # 打印完整回复
        print(response)
        
        # 更新历史
        history[-1] = (user_input, response)

# 8. 执行简单测试
print("\n" + "="*50)
print("测试模型能力（每个输入都会添加<|im_end|>标记以测试预训练模型）:")
print("="*50)

# 测试对话格式
print("\n测试对话格式:")
test_prompt = "用户: 你好，请介绍一下你自己\n助手: "
print(f"输入: {test_prompt}")
print(f"实际输入到模型: {test_prompt}<|im_end|>")
test_response = generate(test_prompt, max_new_tokens=100, temperature=0.7)
print(f"输出: {test_response}")

# 测试纯文本理解能力
print("\n测试纯文本理解能力:")
plain_test_prompt = "北京是中国的首都"
print(f"输入: {plain_test_prompt}")
print(f"实际输入到模型: {plain_test_prompt}<|im_end|>")
plain_test_response = generate(plain_test_prompt, max_new_tokens=100, temperature=0.7)
print(f"输出: {plain_test_response}")

# 测试知识问答能力
print("\n测试知识问答能力:")
qa_test_prompt = "请介绍一下人工智能的发展历程"
print(f"输入: {qa_test_prompt}")
print(f"实际输入到模型: {qa_test_prompt}<|im_end|>")
qa_test_response = generate(qa_test_prompt, max_new_tokens=150, temperature=0.7)
print(f"输出: {qa_test_response}")

# 9. 开始交互式对话
print("\n" + "="*50)
print("开始交互式对话...")
print("="*50)
interactive_chat()