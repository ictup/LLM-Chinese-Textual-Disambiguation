"""
非RAG版本的异步并行千问应用
- 保留异步并行处理多个查询的功能
- 保留彩色输出和用户界面
- 移除所有检索增强生成(RAG)相关代码
- 直接与千问API交互
- 批量模式下总是使用独立对话，确保问答互不干扰
"""

import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import json
from openai import AsyncOpenAI, OpenAI

# 设置环境变量以避免OpenMP相关错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 彩色输出在终端中
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_colored(text, color):
    """用彩色打印文本"""
    print(f"{color}{text}{Colors.ENDC}")

# 对话记忆管理
CONVERSATIONS = {}  # 用于存储多轮对话历史

async def generate_response_async(input_query, api_key, conversation_id=None, yes_no_only=False, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus"):
    """异步方式生成回复（使用对话历史）"""
    global CONVERSATIONS
    
    # 创建新对话或获取现有对话
    if conversation_id is None:
        conversation_id = f"conv_{int(time.time())}_{hash(input_query) % 10000}"  # 添加查询哈希值以增加唯一性
    
    if conversation_id not in CONVERSATIONS:
        CONVERSATIONS[conversation_id] = []
        
    # 添加用户输入到对话历史
    CONVERSATIONS[conversation_id].append({"role": "user", "content": input_query})
    
    try:
        # 创建异步OpenAI客户端
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # 准备系统提示
        if yes_no_only:
            system_prompt = '''你是一个有帮助的聊天机器人。你只能回复"是"或者"否"。'''
        else:
            system_prompt = '''你是一个专业的语言分析机器人，专门负责判断句子是否有歧义。请按照以下规则回答:
1. 首先判断句子是否存在歧义，回答"是"或"否"
2. 如果有歧义，请说明:句子可能的不同解读方式（列出至少两种不同理解）

请确保你的分析清晰、准确并简洁，便于人工审核。如果确实无法判断，直接回复无法判断。
'''
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(CONVERSATIONS[conversation_id])
        
        # 异步调用API
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        
        # 获取回复
        assistant_response = completion.choices[0].message.content
        
        # 将助手回复添加到对话历史
        CONVERSATIONS[conversation_id].append({"role": "assistant", "content": assistant_response})
        
        # 如果对话历史过长，仅保留最近的多轮对话
        if len(CONVERSATIONS[conversation_id]) > 20:  # 保留最近10轮对话
            CONVERSATIONS[conversation_id] = CONVERSATIONS[conversation_id][-20:]
        
        return {
            'query': input_query,
            'response': assistant_response,
            'conversation_id': conversation_id
        }
        
    except Exception as e:
        error_msg = f"调用语言模型时出错: {e}\n请确认API配置是否正确，以及网络连接是否正常。"
        return {
            'query': input_query,
            'response': error_msg,
            'conversation_id': conversation_id,
            'error': str(e)
        }

async def process_query_async(query, api_key, conversation_id=None, yes_no_only=False, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus"):
    """异步处理单个查询"""
    # 生成回复
    result = await generate_response_async(
        input_query=query,
        api_key=api_key,
        conversation_id=conversation_id,
        yes_no_only=yes_no_only,
        base_url=base_url,
        model_name=model_name
    )
    
    return result

async def process_multiple_queries_async(queries, api_key, always_independent=True, conversation_ids=None, yes_no_only=False, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus"):
    """并行处理多个查询"""
    # 如果设置为总是独立对话，则忽略传入的conversation_ids
    if always_independent:
        conversation_ids = [None] * len(queries)  # 为每个查询创建新对话
    elif conversation_ids is None:
        conversation_ids = [None] * len(queries)
    elif len(conversation_ids) != len(queries):
        raise ValueError("queries和conversation_ids的长度必须相同")
    
    # 创建异步任务
    tasks = [
        process_query_async(
            query=query,
            api_key=api_key,
            conversation_id=conv_id,
            yes_no_only=yes_no_only,
            base_url=base_url,
            model_name=model_name
        )
        for query, conv_id in zip(queries, conversation_ids)
    ]
    
    # 并行执行所有任务
    return await asyncio.gather(*tasks)

def save_conversations(file_path):
    """保存所有对话历史到文件"""
    global CONVERSATIONS
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(CONVERSATIONS, f, ensure_ascii=False, indent=2)
    print_colored(f"对话历史已保存到 {file_path}", Colors.GREEN)

def load_conversations(file_path):
    """从文件加载对话历史"""
    global CONVERSATIONS
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            CONVERSATIONS = json.load(f)
        print_colored(f"已从 {file_path} 加载对话历史", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"加载对话历史时出错: {e}", Colors.RED)
        return False

async def process_single_query(query, api_key, conversation_id=None, yes_no_only=False, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus"):
    """处理单个查询并打印结果"""
    print_colored(f"\n处理查询: '{query}'", Colors.YELLOW)
    
    # 异步处理查询
    result = await process_query_async(
        query=query,
        api_key=api_key,
        conversation_id=conversation_id,
        yes_no_only=yes_no_only,
        base_url=base_url,
        model_name=model_name
    )
    
    # 打印结果
    print_colored("聊天机器人回应:", Colors.HEADER)
    print_colored(result['response'], Colors.ENDC)
    
    return result

async def batch_mode():
    """以批处理模式运行应用程序，用于一次处理多个查询"""
    print_colored("===== 千问直接对话系统 (无RAG版) =====", Colors.HEADER)
    print_colored("此版本不使用检索增强生成，直接与千问API交互", Colors.BLUE)
    print_colored("批量模式下总是使用独立对话，确保每个问答互不干扰", Colors.YELLOW)
    
    # 询问API密钥
    api_key = input("请输入模型API密钥 (dashscope): ")
    if not api_key:
        api_key = "sk-e0ed7781c4724b898dc7a99b1b485f29"  # 默认密钥
        print_colored(f"使用默认API密钥: {api_key[:5]}...", Colors.YELLOW)
    
    # 设置基础URL和模型
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen-plus"
    model_name = input(f"请输入模型名称 (默认: {default_model}): ") or default_model
    
    # 是否限制为是/否回答
    yes_no_mode = input("是否限制模型只能回答'是'或'否'? (y/n, 默认n): ").lower() == 'y'
    if yes_no_mode:
        print_colored("已启用仅'是'或'否'回答模式", Colors.YELLOW)
    
    # 询问是否加载历史对话
    load_history = input("是否加载历史对话? (y/n, 默认n): ").lower() == 'y'
    if load_history:
        history_path = input("请输入历史文件路径: ")
        if history_path:
            load_conversations(history_path)
    
    # 当前会话ID
    current_conversation_id = None
    
    # 主应用循环
    while True:
        print_colored("\n请选择模式:", Colors.HEADER)
        print_colored("1. 单一对话模式", Colors.BLUE)
        print_colored("2. 批量查询模式", Colors.GREEN)
        print_colored("3. 保存对话历史", Colors.YELLOW)
        print_colored("4. 创建新对话", Colors.YELLOW)
        print_colored("5. 退出程序", Colors.RED)
        
        choice = input("请输入选择 (1/2/3/4/5): ")
        
        if choice == '5':
            print_colored("退出程序...", Colors.RED)
            save_option = input("是否保存对话历史? (y/n, 默认y): ")
            if save_option.lower() != 'n':
                save_path = input("请输入保存路径 (默认: conversations.json): ") or "conversations.json"
                save_conversations(save_path)
            break
            
        elif choice == '4':
            # 创建新对话
            current_conversation_id = None
            print_colored("已创建新对话", Colors.GREEN)
            
        elif choice == '3':
            # 保存对话历史
            save_path = input("请输入保存路径 (默认: conversations.json): ") or "conversations.json"
            save_conversations(save_path)
            
        elif choice == '1':
            # 单一对话模式
            print_colored(f"\n进入单一对话模式. 输入 'exit' 返回主菜单.", Colors.BLUE)
            if current_conversation_id:
                print_colored(f"当前对话ID: {current_conversation_id}", Colors.YELLOW)
            else:
                print_colored("新对话", Colors.YELLOW)
            
            while True:
                query = input("\n请输入问题: ")
                if query.lower() == 'exit':
                    break
                    
                start_time = time.time()
                result = await process_single_query(
                    query=query,
                    api_key=api_key,
                    conversation_id=current_conversation_id,
                    yes_no_only=yes_no_mode,
                    base_url=base_url,
                    model_name=model_name
                )
                end_time = time.time()
                
                # 更新当前会话ID
                current_conversation_id = result['conversation_id']
                
                print_colored(f"处理时间: {end_time - start_time:.2f} 秒", Colors.YELLOW)
                
        elif choice == '2':
            # 批量查询模式
            print_colored("\n进入批量查询模式.", Colors.GREEN)
            print_colored("批量模式下为每个查询使用独立对话，确保每个问答互不干扰", Colors.YELLOW)
            print_colored("请选择批量查询输入方式:", Colors.HEADER)
            print_colored("1. 手动输入多个查询", Colors.BLUE)
            print_colored("2. 从文本文件读取查询", Colors.BLUE)
            
            input_method = input("请选择 (1/2): ")
            
            queries = []
            output_file = None
            
            if input_method == '1':
                # 手动输入
                print("请输入多个查询，每行一个查询。输入空行完成输入.")
                while True:
                    query = input("> ")
                    if not query:  # 空行完成输入
                        break
                    queries.append(query)
            
            elif input_method == '2':
                # 从文件读取
                queries_file = input("请输入查询文件路径: ")
                if not os.path.exists(queries_file):
                    print_colored(f"错误: 找不到文件 {queries_file}", Colors.RED)
                    continue
                
                # 从文件读取查询
                try:
                    with open(queries_file, 'r', encoding='utf-8') as f:
                        queries = [line.strip() for line in f.readlines() if line.strip()]
                    print_colored(f"已从文件加载 {len(queries)} 个查询", Colors.GREEN)
                    
                    # 询问是否要将结果保存到文件
                    save_option = input("是否将结果保存到文件中? (y/n, 默认y): ").lower()
                    if save_option != 'n':
                        # 默认输出文件名在扩展名前添加"_results"
                        default_output = os.path.splitext(queries_file)[0] + "_results" + os.path.splitext(queries_file)[1]
                        output_path = input(f"请输入输出文件路径 (默认: {default_output}): ") or default_output
                        output_file = output_path
                except Exception as e:
                    print_colored(f"读取文件时出错: {e}", Colors.RED)
                    continue
            else:
                print_colored("无效的选择，返回主菜单。", Colors.RED)
                continue
            
            if not queries:
                print_colored("没有输入查询！", Colors.RED)
                continue
                
            print_colored(f"准备处理 {len(queries)} 个查询...", Colors.BLUE)
            
            # 设置为总是使用独立对话 - 不再询问用户
            independent_conversations = True
            print_colored("每个查询使用独立对话，确保问答互不干扰", Colors.YELLOW)
            
            # 为每个查询创建独立的对话ID
            conversation_ids = [None] * len(queries)  # None表示创建新对话
            
            # 并行处理所有查询
            start_time = time.time()
            results = await process_multiple_queries_async(
                queries=queries,
                api_key=api_key,
                always_independent=True,  # 总是使用独立对话
                conversation_ids=conversation_ids,
                yes_no_only=yes_no_mode,
                base_url=base_url,
                model_name=model_name
            )
            end_time = time.time()
            
            # 准备输出数据（用于潜在的文件保存）
            output_lines = []
            
            # 打印结果
            for i, result in enumerate(results):
                print_colored(f"\n--- 查询 {i+1}: '{result['query']}' ---", Colors.HEADER)
                print_colored("聊天机器人回应:", Colors.HEADER)
                print_colored(result['response'], Colors.ENDC)
                
                # 为文件准备输出行
                output_lines.append(f"{result['query']} |||| {result['response']}")
                
            print_colored(f"\n总处理时间: {end_time - start_time:.2f} 秒 (平均每个查询 {(end_time - start_time)/len(queries):.2f} 秒)", Colors.YELLOW)
            
            # 如果需要，将结果保存到文件
            if output_file:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for line in output_lines:
                            f.write(line + "\n")
                    print_colored(f"结果已保存到文件: {output_file}", Colors.GREEN)
                except Exception as e:
                    print_colored(f"保存结果时出错: {e}", Colors.RED)
        else:
            print_colored("无效的选择，请重试.", Colors.RED)

async def main_async():
    """应用程序的异步主入口点"""
    await batch_mode()

def main():
    """应用程序的非异步主入口点"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
