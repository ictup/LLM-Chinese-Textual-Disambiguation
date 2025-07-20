"""
RAG Application - Asynchronous version that can process multiple queries in parallel
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from rag_util_module import (
    check_cuda_availability, 
    test_gpu_memory,
    initialize_embedding_model, 
    load_dataset, 
    add_chunks_to_database, 
    retrieve, 
    generate_response,
    process_query_async,
    process_multiple_queries_async
)

# For colored output in terminal
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
    """Print text with color in terminal"""
    print(f"{color}{text}{Colors.ENDC}")

async def process_single_query(query, model, model_name, use_gpu, api_key, base_url, llm_model):
    """Process a single query and print results"""
    print_colored(f"\n处理查询: '{query}'", Colors.YELLOW)
    
    # Process query asynchronously
    result = await process_query_async(
        input_query=query,
        model=model,
        model_name=model_name,
        use_gpu=use_gpu,
        api_key=api_key,
        base_url=base_url,
        llm_model=llm_model
    )
    
    # Print results
    print_colored(result['device_info'], Colors.BLUE)
    print_colored(result['knowledge_info'], Colors.GREEN)
    print_colored("聊天机器人回应:", Colors.HEADER)
    print_colored(result['response'], Colors.ENDC)
    return result

async def batch_mode():
    """Run the application in batch mode for processing multiple queries at once"""
    # Set environment variable for macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Check for CUDA availability
    check_cuda_availability()
    
    # Default model settings
    EMBEDDING_MODEL = 'BAAI/bge-large-zh-v1.5'  # 使用BAAI的中文嵌入模型
    
    # Set up file paths
    folder_path = input("请输入数据文件夹路径 (默认为当前目录): ") or "."
    file_path = input("请输入数据文件名 (例如: cat-facts-zh.txt): ")
    full_path = os.path.join(folder_path, file_path)
    
    # Check if the file exists
    if not os.path.exists(full_path):
        print(f"错误: 找不到文件 {full_path}")
        return
    
    # Ask whether to use GPU
    use_gpu = True  # Default to using GPU
    if torch.cuda.is_available():
        gpu_choice = input(f"检测到GPU: {torch.cuda.get_device_name(0)}，是否使用GPU进行向量化? (y/n，默认y): ").lower()
        use_gpu = not (gpu_choice == 'n')
        
        # If using GPU, test memory allocation
        if use_gpu:
            use_gpu = test_gpu_memory()
    else:
        print("未检测到可用的GPU/CUDA设备，将使用CPU运行")
        use_gpu = False
    
    # Set batch size
    default_batch_size = 64 if use_gpu else 32
    batch_size_input = input(f"请输入批处理大小 (默认: {default_batch_size}，较小的值会减少内存使用): ")
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else default_batch_size
    
    # Ask for API key
    api_key = input("请输入模型API密钥 (dashscope): ")
    if not api_key:
        print("警告: 未提供API密钥，回答功能将无法使用。")
        api_key = "sk-e0ed7781c4724b898dc7a99b1b485f29"  # Default key from the original code
    
    # Load dataset
    dataset = load_dataset(full_path)
    
    # Initialize embedding model
    model = initialize_embedding_model(EMBEDDING_MODEL)
    
    # Add data to vector database
    add_chunks_to_database(dataset, model, use_gpu=use_gpu, batch_size=batch_size)
    
    # Define base URL
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model = "qwen-plus"
    
    # Main application loop
    while True:
        print_colored("\n请选择模式:", Colors.HEADER)
        print_colored("1. 单一查询模式", Colors.BLUE)
        print_colored("2. 批量查询模式", Colors.GREEN)
        print_colored("3. 退出程序", Colors.RED)
        
        choice = input("请输入选择 (1/2/3): ")
        
        if choice == '3':
            print_colored("退出程序...", Colors.RED)
            break
            
        elif choice == '1':
            # Single query mode
            print_colored("\n进入单一查询模式. 输入 'exit' 返回主菜单.", Colors.BLUE)
            
            while True:
                query = input("\n请提出一个问题: ")
                if query.lower() == 'exit':
                    break
                    
                start_time = time.time()
                await process_single_query(
                    query=query,
                    model=model,
                    model_name=EMBEDDING_MODEL,
                    use_gpu=use_gpu,
                    api_key=api_key,
                    base_url=base_url,
                    llm_model=llm_model
                )
                end_time = time.time()
                print_colored(f"处理时间: {end_time - start_time:.2f} 秒", Colors.YELLOW)
                
        elif choice == '2':
            # Batch query mode
            print_colored("\n进入批量查询模式.", Colors.GREEN)
            print_colored("请选择批量查询输入方式:", Colors.HEADER)
            print_colored("1. 手动输入多个查询", Colors.BLUE)
            print_colored("2. 从文本文件读取查询", Colors.BLUE)
            
            input_method = input("请选择 (1/2): ")
            
            queries = []
            output_file = None
            
            if input_method == '1':
                # Manual input
                print("请输入多个查询，每行一个查询。输入空行完成输入.")
                while True:
                    query = input("> ")
                    if not query:  # Empty line to finish input
                        break
                    queries.append(query)
            
            elif input_method == '2':
                # Read from file
                queries_file = input("请输入查询文件路径: ")
                if not os.path.exists(queries_file):
                    print_colored(f"错误: 找不到文件 {queries_file}", Colors.RED)
                    continue
                
                # Read queries from file
                try:
                    with open(queries_file, 'r', encoding='utf-8') as f:
                        queries = [line.strip() for line in f.readlines() if line.strip()]
                    print_colored(f"已从文件加载 {len(queries)} 个查询", Colors.GREEN)
                    
                    # Ask if user wants to save results to file
                    save_option = input("是否将结果保存到文件中? (y/n, 默认y): ").lower()
                    if save_option != 'n':
                        # Default output filename adds "_results" before extension
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
            
            # Process all queries in parallel
            start_time = time.time()
            results = await process_multiple_queries_async(
                queries=queries,
                model=model,
                model_name=EMBEDDING_MODEL,
                use_gpu=use_gpu,
                api_key=api_key,
                base_url=base_url,
                llm_model=llm_model
            )
            end_time = time.time()
            
            # Prepare output data (for potential file saving)
            output_lines = []
            
            # Print results
            for i, result in enumerate(results):
                print_colored(f"\n--- 查询 {i+1}: '{result['query']}' ---", Colors.HEADER)
                print_colored(result['device_info'], Colors.BLUE)
                print_colored(result['knowledge_info'], Colors.GREEN)
                print_colored("聊天机器人回应:", Colors.HEADER)
                print_colored(result['response'], Colors.ENDC)
                
                # Prepare output line for file
                output_lines.append(f"{result['query']} |||| {result['response']}")
                
            print_colored(f"\n总处理时间: {end_time - start_time:.2f} 秒 (平均每个查询 {(end_time - start_time)/len(queries):.2f} 秒)", Colors.YELLOW)
            
            # Save results to file if required
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
    """Main async entry point for the application"""
    await batch_mode()

def main():
    """Non-async main entry point for the application"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()