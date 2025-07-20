"""
RAG Utils - A module containing functions for text embedding and retrieval
with async support for parallel processing
"""
import os
import re
import torch
import asyncio
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import json
from openai import AsyncOpenAI, OpenAI
import concurrent.futures

# Initialize global vector database
VECTOR_DB = []  # Each element is a tuple (chunk, embedding)

def check_cuda_availability():
    """Check if CUDA is available and print device information"""
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("GPU设备数量:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))
            print(f"GPU {i} 内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        print("警告: 未检测到可用的GPU/CUDA。程序将使用CPU运行，这会显著降低性能。")
        print_gpu_diagnostics()

def print_gpu_diagnostics():
    """Print diagnostic information when GPU is not available"""
    print("GPU诊断:")
    print("  - PyTorch检测不到CUDA，请确认:")
    print("    1. 您已安装GPU版本的PyTorch (而不是CPU版本)")
    print("    2. 已安装CUDA并正确配置环境变量")
    print("    3. GPU驱动程序已正确安装")
    print("  - 尝试运行以下命令安装GPU版本的PyTorch:")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("    (请根据您的CUDA版本选择正确的安装指令)")

def test_gpu_memory():
    """Test GPU memory allocation"""
    if torch.cuda.is_available():
        print("测试GPU内存分配...")
        try:
            test_tensor = torch.zeros((1, 1000, 1000), device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
            print("GPU内存测试成功!")
            return True
        except RuntimeError as e:
            print(f"GPU内存测试失败: {e}")
            print("将默认使用CPU")
            return False
    return False

def normalize_text(text):
    """Normalize text, remove punctuation and extra spaces"""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def initialize_embedding_model(model_name='BAAI/bge-large-zh-v1.5'):
    """Initialize embedding model"""
    print("正在加载嵌入模型...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"模型将初始化在 {device} 设备上")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    print(f"模型 {model_name} 已成功加载")
    
    # For BGE models, inform user about the special prefix
    if "BAAI/bge" in model_name:
        print("注意: 使用BGE模型时，查询语句会自动添加'查询：'前缀以提高性能")
    
    return model

def encode_text(model, sentences, batch_size=128, use_gpu=True):
    """Encode text using embedding model"""
    print("\n正在编码句子...")
    
    # Check if GPU is available and requested
    if torch.cuda.is_available() and use_gpu:
        device = 'cuda'
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        if use_gpu and not torch.cuda.is_available():
            print("警告: 请求使用GPU但未检测到CUDA设备，将使用CPU替代")
        print("使用CPU进行编码")
    
    try:
        # Move model to specified device
        model = model.to(device)
        print(f"模型已成功移至 {device} 设备")
        
        # Set smaller batch size to prevent memory issues
        if device == 'cuda' and batch_size > 64:
            print(f"在GPU上使用批处理大小 {batch_size}，如果内存不足，请考虑减小此值")
            
        embeddings = model.encode(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device
        )
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
        
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"GPU内存不足错误: {e}")
            print("尝试减小批处理大小并重试...")
            if batch_size > 16:
                # Recursive call with smaller batch size
                return encode_text(model, sentences, batch_size=batch_size//2, use_gpu=use_gpu)
            else:
                print("批处理大小已经很小，将回退到CPU...")
                model = model.to('cpu')
                embeddings = model.encode(
                    sentences, 
                    batch_size=16, 
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device='cpu'
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings
        else:
            raise  # Re-raise other errors

def load_dataset(file_path):
    """Load data from a file"""
    dataset = []
    with open(file_path, encoding='utf-8') as file:
        dataset = file.readlines()
        print(f'已加载 {len(dataset)} 条记录')
    return dataset

def add_chunks_to_database(chunks, model, use_gpu=True, batch_size=64):
    """Add multiple text chunks to the vector database"""
    global VECTOR_DB
    
    # Batch encode all texts for efficiency
    print(f"使用{'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'}进行向量化处理...")
    embeddings = encode_text(model, chunks, batch_size=batch_size, use_gpu=use_gpu)
    
    # Convert encoding results to Python lists and store in the database
    for i, chunk in enumerate(chunks):
        embedding_list = embeddings[i].cpu().numpy().tolist()
        VECTOR_DB.append((chunk, embedding_list))
    
    print(f"已将 {len(chunks)} 条记录添加到向量数据库")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, model, embedding_model_name='BAAI/bge-large-zh-v1.5', top_n=3, threshold=0.5, use_gpu=True):
    """Retrieve similar texts from the vector database"""
    global VECTOR_DB
    
    # Add special prefix for BGE models
    if "BAAI/bge" in embedding_model_name:
        query = "查询：" + query
    
    # Check if GPU is available and requested
    if torch.cuda.is_available() and use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        if use_gpu and not torch.cuda.is_available():
            print("警告: 检索时请求使用GPU但未检测到CUDA，将使用CPU")
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Encode query
    query_embedding = model.encode(
        query, 
        convert_to_tensor=True,
        device=device
    )
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
    query_embedding = query_embedding.cpu().numpy().tolist()
    
    # Calculate similarities
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Filter results
    filtered_results = [(chunk, sim) for chunk, sim in similarities[:top_n] if sim >= threshold]
    return filtered_results

def generate_response(input_query, model, retrieved_knowledge, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus"):
    """Generate a response using LLM based on retrieved knowledge (synchronous version)"""
    # Build prompt
    if retrieved_knowledge:
        chunks_text = '\n'.join([' - {}'.format(chunk) for chunk, similarity in retrieved_knowledge])
        instruction_prompt = f'''你是一个有帮助的聊天机器人。这里是检索到的可能有帮助的相关信息, 只允许使用以下的上下文信息来回答问题, 并且不要编造任何新信息:
{chunks_text}
'''
    else:
        # Prompt when no relevant information is found
        instruction_prompt = '''你是一个有帮助的聊天机器人。由于没有在数据库种找到与用户问题相关的信息, 你必须告诉用户你没有检索到这方面的信息并且不要尝试回答问题或编造信息。
'''
    
    # Call language model
    try:
        # Create OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # Create chat completion request
        completion = client.chat.completions.create(
            model=model_name,  # Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query}
            ],
        )
        
        # Return response
        assistant_response = completion.choices[0].message.content
        return assistant_response
        
    except Exception as e:
        error_msg = f"调用语言模型时出错: {e}\n请确认API配置是否正确，以及网络连接是否正常。"
        
        # If API call fails, provide a simple local response
        if retrieved_knowledge:
            error_msg += "\n\n由于API调用失败，以下是从检索到的知识中提取的信息:"
            for i, (chunk, similarity) in enumerate(retrieved_knowledge):
                error_msg += f"\n{i+1}. 相似度 {similarity:.4f}: {chunk}"
        else:
            error_msg += "\n\n未找到相关知识，且API调用失败。请尝试其他问题或检查API配置。"
        
        return error_msg

async def generate_response_async(input_query, model, retrieved_knowledge, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus"):
    """Generate a response using LLM based on retrieved knowledge (async version)"""
    # Build prompt
    if retrieved_knowledge:
        chunks_text = '\n'.join([' - {}'.format(chunk) for chunk, similarity in retrieved_knowledge])
        instruction_prompt = f'''你是一个专业的语言分析机器人，专门负责判断句子是否有歧义。这里是可能有帮助的相关信息，请只使用以下上下文信息进行分析，不要编造任何新信息:
{chunks_text}

请按照以下规则回答:
1. 首先判断句子是否存在歧义，回答"是"或"否"
2. 如果有歧义，请详细说明: 句子可能的不同解读方式（列出至少两种不同理解）

请确保你的分析清晰、准确并简洁，便于人工审核。如果确实无法判断，请说明原因。
'''
    else:
        # Prompt when no relevant information is found
        instruction_prompt = '''你是一个专业的语言分析机器人，专门负责判断句子是否有歧义。由于没有在数据库种找到与用户问题相关的信息, 你必须告诉用户你没有检索到这方面的歧义信息, 并且不要尝试回答问题或编造信息。
'''
    
    # Call language model
    try:
        # Create AsyncOpenAI client
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # Create chat completion request asynchronously
        completion = await client.chat.completions.create(
            model=model_name,  # Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query}
            ],
        )
        
        # Return response
        assistant_response = completion.choices[0].message.content
        return assistant_response
        
    except Exception as e:
        error_msg = f"调用语言模型时出错: {e}\n请确认API配置是否正确，以及网络连接是否正常。"
        
        # If API call fails, provide a simple local response
        if retrieved_knowledge:
            error_msg += "\n\n由于API调用失败，以下是从检索到的知识中提取的信息:"
            for i, (chunk, similarity) in enumerate(retrieved_knowledge):
                error_msg += f"\n{i+1}. 相似度 {similarity:.4f}: {chunk}"
        else:
            error_msg += "\n\n未找到相关知识，且API调用失败。请尝试其他问题或检查API配置。"
        
        return error_msg

async def process_query_async(input_query, model, model_name, use_gpu, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", llm_model="qwen-plus", threshold=0.4):
    """Process a query asynchronously - from retrieval to response generation"""
    # Retrieve relevant knowledge
    retrieved_knowledge = retrieve(
        query=input_query, 
        model=model, 
        embedding_model_name=model_name,
        threshold=threshold, 
        use_gpu=use_gpu
    )
    
    # Display device information
    if torch.cuda.is_available() and use_gpu:
        cur_device = next(model.parameters()).device
        device_info = f"模型当前在设备: {cur_device}"
    else:
        device_info = "模型当前在CPU上运行"
    
    # Format retrieved knowledge
    knowledge_info = '检索到的知识:\n'
    for chunk, similarity in retrieved_knowledge:
        knowledge_info += f' - (相似度: {similarity:.4f}) {chunk}\n'
    
    # Generate response asynchronously
    response = await generate_response_async(
        input_query=input_query,
        model=model,
        retrieved_knowledge=retrieved_knowledge,
        api_key=api_key,
        base_url=base_url,
        model_name=llm_model
    )
    
    # Return all information
    return {
        'query': input_query,
        'device_info': device_info,
        'knowledge_info': knowledge_info,
        'response': response
    }

async def process_multiple_queries_async(queries, model, model_name, use_gpu, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", llm_model="qwen-plus", threshold=0.4):
    """Process multiple queries in parallel using asyncio"""
    # Create tasks for all queries
    tasks = [
        process_query_async(
            query, model, model_name, use_gpu, api_key, base_url, llm_model, threshold
        ) 
        for query in queries
    ]
    
    # Run all tasks concurrently and return results
    return await asyncio.gather(*tasks)