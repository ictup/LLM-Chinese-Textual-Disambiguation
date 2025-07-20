# README

Server
```python
vllm serve Qwen/Qwen/Qwen3-0.7B  --dtype=half --gpu_memory_utilization=0.9 --max_model_len=1024 --port 8999
vllm serve Qwen/Qwen/Qwen3-1.7B  --dtype=half --gpu_memory_utilization=0.9 --max_model_len=1024 --port 8999
vllm serve Qwen/Qwen/Qwen3-8B  --dtype=half --gpu_memory_utilization=0.9 --max_model_len=1024 --port 8999
```