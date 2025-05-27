# Ensemble-Hub API 使用指南

Enhanced API v2.0 支持灵活的集成方法选择和配置。

## 🚀 启动 API 服务器

### 基础启动
```bash
# 使用默认配置在项目根目录下启动
python -m ensemblehub.api

# 或使用 uvicorn（仅支持服务器配置，不支持集成方法配置）
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

### 命令行配置启动
**注意：集成方法配置仅在使用 `python -m ensemblehub.api` 启动时有效，uvicorn 启动方式不支持这些自定义参数。**

```bash
# 配置服务器地址和端口
python -m ensemblehub.api --host 0.0.0.0 --port 8080

# 配置模型选择和集成方法
python -m ensemblehub.api --model_selection_method zscore --ensemble_method progressive

# 配置循环推理（不使用模型选择）
python -m ensemblehub.api --model_selection_method all --ensemble_method loop --max_rounds 5

# 配置渐进式集成
python -m ensemblehub.api --ensemble_method progressive --progressive_mode length \
  --length_thresholds 50,100,200 --max_rounds 3

# 配置随机选择集成
python -m ensemblehub.api --model_selection_method all --ensemble_method random --max_rounds 3

# 配置循环选择集成（轮询模式）
python -m ensemblehub.api --model_selection_method all --ensemble_method loop \
  --max_rounds 5 --max_repeat 2

# 配置自定义模型
python -m ensemblehub.api --model_specs '[{"path":"model1","engine":"hf"},{"path":"model2","engine":"hf"}]'

# 显示模型归属信息
python -m ensemblehub.api --show_attribution

# 显示详细输入参数（用于调试）
python -m ensemblehub.api --show_input_details

# 完整配置示例
python -m ensemblehub.api \
  --host 0.0.0.0 --port 8080 \
  --model_selection_method zscore \
  --ensemble_method progressive \
  --progressive_mode mixed \
  --length_thresholds 100,200 \
  --special_tokens "<step>,<think>" \
  --max_rounds 5 \
  --score_threshold -2.0 \
  --max_repeat 3 \
  --show_attribution \
  --show_input_details
```

### 可用的命令行参数

#### 服务器配置
- `--host`: 服务器主机地址 (默认: 127.0.0.1)
- `--port`: 服务器端口 (默认: 8000)

#### 集成配置
- `--model_selection_method`: 模型选择方法
  - `zscore`: 基于 Z-score 的统计选择 (默认)
  - `all`: 使用所有模型
  - `random`: 随机选择模型
- `--ensemble_method`: 集成方法
  - `simple`: 简单奖励模型集成 (默认)
  - `progressive`: 渐进式集成
  - `random`: 随机集成
  - `loop`: 循环/轮询集成
- `--max_rounds`: 最大推理轮数 (默认: 10)
- `--score_threshold`: 分数阈值 (默认: -1.5)
- `--max_repeat`: 最大重复次数 (默认: 3)

#### 渐进式集成特定配置
- `--progressive_mode`: 渐进模式
  - `length`: 基于长度的模型切换
  - `token`: 基于特殊令牌的模型切换
  - `mixed`: 混合模式 (默认)
- `--length_thresholds`: 长度阈值列表，逗号分隔 (如: 50,100,200)
- `--special_tokens`: 特殊令牌列表，逗号分隔 (如: <step>,<think>)

#### 模型配置
- `--model_specs`: JSON 格式的模型规格列表

#### 调试和输出配置
- `--show_attribution`: 显示模型归属信息（哪个模型生成了哪部分输出）
- `--show_input_details`: 显示详细的输入参数（用于调试 API 请求）

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/status

## 📋 主要 API 端点

### 1. 基础信息
- `GET /` - API 信息和端点列表
- `GET /status` - 健康检查和可用方法
- `GET /v1/ensemble/methods` - 列出所有可用的集成方法

### 2. 配置管理
- `GET /v1/ensemble/config` - 获取当前配置
- `POST /v1/ensemble/config` - 更新配置

### 3. 推理端点
- `POST /v1/chat/completions` - OpenAI 兼容的聊天完成
- `POST /v1/loop/completions` - 专用循环推理端点（轮询模式）
- `POST /v1/ensemble/inference` - 直接集成推理
- `POST /v1/ensemble/batch` - 批量推理

### 4. 预设端点
- `POST /v1/ensemble/presets/simple` - 简单集成
- `POST /v1/ensemble/presets/selection_only` - 仅模型选择
- `POST /v1/ensemble/presets/aggregation_only` - 仅输出聚合

## 🔧 使用示例

### 1. 基础聊天完成（使用默认配置）

```bash
# 使用 prompt 字段（文本完成格式）
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "prompt": "What is 2+2?",
  "max_tokens": 100
}'

# 使用 messages 字段（聊天格式）
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 100
}'
```

### 2. 带集成配置的聊天完成

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "Solve this math problem: 15 × 23"}
  ],
  "max_tokens": 200,
  "ensemble_config": {
    "model_selection_method": "zscore",
    "aggregation_method": "reward_based",
    "aggregation_level": "sentence",
    "use_model_selection": true,
    "use_output_aggregation": true
  }
}'
```

### 3. 渐进式集成示例

#### 基于长度的渐进式集成
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "progressive-ensemble",
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "Explain how to solve 15 + 27"}
  ],
  "max_tokens": 256,
  "ensemble_config": {
    "ensemble_method": "progressive",
    "progressive_mode": "length",
    "length_thresholds": [500, 1000, 1500],
    "model_selection_method": "all",
    "show_attribution": true
  }
}'
```

#### 基于特殊 Token 的渐进式集成
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "progressive-token",
  "messages": [
    {"role": "user", "content": "Think step by step: What is the derivative of x^2?"}
  ],
  "ensemble_config": {
    "ensemble_method": "progressive",
    "progressive_mode": "token",
    "special_tokens": ["<\\think>", "<\\analyze>"],
    "show_attribution": true
  }
}'
```

### 4. 循环推理端点（轮询模式）

```bash
curl -X POST "http://localhost:8000/v1/loop/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "prompt": "解释量子计算的基本原理",
  "max_tokens": 300,
  "ensemble_config": {
    "max_rounds": 5,
    "score_threshold": -1.5
  }
}'
```

### 5. 批量请求示例

#### 批量处理多个问题
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "batch-ensemble",
  "messages": [
    [
      {"role": "user", "content": "What is 5 + 3?"}
    ],
    [
      {"role": "user", "content": "What is 10 * 7?"}
    ],
    [
      {"role": "system", "content": "You are a history expert."},
      {"role": "user", "content": "When was the Declaration of Independence signed?"}
    ]
  ],
  "max_tokens": 150,
  "ensemble_config": {
    "ensemble_method": "simple",
    "model_selection_method": "all",
    "show_attribution": true
  }
}'
```

#### 使用 Legacy Prompt 字段的批量请求
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "legacy-batch",
  "prompt": [
    "Calculate 8 * 9",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence"
  ],
  "max_tokens": 100
}'
```

### 6. 直接集成推理

```bash
curl -X POST "http://localhost:8000/v1/ensemble/inference" \
-H "Content-Type: application/json" \
-d '{
  "instruction": "You are a helpful math tutor.",
  "input": "Explain how to solve quadratic equations",
  "ensemble_config": {
    "model_selection_method": "all",
    "aggregation_method": "reward_based",
    "aggregation_level": "sentence",
    "model_selection_params": {},
    "aggregation_params": {
      "max_repeat": 3
    }
  },
  "max_rounds": 10,
  "score_threshold": -1.5,
  "max_tokens": 500
}'
```

### 5. 仅使用模型选择（不聚合输出）

```bash
curl -X POST "http://localhost:8000/v1/ensemble/presets/selection_only" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "What is machine learning?",
  "model_selection_method": "zscore",
  "max_tokens": 300
}'
```

### 6. 仅使用输出聚合（所有模型）

```bash
curl -X POST "http://localhost:8000/v1/ensemble/presets/aggregation_only" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Explain quantum computing",
  "aggregation_method": "round_robin",
  "aggregation_level": "sentence",
  "max_tokens": 400
}'
```

### 7. 批量推理

```bash
curl -X POST "http://localhost:8000/v1/ensemble/batch" \
-H "Content-Type: application/json" \
-d '{
  "examples": [
    {
      "instruction": "You are a helpful assistant.",
      "input": "What is 5+5?",
      "ensemble_config": {
        "model_selection_method": "all",
        "aggregation_method": "reward_based"
      }
    },
    {
      "instruction": "You are a math expert.",
      "input": "What is 10×10?",
      "ensemble_config": {
        "model_selection_method": "zscore",
        "aggregation_method": "random"
      }
    }
  ],
  "batch_size": 2
}'
```

## ⚙️ 配置选项

### 模型选择方法 (model_selection_method)
- `"zscore"` - 基于 Z-score 的模型选择（困惑度和置信度）
- `"all"` - 使用所有可用模型
- `"random"` - 随机选择模型子集
- `"llm_blender"` - LLM-Blender 方法（如果实现）

### 输出聚合方法 (aggregation_method)
- `"reward_based"` - 基于奖励模型分数选择输出
- `"random"` - 随机选择生成的输出
- `"round_robin"` - 轮询选择模型输出

### 聚合级别 (aggregation_level)
- `"sentence"` - 句子/段落级别聚合（生成过程中）
- `"token"` - 令牌级别聚合（例如 GaC）
- `"response"` - 完整响应级别聚合（例如投票）

## 🔗 Python 客户端示例

### 基础客户端类
```python
import requests
import json

class EnsembleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def chat_completion(self, messages=None, prompt=None, ensemble_config=None, **kwargs):
        """发送聊天完成请求"""
        payload = {
            "model": kwargs.get("model", "ensemble"),
            "max_tokens": kwargs.get("max_tokens", 256)
        }
        
        if messages:
            payload["messages"] = messages
        elif prompt:
            payload["prompt"] = prompt
        else:
            raise ValueError("Either messages or prompt must be provided")
            
        if ensemble_config:
            payload["ensemble_config"] = ensemble_config
            
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        return response.json()
    
    def batch_completion(self, conversations, ensemble_config=None, **kwargs):
        """批量处理多个对话"""
        payload = {
            "model": "batch-ensemble",
            "messages": conversations,  # List[List[Message]]
            "max_tokens": kwargs.get("max_tokens", 256)
        }
        
        if ensemble_config:
            payload["ensemble_config"] = ensemble_config
        
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        return response.json()
```

### 单个请求示例
```python
# 初始化客户端
client = EnsembleClient()

# 基础聊天请求
messages = [
    {"role": "user", "content": "What is artificial intelligence?"}
]
result = client.chat_completion(messages=messages)
print(result["choices"][0]["message"]["content"])

# 使用渐进式集成
messages = [
    {"role": "user", "content": "Solve: 2x + 5 = 15"}
]

config = {
    "ensemble_method": "progressive",
    "progressive_mode": "length",
    "length_thresholds": [500, 1000],
    "show_attribution": True
}

result = client.chat_completion(messages=messages, ensemble_config=config)
print(json.dumps(result, indent=2))
```

### 批量请求示例
```python
# 批量处理多个问题
conversations = [
    [{"role": "user", "content": "What is 15 + 27?"}],
    [{"role": "user", "content": "Calculate 8 * 9"}],
    [{"role": "user", "content": "What is the square root of 144?"}]
]

config = {
    "ensemble_method": "progressive",
    "progressive_mode": "token",
    "special_tokens": ["<\\think>"],
    "show_attribution": True
}

batch_result = client.batch_completion(conversations, config)

# 处理结果
for i, choice in enumerate(batch_result["choices"]):
    print(f"Conversation {i}:")
    print(f"  Response: {choice['message']['content']}")
    if choice.get("metadata", {}).get("attribution"):
        attr = choice["metadata"]["attribution"]
        print(f"  Attribution: {attr['summary']}")
    print()
```

## 📊 响应格式

### 单个请求响应格式
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710123456,
  "model": "ensemble",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Generated response text"
      },
      "finish_reason": "stop",
      "metadata": {
        "selected_models": ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
        "method": "all+progressive",
        "attribution": {
          "summary": "[R01:Qwen2.5-1.5B-Instruct] → [R02:Qwen2.5-0.5B-Instruct]",
          "detailed": [
            {
              "text": "First part of response",
              "model": "Qwen/Qwen2.5-1.5B-Instruct",
              "round": 1,
              "length": 50
            },
            {
              "text": "Second part of response",
              "model": "Qwen/Qwen2.5-0.5B-Instruct",
              "round": 2,
              "length": 45
            }
          ]
        }
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 95,
    "total_tokens": 105
  }
}
```

### 批量请求响应格式
批量请求会在 `choices` 数组中返回多个结果，每个结果对应一个输入对话：

```json
{
  "id": "chatcmpl-batch-...",
  "object": "chat.completion",
  "created": 1710123456,
  "model": "batch-ensemble",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Answer to first question"
      },
      "finish_reason": "stop",
      "metadata": {...}
    },
    {
      "index": 1,
      "message": {
        "role": "assistant",
        "content": "Answer to second question"
      },
      "finish_reason": "stop",
      "metadata": {...}
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 100,
    "total_tokens": 120
  }
}
```

### Ensemble Inference 响应
```json
{
  "id": "ensemble-uuid",
  "created": 1234567890,
  "result": {
    "output": "生成的文本...",
    "selected_models": ["model1", "model2"],
    "method": "zscore+reward_based",
    "config": {...}
  },
  "config": {...}
}
```

## 🛠️ 运行时配置更新

```bash
# 更新模型配置
curl -X POST "http://localhost:8000/v1/ensemble/config" \
-H "Content-Type: application/json" \
-d '{
  "model_specs": [
    {"path": "new-model-1", "engine": "hf", "device": "cuda:0"},
    {"path": "new-model-2", "engine": "hf", "device": "cuda:1"}
  ]
}'

# 更新默认集成配置
curl -X POST "http://localhost:8000/v1/ensemble/config" \
-H "Content-Type: application/json" \
-d '{
  "default_ensemble_config": {
    "model_selection_method": "zscore",
    "aggregation_method": "reward_based",
    "use_model_selection": true,
    "use_output_aggregation": true
  }
}'
```

## 🧪 测试 API

```bash
# 运行 API 测试
python test/test_api.py

# 或手动测试健康检查
curl http://localhost:8000/status
```

## 🛠️ 故障排除

### vLLM CUDA 内存分配错误

如果你在使用 vLLM 引擎时遇到以下错误：
```
captures_underway.empty() INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":3085
```

**解决方案：**

1. **使用命令行参数修复（推荐）：**
   ```bash
   python -m ensemblehub.api --vllm_enforce_eager --vllm_disable_chunked_prefill
   ```

2. **切换到 HuggingFace 引擎：**
   ```bash
   # 将模型配置从 "engine": "vllm" 改为 "engine": "hf"
   python -m ensemblehub.api --model_specs 'model_path:hf:cuda:0'
   ```

3. **环境变量设置：**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   python -m ensemblehub.api
   ```

### vLLM 层名称冲突错误

如果你在使用 vLLM 引擎时遇到以下错误：
```
Duplicate layer name: model.layers.X.self_attn.attn
```

**解决方案：**

1. **使用优化的 vLLM 配置（推荐）：**
   ```bash
   # 使用 LlamaFactory 风格的配置，适合单卡大模型
   python -m ensemblehub.api --ensemble_method random --model_selection_method all
   ```

2. **切换到 HuggingFace 引擎（最稳定）：**
   ```bash
   # 将模型配置从 "engine": "vllm" 改为 "engine": "hf" 
   python -m ensemblehub.api --model_specs 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:hf:cuda:0'
   ```

3. **调试模式设置：**
   ```bash
   # 如果仍有问题，使用调试模式启动
   export CUDA_LAUNCH_BLOCKING=1
   python -m ensemblehub.api --ensemble_method random
   ```

### HuggingFace Meta Tensor 错误

如果你在使用 HuggingFace 引擎时遇到以下错误：
```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

**解决方案：**

1. **使用 eager attention（推荐）：**
   ```bash
   python -m ensemblehub.api --hf_use_eager_attention
   ```

2. **禁用 device_map：**
   ```bash
   python -m ensemblehub.api --hf_disable_device_map
   ```

3. **降级 transformers 版本：**
   ```bash
   pip install transformers==4.35.0
   python -m ensemblehub.api
   ```

4. **使用自动设备分配：**
   ```bash
   # 将设备从 "cuda:X" 改为 "auto"
   python -m ensemblehub.api --model_specs 'model_path:hf:auto'
   ```

### GPU 内存不足错误

如果你遇到以下错误：
```
CUDA error: out of memory
```

**解决方案：**

1. **使用量化减少内存占用（推荐）：**
   ```bash
   # 使用 8-bit 量化
   python -m ensemblehub.api --hf_use_8bit
   
   # 使用 4-bit 量化（更节省内存）
   python -m ensemblehub.api --hf_use_4bit
   ```

2. **减少同时加载的模型数量：**
   ```bash
   # 只使用较小的模型
   python -m ensemblehub.api --model_specs 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:hf:cuda:0,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:hf:cuda:1'
   ```

3. **使用 CPU 运行大模型：**
   ```bash
   # 将大模型移到 CPU 上
   python -m ensemblehub.api --model_specs 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:hf:cpu'
   ```

4. **清理 GPU 缓存：**
   ```bash
   # 在运行前清理 GPU 缓存
   export CUDA_LAUNCH_BLOCKING=1
   python -c "import torch; torch.cuda.empty_cache()"
   python -m ensemblehub.api
   ```

### 常见问题

**Q: API 启动后无法访问？**
A: 检查防火墙设置，确保端口未被占用：
```bash
curl http://localhost:8000/status
```

**Q: 模型加载失败？**
A: 检查模型路径和设备配置，确保有足够的 GPU 内存：
```bash
nvidia-smi  # 检查 GPU 使用情况
```

**Q: 集成方法配置无效？**
A: 确保使用 `python -m ensemblehub.api` 而不是 `uvicorn` 来启动：
```bash
# ✅ 正确
python -m ensemblehub.api --ensemble_method loop

# ❌ 不支持自定义配置
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

## 🔗 lm-evaluation-harness 兼容性

Ensemble-Hub API 完全兼容 lm-evaluation-harness，支持所有标准参数：

### 使用 lm-eval 测试

```bash
# 基础测试
lm_eval --model openai-completions \
  --tasks gsm8k \
  --model_args model=ensemble,base_url=http://localhost:8000/v1/chat/completions,tokenizer_backend=None \
  --batch_size 2 \
  --num_fewshot 5

# 完整参数示例
OPENAI_API_KEY=dummy lm_eval \
  --model openai-completions \
  --tasks gsm8k,hendrycks_math \
  --model_args model=ensemble,base_url=http://localhost:8000/v1/chat/completions,tokenizer_backend=None \
  --batch_size 16 \
  --num_fewshot 5 \
  --limit 100
```

### 支持的 lm-eval 参数

API 完全支持以下 lm-evaluation-harness 参数：

- **`max_tokens`**: 最大生成令牌数
- **`temperature`**: 采样温度（0 表示贪婪解码）
- **`stop`**: 停止序列列表（如 `["Question:", "</s>", "<|im_end|>"]`）
- **`seed`**: 随机种子，用于结果可重现性

### 调试 lm-eval 请求

当需要调试 lm-evaluation-harness 发送的请求时：

```bash
# 启动 API 并显示输入详情
python -m ensemblehub.api --show_input_details

# 运行 lm-eval，API 日志将显示完整请求内容
```

示例日志输出：
```
================================================================================
Received API request:
Model: ensemble
Messages: None
Prompt: Question: Janet's ducks lay 16 eggs per day...
Temperature: 0.0
Max tokens: 256
Stop: ['Question:', '</s>', '<|im_end|>']
Seed: 1234
================================================================================
```

## 🔄 自动批量检测规则

API 会自动检测请求类型，无需使用不同的端点：

1. **单个请求**: 
   - `messages` 是 `List[Message]` 格式
   - `prompt` 是单个字符串

2. **批量请求**: 
   - `messages` 是 `List[List[Message]]` 格式
   - `prompt` 是字符串列表 `List[str]`

同一个 `/v1/chat/completions` 端点可以处理所有情况，自动识别并正确处理。

这个增强的 API 提供了完全的灵活性，让你可以根据需要选择和配置不同的集成方法！