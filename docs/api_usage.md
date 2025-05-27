# Ensemble-Hub API Usage Guide

Enhanced API v3.0 supports flexible ensemble method selection and configuration, provides OpenAI-compatible interface and automatic batch detection.

## 🚀 Starting the API Server

### Basic Startup
```bash
# Start with default configuration in project root directory
python -m ensemblehub.api

# Or use uvicorn (supports only server configuration, not ensemble method configuration)
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

### Command Line Configuration
**Note: Ensemble method configuration is only available when using `python -m ensemblehub.api`, not with uvicorn.**

```bash
# Configure server address and port
python -m ensemblehub.api --host 0.0.0.0 --port 8080

# Configure model selection and ensemble method
python -m ensemblehub.api --model_selection_method zscore --ensemble_method progressive

# Configure loop inference (without model selection)
python -m ensemblehub.api --model_selection_method all --ensemble_method loop --max_rounds 5

# Configure progressive ensemble
python -m ensemblehub.api --ensemble_method progressive --progressive_mode length \
  --length_thresholds 50,100,200 --max_rounds 3

# Configure random selection ensemble
python -m ensemblehub.api --model_selection_method all --ensemble_method random --max_rounds 3

# Configure round-robin ensemble
python -m ensemblehub.api --model_selection_method all --ensemble_method loop \
  --max_rounds 5 --max_repeat 2

# Configure custom models
python -m ensemblehub.api --model_specs '[{"path":"model1","engine":"hf"},{"path":"model2","engine":"hf"}]'

# Show model attribution
python -m ensemblehub.api --show_attribution

# Show detailed input parameters (for debugging)
python -m ensemblehub.api --show_input_details

# Enable thinking mode
python -m ensemblehub.api --enable_thinking

# Configure vLLM with memory optimization
python -m ensemblehub.api --vllm_enforce_eager --vllm_disable_chunked_prefill \
  --vllm_max_model_len 16384 --vllm_gpu_memory_utilization 0.9

# Configure HuggingFace with quantization
python -m ensemblehub.api --hf_use_8bit --hf_use_eager_attention

# Configure HuggingFace with 4-bit quantization for large models
python -m ensemblehub.api --hf_use_4bit --hf_disable_device_map

# Complete configuration example
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

### Available Command Line Arguments

#### Server Configuration
- `--host`: Server host address (default: 127.0.0.1)
- `--port`: Server port (default: 8000)

#### Ensemble Configuration
- `--model_selection_method`: Model selection method
  - `zscore`: Z-score based statistical selection (default)
  - `all`: Use all models
  - `random`: Random model selection
- `--ensemble_method`: Ensemble method
  - `simple`: Simple reward-based ensemble (default)
  - `progressive`: Progressive ensemble
  - `random`: Random ensemble
  - `loop`: Round-robin ensemble
- `--max_rounds`: Maximum inference rounds (default: 500)
- `--score_threshold`: Score threshold for early stopping (default: -2.0)
- `--max_repeat`: Maximum repeat count (default: 3)

#### Progressive Ensemble Specific Configuration
- `--progressive_mode`: Progressive mode
  - `length`: Length-based model switching
  - `token`: Token-based model switching
  - `mixed`: Mixed mode (default)
- `--length_thresholds`: Comma-separated length thresholds (e.g., 50,100,200)
- `--special_tokens`: Comma-separated special tokens (e.g., <step>,<think>)

#### Model Configuration
- `--model_specs`: Model specifications in JSON format

#### Debug and Output Configuration
- `--show_attribution`: Show model attribution (which model generated which part)
- `--show_input_details`: Show detailed input parameters (for debugging API requests)
- `--enable_thinking`: Enable thinking mode (for models that support it, e.g., DeepSeek-R1)

#### vLLM Specific Options
- `--vllm_enforce_eager`: Disable CUDA graphs in vLLM (fixes memory allocation errors)
- `--vllm_disable_chunked_prefill`: Disable chunked prefill in vLLM (fixes conflicts)
- `--vllm_max_model_len`: Maximum model length for vLLM (default: 32768, reduces OOM)
- `--vllm_gpu_memory_utilization`: GPU memory utilization for vLLM (default: 0.8)
- `--vllm_disable_sliding_window`: Disable sliding window attention (fixes layer name conflicts)

#### HuggingFace Specific Options
- `--hf_use_eager_attention`: Use eager attention implementation (default: True, fixes meta tensor errors)
- `--hf_disable_device_map`: Disable device_map for specific device assignment (fixes meta tensor errors)
- `--hf_use_8bit`: Use 8-bit quantization for large models (saves GPU memory)
- `--hf_use_4bit`: Use 4-bit quantization for large models (saves more GPU memory)
- `--hf_low_cpu_mem`: Use low CPU memory loading (default: True)

After starting the service, access:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/status

## 📋 Main API Endpoints

### 1. Basic Information
- `GET /` - API information and endpoint list
- `GET /status` - Health check and available methods
- `GET /v1/ensemble/methods` - List all available ensemble methods

### 2. Configuration Management
- `GET /v1/ensemble/config` - Get current configuration
- `POST /v1/ensemble/config` - Update configuration

### 3. Inference Endpoints
- `POST /v1/chat/completions` - OpenAI-compatible chat completion
- `POST /v1/loop/completions` - Dedicated loop inference endpoint (round-robin mode)
- `POST /v1/ensemble/inference` - Direct ensemble inference
- `POST /v1/ensemble/batch` - Batch inference

### 4. Preset Endpoints
- `POST /v1/ensemble/presets/simple` - Simple ensemble
- `POST /v1/ensemble/presets/selection_only` - Model selection only
- `POST /v1/ensemble/presets/aggregation_only` - Output aggregation only

## 🔧 Usage Examples

### 1. Basic Chat Completion (using default configuration)

```bash
# Using prompt field (text completion format)
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "prompt": "What is 2+2?",
  "max_tokens": 100
}'

# Using messages field (chat format)
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

### 2. Chat Completion with Ensemble Configuration

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

### 3. Enable Thinking Mode Example

```bash
# Enable thinking mode for models that support it (e.g., DeepSeek-R1)
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "messages": [
    {"role": "user", "content": "Solve this complex math problem: Find the derivative of f(x) = x^3 * sin(x)"}
  ],
  "max_tokens": 500,
  "ensemble_config": {
    "model_selection_method": "all",
    "ensemble_method": "simple",
    "enable_thinking": true,
    "show_attribution": true
  }
}'
```

### 4. Progressive Ensemble Examples

#### Length-based Progressive Ensemble
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

#### Token-based Progressive Ensemble
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

### 5. Loop Inference Endpoint (Round-Robin Mode)

```bash
curl -X POST "http://localhost:8000/v1/loop/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "prompt": "Explain the basic principles of quantum computing",
  "max_tokens": 300,
  "ensemble_config": {
    "max_rounds": 5,
    "score_threshold": -1.5
  }
}'
```

### 6. Batch Request Examples

#### Batch Processing Multiple Questions
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

#### Batch Request Using Legacy Prompt Field
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

### 7. Direct Ensemble Inference

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

### 8. Model Selection Only (No Output Aggregation)

```bash
curl -X POST "http://localhost:8000/v1/ensemble/presets/selection_only" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "What is machine learning?",
  "model_selection_method": "zscore",
  "max_tokens": 300
}'
```

### 9. Output Aggregation Only (All Models)

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

### 10. Batch Inference

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

## 📝 API 参数详解

### ChatCompletionRequest 参数

#### 核心参数
- **`model`** (string, default: "ensemble"): Model identifier
- **`messages`** (List[Message] | List[List[Message]], optional): Chat messages
  - Single request: `[{"role": "user", "content": "Hello"}]`
  - Batch request: `[[{"role": "user", "content": "Q1"}], [{"role": "user", "content": "Q2"}]]`
- **`prompt`** (string | List[string], optional): Legacy prompt field for backward compatibility
  - Single: `"What is 2+2?"`
  - Batch: `["What is 2+2?", "What is 3+3?"]`

#### 生成参数
- **`max_tokens`** (int, default: 256): Maximum tokens to generate
- **`temperature`** (float, default: 1.0): Sampling temperature (0 = greedy, 1 = normal sampling)
- **`stop`** (List[string], optional): Stop sequences, e.g., `["\\n", "Question:"]`
- **`stream`** (bool, default: false): Stream responses (not yet implemented)
- **`seed`** (int, optional): Random seed for reproducibility

#### EnsembleConfig 参数
- **`model_selection_method`** (string, default: "all"): Model selection strategy
  - `"zscore"`: Statistical selection based on perplexity and confidence
  - `"all"`: Use all available models
  - `"random"`: Random model selection
- **`ensemble_method`** (string, default: "simple"): Output aggregation method
  - `"simple"`: Reward-based selection
  - `"progressive"`: Progressive ensemble with model switching
  - `"random"`: Random sentence selection
  - `"loop"`: Round-robin selection
- **`progressive_mode`** (string, default: "length"): Mode for progressive ensemble
  - `"length"`: Switch models based on output length
  - `"token"`: Switch models based on special tokens
- **`length_thresholds`** (List[int], optional): Length thresholds for progressive mode
  - Example: `[500, 1000, 1500]`
- **`special_tokens`** (List[string], optional): Special tokens for progressive mode
  - Example: `["<\\\\think>", "<\\\\step>"]`
- **`max_rounds`** (int, default: 500): Maximum generation rounds
- **`score_threshold`** (float, default: -2.0): Score threshold for early stopping
- **`show_attribution`** (bool, default: false): Include model attribution in response
- **`enable_thinking`** (bool, default: false): Enable thinking mode for compatible models

## ⚙️ 配置选项

### Model Selection Method (model_selection_method)
- `"zscore"` - Z-score based model selection (perplexity and confidence)
- `"all"` - Use all available models
- `"random"` - Random model subset selection
- `"llm_blender"` - LLM-Blender method (if implemented)

### Output Aggregation Method (aggregation_method)
- `"reward_based"` - Select output based on reward model scores
- `"random"` - Random selection from generated outputs
- `"round_robin"` - Round-robin model output selection

### Aggregation Level (aggregation_level)
- `"sentence"` - Sentence/paragraph level aggregation (during generation)
- `"token"` - Token level aggregation (e.g., GaC)
- `"response"` - Full response level aggregation (e.g., voting)

## 🔗 Python 客户端示例

### Basic Client Class
```python
import requests
import json

class EnsembleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def chat_completion(self, messages=None, prompt=None, ensemble_config=None, **kwargs):
        """Send chat completion request"""
        payload = {
            "model": kwargs.get("model", "ensemble"),
            "max_tokens": kwargs.get("max_tokens", 256)
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        
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
        """Process multiple conversations in batch"""
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

### Single Request Examples
```python
# Initialize client
client = EnsembleClient()

# Basic chat request
messages = [
    {"role": "user", "content": "What is artificial intelligence?"}
]
result = client.chat_completion(messages=messages)
print(result["choices"][0]["message"]["content"])

# Using progressive ensemble
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

# Using thinking mode
messages = [
    {"role": "user", "content": "Explain the chain rule in calculus with an example"}
]

config = {
    "model_selection_method": "all",
    "ensemble_method": "simple",
    "enable_thinking": True,
    "show_attribution": True
}

result = client.chat_completion(
    messages=messages, 
    ensemble_config=config,
    max_tokens=1000,
    temperature=0.7
)
print(result["choices"][0]["message"]["content"])
```

### Batch Request Examples
```python
# Process multiple questions in batch
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

# Process results
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

## 🛠️ Runtime Configuration Update

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

### vLLM CUDA Memory Allocation Error

If you encounter the following error when using the vLLM engine:
```
captures_underway.empty() INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":3085
```

**Solutions:**

1. **Fix with command-line arguments (recommended):**
   ```bash
   python -m ensemblehub.api --vllm_enforce_eager --vllm_disable_chunked_prefill
   ```

2. **Switch to HuggingFace engine:**
   ```bash
   # Change model configuration from "engine": "vllm" to "engine": "hf"
   python -m ensemblehub.api --model_specs 'model_path:hf:cuda:0'
   ```

3. **Environment variable settings:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   python -m ensemblehub.api
   ```

### vLLM Layer Name Conflict Error

If you encounter the following error when using the vLLM engine:
```
Duplicate layer name: model.layers.X.self_attn.attn
```

**解决方案：**

1. **Use optimized vLLM configuration (recommended):**
   ```bash
   # Use LlamaFactory-style configuration, suitable for single-GPU large models
   python -m ensemblehub.api --ensemble_method random --model_selection_method all
   ```

2. **Switch to HuggingFace engine (most stable):**
   ```bash
   # Change model configuration from "engine": "vllm" to "engine": "hf" 
   python -m ensemblehub.api --model_specs 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:hf:cuda:0'
   ```

3. **Debug mode settings:**
   ```bash
   # If problems persist, start with debug mode
   export CUDA_LAUNCH_BLOCKING=1
   python -m ensemblehub.api --ensemble_method random
   ```

### HuggingFace Meta Tensor Error

If you encounter the following error when using the HuggingFace engine:
```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

**解决方案：**

1. **Use eager attention (recommended):**
   ```bash
   python -m ensemblehub.api --hf_use_eager_attention
   ```

2. **Disable device_map:**
   ```bash
   python -m ensemblehub.api --hf_disable_device_map
   ```

3. **Downgrade transformers version:**
   ```bash
   pip install transformers==4.35.0
   python -m ensemblehub.api
   ```

4. **Use automatic device allocation:**
   ```bash
   # Change device from "cuda:X" to "auto"
   python -m ensemblehub.api --model_specs 'model_path:hf:auto'
   ```

### GPU Out of Memory Error

If you encounter the following error:
```
CUDA error: out of memory
```

**解决方案：**

1. **Use quantization to reduce memory usage (recommended):**
   ```bash
   # Use 8-bit quantization
   python -m ensemblehub.api --hf_use_8bit
   
   # Use 4-bit quantization (more memory-efficient)
   python -m ensemblehub.api --hf_use_4bit
   ```

2. **Reduce the number of simultaneously loaded models:**
   ```bash
   # Use only smaller models
   python -m ensemblehub.api --model_specs 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:hf:cuda:0,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:hf:cuda:1'
   ```

3. **Run large models on CPU:**
   ```bash
   # Move large models to CPU
   python -m ensemblehub.api --model_specs 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:hf:cpu'
   ```

4. **Clear GPU cache:**
   ```bash
   # Clear GPU cache before running
   export CUDA_LAUNCH_BLOCKING=1
   python -c "import torch; torch.cuda.empty_cache()"
   python -m ensemblehub.api
   ```

### Common Issues

**Q: Cannot access API after startup?**
A: Check firewall settings and ensure the port is not occupied:
```bash
curl http://localhost:8000/status
```

**Q: Model loading failed?**
A: Check model path and device configuration, ensure sufficient GPU memory:
```bash
nvidia-smi  # Check GPU usage
```

**Q: Ensemble method configuration not working?**
A: Ensure you use `python -m ensemblehub.api` instead of `uvicorn` to start:
```bash
# ✅ Correct
python -m ensemblehub.api --ensemble_method loop

# ❌ Does not support custom configuration
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

## 🔗 lm-evaluation-harness Compatibility

Ensemble-Hub API is fully compatible with lm-evaluation-harness, supporting all standard parameters:

### Testing with lm-eval

```bash
# Export dummy API key (required by lm-eval)
export OPENAI_API_KEY=dummy

# Basic test
lm_eval --model openai-completions \
  --tasks gsm8k \
  --model_args model=ensemble,base_url=http://localhost:8000/v1/chat/completions,tokenizer_backend=None \
  --batch_size 2 \
  --num_fewshot 5

# Full parameter example
lm_eval --model openai-completions \
  --tasks gsm8k,hendrycks_math \
  --model_args model=ensemble,base_url=http://localhost:8000/v1/chat/completions,tokenizer_backend=None \
  --batch_size 16 \
  --num_fewshot 5 \
  --limit 100

# Test with specific ensemble configuration
export OPENAI_API_KEY=dummy
lm_eval --model openai-completions \
  --tasks hendrycks_math \
  --model_args model=ensemble,base_url=http://localhost:8000/v1/chat/completions,tokenizer_backend=None \
  --batch_size 8 \
  --num_fewshot 4 \
  --seed 42
```

### Supported lm-eval Parameters

The API fully supports the following lm-evaluation-harness parameters:

- **`max_tokens`**: Maximum number of tokens to generate
- **`temperature`**: Sampling temperature (0 for greedy decoding)
- **`stop`**: List of stop sequences (e.g., `["Question:", "</s>", "<|im_end|>"]`)
- **`seed`**: Random seed for reproducibility

### Debugging lm-eval Requests

When you need to debug requests sent by lm-evaluation-harness:

```bash
# Start API with input details display
python -m ensemblehub.api --show_input_details

# Run lm-eval, API logs will show complete request content
```

Example log output:
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

## 🔄 Automatic Batch Detection Rules

The API automatically detects request types without requiring different endpoints:

1. **Single Request**: 
   - `messages` is in `List[Message]` format
   - `prompt` is a single string

2. **Batch Request**: 
   - `messages` is in `List[List[Message]]` format
   - `prompt` is a list of strings `List[str]`

The same `/v1/chat/completions` endpoint handles all cases, automatically recognizing and processing correctly.

This enhanced API provides complete flexibility, allowing you to select and configure different ensemble methods as needed!