# Ensemble-Hub API 使用指南

Enhanced API v2.0 支持灵活的集成方法选择和配置。

## 🚀 启动 API 服务器

```bash
# 在项目根目录下
python ensemblehub/api.py

# 或使用 uvicorn
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

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
- `POST /v1/ensemble/inference` - 直接集成推理
- `POST /v1/ensemble/batch` - 批量推理

### 4. 预设端点
- `POST /v1/ensemble/presets/simple` - 简单集成
- `POST /v1/ensemble/presets/selection_only` - 仅模型选择
- `POST /v1/ensemble/presets/aggregation_only` - 仅输出聚合

## 🔧 使用示例

### 1. 基础聊天完成（使用默认配置）

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "prompt": "What is 2+2?",
  "max_tokens": 100
}'
```

### 2. 带集成配置的聊天完成

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "prompt": "Solve this math problem: 15 × 23",
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

### 3. 直接集成推理

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

### 4. 仅使用模型选择（不聚合输出）

```bash
curl -X POST "http://localhost:8000/v1/ensemble/presets/selection_only" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "What is machine learning?",
  "model_selection_method": "zscore",
  "max_tokens": 300
}'
```

### 5. 仅使用输出聚合（所有模型）

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

### 6. 批量推理

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

```python
import requests

class EnsembleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def chat_completion(self, prompt, ensemble_config=None, **kwargs):
        """发送聊天完成请求"""
        payload = {
            "prompt": prompt,
            "ensemble_config": ensemble_config,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        return response.json()
    
    def simple_ensemble(self, prompt, ensemble_method="reward_based", model_selection="all"):
        """使用简单集成预设"""
        payload = {
            "prompt": prompt,
            "ensemble_method": ensemble_method,
            "model_selection_method": model_selection
        }
        response = requests.post(f"{self.base_url}/v1/ensemble/presets/simple", json=payload)
        return response.json()

# 使用示例
client = EnsembleClient()

# 基础调用
result = client.chat_completion("What is artificial intelligence?")
print(result["choices"][0]["text"])

# 自定义配置
config = {
    "model_selection_method": "zscore",
    "aggregation_method": "reward_based",
    "use_model_selection": True,
    "use_output_aggregation": True
}
result = client.chat_completion("Solve: 2x + 3 = 7", ensemble_config=config)
print(result["choices"][0]["text"])
```

## 📊 响应格式

### Chat Completions 响应
```json
{
  "id": "cmpl-uuid",
  "object": "text_completion",
  "created": 1234567890,
  "model": "ensemble",
  "choices": [
    {
      "index": 0,
      "text": "生成的文本...",
      "finish_reason": "stop",
      "metadata": {
        "selected_models": ["model1", "model2"],
        "method": "zscore+reward_based",
        "ensemble_config": {...}
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
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

这个增强的 API 提供了完全的灵活性，让你可以根据需要选择和配置不同的集成方法！