# API Usage Examples

新的统一API支持自动批量检测，只需要一个端点 `/v1/chat/completions` 即可处理单个请求和批量请求。

## 启动API服务器

```bash
cd /path/to/Ensemble-Hub
python -m ensemblehub.api
```

或者：

```bash
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

## 1. 单个请求示例

### 基础请求（使用默认配置）
```bash
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

### 使用渐进式集成
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

### 使用特殊token切换的渐进式集成
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

## 2. 批量请求示例

### 批量处理多个问题
```bash
curl -X POST "http://localhost:9876/v1/chat/completions" \
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

### 使用legacy prompt字段的批量请求
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

## 3. Python客户端示例

### 单个请求
```python
import requests
import json

def ensemble_completion(messages, ensemble_config=None):
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "ensemble",
        "messages": messages,
        "max_tokens": 256
    }
    
    if ensemble_config:
        payload["ensemble_config"] = ensemble_config
    
    response = requests.post(url, json=payload)
    return response.json()

# 示例使用
messages = [
    {"role": "user", "content": "Solve: 2x + 5 = 15"}
]

config = {
    "ensemble_method": "progressive",
    "progressive_mode": "length",
    "length_thresholds": [500, 1000],
    "show_attribution": True
}

result = ensemble_completion(messages, config)
print(json.dumps(result, indent=2))
```

### 批量请求
```python
def batch_completion(conversations, ensemble_config=None):
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "batch-ensemble",
        "messages": conversations,  # List of List[Message]
        "max_tokens": 256
    }
    
    if ensemble_config:
        payload["ensemble_config"] = ensemble_config
    
    response = requests.post(url, json=payload)
    return response.json()

# 批量请求示例
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

batch_result = batch_completion(conversations, config)

# 处理结果
for i, choice in enumerate(batch_result["choices"]):
    print(f"Conversation {i}:")
    print(f"  Response: {choice['message']['content']}")
    if choice.get("metadata", {}).get("attribution"):
        attr = choice["metadata"]["attribution"]
        print(f"  Attribution: {attr['summary']}")
    print()
```

## 4. 响应格式

### 单个请求响应
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
            }
          ]
        }
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

### 批量请求响应
批量请求会在`choices`数组中返回多个结果，每个结果对应一个输入对话。

## 5. 配置管理

### 查看当前配置
```bash
curl "http://localhost:8000/v1/ensemble/config"
```

### 更新模型配置
```bash
curl -X POST "http://localhost:8000/v1/ensemble/config" \
  -H "Content-Type: application/json" \
  -d '{
    "model_specs": [
      {"path": "Qwen/Qwen2.5-1.5B-Instruct", "engine": "hf", "device": "cpu"},
      {"path": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "hf", "device": "cpu"}
    ],
    "default_ensemble_config": {
      "ensemble_method": "progressive",
      "model_selection_method": "all",
      "show_attribution": true
    }
  }'
```

## 6. 可用的集成方法

- **simple**: 基于奖励分数的句子级选择
- **progressive**: 渐进式模型切换（支持长度和token模式）
- **random**: 随机选择模型
- **loop**: 轮询选择模型

## 7. 模型选择方法

- **all**: 使用所有可用模型
- **zscore**: 基于z-score的统计选择
- **random**: 随机选择模型子集

## 自动批量检测规则

API会自动检测请求类型：

1. **单个请求**: `messages` 是 `List[Message]` 格式
2. **批量请求**: `messages` 是 `List[List[Message]]` 格式

无需不同的端点，同一个 `/v1/chat/completions` 端点处理所有情况。