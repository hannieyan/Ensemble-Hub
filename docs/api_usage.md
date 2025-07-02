# Ensemble-Hub API Usage Guide

## 📖 Overview

Ensemble-Hub API provides an OpenAI-compatible interface for ensemble model inference. Configure via YAML files, start with a single command, and use standard OpenAI endpoints.

### Key Features

- **YAML Configuration**: All settings defined in configuration files
- **OpenAI Compatible**: Standard `/v1/chat/completions` and `/v1/completions` endpoints
- **Automatic Batch Detection**: Single and batch requests handled seamlessly
- **Multiple Ensemble Methods**: Loop, progressive, reward-based, and more

## 🛠️ Complete API Parameters

### Starting the API Server

```bash
# Start with YAML configuration (recommended)
python ensemblehub/api.py examples/all_loop.yaml

# Start with default configuration
python ensemblehub/api.py

# Set server host/port via environment variables
API_HOST=0.0.0.0 API_PORT=8080 python ensemblehub/api.py examples/all_loop.yaml
```

### Configuration

**Server Configuration**
- `API_HOST`: Server host address (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)

All other configuration is done via YAML files. See `examples/` directory for examples.

### API Request Parameters

**Core Parameters**
- `model` (string, default: "ensemble"): Model identifier
- `messages` (List[Message] | List[List[Message]]): Chat messages
  - Single: `[{"role": "user", "content": "Hello"}]`
  - Batch: `[[{"role": "user", "content": "Q1"}], [{"role": "user", "content": "Q2"}]]`
- `prompt` (string | List[string]): Legacy prompt field
  - Single: `"What is 2+2?"`
  - Batch: `["What is 2+2?", "What is 3+3?"]`

**Generation Parameters**
- `max_tokens` (int, default: 256): Maximum tokens to generate
- `temperature` (float, default: 1.0): Sampling temperature
- `stop` (List[string]): Stop sequences
- `stream` (bool, default: false): Stream responses (not yet implemented)
- `seed` (int): Random seed for reproducibility

**EnsembleConfig Parameters**
- `model_selection_method` (string): Model selection strategy [`zscore`, `all`, `random`]
- `ensemble_method` (string): Output aggregation method [`simple`, `progressive`, `random`, `loop`]
- `progressive_mode` (string): Mode for progressive ensemble [`length`, `token`]
- `length_thresholds` (List[int]): Length thresholds for progressive mode
- `special_tokens` (List[string]): Special tokens for progressive mode
- `max_rounds` (int, default: 500): Maximum generation rounds
- `score_threshold` (float, default: -2.0): Score threshold for early stopping
- `show_attribution` (bool, default: false): Include model attribution
- `enable_thinking` (bool, default: false): Enable thinking mode
- `save_results` (bool, default: false): Save all API requests/responses to disk for debugging

### API Endpoints

**Main Endpoints**
- `POST /v1/completions` - OpenAI-compatible chat completion
- `GET /status` - Health check and available methods
- `GET /v1/ensemble/config` - Get current configuration
- `POST /v1/ensemble/config` - Update configuration

**Additional Endpoints**
- `POST /v1/loop/completions` - Dedicated loop inference
- `POST /v1/ensemble/inference` - Direct ensemble inference
- `POST /v1/ensemble/batch` - Batch inference
- `POST /v1/ensemble/presets/simple` - Simple ensemble
- `POST /v1/ensemble/presets/selection_only` - Model selection only
- `POST /v1/ensemble/presets/aggregation_only` - Output aggregation only

## 💻 Common Usage Examples

### Basic Usage

```bash
# Start with default configuration
python ensemblehub/api.py

# Start with example configurations
python ensemblehub/api.py examples/all_loop.yaml
python ensemblehub/api.py examples/all_progressive.yaml

# Configure server address and port
API_HOST=0.0.0.0 API_PORT=8080 python ensemblehub/api.py examples/all_loop.yaml
```

### API Request Examples

#### Basic Chat Completion

```bash
# Using messages field (chat format)
curl -X POST "http://localhost:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 100
}'

# Using prompt field (text completion format)
curl -X POST "http://localhost:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "prompt": "What is 2+2?",
  "max_tokens": 100
}'
```

#### With Ensemble Configuration

```bash
curl -X POST "http://localhost:8000/v1/completions" \
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
    "ensemble_method": "simple",
    "show_attribution": true
  }
}'
```

#### Progressive Ensemble

```bash
curl -X POST "http://localhost:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "messages": [
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

#### Batch Request

```bash
curl -X POST "http://localhost:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "ensemble",
  "messages": [
    [
      {"role": "user", "content": "What is 5 + 3?"}
    ],
    [
      {"role": "user", "content": "What is 10 * 7?"}
    ],
    [
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

### Using with lm-evaluation-harness

```bash
# Start API server
python ensemblehub/api.py examples/all_loop.yaml

# Run evaluation in another terminal
export OPENAI_API_KEY=dummy
lm_eval --model openai-completions \
  --tasks arc_challenge_chat \
  --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
  --batch_size 2 \
  --num_fewshot 5
```

### Python Client Examples

```python
import requests

class EnsembleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def chat_completion(self, messages=None, prompt=None, ensemble_config=None, **kwargs):
        payload = {
            "model": kwargs.get("model", "ensemble"),
            "max_tokens": kwargs.get("max_tokens", 256)
        }
        
        if messages:
            payload["messages"] = messages
        elif prompt:
            payload["prompt"] = prompt
            
        if ensemble_config:
            payload["ensemble_config"] = ensemble_config
            
        response = requests.post(f"{self.base_url}/v1/completions", json=payload)
        return response.json()

# Usage
client = EnsembleClient()

# Basic request
result = client.chat_completion(
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(result["choices"][0]["message"]["content"])

# With ensemble configuration
result = client.chat_completion(
    messages=[{"role": "user", "content": "Solve: 2x + 5 = 15"}],
    ensemble_config={
        "ensemble_method": "progressive",
        "progressive_mode": "length",
        "length_thresholds": [500, 1000],
        "show_attribution": True
    }
)
```

## 📊 Response Format

### Single Request Response
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
        "selected_models": ["Qwen/Qwen2.5-1.5B-Instruct"],
        "method": "zscore+simple",
        "attribution": {
          "summary": "[R01:Qwen2.5-1.5B-Instruct]",
          "detailed": [
            {
              "text": "Generated response",
              "model": "Qwen/Qwen2.5-1.5B-Instruct",
              "round": 1,
              "length": 95
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

### Batch Request Response
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

## 🛠️ Troubleshooting

### Common Issues

**vLLM CUDA Memory Error**: Use `--vllm_enforce_eager --vllm_disable_chunked_prefill`

**HuggingFace Meta Tensor Error**: Use `--hf_use_eager_attention`

**GPU Out of Memory**: Use `--hf_use_8bit` or `--hf_use_4bit` for quantization

**Cannot access API**: Check firewall and port availability with `curl http://localhost:8000/status`

**Configuration not loading**: Check YAML file path and syntax

## 📄 Request/Response Logging

When `save_results: true` is enabled in the configuration, all API requests and responses are saved to `saves/logs/api_session_{timestamp}.jsonl` in JSONL format:

**Text Completion Example:**
```json
{
  "timestamp": "2025-07-02T11:56:53.146701",
  "request_id": "chatcmpl-e4da3f07-f892-473a-9588-3a18bf88b204",
  "endpoint": "/v1/completions",
  "request": {
    "model": "ensemble",
    "prompt": "What is 2+2?",
    "max_tokens": 50
  },
  "response": {
    "id": "chatcmpl-e4da3f07-f892-473a-9588-3a18bf88b204",
    "object": "text.completion",
    "choices": [
      {
        "index": 0,
        "text": "2+2 equals 4. This is a basic arithmetic operation.",
        "finish_reason": "stop",
        "metadata": {
          "selected_models": ["Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"],
          "method": "all+progressive",
          "attribution": {
            "summary": "[R00:Qwen3-4B] → [R00:Qwen3-1.7B]",
            "detailed": [
              {
                "text": "2+2 equals 4.",
                "model": "Qwen/Qwen3-4B",
                "round": 0,
                "length": 12
              },
              {
                "text": " This is a basic arithmetic operation.",
                "model": "Qwen/Qwen3-1.7B",
                "round": 0,
                "length": 37
              }
            ]
          }
        }
      }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 10, "total_tokens": 14}
  }
}
```

**Chat Completion Example:**
```json
{
  "timestamp": "2025-07-02T12:15:30.548291",
  "request_id": "chatcmpl-f8e2a316-c894-582b-a699-4b29cf99c315",
  "endpoint": "/v1/chat/completions",
  "request": {
    "model": "ensemble",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  },
  "response": {
    "id": "chatcmpl-f8e2a316-c894-582b-a699-4b29cf99c315",
    "object": "chat.completion",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "Quantum computing uses quantum mechanics principles to process information differently than classical computers..."
        },
        "finish_reason": "length",
        "metadata": {
          "selected_models": ["Qwen/Qwen3-4B"],
          "method": "all+simple",
          "attribution": {
            "summary": "[R01:Qwen3-4B]",
            "detailed": [
              {
                "text": "Quantum computing uses quantum mechanics principles to process information differently than classical computers...",
                "model": "Qwen/Qwen3-4B",
                "round": 1,
                "length": 95
              }
            ]
          }
        }
      }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 95, "total_tokens": 110}
  }
}
```

## 📖 Additional Resources

- API Documentation: http://localhost:8000/docs (after starting the server)
- Health Check: http://localhost:8000/status
- Run API tests: `python test/test_api.py`

This enhanced API provides complete flexibility, allowing you to select and configure different ensemble methods as needed!