# Ensemble-Hub API Usage Guide

## 📖 Overview

Ensemble-Hub API provides an OpenAI-compatible interface for ensemble model inference. It supports multiple model selection strategies, various output aggregation methods, and flexible configuration options. The API can automatically detect single and batch requests, making it compatible with tools like lm-evaluation-harness.

### Key Concepts

- **Model Selection**: Choose which models to use based on statistical methods (e.g., z-score), use all models, or random selection
- **Output Aggregation**: Combine outputs from multiple models at different levels (sentence, token, or response)
- **Ensemble Methods**: Pre-configured combinations of model selection and aggregation strategies
- **Progressive Ensemble**: Dynamically switch between models based on length or special tokens
- **Batch Processing**: Process multiple requests efficiently in a single API call

## 🛠️ Complete API Parameters

### Starting the API Server

```bash
# Basic startup
python -m ensemblehub.api

# With uvicorn (limited configuration)
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

**Note**: Full configuration is only available with `python -m ensemblehub.api`.

### Command Line Arguments

**Server Configuration**
- `--host`: Server host address (default: 127.0.0.1)
- `--port`: Server port (default: 8000)

**Ensemble Configuration**
- `--model_selection_method`: Model selection strategy [`zscore`, `all`, `random`] (default: zscore)
- `--ensemble_method`: Ensemble method [`simple`, `progressive`, `random`, `loop`] (default: simple)
- `--max_rounds`: Maximum inference rounds (default: 500)
- `--score_threshold`: Score threshold for early stopping (default: -2.0)
- `--max_repeat`: Maximum repeat count (default: 3)

**Progressive Ensemble Configuration**
- `--progressive_mode`: Mode [`length`, `token`, `mixed`] (default: mixed)
- `--length_thresholds`: Comma-separated thresholds (e.g., 50,100,200)
- `--special_tokens`: Comma-separated tokens (e.g., <step>,<think>)

**Model Configuration**
- `--model_specs`: Model specifications in JSON format

**Debug and Output**
- `--show_attribution`: Show which model generated which part
- `--show_input_details`: Show detailed input parameters
- `--enable_thinking`: Enable thinking mode for compatible models
- `--disable_internal_template`: Disable internal template formatting

**vLLM Engine Options**
- `--vllm_enforce_eager`: Disable CUDA graphs (fixes memory errors)
- `--vllm_disable_chunked_prefill`: Disable chunked prefill
- `--vllm_max_model_len`: Maximum model length (default: 32768)
- `--vllm_gpu_memory_utilization`: GPU memory utilization (default: 0.8)
- `--vllm_disable_sliding_window`: Disable sliding window attention

**HuggingFace Engine Options**
- `--hf_use_eager_attention`: Use eager attention (default: True)
- `--hf_disable_device_map`: Disable device_map
- `--hf_use_8bit`: Use 8-bit quantization
- `--hf_use_4bit`: Use 4-bit quantization
- `--hf_low_cpu_mem`: Use low CPU memory loading (default: True)

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

## 💻 Common Command Line Examples

### Basic Usage

```bash
# Start with default configuration
python -m ensemblehub.api

# Configure server address and port
python -m ensemblehub.api --host 0.0.0.0 --port 8080

# Show model attribution
python -m ensemblehub.api --show_attribution

# Enable thinking mode
python -m ensemblehub.api --enable_thinking
```

### Model Selection and Ensemble Methods

```bash
# Z-score based model selection with simple ensemble
python -m ensemblehub.api --model_selection_method zscore --ensemble_method simple

# Use all models with loop (round-robin) ensemble
python -m ensemblehub.api --model_selection_method all --ensemble_method loop

# Random model selection
python -m ensemblehub.api --model_selection_method all --ensemble_method random --max_rounds 3
```

### Progressive Ensemble

```bash
# Length-based progressive ensemble
python -m ensemblehub.api --ensemble_method progressive --progressive_mode length \
  --length_thresholds 50,100,200 --max_rounds 3

# Token-based progressive ensemble
python -m ensemblehub.api --ensemble_method progressive --progressive_mode token \
  --special_tokens "<step>,<think>" --show_attribution

# Mixed mode progressive ensemble
python -m ensemblehub.api --ensemble_method progressive --progressive_mode mixed \
  --length_thresholds 100,200 --special_tokens "<step>,<think>"
```

### Memory Optimization

```bash
# vLLM with memory optimization
python -m ensemblehub.api --vllm_enforce_eager --vllm_disable_chunked_prefill \
  --vllm_max_model_len 16384 --vllm_gpu_memory_utilization 0.9

# HuggingFace with 8-bit quantization
python -m ensemblehub.api --hf_use_8bit --hf_use_eager_attention

# HuggingFace with 4-bit quantization
python -m ensemblehub.api --hf_use_4bit --hf_disable_device_map
```

### Custom Model Configuration

```bash
# Configure custom models
python -m ensemblehub.api --model_specs '[{"path":"model1","engine":"hf"},{"path":"model2","engine":"hf"}]'

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
# Export dummy API key (required by lm-eval)
export OPENAI_API_KEY=dummy

# Basic test
lm_eval --model local-completions \
  --tasks gsm8k \
  --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
  --batch_size 2 \
  --num_fewshot 5

# Full evaluation with specific ensemble configuration
python -m ensemblehub.api \
  --model_selection_method all \
  --ensemble_method loop \
  --enable_thinking \
  --show_attribution

# In another terminal
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

**Ensemble configuration not working**: Use `python -m ensemblehub.api` instead of `uvicorn`

## 📖 Additional Resources

- API Documentation: http://localhost:8000/docs (after starting the server)
- Health Check: http://localhost:8000/status
- Run API tests: `python test/test_api.py`

This enhanced API provides complete flexibility, allowing you to select and configure different ensemble methods as needed!