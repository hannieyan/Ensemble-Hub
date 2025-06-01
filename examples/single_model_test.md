# Single Model Testing with Ensemble-Hub

This guide shows how to test individual models using the Ensemble-Hub API framework, which is useful for benchmarking and evaluation.

## Overview

When testing a single model, the framework automatically:
- Skips output aggregation (no need to combine outputs from multiple models)
- Uses the model's maximum supported token length (or a safe default)
- Provides OpenAI-compatible API for easy integration with lm-evaluation-harness

## Starting the API Server

### Basic Command (Single Model)

```bash
# Using a small model (Qwen 0.5B)
python -m ensemblehub.api \
    --model_specs "Qwen/Qwen2.5-0.5B-Instruct:hf:cpu" \
    --disable_internal_template \
    --show_attribution \
    --host 0.0.0.0 \
    --port 8000
```

### With GPU Support

```bash
# Using a larger model on GPU
python -m ensemblehub.api \
    --model_specs "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:hf:cuda:0" \
    --disable_internal_template \
    --show_attribution \
    --host 0.0.0.0 \
    --port 8000
```

### With Quantization (to save memory)

```bash
# 8-bit quantization
python -m ensemblehub.api \
    --model_specs "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B:hf:cuda:0" \
    --hf_use_8bit \
    --disable_internal_template \
    --show_attribution \
    --host 0.0.0.0 \
    --port 8000

# 4-bit quantization (even more memory efficient)
python -m ensemblehub.api \
    --model_specs "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:hf:cuda:0" \
    --hf_use_4bit \
    --disable_internal_template \
    --show_attribution \
    --host 0.0.0.0 \
    --port 8000
```

### With vLLM Backend (for better performance)

```bash
# Using vLLM engine
python -m ensemblehub.api \
    --model_specs "Qwen/Qwen2.5-7B-Instruct:vllm:cuda:0" \
    --vllm_gpu_memory_utilization 0.9 \
    --disable_internal_template \
    --show_attribution \
    --host 0.0.0.0 \
    --port 8000
```

## Using with lm-evaluation-harness

Once the API server is running, you can use lm-evaluation-harness to evaluate the model:

### Install lm-evaluation-harness

```bash
pip install lm-eval
```

### Run Evaluations

```bash
# Basic evaluation on ARC Challenge (0-shot)
lm_eval --model openai-completions \
    --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
    --tasks arc_challenge_chat \
    --batch_size 8 \
    --log_samples \
    --output_path results

# Multiple tasks with few-shot
lm_eval --model openai-completions \
    --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
    --tasks gsm8k,mmlu_abstract_algebra \
    --num_fewshot 5 \
    --batch_size 4 \
    --log_samples \
    --output_path results

# Comprehensive evaluation
lm_eval --model openai-completions \
    --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
    --tasks leaderboard \
    --batch_size 8 \
    --log_samples \
    --output_path results
```

## Command Line Options

### Model Specification Format

```
model_path:engine:device
```

- `model_path`: HuggingFace model ID or local path
- `engine`: Either `hf` (HuggingFace) or `vllm`
- `device`: Either `cpu` or `cuda:N` where N is the GPU index

### Useful API Flags

- `--show_input_details`: Show detailed request information in logs
- `--show_attribution`: Include model attribution in responses (useful for tracking which model generated the output)
- `--enable_thinking`: Enable thinking mode for models that support it
- `--disable_internal_template`: Disable the internal chat template handling (recommended for lm-eval compatibility)

### Generation Parameters

You can control generation behavior with:

- `--max_rounds`: Maximum generation rounds (default: 500)
- `--score_threshold`: Score threshold for early stopping (default: -2.0)

## Testing Different Models

### Small Models (Good for testing)

```bash
# Qwen 0.5B
--model_specs "Qwen/Qwen2.5-0.5B-Instruct:hf:cpu"

# Qwen 1.5B
--model_specs "Qwen/Qwen2.5-1.5B-Instruct:hf:cpu"
```

### Medium Models

```bash
# DeepSeek R1 7B
--model_specs "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:hf:cuda:0"

# Qwen 7B
--model_specs "Qwen/Qwen2.5-7B-Instruct:hf:cuda:0"
```

### Large Models (with quantization)

```bash
# DeepSeek R1 14B with 8-bit
--model_specs "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B:hf:cuda:0" --hf_use_8bit

# DeepSeek R1 32B with 4-bit
--model_specs "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:hf:cuda:0" --hf_use_4bit
```

## Example: Complete Workflow

### 1. Start the API server with a single model:

```bash
python -m ensemblehub.api \
    --model_specs "Qwen/Qwen2.5-7B-Instruct:hf:cuda:0" \
    --disable_internal_template \
    --show_attribution \
    --show_input_details \
    --host 0.0.0.0 \
    --port 8000
```

### 2. In another terminal, run the evaluation:

```bash
lm_eval --model openai-completions \
    --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
    --tasks arc_challenge_chat,hellaswag,truthfulqa \
    --batch_size 8 \
    --log_samples \
    --output_path results/qwen-7b
```

### 3. Check the results in `results/qwen-7b/` directory

## Troubleshooting

### Out of Memory Errors

- Use quantization: `--hf_use_8bit` or `--hf_use_4bit`
- Reduce batch size in lm_eval: `--batch_size 1`
- Use CPU instead of GPU: change `cuda:0` to `cpu`

### Slow Performance

- Use vLLM backend instead of HuggingFace: change `hf` to `vllm`
- Enable GPU if available
- Use smaller models for testing

### Connection Errors

- Make sure the API server is running
- Check the port is not blocked by firewall
- Verify the base_url in lm_eval command matches your server

## Advanced Usage

### Custom Generation Parameters

You can pass custom generation parameters through the API:

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "ensemble",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
})
```

### Monitoring Performance

The API logs useful information about:
- Model loading
- Token usage
- Generation time
- Memory usage (when using vLLM)

Watch the API server logs to monitor performance and debug issues.