![# LLaMA Factory](assets/Ensemble-Hub.gif)

# Ensemble-Hub

**Ensemble-Hub** is an open-source toolkit for large language model (LLM) ensemble inference. 
It is designed to support and unify multiple ensemble strategies for LLMs, including existing methods such as [LLM-Blender](https://github.com/yuchenlin/LLM-Blender), [GaC](https://github.com/yaoching0/GaC), and [UniTE](https://github.com/starrYYxuan/UniTE). 
The project is under active development.

## 🌟 Project goals

| **Why?**                                                  | **How?**                                                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Boost answer quality** by letting several LLMs compete. | Each round, every generator writes a short segment → a reward model (Qwen 2.5-Math-PRM-7B) scores them → best segment is kept. |
| **Stay fast & memory-friendly** with model caching.       | ModelPool loads each generator/reward model once, then re-uses it for every call (CLI, notebook or API).                       |
| **Provide plug-and-play usage** for research & services.  | Python helper `run_ensemble()` **or** a production-grade FastAPI server (`ensemble_api_server.py`).                            |


## 💡 Core features

* **Unlimited generators** – mix and match multiple models (HF *and* vLLM backends supported).
* **Reward-guided selection** – uses a reward model (e.g. Qwen2.5-Math-PRM-7B) to score candidates and pick the best output each round.
* **EOS-based early stop** – if a model outputs its end-of-sequence token, the loop exits early.
* **Context accumulation** – optionally carry forward previously chosen segments into the next round (builds a running conversation context).
* **Clean prompt template** – minimal prompt format with no extraneous instructions (no stray “600 words” artifacts).
* **Singleton caches** – models load once and are reused on repeated calls (even across API requests).


## 🎯 Ensemble Methods

Ensemble-Hub supports multiple ensemble strategies that can be easily configured:

### Model Selection Methods
- **`zscore`**: Statistical selection based on perplexity and confidence scores
- **`all`**: Use all available models (no selection)
- **`random`**: Randomly select a subset of models

### Output Aggregation Methods
- **`simple`**: Reward-based selection using scoring models (default)
- **`progressive`**: Length or token-based model switching during generation
- **`random`**: Random selection from model outputs
- **`loop`**: Round-robin cycling through models (循环推理)

### Progressive Ensemble Options
- **Length-based**: Switch models based on output length thresholds
- **Token-based**: Switch models when encountering special tokens
- **Mixed mode**: Combine both approaches

### Configuration Examples
```bash
# Reward-based ensemble with statistical model selection
python -m ensemblehub.api --model_selection_method zscore --ensemble_method simple

# Round-robin through all models
python -m ensemblehub.api --model_selection_method all --ensemble_method loop

# Progressive ensemble with length switching
python -m ensemblehub.api --ensemble_method progressive --progressive_mode length \
  --length_thresholds 100,300,500
```


## 🗂 Repository layout

```
Ensemble-Hub/
├── ensemblehub/                 # Main package
│   ├── api.py                   # FastAPI server with command line configuration
│   ├── ensemble.py              # Core ensemble framework
│   ├── generator.py             # Model generators (HF, vLLM backends)
│   ├── scorer.py                # Reward models and scoring
│   ├── inference.py             # Inference pipeline
│   ├── utils.py                 # Utility functions
│   └── ensemble_methods/        # Ensemble method implementations
│       ├── model_selection/     # Model selection strategies
│       └── output_aggregation/  # Output aggregation methods
├── data/                        # Datasets (AIME, GSM8K, MATH, etc.)
├── docs/                        # Documentation
│   └── api_usage.md            # Complete API usage guide
├── test/                        # Test suite
├── scripts/                     # Utility scripts
├── requirements.txt             # Dependencies
└── README.md                    # You're here!
```

##  Getting Started

### 🔧 Installation

```bash
conda create -n ensemble python=3.12

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..

git clone https://github.com/Fzkuji/Ensemble-Hub.git
cd Ensemble-Hub

pip install -r requirements.txt
```


### 💻 Quickstart

> [!NOTE]
> Please update ensemblehub/inference.py to custom your LLM ensemble.

```shell
python -m ensemblehub.inference \
    --input_path data/AIME2024/aime/aime24.json \
    --output_path saves/aime24.jsonl \
    --max_examples 500
```

*Under the hood: models are loaded once → the reward model scores each round → loop stops when the selected segment ends with an EOS token.*

### 🚀 Start the REST API

#### Quick Start with Default Configuration

```bash
# Basic startup with default models
python -m ensemblehub.api

# Or use uvicorn (without ensemble configuration)
uvicorn ensemblehub.api:app --host 0.0.0.0 --port 8000
```

#### Advanced Configuration with Command Line Arguments

```bash
# Configure different ensemble methods
python -m ensemblehub.api --model_selection_method all --ensemble_method random

# Loop/Round-robin inference (循环推理)
python -m ensemblehub.api --model_selection_method all --ensemble_method loop --max_rounds 5

# Progressive ensemble with length-based switching
python -m ensemblehub.api --ensemble_method progressive --progressive_mode length \
  --length_thresholds 50,100,200 --max_rounds 3

# Statistical model selection with reward-based aggregation
python -m ensemblehub.api --model_selection_method zscore --ensemble_method simple \
  --score_threshold -1.5 --max_rounds 10

# Custom server configuration
python -m ensemblehub.api --host 0.0.0.0 --port 9876 \
  --ensemble_method loop --show_attribution
```

**Available Configuration Options:**
- **Model Selection**: `zscore` (statistical), `all` (use all models), `random`
- **Ensemble Methods**: `simple` (reward-based), `progressive`, `random`, `loop` (round-robin)
- **Progressive Options**: `--progressive_mode`, `--length_thresholds`, `--special_tokens`
- **General**: `--max_rounds`, `--score_threshold`, `--show_attribution`

> **Note**: Command line ensemble configuration only works with `python -m ensemblehub.api`. When using `uvicorn`, only server settings (host/port) are configurable.

#### Testing the API

1. **Health Check**

   ```bash
   curl http://localhost:8000/status
   # ➜ {"status":"ready", "available_methods": [...]}
   ```

2. **Basic Chat Completion**

   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "ensemble",
           "prompt": "What is the capital of France?",
           "max_tokens": 50
       }'
   ```

3. **Ensemble with Custom Configuration**

   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "ensemble",
           "prompt": "Solve: 2x + 3 = 7",
           "max_tokens": 100,
           "ensemble_config": {
               "model_selection_method": "zscore",
               "aggregation_method": "reward_based",
               "use_model_selection": true,
               "use_output_aggregation": true
           }
       }'
   ```

4. **Loop/Round-robin Endpoint** (dedicated endpoint for 循环推理)

   ```bash
   curl -X POST http://localhost:8000/v1/loop/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "ensemble",
           "prompt": "Explain quantum computing",
           "max_tokens": 200
       }'
   ```

For complete API documentation, visit: http://localhost:8000/docs

#### LM-Evaluation-Harness Integration

The API is fully compatible with lm-evaluation-harness for benchmarking ensemble methods:

1. **Install lm-evaluation-harness:**
   ```shell
   git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
   cd lm-evaluation-harness
   pip install -e .
   ```

2. **Run ensemble evaluation:**
   ```bash
   # Start API with specific ensemble configuration
   python -m ensemblehub.api --ensemble_method loop --model_selection_method all &
   
   # Run evaluation against the ensemble API
   lm_eval \
     --model openai-completions \
     --tasks gsm8k \
     --model_args model=ensemble,base_url=http://localhost:8000,v1=True,tokenizer_backend=None \
     --batch_size 1
   ```

3. **Compare different ensemble methods:**
   ```bash
   # Test different configurations
   python -m ensemblehub.api --ensemble_method random --port 8001 &
   python -m ensemblehub.api --ensemble_method simple --port 8002 &
   python -m ensemblehub.api --ensemble_method progressive --port 8003 &
   
   # Run evaluations on different ports to compare methods
   ```

## 🛠️ Troubleshooting

### vLLM CUDA Memory Allocation Error

If you encounter this error when using vLLM:
```
captures_underway.empty() INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":3085
```

**Solutions:**

1. **Use command line flags (Recommended):**
   ```bash
   python -m ensemblehub.api --vllm_enforce_eager --vllm_disable_chunked_prefill
   ```

2. **Switch to HuggingFace engine:**
   ```bash
   # Change from "engine": "vllm" to "engine": "hf" in model specs
   python -m ensemblehub.api --model_specs 'model_path:hf:cuda:0'
   ```

3. **Set environment variable:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   python -m ensemblehub.api
   ```

### HuggingFace Meta Tensor Error

If you encounter this error when using HuggingFace:
```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead
```

**Solutions:**

1. **Use eager attention (Recommended):**
   ```bash
   python -m ensemblehub.api --hf_use_eager_attention
   ```

2. **Use auto device assignment:**
   ```bash
   python -m ensemblehub.api --model_specs 'model_path:hf:auto'
   ```

3. **Downgrade transformers:**
   ```bash
   pip install transformers==4.35.0
   ```

### Common Issues

- **Memory errors**: Reduce batch size or use smaller models
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Model loading fails**: Check model paths and GPU memory availability

## ✍️ Extending

* **More backends** – plug in other model sources by subclassing `BaseGenerator` and registering it in the `ModelPool` (e.g. to use an OpenAI API model).
* **Streaming answers** – wrap `run_ensemble()` in an async generator to yield partial results, and return via SSE or websockets for real-time streaming.
* **Custom reward models** – implement a new scorer class (similar to `PRMScorer`) and swap it in via `ModelPool.get_reward` to test different reward functions.

## 📌 To-Do

- [x] Multi-model inference
- [x] Reward model selection
- [x] HuggingFace backend
- [x] FastAPI server with OpenAI-compatible endpoints
- [x] Command line configuration for ensemble methods
- [x] Model attribution tracking
- [x] Progressive ensemble methods
- [x] LM-evaluation-harness compatibility
- [ ] vLLM backends
- [ ] API support for closed-source models
- [ ] Streaming API interface (SSE)
- [ ] Web interface for ensemble configuration
- [ ] Advanced scorer aggregation methods

## 📜 License

Apache-2.0. See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

Relies on **DeepSeek**, **Qwen** model weights, Hugging Face Transformers, [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the incredible open-source community.
