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
- **`reward_based`**: Reward-based selection using scoring models (default)
- **`progressive`**: Length or token-based model switching during generation
  - Length-based: switch models based on output length thresholds
  - Token-based: switch models when encountering special tokens
- **`random`**: Random selection from model outputs
- **`loop`**: Round-robin cycling through models
- **`gac`**: GAC token-level aggregation
- **`distribution`**: Distribution-based token aggregation

## 🗂 Repository layout

```
Ensemble-Hub/
├── ensemblehub/                         # Main package
│   ├── api/                             # FastAPI server module
│   │   ├── __main__.py                  # Command line entry point
│   │   └── app.py                       # FastAPI application
│   ├── ensemble_methods/                # Ensemble method implementations
│   │   ├── ensemble.py                  # Unified ensemble framework
│   │   ├── model_selection/             # Model selection strategies
│   │   │   ├── base.py                  # Base selector interface
│   │   │   ├── statistical.py           # Z-score, random selection
│   │   │   └── learned.py               # LLM-Blender, meta-learning
│   │   └── output_aggregation/          # Output aggregation methods
│   │       ├── token_level/             # Token-level aggregation (GAC, distribution)
│   │       ├── sentence_level/          # Sentence-level aggregation
│   │       │   ├── loop_selector.py     # Round-robin selection
│   │       │   ├── random_selector.py   # Random selection
│   │       │   ├── reward_based.py      # Reward-based selection
│   │       │   └── progressive_selector.py # Progressive selection
│   │       └── response_level/          # Response-level aggregation
│   ├── generators/                      # Model generators (HF, vLLM backends)
│   │   ├── base.py                      # Base generator interface
│   │   ├── hf.py                        # Hugging Face transformers
│   │   ├── vllm.py                      # vLLM backend
│   │   └── pool.py                      # Generator pool management
│   ├── scorers/                         # Reward models and scoring
│   │   └── base.py                      # Base scorer interface
│   ├── inference.py                     # High-level inference pipeline
│   └── utils.py                         # Utility functions
├── data/                                # Datasets (AIME, GSM8K, MATH, etc.)
├── docs/                                # Documentation
│   ├── api_usage.md                     # Complete API usage guide
│   ├── benchmark_single_model.md        # Single model benchmarking
│   └── progressive_selector_usage.md    # Progressive selector guide
├── examples/                            # Usage examples
│   └── test_single_model.py             # Single model testing
├── scripts/                             # Utility scripts
│   ├── vllm_infer.py                    # vLLM inference script
│   └── grader.py                        # Answer grading
├── requirements.txt                     # Dependencies
└── README.md                            # You're here!
```

##  Getting Started

### 🔧 Installation

```bash
conda create -n ensemble python=3.12
conda activate ensemble

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

git clone https://github.com/Fzkuji/Ensemble-Hub.git
cd Ensemble-Hub

pip install -r requirements.txt
```


### 💻 Quickstart

> [!NOTE]
> Please update ensemblehub/inference.py to custom your ensembled LLMs.

```shell
python -m ensemblehub.inference \
   --input_path data/AIME2024/aime/aime24.json \
   --output_path saves/aime24.jsonl \
   --max_examples 500 \
   --batch_size 4 \
   --output_aggregation_method loop \
   --max_tokens 2048
```

*Under the hood: models are loaded once → the reward model scores each round → loop stops when the selected segment ends with an EOS token.*

### 🚀 Start the FastAPI

#### Quick Start with Simple Configuration

```bash
python -m ensemblehub.api \
   --model_specs '[
     {"path":"Qwen/Qwen2.5-0.5B-Instruct","engine":"hf","device":"cuda:0"},
     {"path":"Qwen/Qwen2.5-1.5B-Instruct","engine":"hf","device":"cuda:1","quantization":"4bit"}
   ]' \
   --show_output_details \
   --output_aggregation_method random
```

#### Evaluate with lm-evaluation-harness

```bash
export OPENAI_API_KEY=dummy_key
lm_eval --model openai-completions \
   --tasks arc_challenge_chat \
   --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
   --batch_size 2 \
   --num_fewshot 5
```

#### Advanced Configuration with Command Line Arguments

```bash
# Configure different ensemble methods
python -m ensemblehub.api --model_selection_method all --output_aggregation_method random

# Loop/Round-robin inference (循环推理)
python -m ensemblehub.api --model_selection_method all --output_aggregation_method loop --max_rounds 5

# Progressive ensemble with length-based switching
python -m ensemblehub.api --output_aggregation_method progressive --progressive_mode length \
  --length_thresholds 50,100,200 --max_rounds 3

# Statistical model selection with reward-based aggregation
python -m ensemblehub.api --model_selection_method zscore --output_aggregation_method reward_based \
  --score_threshold -1.5 --max_rounds 10

# Custom server configuration
python -m ensemblehub.api --host 0.0.0.0 --port 9876 \
  --output_aggregation_method loop --show_output_details

# Enable thinking mode for reasoning models (e.g., DeepSeek-R1)
python -m ensemblehub.api --model_selection_method all --output_aggregation_method loop --enable_thinking
```

**Available Configuration Options:**
- **Model Selection**: `zscore` (statistical), `all` (use all models), `random`
- **Output Aggregation Methods**: `reward_based`, `progressive`, `random`, `loop`
- **Progressive Options**: `--progressive_mode`, `--length_thresholds`, `--special_tokens`
- **General**: `--max_rounds`, `--score_threshold`, `--show_input_details`, `--show_output_details`
- **Thinking Mode**: `--enable_thinking` (enables reasoning models' thinking process)

> **Note**: Command line ensemble configuration only works with `python -m ensemblehub.api`. When using `uvicorn`, only server settings (host/port) are configurable.

#### Testing the API

1. **Health Check**

   ```bash
   curl http://localhost:8000/status
   # ➜ {"status":"ready", "available_methods": [...]}
   ```

2. **Chat Completion (New OpenAI Format)**

   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "ensemble",
           "messages": [
               {"role": "user", "content": "What is the capital of France?"}
           ],
           "max_tokens": 50
       }'
   ```

3. **Text Completion (Legacy OpenAI Format)**

   ```bash
   curl -X POST http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "ensemble",
           "prompt": "What is the capital of France?",
           "max_tokens": 50
       }'
   ```

For complete API documentation, visit: http://localhost:8000/docs



## 🛠️ Troubleshooting


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

## 📝 Changelog

### Recent Updates

- **Enable Thinking Mode**: Refactored `enable_thinking` parameter to be configured at model initialization level instead of generation time. This allows better integration with LLaMA-Factory's template system and supports reasoning models like DeepSeek-R1.
- **Consistent Length Handling**: Updated tokenizer calls to use `cutoff_len` from DataArguments for consistent max_length handling across all generation methods.
- **API Improvements**: Added `--enable_thinking` command line flag for easy configuration of reasoning models.

## 📜 License

Apache-2.0. See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

Relies on **DeepSeek**, **Qwen** model weights, Hugging Face Transformers, [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the incredible open-source community.
