# Ensemble-Hub

**Ensemble-Hub** is an open-source toolkit to **boost your LLM answers** by letting multiple language models work together. Instead of betting everything on one model, Ensemble-Hub has a whole team of models brainstorm and compete – then picks the best parts from each. The result? Answers that are often more accurate, detailed, and reliable. 🎉

How does it work? Under the hood, you provide any number of generator models (we've tested with HuggingFace Transformers and the ultra-fast vLLM) plus a reward model as the judge. Each round, every generator writes a short answer segment; the reward model (e.g. a Qwen-7B fine-tuned preference model) scores them; and the best segment is kept. This repeats until the answer is complete (or an end-of-sequence token is reached).

Ensemble-Hub is **easy to use** in both research and production settings. You can call the high-level `run_ensemble()` function in a Python script or notebook for quick experiments. When it's time to go live, switch to the provided FastAPI server (`ensemble_api_server.py`) for a **plug-and-play REST API**. All models are loaded once and cached (thanks to a singleton ModelPool), so you get speed and memory efficiency out of the box.

## 🌟 Project goals

| **Why?**                                                  | **How?**                                                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Boost answer quality** by letting several LLMs compete. | Each round, every generator writes a short segment → a reward model (Qwen 2.5-Math-PRM-7B) scores them → best segment is kept. |
| **Stay fast & memory-friendly** with model caching.       | ModelPool loads each generator/reward model once, then re-uses it for every call (CLI, notebook or API).                       |
| **Provide plug-and-play usage** for research & services.  | Python helper `run_ensemble()` **or** a production-grade FastAPI server (`ensemble_api_server.py`).                            |

## 🗂 Repository layout

```
ensemble-inference/
├── ensemble_inference.py        # High-level interface (run_ensemble, ModelPool)
├── ensemble_api_server.py       # FastAPI server for REST API
├── v6/                          # Latest core modules
│   ├── generator.py             # Generator classes (HF & vLLM backends)
│   ├── scorer.py                # Reward model classes (PRMScorer, APIScorer, etc.)
│   ├── ensemble.py              # Multi-model reasoning loop (uses generators & scorers)
│   └── data/                    # Prompt templates, dataset converters, etc.
├── notebooks/
│   └── quick_demo.ipynb         # Colab/Jupyter walkthrough
├── configs/
│   └── example.yaml             # Demo config – three DeepSeek models + reward
├── requirements.txt             # Minimal dependencies
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
python -m ensemblehub.inference
    --input_path data/AIME2024/aime/aime24.json \
    --output_path saves/aime24.jsonl \
    --max_examples 500
```

*Under the hood: models are loaded once → the reward model scores each round → loop stops when the selected segment ends with an EOS token.*

## 🛰 Start the REST API

1. **Create a YAML config** (see `configs/example.yaml` for a template)

   ```yaml
   models:
     - path: /models/DeepSeek-R1-Distill-Qwen-1.5B
       engine: hf
     - path: /models/DeepSeek-R1-Distill-Qwen-7B
       engine: hf
   reward_path: /models/Qwen2.5-Math-PRM-7B
   ```

2. **Launch the server**

   ```bash
   python ensemble_api_server.py \
       --config configs/example.yaml \
       --host 0.0.0.0 --port 8000
   ```

3. **Ping the server**

   ```bash
   curl http://localhost:8000/status
   # ➜ {"status":"ready"}
   ```

4. **Ask a question**

   ```bash
   curl -X POST http://localhost:8000/api/generate \
        -H "Content-Type: application/json" \
        -d '{"question":"What is RLHF?", "max_rounds":4}'
   ```

## 💡 Core features

* **Unlimited generators** – mix and match multiple models (HF *and* vLLM backends supported).
* **Reward-guided selection** – uses a reward model (e.g. Qwen2.5-Math-PRM-7B) to score candidates and pick the best output each round.
* **EOS-based early stop** – if a model outputs its end-of-sequence token, the loop exits early.
* **Context accumulation** – optionally carry forward previously chosen segments into the next round (builds a running conversation context).
* **Clean prompt template** – minimal prompt format with no extraneous instructions (no stray “600 words” artifacts).
* **Singleton caches** – models load once and are reused on repeated calls (even across API requests).

## ✍️ Extending

* **More backends** – plug in other model sources by subclassing `BaseGenerator` and registering it in the `ModelPool` (e.g. to use an OpenAI API model).
* **Streaming answers** – wrap `run_ensemble()` in an async generator to yield partial results, and return via SSE or websockets for real-time streaming.
* **Custom reward models** – implement a new scorer class (similar to `PRMScorer`) and swap it in via `ModelPool.get_reward` to test different reward functions.

## 📌 To-Do

-[x] Multi-model inference
    
-[x] Reward model selection
    
-[x] HuggingFace backends

-[ ] vLLM backends
    
-[ ] API support for closed-source models
    
-[ ] Streaming API interface (FastAPI)
    
-[ ] Improved scorer aggregation
    
-[ ] Config-driven pipelines

## 📜 License

Apache-2.0. See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

Relies on **DeepSeek**, **Qwen** model weights, Hugging Face Transformers, [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the incredible open-source community.
