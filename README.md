## **Ensemble-Inference 🚀**

  

_Multi-model reasoning with reward-guided selection, notebook & API ready._

---

### **🌟 Project goals**

|**Why?**|**How?**|
|---|---|
|**Boost answer quality** by letting several LLMs “compete”.|Each round, every generator writes a short segment → a reward model (Qwen 2.5-Math-PRM-7B) scores them → best segment is kept.|
|**Stay fast & memory-friendly** with model caching.|ModelPool loads each generator/reward model once, then re-uses it for every call (CLI, notebook or API).|
|**Provide plug-and-play usage** for research & services.|Python helper run_ensemble() **or** a production-grade FastAPI server (ensemble_api_server.py).|

  

---

### **🗂 Repository layout**

```
ensemble-inference/
├── ensemble_inference.py       # Core logic (cached pool, template, EOS early-stop)
├── ensemble_api_server.py      # FastAPI wrapper
├── configs/
│   └── example.yaml            # Demo config – three DeepSeek models + reward
├── notebooks/
│   └── quick_demo.ipynb        # Colab/Jupyter walkthrough
├── requirements.txt            # Minimal deps
└── README.md                   # → you are here
```

  

---

### **🔧 Environment**

|**Package**|**Version tested**|**Notes**|
|---|---|---|
|Python|≥ 3.9||
|PyTorch|≥ 2.2|+ CUDA 11/12 GPU recommended|
|transformers|≥ 4.40|HF backend|
|fastapi & uvicorn|≥ 0.110|API server|
|pyyaml|any|load config.yaml|
|**Optional**|||
|vllm|≥ 0.4|ultra-fast inference backend|

```
# CUDA-enabled install (edit as needed)
pip install -r requirements.txt
# add vLLM if you want that backend
pip install vllm
```

  

---

### **▶️ Quick start (Python / notebook)**

```
from ensemble_inference import run_ensemble

model_specs = [
    {"path": "/models/DeepSeek-R1-Distill-Qwen-1.5B",  "engine": "hf"},
    {"path": "/models/DeepSeek-R1-Distill-Qwen-7B",   "engine": "hf"},
    {"path": "/models/DeepSeek-R1-Distill-Qwen-14B",  "engine": "vllm"},
]

answer = run_ensemble(
    "Explain gradient accumulation in simple terms.",
    model_specs=model_specs,
    max_rounds=5,
    accumulate_context=True      # let the conversation build
)
print(answer)
```

_Under the hood: models are pulled once → PRM scores each round → loop stops when a selected segment ends with its own EOS token._

---

### **🛰 Start the REST API**

1. **Create a YAML config** (configs/example.yaml)
```
models:
  - path: /models/DeepSeek-R1-Distill-Qwen-1.5B
    engine: hf
  - path: /models/DeepSeek-R1-Distill-Qwen-7B
    engine: hf
reward_path: /models/Qwen2.5-Math-PRM-7B
```

2. **Launch**

```
python ensemble_api_server.py \
    --config configs/example.yaml \
    --host 0.0.0.0 --port 8000
```

3. **Ping**
```
curl http://localhost:8000/status
# ➜ {"status":"ready"}
```

4. **Ask a question**
```
curl -X POST http://localhost:8000/api/generate \
     -H "Content-Type: application/json" \
     -d '{"question":"What is RLHF?", "max_rounds":4}'
```


---

### **💡 Core features**

- **Unlimited generators** – mix HF & vLLM backends (engine: hf|vllm).
- **Reward-guided selection** – Qwen 2.5-Math-PRM-7B, official step-probability scoring.
- **EOS-based early stop** – each model’s own eos_token_id triggers loop exit.
- **Context accumulation** (opt-in/out).
- **Clean prompt template** – no stray “600 words” artefacts.
- **Singleton caches** – zero reload on repeated calls, even across API requests.

---

### **✍️ Extending**

- **More backends** – subclass BaseGenerator, register in ModelPool.
- **Streaming answers** – wrap run_ensemble in an async generator & return SSE / websockets.
- **Custom reward models** – implement PRMScorer-like class & swap in ModelPool.get_reward.

---

### **📜 License**

Apache-2.0. See LICENSE.

---

### **🙏 Acknowledgements**

Relies on **DeepSeek**, **Qwen** model weights, HuggingFace Transformers and the incredible open-source community.