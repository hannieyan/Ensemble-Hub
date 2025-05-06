"""FastAPI wrapper for ensemble_inference.run_ensemble
====================================================

Launch an HTTP server that exposes two endpoints:
* `GET /status`         → `{ "status": "ready" }`
* `POST /api/generate`  → run the ensemble reasoning over the given question
  and return `{ "answer": <string> }`

Models are **loaded once at startup** using the same `ModelPool` logic from
`ensemble_inference.py`, so subsequent calls are fast.

Configuration
-------------
The server accepts a **YAML** file that lists the generator models and the
reward model. Example `config.yaml`:
```yaml
models:
  - path: /root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B
    engine: hf
  - path: /root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B
    engine: hf
  - path: /root/autodl-tmp/DeepSeek-R1-Distill-Qwen-14B
    engine: vllm
reward_path: /root/autodl-tmp/Qwen2.5-Math-PRM-7B
```

Run:
```bash
python ensemble_api_server.py --config config.yaml --host 0.0.0.0 --port 8000
```

Request schema (`POST /api/generate`):
```json
{
  "question": "Explain gradient accumulation.",
  "max_rounds": 5,                // optional
  "score_threshold": 0.5,         // optional
  "accumulate_context": true,     // optional
  "model_specs": [                // optional override; same format as YAML
    {"path":"...","engine":"hf"},
    {"path":"...","engine":"vllm"}
  ]
}
```
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ensemble_inference import ModelPool, run_ensemble  # type: ignore

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    question: str
    max_rounds: int = Field(5, ge=1)
    score_threshold: float = Field(0.5, ge=0.0, le=10.0)
    accumulate_context: bool = True
    model_specs: Optional[List[Dict[str, str]]] = None  # optional override

class GenerateResponse(BaseModel):
    answer: str

# ---------------------------------------------------------------------------
# Config loading util
# ---------------------------------------------------------------------------

def load_config(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("models", [])
    reward_path = cfg.get("reward_path", "/root/autodl-tmp/Qwen2.5-Math-PRM-7B")
    return models, reward_path

# ---------------------------------------------------------------------------
# FastAPI app with startup model loading
# ---------------------------------------------------------------------------

def create_app(config_path: str) -> FastAPI:
    models_cfg, reward_path = load_config(config_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # prime the pool so first request is instant
        for spec in models_cfg:
            ModelPool.get_generator(spec["path"], spec.get("engine", "hf"))
        ModelPool.get_reward(reward_path)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/status")
    async def status():
        return {"status": "ready"}

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        # Use override list if provided, else YAML list
        model_specs = req.model_specs or models_cfg
        if not model_specs:
            raise HTTPException(status_code=400, detail="No model specs provided")

        answer = run_ensemble(
            req.question,
            model_specs=model_specs,
            max_rounds=req.max_rounds,
            score_threshold=req.score_threshold,
            accumulate_context=req.accumulate_context,
        )
        return GenerateResponse(answer=answer)

    return app

# ---------------------------------------------------------------------------
# CLI launcher
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start ensemble inference API server")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn workers")
    args = parser.parse_args()

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
