# api_server.py
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time, uuid

from ensemblehub.utils import run_zscore_ensemble, ModelStatStore, SYSTEM_PROMPT


# 请求体格式定义
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 256
    temperature: float = 1.0
    stop: Optional[List[str]] = None


app = FastAPI()
stat_store = ModelStatStore()

# 模型配置（可以根据你 inference.py 的逻辑进行修改）
model_specs = [
    # {"path": "Qwen/Qwen2.5-Math-1.5B-Instruct",           "engine": "hf", "device": "cuda:0"},
    {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf", "device": "cuda:0"},
    # {"path": "Qwen/Qwen3-4B",                             "engine": "hf", "device": "cuda:2"},
    # {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf", "device": "cuda:6"},
    {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "engine": "hf", "device": "cuda:2"},
    {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "engine": "hf", "device": "cuda:3"},
]
reward_spec = [
    # {"path": "Qwen/Qwen2.5-Math-PRM-7B",                  "engine": "hf_rm",  "device": "cuda:0", "weight": 0.2},
    # {"path": "http://localhost:8000/v1/score/evaluation", "engine": "api",                        "weight": 0.4},
    # {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf_gen", "device": "cuda:0", "weight": 1.0},
]


@app.get("/status")
def status():
    return {"status": "ready"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    example = {
        "instruction": "",
        "input": req.prompt
    }

    result = run_zscore_ensemble(
        example=example,
        dataset_problems=[],
        model_specs=model_specs,
        reward_spec=reward_spec,
        stat_store=stat_store,
    )
    output = result["output"]

    finish_reason = "length"
    if req.stop:
        for s in req.stop:
            if s in output:
                output = output.split(s)[0]
                finish_reason = "stop"
                break

    if len(output.split()) > req.max_tokens:
        output = " ".join(output.split()[:req.max_tokens])
        finish_reason = "length"

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": output,
                "finish_reason": finish_reason
            }
        ]
    }




