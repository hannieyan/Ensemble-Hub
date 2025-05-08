from typing import Optional, List

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

STEP_TOKEN = "<extra_0>"  # Token separator used by reward model
SYSTEM_PROMPT = "Solve the following math problem step by step. Write your reasoning clearly using LaTeX. Box the final answer using \\boxed{}."


# ---------------------------------------------------------------------------
# Utility: extract token-level reward scores from logits
# ---------------------------------------------------------------------------

def _step_rewards(logits: torch.Tensor, mask: torch.Tensor):
    """
    Compute step-wise probabilities using softmax over logits.
    Only consider positions where mask is non-zero (STEP_TOKEN positions).
    """
    probs = F.softmax(logits, dim=-1) * mask.unsqueeze(-1)
    arr: List[List[float]] = []
    for sample in probs:
        pos = sample[sample != 0].view(-1, 2)[:, 1]
        arr.append(pos.cpu().tolist())
    return arr




# ---------------------------------------------------------------------------
# Abstract base class for scorer
# ---------------------------------------------------------------------------

class BaseScorer:
    def score(self, question: str, answer: List[str]) -> float:
        raise NotImplementedError("Score method must be implemented by subclass.")


# ---------------------------------------------------------------------------
# APIScorer: API-based reward model scorer
# ---------------------------------------------------------------------------
import requests

class APIScorer(BaseScorer):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def score(self, prompt: str, completions: List[str]) -> List[float]:
        """
        Sends a POST request with the input completions to an external scoring API.
        The prompt is concatenated with each completion before being sent.
        Expected API format:
            POST {endpoint}
            {
                "model": "your_model_name_or_id",
                "messages": ["prompt + completion 1", "prompt + completion 2", ...]
            }
        """
        messages = [prompt + c for c in completions]
        payload = {
            "model": "your_reward_model_id",
            "messages": messages
        }
        try:
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            scores = result.get("scores")
            return scores
        except Exception as e:
            print(f"[APIScorer] Request failed: {e}")
            return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# PRMScorer: reward model used for evaluating step-level outputs
# ---------------------------------------------------------------------------

class PRMScorer(BaseScorer):
    def __init__(self, path: str, device: str = "auto"):
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.mod = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        ).eval()
        self.sep_id = self.tok.encode(STEP_TOKEN)[0]


    @torch.inference_mode()
    def score(self, prompt: str, completions: List[str]) -> List[float]:
        """
        Efficiently score a batch of completions using the format:
        [prompt + STEP_TOKEN + completion + STEP_TOKEN]
        """
        inputs = [prompt + STEP_TOKEN + c + STEP_TOKEN for c in completions]
        enc = self.tok(inputs, return_tensors="pt", padding=True, truncation=True).to(self.mod.device)
        mask = enc["input_ids"] == self.sep_id
        logits = self.mod(**enc).logits
        probs = _step_rewards(logits, mask)
        return [float(sum(p) / len(p) * 10.0) if p else 0.0 for p in probs]




