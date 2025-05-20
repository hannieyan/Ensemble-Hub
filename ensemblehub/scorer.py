import logging
import math
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from ensemblehub.generator import BaseGenerator, HFGenerator

logger = logging.getLogger(__name__)

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
    def score(self, question: str, answer: List[str]) -> List[float]:
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
    def __init__(self, model_path: str, device: str = "auto"):
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mod = AutoModel.from_pretrained(
            model_path,
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




# ---------------------------------------------------------------------------
# GeneratorScorer: scorer based on a generator model
# ---------------------------------------------------------------------------

class GeneratorScorer(BaseScorer):
    def __init__(
        self,
        generator_instance: 'BaseGenerator',
        name: Optional[str] = None
    ):
        if not hasattr(generator_instance, 'calculate_ppl') or \
                not hasattr(generator_instance, 'calculate_confidence'):
            raise ValueError("Provided generator_instance must have calculate_ppl and calculate_confidence methods.")

        self.generator = generator_instance
        self.name = name if name else f"generator_scorer_{generator_instance.name}"

        logger.info(f"GeneratorScorer initialized with generator: {self.generator.name} as Scorer: {self.name}")

    def score(self, prompt: str, completions: List[str]) -> List[float]:
        scores = []
        for completion_text in completions:
            ppl_val = None
            conf_val = None

            try:
                ppl_val = self.generator.calculate_ppl(prompt, completion_text)
            except Exception as e_ppl:
                logger.error(f"Error during PPL calculation by {self.name} for completion '{completion_text[:30]}...': {e_ppl}")

            try:
                conf_val = self.generator.calculate_confidence(prompt, completion_text)
            except Exception as e_conf:
                logger.error(f"Error during Confidence calculation by {self.name} for completion '{completion_text[:30]}...': {e_conf}")

            # Combine PPL and Confidence
            # PPL: lower is better (1 to inf). Confidence: higher is better (0 to 1).
            # Transform PPL to be "higher is better", e.g., -log(PPL).

            ppl_score_component = -float('inf')  # Default for unusable PPL
            if ppl_val is not None and ppl_val > 0 and not math.isinf(ppl_val) and not math.isnan(ppl_val):
                ppl_score_component = -math.log(ppl_val)
            elif ppl_val == 0:  # Theoretical perfect PPL
                ppl_score_component = float('inf')

            conf_score_component = -float('inf')  # Default for unusable Confidence
            if conf_val is not None and not math.isinf(conf_val) and not math.isnan(conf_val):
                # Assuming confidence is already "higher is better" and roughly scaled (e.g. 0-1)
                conf_score_component = conf_val

                # Averaging logic
            valid_components = 0
            current_score_sum = 0.0

            if ppl_score_component > -float('inf'):  # Check if it's not the initial unusable value
                current_score_sum += ppl_score_component
                valid_components += 1

            if conf_score_component > -float('inf'):
                current_score_sum += conf_score_component
                valid_components += 1

            final_score = current_score_sum / valid_components
            scores.append(final_score)

        return scores




# ---------------------------------------------------------------------------
# ScorerPool: manages multiple reward scorers and scoring logic
# ---------------------------------------------------------------------------

class ScorerPool:
    _scorer_cache: Dict[str, Tuple[BaseScorer, float]] = {}

    @classmethod
    def get_scorer(cls, spec: Dict[str, str]) -> BaseScorer:
        """
        Load a reward scorer and cache it. Also print the key.
        """
        key = f"{spec['engine']}::{spec['path']}"
        if key in cls._scorer_cache:
            return cls._scorer_cache[key][0]

        engine = spec["engine"]
        weight = spec.get("weight", 1.0)
        if engine == "hf_rm":
            scorer = PRMScorer(model_path=spec["path"], device=spec.get("device", "cuda"))
            cls._scorer_cache[key] = (scorer, weight)
        elif engine == "api":
            scorer = APIScorer(endpoint=spec["path"])
            cls._scorer_cache[key] = (scorer, weight)
        elif engine == "hf_gen":
            generator = HFGenerator(spec["path"], device=spec.get("device", "cuda"))
            scorer = cls.register_scorer(generator, weight=spec.get("weight", 1.0))
        else:
            raise ValueError(f"Unknown scorer engine: {engine}")

        print(f"[ScorerPool] Added scorer: {key}")
        return scorer

    @classmethod
    def del_scorer(cls, key: str):
        if key in cls._scorer_cache:
            del cls._scorer_cache[key]
            print(f"[ScorerPool] Removed scorer: {key}")

    @classmethod
    def score(cls, prompt: str, completions: List[str], keys: Optional[List[str]] = None) -> List[float]:
        """
        Score completions using selected or all scorers, apply min-max normalization and weighted average.
        """
        selected_items = (
            {k: cls._scorer_cache[k] for k in keys if k in cls._scorer_cache}
            if keys is not None else cls._scorer_cache
        )

        if not selected_items:
            return [- 0.5] * len(completions)

        all_weighted_scores = []
        total_weight = sum(weight for (_, weight) in selected_items.values())

        for key, (scorer, weight) in selected_items.items():
            scores = scorer.score(prompt, completions)
            normalized_w = weight / total_weight if total_weight > 0 else 0.0
            all_weighted_scores.append([s * normalized_w for s in scores])

        final_scores = []
        for i in range(len(completions)):
            combined = sum(scores[i] for scores in all_weighted_scores)
            final_scores.append(combined)

        return final_scores

    @classmethod
    def register_scorer(cls, generator_instance: 'BaseGenerator', weight: float, scorer_name: Optional[str] = None):
        if not scorer_name:
            scorer_name = generator_instance.name

        if not (hasattr(generator_instance, 'calculate_ppl') and callable(generator_instance.calculate_ppl) and
                hasattr(generator_instance, 'calculate_confidence') and callable(generator_instance.calculate_confidence)):
            logger.error(f"Gen '{generator_instance.name}' lacks methods for GeneratorScorer. Not registered.")
            return

        engine_type = "hf_gen"
        key = f"{engine_type}::{scorer_name}"
        if key in cls._scorer_cache: logger.warning(f"Scorer key '{key}' exists. Overwriting.")

        scorer = GeneratorScorer(generator_instance, name=scorer_name)
        cls._scorer_cache[key] = (scorer, float(weight))
        logger.info(f"[ScorerPool] Registered GeneratorScorer: '{key}' w={weight} from gen '{generator_instance.name}'.")

        return scorer