import logging
import math
from typing import List, Dict, Callable
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from v6.ensemble import ModelPool, EnsembleReasoner, ConversationTemplate



# Optional vLLM backend -----------------------------------------------------
try:
    from vllm import LLM, SamplingParams  # type: ignore

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging / constants
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ensemble_inference")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EOS_TEXT = ""  # Most Qwen / Llama models use empty string as EOS
STEP_TOKEN = "<extra_0>"  # Token separator used by reward model
SYSTEM_PROMPT = "You are a helpful assistant."
STOP_TOKENS_TEXT = {".", "\n"}  # Stop decoding after these tokens


# Assumes SYSTEM_PROMPT and ConversationTemplate already defined

class ModelStatStore:
    def __init__(self):
        self._stats: Dict[str, Dict[str, float]] = {}

    def has(self, model_path: str) -> bool:
        return model_path in self._stats

    def get(self, model_path: str) -> Dict[str, float]:
        return self._stats[model_path]

    def set(self, model_path: str, stats: Dict[str, float]):
        self._stats[model_path] = stats

    def maybe_compute(self, model_path: str, model, tokenizer, device, dataset: List[str]):
        if not self.has(model_path):
            stats = compute_model_stats_on_dataset(model, tokenizer, device, dataset)
            self.set(model_path, stats)
        return self.get(model_path)

def compute_model_stats_on_dataset(model, tokenizer, device, dataset: List[str]) -> Dict[str, float]:
    all_ppls, all_confs = [], []
    for problem in dataset:
        inputs = tokenizer(problem, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            mask = labels != tokenizer.pad_token_id
            token_log_probs = token_log_probs[mask]
            avg_nll = -token_log_probs.mean().item()
            perplexity = math.exp(avg_nll)

            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values.squeeze(0)
            mask_flat = mask.squeeze(0)
            confidence = max_probs[mask_flat].mean().item()

            all_ppls.append(perplexity)
            all_confs.append(confidence)

    return {
        "ppl_mean": float(torch.tensor(all_ppls).mean()),
        "ppl_std": float(torch.tensor(all_ppls).std()),
        "conf_mean": float(torch.tensor(all_confs).mean()),
        "conf_std": float(torch.tensor(all_confs).std()),
    }

def score_question_for_model(question: str, model, tokenizer, device: str, prompt_builder: Callable) -> Dict[str, float]:
    prompt = prompt_builder(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        mask = labels != tokenizer.pad_token_id
        token_log_probs = token_log_probs[mask]
        avg_nll = -token_log_probs.mean().item()
        ppl = math.exp(avg_nll)

        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values.squeeze(0)
        mask_flat = mask.squeeze(0)
        conf = max_probs[mask_flat].mean().item()

    return {"ppl": ppl, "conf": conf}

def determine_model_count(question_scores: List[Dict[str, float]], model_stats: Dict[str, Dict[str, float]]) -> int:
    over_threshold = 0
    for score, (model_path, stats) in zip(question_scores, model_stats.items()):
        if score["ppl"] > stats["ppl_mean"] + 2:
            over_threshold += 1
    if over_threshold >= len(question_scores) * 0.90:
        return 3
    else:
        return 2

def select_top_models_by_z_score(question: str, model_specs: List[Dict], prompt_builder, model_stats: Dict[str, Dict[str, float]], model_pool, model_count: int = -1) -> List[Dict]:
    results = []
    question_scores = []
    for spec in model_specs:
        model = model_pool.get_generator(spec["path"], spec.get("engine", "hf"), spec.get("device")).model
        tokenizer = model_pool.get_generator(spec["path"], spec.get("engine", "hf"), spec.get("device")).tokenizer
        score = score_question_for_model(question, model, tokenizer, spec["device"], prompt_builder)
        stats = model_stats[spec["path"]]
        z_ppl = (stats["ppl_mean"] - score["ppl"]) / stats["ppl_std"]
        z_conf = (score["conf"] - stats["conf_mean"]) / stats["conf_std"]
        total_score = z_ppl + z_conf
        results.append((total_score, spec))
        question_scores.append(score)

    if model_count == -1:
        model_count = determine_model_count(question_scores, model_stats)

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return [spec for _, spec in results[:model_count]]

def run_zscore_ensemble(
    example: Dict,
    dataset_problems: List[str],
    model_specs: List[Dict],
    reward_spec: Dict,
    stat_store: ModelStatStore,
    max_rounds: int = 500,
    score_threshold: float = 0.5
) -> str:

    logger.info("[Stage 1] Computing or retrieving reference statistics for all models...")
    model_pool = ModelPool()
    model_stats = {}
    for spec in model_specs:
        model_path = spec["path"]
        generator = model_pool.get_generator(spec["path"], spec.get("engine", "hf"), spec.get("device"))
        stats = stat_store.maybe_compute(model_path, generator.model, generator.tokenizer, generator.device, dataset_problems)
        model_stats[model_path] = stats
        logger.info(
            f"→ Stats for {model_path}: "
            f"PPL µ={stats['ppl_mean']:.2f}, σ={stats['ppl_std']:.2f} | "
            f"Conf µ={stats['conf_mean']:.2f}, σ={stats['conf_std']:.2f}"
        )


    logger.info("[Stage 2] Selecting top models based on z-score (auto model count)...")
    prompt_builder = lambda q: ConversationTemplate(SYSTEM_PROMPT, q).render()
    selected_specs = select_top_models_by_z_score(
        question=example["input"],
        model_specs=model_specs,
        prompt_builder=prompt_builder,
        model_stats=model_stats,
        model_pool=model_pool,
        model_count=-1
    )
    logger.info(f"✅ Selected models: {[s['path'] for s in selected_specs]}")

    logger.info("[Stage 3] Loading selected generators and reward model...")
    generators = [
        model_pool.get_generator(spec["path"], spec.get("engine", "hf"), spec.get("device"))
        for spec in selected_specs
    ]
    scorer = model_pool.get_reward(reward_spec["path"], device=reward_spec["device"])

    logger.info("[Stage 4] Running ensemble reasoner...")
    reasoner = EnsembleReasoner(
        generators=generators,
        scorer=scorer,
        max_rounds=max_rounds,
        score_threshold=score_threshold
    )
    return reasoner(example)