import logging
import math
from typing import List, Dict, Callable
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


from ensemblehub.statistics.compute_model_stats import ModelStatStore

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
        if score["ppl"] > stats["ppl_mean"]:
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
    reward_spec: List[Dict],
    stat_store: ModelStatStore,
    ensemble_method: str = "simple",
    max_rounds: int = 500,
    score_threshold: float = -2
) -> Dict[str, any]:
    """
    Legacy wrapper for backward compatibility.
    Maintains the original interface while using the new ensemble framework.
    """
    logger.info("Using legacy run_zscore_ensemble (consider migrating to new EnsembleFramework)")
    
    # Convert stat_store to model_stats format if needed
    if hasattr(stat_store, 'get_all_stats'):
        # If stat_store has the method, use it
        try:
            model_stats_from_store = stat_store.get_all_stats()
            # Merge with default stats
            default_stats = get_default_model_stats()
            default_stats.update(model_stats_from_store)
        except Exception as e:
            logger.warning(f"Failed to get stats from stat_store: {e}, using defaults")
            default_stats = get_default_model_stats()
    else:
        default_stats = get_default_model_stats()
    
    return run_ensemble(
        example=example,
        model_specs=model_specs,
        reward_spec=reward_spec,
        ensemble_method=ensemble_method,
        model_selection_method="zscore",
        max_rounds=max_rounds,
        score_threshold=score_threshold
    )

