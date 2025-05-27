import logging
import math
from typing import List, Dict, Callable
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from ensemblehub.conversation import ConversationTemplate
from ensemblehub.generators import GeneratorPool
from ensemblehub.scorer import ScorerPool
from ensemblehub.statistics.compute_model_stats import ModelStatStore

# New architecture imports
from ensemblehub.ensemble_methods import (
    EnsembleFramework,
    EnsembleConfig,
    run_simple_ensemble,
    ZScoreSelector, 
    AllModelsSelector, 
    RandomSelector,
    RewardBasedSelector,
    RandomSentenceSelector,
    RoundRobinSelector
)

# Import ProgressiveSelector
from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector

# Legacy support - map ensemble methods to new classes
ensemble_map = {
    "simple": RewardBasedSelector,
    "random": RandomSentenceSelector, 
    "loop": RoundRobinSelector,
    "reward_based": RewardBasedSelector,
    "round_robin": RoundRobinSelector,
    "progressive": ProgressiveSelector,
}

# Model selection methods
model_selection_map = {
    "zscore": ZScoreSelector,
    "all": AllModelsSelector,
    "random": RandomSelector,
}

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

def run_ensemble(
    example: Dict,
    model_specs: List[Dict],
    reward_spec: List[Dict],
    ensemble_method: str = "reward_based",
    model_selection_method: str = "zscore", 
    max_rounds: int = 500,
    score_threshold: float = -2.0,
    progressive_mode: str = "length",
    length_thresholds: List[int] = None,
    special_tokens: List[str] = None,
    **kwargs
) -> Dict[str, any]:
    """
    New unified ensemble function using the refactored architecture.
    Now uses the EnsembleFramework for better organization.
    
    Args:
        example: Input example with "instruction", "input", "output"
        model_specs: List of model specifications
        reward_spec: List of reward model specifications  
        ensemble_method: Output aggregation method ("reward_based", "random", "round_robin")
        model_selection_method: Model selection method ("zscore", "all", "random")
        max_rounds: Maximum generation rounds
        score_threshold: Score threshold for early stopping
    
    Returns:
        Dict with "output" and "selected_models"
    """
    logger.info("[New Framework] Initializing unified ensemble framework...")
    
    # Initialize pools
    model_pool = GeneratorPool()
    scorers = ScorerPool()
    
    # Load model statistics
    model_stats = get_default_model_stats()
    
    # Load external scorers
    for spec in reward_spec:
        try:
            scorers.get_scorer(spec)
            logger.info(f"✅ Loaded scorer: {spec.get('path', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load scorer {spec.get('path', 'unknown')}: {e}")
    
    # Map legacy method names to new names
    method_mapping = {
        "simple": "reward_based",
        "random": "random", 
        "loop": "round_robin",
        "progressive": "progressive"
    }
    aggregation_method = method_mapping.get(ensemble_method, ensemble_method)
    
    # Handle progressive-specific parameters (only constructor params, not runtime params)
    aggregation_params = {}
    if ensemble_method == "progressive":
        aggregation_params.update({
            "switch_mode": progressive_mode,
            "length_thresholds": length_thresholds or [1000, 2000, 3000],
            "special_tokens": special_tokens or [r"<\think>"]
        })
    
    # Note: max_rounds and score_threshold are runtime parameters, not constructor parameters
    
    # Create ensemble framework
    config = EnsembleConfig(
        use_model_selection=True,
        model_selection_method=model_selection_method,
        model_selection_params={"model_count": -1} if model_selection_method == "zscore" else {},
        use_output_aggregation=True,
        aggregation_method=aggregation_method,
        aggregation_level="sentence",
        aggregation_params=aggregation_params
    )
    
    framework = EnsembleFramework(config)
    
    # Run ensemble
    result = framework.run_ensemble(
        example=example,
        model_specs=model_specs,
        generators=model_pool,
        scorers=scorers,
        model_stats=model_stats,
        max_rounds=max_rounds,
        score_threshold=score_threshold,
        **kwargs
    )
    
    # Convert to legacy format - preserve all data including attribution
    legacy_result = {
        "output": result["output"],
        "selected_models": result["selected_models"],
        "method": f"{model_selection_method}+{ensemble_method}"
    }
    
    # Add attribution data if available
    if "attribution" in result:
        legacy_result["attribution"] = result["attribution"]
    
    return legacy_result


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


def get_default_model_stats() -> Dict[str, Dict[str, float]]:
    """
    Get default model statistics. In production, this should load from a file.
    """
    return {
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "ppl_mean": 9.795982360839844,
            "ppl_std": 22.284496307373047,
            "conf_mean": 0.6799513101577759,
            "conf_std": 0.08082679659128189,
            "weight": 0.2,
            "size": 0.5
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
            "ppl_mean": 9.795982360839844,
            "ppl_std": 22.284496307373047,
            "conf_mean": 0.6799513101577759,
            "conf_std": 0.08082679659128189,
            "weight": 0.2,
            "size": 1.5
        },
        "Qwen/Qwen3-4B": {
            "ppl_mean": 6.160105228424072,
            "ppl_std": 6.118084907531738,
            "conf_mean": 0.8231604099273682,
            "conf_std": 0.07646501809358597,
            "weight": 1.0,
            "size": 4.0
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "ppl_mean": 16.57339096069336,
            "ppl_std": 50.37682342529297,
            "conf_mean": 0.6976740956306458,
            "conf_std": 0.10360505431890488,
            "weight": 0.5,
            "size": 7.0
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
            "ppl_mean": 8.22177505493164,
            "ppl_std": 14.440741539001465,
            "conf_mean": 0.7438507676124573,
            "conf_std": 0.0863514393568039,
            "weight": 1.0,
            "size": 14.0
        },
        "Qwen/Qwen2.5-Math-7B-Instruct": {
            'ppl_mean': 4.232998847961426,
            'ppl_std': 3.664811611175537,
            'conf_mean': 0.7785097360610962,
            'conf_std': 0.09053431451320648,
            "weight": 1.0,
            "size": 7.0
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
            "ppl_mean": 4.0472869873046875,
            "ppl_std": 3.9851391315460205,
            "conf_mean": 0.7702987194061279,
            "conf_std": 0.0831739529967308,
            "weight": 1.0,
            "size": 32.0
        }
    }