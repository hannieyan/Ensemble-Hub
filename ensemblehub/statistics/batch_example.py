"""
Batch statistics computation for multiple HuggingFace models
on the full EleutherAI/hendrycks_math dataset (train split).
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from compute_model_stats import ModelStatStore
from tqdm.auto import tqdm


def compute_stats_for_models(model_names, problems, device):
    """Compute and cache stats for each model in model_names on the given problems."""
    stat_store = ModelStatStore()

    for model_name in tqdm(model_names, desc="Models"):
        print(f"\n[INFO] Processing model: {model_name}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Compute statistics (mean/std of PPL & confidence)
        stats = stat_store.compute(
            model_path=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset=problems,
        )

        # Print results
        print(f"=== Stats for {model_name} ===")
        for k, v in stats.items():
            print(f"{k:>10s}: {v:8.4f}")

    print("\n[INFO] Cached statistics:")
    for model, values in stat_store._stats.items():
        print(f"{model}: {values}")

    # save all model stats to Dict[str, Dict[str, float]]
    model_stats = {
        model: {
            "ppl_mean": values["ppl_mean"],
            "ppl_std": values["ppl_std"],
            "conf_mean": values["conf_mean"],
            "conf_std": values["conf_std"],
        }
        for model, values in stat_store._stats.items()
    }

    return model_stats


def load_hendrycks_math_questions():
    """Load *all* train problems from every valid Hendrycks Math subset."""
    subsets = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    problems = []
    for subset in tqdm(subsets, desc="Subsets"):
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split="train")
        field = "problem" if "problem" in ds.column_names else "question"
        problems.extend([x[field] for x in ds])
    return problems


def main():
    model_names = [
        "Qwen/Qwen3-4B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ]
    device = "cuda" if torch.cuda.is_available() else "mps"

    print("[INFO] Loading Hendrycks Math dataset...")
    problems = load_hendrycks_math_questions()

    model_stats = compute_stats_for_models(model_names, problems, device)

    print(model_stats)

if __name__ == "__main__":
    main()
