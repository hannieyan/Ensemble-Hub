# ----------------------- Compute statistics -----------------------

# Standard library imports

# Third-party imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load dataset from HuggingFace hub
from datasets import load_dataset

# Local application imports
from v6.statistics.compute_model_stats import ModelStatStore


# Load MATH-500 problem texts as reference dataset
dataset = load_dataset(
    "HuggingFaceH4/MATH-500",
    split="test"
)
problems = [x["problem"] for x in dataset]


def main():
    """
    Main function to load a language model, compute statistics on the MATH-500 dataset,
    and print the results. Uses caching to avoid redundant computations.
    """

    # Set the model name and select device (GPU if available, else CPU)
    model_name = "Qwen/Qwen3-4B"           # e.g. "Qwen/Qwen2.5-Math-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "mps"

    # Load tokenizer and model onto the selected device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Initialize the statistics store
    stat_store = ModelStatStore()

    # Compute or retrieve cached statistics for the model on the dataset
    stats = stat_store.compute(
        model_path=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dataset=problems,
    )

    # Print computed statistics
    print(f"\n=== Stats for {model_name} on MATH-500 ===")
    for k, v in stats.items():
        print(f"{k:>10s}: {v:8.4f}")

    # Display the internal cache of statistics for reference
    print("\n[INFO] Stats have been cached in the following format:")
    print(stat_store._stats)


if __name__ == "__main__":
    main()