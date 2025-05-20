# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import re

import fire
from datasets import load_dataset
from transformers import AutoTokenizer

def compute_token_count(sample, tokenizer):
    """Compute token count for the sample using the provided tokenizer."""
    # Count tokens using the tokenizer
    predict_tokens = tokenizer.encode(sample["predict"])
    predict_token_count = len(predict_tokens)
    
    return {
        "predict_token_count": predict_token_count
    }

def main(filename: str, model_name: str = "gpt2"):
    """
    Calculate average token count for predictions.
    
    Args:
        filename: Path to the JSON file with predictions
        model_name: Name of the tokenizer to use (default: "gpt2")
    """
    start_time = time.time()
    print(f"Loading dataset from {filename}...")
    dataset = load_dataset("json", data_files=filename, split="train")
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Computing token counts...")
    dataset = dataset.map(
        lambda sample: compute_token_count(sample, tokenizer),
        num_proc=1  # Set to 1 since tokenizers might not be thread-safe
    )
    
    # Calculate average token count
    total_samples = len(dataset)
    total_tokens = sum(sample["predict_token_count"] for sample in dataset)
    average_token_count = total_tokens / total_samples
    
    # Print summary
    print("\n=== Token Count Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Total tokens in predictions: {total_tokens}")
    print(f"Average tokens per prediction: {average_token_count:.2f}")
    
    # Save metrics to file
    token_metrics = {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "average_token_count": round(average_token_count, 2)
    }
    
    with open("token_metrics.json", "w", encoding="utf-8") as f:
        json.dump(token_metrics, f, indent=4)
    
    # Generate token count distribution data
    token_counts = [int(sample["predict_token_count"]) for sample in dataset]
    
    with open("token_distribution.json", "w", encoding="utf-8") as f:
        json.dump(token_counts, f, indent=4)
    
    print(f"\nDone in {time.time() - start_time:.3f}s.")
    print(f"Token metrics saved to token_metrics.json")
    print(f"Token distribution saved to token_distribution.json")

if __name__ == "__main__":
    fire.Fire(main)