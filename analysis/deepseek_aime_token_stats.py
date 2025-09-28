#!/usr/bin/env python3
"""Run DeepSeek-R1-Distill-Qwen-7B on AIME24 dataset and record per-token stats."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Return the dataset as a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Allow optional {"data": [...]} format
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("Unsupported dataset dict structure; expected 'data' key with list value")
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset format: {type(data)!r}")
    return data


def build_prompt(example: Dict[str, Any]) -> str:
    """Construct the prompt string from instruction/input fields."""
    instruction = (example.get("instruction") or "").strip()
    example_input = (example.get("input") or "").strip()
    if instruction and example_input:
        return f"{instruction}\n\n{example_input}"
    return instruction or example_input


def prepare_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Failed to parse tokenizer configuration. Ensure the model is fully downloaded "
            "or configure Hugging Face authentication (e.g. set HF_TOKEN or run `huggingface-cli login`)."
        ) from exc
    if tokenizer.pad_token_id is None:
        # Align pad/eos for causal LM decoding
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def iter_batches(dataset: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(dataset), batch_size):
        yield dataset[idx : idx + batch_size]


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(token_id)
    # convert_ids_to_tokens returns str for single id but can return None for unknowns
    if token is None:
        return "<unk>"
    return token


def collect_stats(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    cfg: GenerationConfig,
) -> List[Dict[str, Any]]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = model.device if getattr(model, "device", None) else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        prompt_outputs = model(**inputs, use_cache=False)

    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    sequences = generation.sequences
    scores = generation.scores
    prompt_logits = prompt_outputs.logits

    batch_results: List[Dict[str, Any]] = []
    for batch_idx, prompt in enumerate(prompts):
        prompt_len = int(inputs["attention_mask"][batch_idx].sum().item())
        generated_ids = sequences[batch_idx, prompt_len:]
        decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        prompt_token_ids = inputs["input_ids"][batch_idx][:prompt_len]
        prompt_stats = []
        for pos in range(1, prompt_len):
            token_id = prompt_token_ids[pos]
            logits = prompt_logits[batch_idx, pos - 1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_prob = log_probs[token_id]
            prob = torch.exp(token_log_prob)
            ppl = torch.exp(-token_log_prob)
            prompt_stats.append(
                {
                    "position": pos,
                    "token": decode_token(tokenizer, int(token_id.item())),
                    "token_id": int(token_id.item()),
                    "log_prob": float(token_log_prob.item()),
                    "confidence": float(prob.item()),
                    "perplexity": float(ppl.item()),
                }
            )

        token_stats = []
        for step, token_id in enumerate(generated_ids):
            step_scores = scores[step][batch_idx]
            log_probs = torch.nn.functional.log_softmax(step_scores, dim=-1)
            token_log_prob = log_probs[token_id]
            prob = torch.exp(token_log_prob)
            ppl = torch.exp(-token_log_prob)
            token_stats.append(
                {
                    "token": decode_token(tokenizer, int(token_id.item())),
                    "token_id": int(token_id.item()),
                    "log_prob": float(token_log_prob.item()),
                    "confidence": float(prob.item()),
                    "perplexity": float(ppl.item()),
                }
            )

        batch_results.append(
            {
                "prompt": prompt,
                "prompt_token_stats": prompt_stats,
                "generated_text": decoded_output,
                "generated_token_stats": token_stats,
            }
        )
    return batch_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-token stats for DeepSeek on AIME dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/math/AIME2024/aime/aime24.json"),
        help="Path to the JSON dataset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/deepseek_aime24_token_stats.jsonl"),
        help="Where to save the JSONL results",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("analysis/deepseek_aime24_predictions.json"),
        help="Where to save aggregated predictions",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling (default: greedy)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if not dataset:
        raise ValueError("Dataset is empty")

    tokenizer, model = prepare_model(args.model_name)

    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    summary_records = []

    with args.output.open("w", encoding="utf-8") as f:
        total_batches = math.ceil(len(dataset) / args.batch_size)
        for batch in tqdm(iter_batches(dataset, args.batch_size), total=total_batches, desc="Batches"):
            prompts = [build_prompt(example) for example in batch]
            results = collect_stats(tokenizer, model, prompts, cfg)
            for example, result in zip(batch, results):
                record = {
                    "example": example,
                    "prompt": result["prompt"],
                    "prompt_token_stats": result["prompt_token_stats"],
                    "generated_text": result["generated_text"],
                    "generated_token_stats": result["generated_token_stats"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                gen_stats = result["generated_token_stats"]
                if gen_stats:
                    avg_ppl = sum(item["perplexity"] for item in gen_stats) / len(gen_stats)
                    avg_conf = sum(item["confidence"] for item in gen_stats) / len(gen_stats)
                else:
                    avg_ppl = None
                    avg_conf = None

                summary_records.append(
                    {
                        "instruction": example.get("instruction"),
                        "input": example.get("input"),
                        "generated_text": result["generated_text"],
                        "avg_perplexity": avg_ppl,
                        "avg_confidence": avg_conf,
                    }
                )

    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary_records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
