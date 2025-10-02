"""Evaluate HF models under adversarial prompt corruptions.

This script perturbs math problems by randomly corrupting a subset of
tokens (numbers or words) and measures whether a Hugging Face
causal language model can still recover the correct answer.  It can
load data either from the Hugging Face Hub (e.g. ``lighteval/MATH``
with the ``math500`` configuration) or from a local JSON/JSONL file
containing ``question``/``answer`` fields.

Example usage::

    python scripts/ablation/initial/perturb_hf_math.py \
        --model_path Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_name lighteval/MATH --dataset_config math500 \
        --num_examples 50 --corruption_rate 0.2

The generated responses are scored with ``scripts/grader.py`` so the
same normalization logic used elsewhere in the project is applied.

To force corruptions within specific character spans (e.g. the first
ten characters), supply ``--corrupt_ranges 0-10,50-70``.  Ranges are
applied on top of the random corruption logic, enabling targeted
ablations over sections of the prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
import math
import os
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT_DIR / "scripts"

for path in (ROOT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    # ``grader`` relies on other helpers inside scripts/, so make sure they
    # are importable before evaluating answers.
    from grader import grade_answer  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import helper
    raise RuntimeError(
        "Failed to import 'grader'. Ensure the repository root is on PYTHONPATH"
    ) from exc


DEFAULT_PROMPT = (
    "You are a mathematics expert. The following problem may contain "
    "corrupted numbers or typos. Carefully infer the intended question, "
    "correct any mistakes, then solve it. Provide your final answer on "
    "the last line in the format 'Answer: <value>'.\n\nProblem:\n{problem}\n"
)


@dataclass
class ExampleResult:
    """Container for a single evaluation result."""

    idx: int
    original_question: str
    corrupted_question: str
    reference_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    corruption_stats: Dict[str, int]
    raw_completion: str
    corruption_rate: float
    range_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        required=True,
        help="Hugging Face model identifier or local path",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        help="Optional local JSON/JSONL file containing evaluation samples",
    )
    parser.add_argument(
        "--dataset_name",
        default="lighteval/MATH",
        help="Datasets Hub identifier (ignored when --dataset_path is set)",
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="Dataset configuration name on the Hub if required",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (Hub datasets only)",
    )
    parser.add_argument(
        "--question_field",
        default="problem",
        help="Column name containing the question text",
    )
    parser.add_argument(
        "--answer_field",
        default="answer",
        help="Column name containing the reference answer",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=50,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=0.15,
        help="Fraction of eligible tokens to corrupt (0-1 range)",
    )
    parser.add_argument(
        "--corruption_rates",
        type=str,
        help="Comma-separated list of corruption rates to sweep (overrides --corruption_rate)",
    )
    parser.add_argument(
        "--corrupt_ranges",
        type=str,
        help="Comma-separated character ranges (start-end) to target for corruption, e.g. '0-100,100-200'",
    )
    parser.add_argument(
        "--range_sweep",
        type=str,
        help="Sweep over multiple range specifications; same format as --corrupt_ranges",
    )
    parser.add_argument(
        "--min_corrupt_tokens",
        type=int,
        default=1,
        help="Ensure at least this many tokens are corrupted when possible",
    )
    parser.add_argument(
        "--max_corrupt_tokens",
        type=int,
        default=5,
        help="Cap the number of corrupted tokens per sample",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum generation length for the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of corrupted prompts to evaluate per generation call",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 uses greedy decoding)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling parameter",
    )
    parser.add_argument(
        "--prompt_template",
        default=DEFAULT_PROMPT,
        help="Template applied to corrupted problems",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for corruption and generation",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run the model on",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Optional Hugging Face device map (e.g. 'auto') for sharding across multiple GPUs",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        help="Optional torch dtype for model weights (e.g. float16, bfloat16)",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU device ids for parallel evaluation (each process runs on one)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Optional JSONL file to store detailed predictions",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        help="Optional JSON file to store per-configuration accuracy summaries",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-example outcomes to stdout",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_local_dataset(path: Path) -> Dataset:
    """Load a JSON/JSONL dataset from disk."""

    records: List[Dict[str, str]] = []
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            else:
                raise ValueError("JSON dataset must contain a list of records")
    else:
        raise ValueError(f"Unsupported dataset extension: {suffix}")

    if not records:
        raise ValueError(f"No samples found in {path}")

    return Dataset.from_list(records)


def load_samples(args: argparse.Namespace) -> Dataset:
    """Load evaluation samples from either a local file or the Hub."""

    if args.dataset_path:
        dataset = load_local_dataset(args.dataset_path)
    else:
        load_kwargs = {"split": args.split}
        if args.dataset_config:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config,
                **load_kwargs,
            )
        else:
            dataset = load_dataset(
                args.dataset_name,
                **load_kwargs,
            )

    if args.num_examples and args.num_examples < len(dataset):
        dataset = dataset.select(range(args.num_examples))

    missing_columns = []
    for field in (args.question_field, args.answer_field):
        if field not in dataset.column_names:
            missing_columns.append(field)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    return dataset


def parse_range_spec(spec: str) -> List[Tuple[int, int]]:
    """Parse ``start-end`` range specifications into a list of tuples."""

    ranges: List[Tuple[int, int]] = []
    for raw in re.split(r"[,;]", spec):
        part = raw.strip()
        if not part:
            continue
        if part.lower() in {"none", "random"}:
            ranges.append((-1, -1))
            continue
        if "-" not in part:
            raise ValueError(f"Invalid range '{part}', expected 'start-end'")
        start_str, end_str = part.split("-", 1)
        start = int(start_str.strip())
        end = int(end_str.strip())
        if end <= start:
            raise ValueError(f"Range '{part}' must have end > start")
        ranges.append((start, end))
    return ranges


def _spans_intersect(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> bool:
    return not (span_a[1] <= span_b[0] or span_b[1] <= span_a[0])


def resolve_dtype(dtype_str: Optional[str]):
    if not dtype_str:
        return None
    key = dtype_str.strip().lower()
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float32", "fp32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_str}")


def _mutate_numeric(token: str) -> str:
    digits = string.digits
    mutated_chars = []
    for ch in token:
        alternatives = [d for d in digits if d != ch]
        mutated_chars.append(random.choice(alternatives) if alternatives else ch)
    mutated = "".join(mutated_chars)
    if mutated == token and token:
        mutated = random.choice([d for d in digits if d != token[0]]) + token[1:]
    return mutated


def _mutate_alpha(token: str) -> str:
    if not token:
        return token
    idx = random.randrange(len(token))
    original = token[idx]
    if original.islower():
        pool = string.ascii_lowercase
    elif original.isupper():
        pool = string.ascii_uppercase
    else:
        pool = string.ascii_letters
    choices = [c for c in pool if c != original]
    replacement = random.choice(choices) if choices else original
    return token[:idx] + replacement + token[idx + 1 :]


def corrupt_question(
    text: str,
    corruption_rate: float,
    min_corrupt: int,
    max_corrupt: int,
    target_spans: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[str, Dict[str, int]]:
    """Introduce random corruptions into the supplied text."""

    matches = list(re.finditer(r"\w+|\W+", text))
    tokens = [m.group(0) for m in matches]

    def eligible(idx: int) -> bool:
        token = tokens[idx]
        if not token.strip() or not token.isalnum():
            return False
        if not target_spans:
            return True
        start, end = matches[idx].span()
        return any(_spans_intersect((start, end), span) for span in target_spans)

    candidate_indices = [i for i in range(len(tokens)) if eligible(i)]
    if not candidate_indices:
        stats = {
            "corrupted_tokens": 0,
            "target_spans": [list(span) for span in (target_spans or [])],
        }
        return text, stats

    if target_spans:
        chosen_indices = candidate_indices
    else:
        num_to_corrupt = max(min_corrupt, int(len(candidate_indices) * corruption_rate))
        if max_corrupt > 0:
            num_to_corrupt = min(num_to_corrupt, max_corrupt)
        num_to_corrupt = min(num_to_corrupt, len(candidate_indices))
        chosen_indices = (
            random.sample(candidate_indices, num_to_corrupt)
            if num_to_corrupt < len(candidate_indices)
            else candidate_indices
        )

    if max_corrupt > 0 and len(chosen_indices) > max_corrupt:
        chosen_indices = random.sample(chosen_indices, max_corrupt)

    for idx in chosen_indices:
        original = tokens[idx]
        if original.isdigit():
            tokens[idx] = _mutate_numeric(original)
        elif original.isalpha():
            tokens[idx] = _mutate_alpha(original)
        else:
            # Alphanumeric mix (e.g. variables like x2); mutate one character.
            pos = random.randrange(len(original))
            pool = string.ascii_letters + string.digits
            replacement = random.choice([c for c in pool if c != original[pos]])
            tokens[idx] = original[:pos] + replacement + original[pos + 1 :]

    corrupted = "".join(tokens)
    stats = {
        "corrupted_tokens": len(chosen_indices),
        "target_spans": [list(span) for span in (target_spans or [])],
    }
    return corrupted, stats


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    device_map: Optional[str] = None,
    torch_dtype_str: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if device_map:
        model_kwargs["device_map"] = device_map
    torch_dtype = resolve_dtype(torch_dtype_str)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not device_map:
        model.to(device)
    model.eval()
    return model, tokenizer


ANSWER_REGEX = re.compile(r"answer\s*[:=]\s*(.+)", flags=re.IGNORECASE)


def _get_model_device(model: AutoModelForCausalLM):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_answer(text: str) -> Optional[str]:
    matches = ANSWER_REGEX.findall(text)
    if matches:
        candidate = matches[-1].strip()
        return candidate.splitlines()[0].strip()
    # Fallback: use the last non-empty line
    for line in reversed(text.splitlines()):
        clean = line.strip()
        if clean:
            return clean
    return None


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    model_device = _get_model_device(model)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def evaluate(
    args: argparse.Namespace,
    dataset: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    corruption_rate: float,
    target_spans: Optional[List[Tuple[int, int]]] = None,
) -> List[ExampleResult]:
    results: List[ExampleResult] = []
    batch_size = max(1, args.batch_size)
    total_examples = len(dataset)
    range_label = format_span_label(target_spans)

    num_batches = (total_examples + batch_size - 1) // batch_size if total_examples else 0
    for batch_idx, start_idx in enumerate(
        tqdm(range(0, total_examples, batch_size), desc="Evaluating", total=num_batches)
    ):
        end_idx = min(start_idx + batch_size, total_examples)
        batch_prompts: List[str] = []
        batch_entries: List[Tuple[int, str, str, Dict[str, int], str]] = []

        for idx in range(start_idx, end_idx):
            sample = dataset[idx]
            question = sample[args.question_field]
            reference = sample[args.answer_field]

            corrupted_question, stats = corrupt_question(
                question,
                corruption_rate,
                args.min_corrupt_tokens,
                args.max_corrupt_tokens,
                target_spans=target_spans,
            )

            prompt = args.prompt_template.format(problem=corrupted_question)
            batch_prompts.append(prompt)
            batch_entries.append((idx, question, reference, stats, corrupted_question))

        completions = generate_responses(
            model,
            tokenizer,
            batch_prompts,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )

        if args.verbose:
            print(f"Processed batch {batch_idx + 1}/{num_batches}")

        for (idx, question, reference, stats, corrupted_question), completion in zip(batch_entries, completions):
            predicted_answer = extract_answer(completion)
            is_correct = grade_answer(predicted_answer, reference)

            if args.verbose:
                print("-" * 80)
                print(f"Example {idx}")
                print(f"Original question: {question}")
                print(f"Corrupted question: {corrupted_question}")
                print(f"Model completion: {completion}")
                print(f"Parsed answer: {predicted_answer}")
                print(f"Ground truth: {reference}")
                print(f"Correct: {is_correct}")
                print(f"Corrupted tokens: {stats['corrupted_tokens']}")
                print(f"Range label: {range_label}")

            results.append(
                ExampleResult(
                    idx=idx,
                    original_question=question,
                    corrupted_question=corrupted_question,
                    reference_answer=reference,
                    predicted_answer=predicted_answer,
                    is_correct=is_correct,
                    corruption_stats=stats,
                    raw_completion=completion,
                    corruption_rate=corruption_rate,
                    range_label=range_label,
                )
            )

    return results


def summarize(results: Sequence[ExampleResult]) -> Dict[str, float]:
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    avg_corrupt = sum(r.corruption_stats.get("corrupted_tokens", 0) for r in results) / max(total, 1)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "avg_corrupted_tokens": avg_corrupt,
    }


def save_results(path: Path, results: Iterable[ExampleResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in results:
            record = {
                "index": item.idx,
                "original_question": item.original_question,
                "corrupted_question": item.corrupted_question,
                "reference_answer": item.reference_answer,
                "predicted_answer": item.predicted_answer,
                "is_correct": item.is_correct,
                "corrupted_tokens": item.corruption_stats.get("corrupted_tokens", 0),
                "completion": item.raw_completion,
                "corruption_rate": item.corruption_rate,
                "range_label": item.range_label,
                "target_spans": item.corruption_stats.get("target_spans", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_span_label(spans: Optional[List[Tuple[int, int]]]) -> str:
    if not spans:
        return "random"
    normalized = []
    for span in spans:
        if span == (-1, -1):
            return "random"
        normalized.append(f"{span[0]}-{span[1]}")
    return ",".join(normalized)


def run_experiments(
    args: argparse.Namespace,
    dataset: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    corruption_rates: Sequence[float],
    range_specs: Sequence[Optional[List[Tuple[int, int]]]],
) -> Tuple[List[ExampleResult], List[Dict[str, float]]]:
    all_results: List[ExampleResult] = []
    summary_rows: List[Dict[str, float]] = []

    experiment_idx = 0
    for rate in corruption_rates:
        for span_config in range_specs:
            if args.verbose:
                print("=" * 80)
                print(f"Running evaluation for corruption_rate={rate}, spans={span_config}")

            set_seed(args.seed + experiment_idx)
            experiment_idx += 1

            results = evaluate(
                args,
                dataset,
                model,
                tokenizer,
                corruption_rate=rate,
                target_spans=span_config,
            )
            metrics = summarize(results)
            metrics["corruption_rate"] = rate
            metrics["range_label"] = format_span_label(span_config)

            print("=" * 80)
            print(f"Summary for rate={rate}, range={metrics['range_label']}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

            summary_rows.append(metrics)
            all_results.extend(results)

    return all_results, summary_rows


def _make_shard_path(base: Path, suffix: str) -> Path:
    return base.with_name(base.stem + suffix + base.suffix)


def _worker_entry(
    args_dict: Dict[str, Any],
    indices: List[int],
    device_id: str,
    corruption_rates: Sequence[float],
    range_specs: Sequence[Optional[List[Tuple[int, int]]]],
    output_path: Optional[Path],
    summary_path: Optional[Path],
):
    worker_args = argparse.Namespace(**args_dict)
    worker_args.device_ids = None
    worker_args.device = f"cuda:{device_id}" if torch.cuda.is_available() else worker_args.device
    worker_args.output_path = output_path
    worker_args.summary_path = summary_path
    worker_args.device_map = None  # ensure single GPU usage per worker

    dataset = load_samples(worker_args)
    if indices:
        dataset = dataset.select(indices)

    model, tokenizer = load_model_and_tokenizer(
        worker_args.model_path,
        worker_args.device,
        device_map=worker_args.device_map,
        torch_dtype_str=worker_args.torch_dtype,
    )

    all_results, summary_rows = run_experiments(
        worker_args,
        dataset,
        model,
        tokenizer,
        corruption_rates,
        range_specs,
    )

    if worker_args.output_path:
        save_results(worker_args.output_path, all_results)

    if worker_args.summary_path:
        worker_args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with worker_args.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)


def run_multi_gpu(
    args: argparse.Namespace,
    dataset: Dataset,
    corruption_rates: Sequence[float],
    range_specs: Sequence[Optional[List[Tuple[int, int]]]],
) -> None:
    if not args.device_ids:
        raise ValueError("device_ids must be provided for multi-GPU execution")

    device_ids = [d.strip() for d in args.device_ids.split(",") if d.strip()]
    if not device_ids:
        raise ValueError("No valid device ids parsed from --device_ids")

    total_examples = len(dataset)
    shard_size = math.ceil(total_examples / len(device_ids)) if device_ids else total_examples

    args_dict = vars(args).copy()
    processes: List[mp.Process] = []
    shard_output_paths: List[Tuple[Optional[Path], Optional[Path]]] = []

    for rank, device_id in enumerate(device_ids):
        start = rank * shard_size
        end = min(start + shard_size, total_examples)
        if start >= end:
            continue

        shard_indices = list(range(start, end))
        output_path = None
        summary_path = None
        if args.output_path:
            output_path = _make_shard_path(args.output_path, f".gpu{device_id}")
        if args.summary_path:
            summary_path = _make_shard_path(args.summary_path, f".gpu{device_id}")

        p = mp.Process(
            target=_worker_entry,
            args=(args_dict, shard_indices, device_id, corruption_rates, range_specs, output_path, summary_path),
        )
        p.start()
        processes.append(p)
        shard_output_paths.append((output_path, summary_path))

    for p in processes:
        p.join()

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as fout:
            for partial_output, _ in shard_output_paths:
                if partial_output and partial_output.exists():
                    with partial_output.open("r", encoding="utf-8") as fin:
                        shutil.copyfileobj(fin, fout)
                    partial_output.unlink()
        print(f"Detailed predictions saved to {args.output_path}")

    if args.summary_path:
        combined_summary: List[Dict[str, float]] = []
        for _, partial_summary in shard_output_paths:
            if partial_summary and partial_summary.exists():
                with partial_summary.open("r", encoding="utf-8") as f:
                    combined_summary.extend(json.load(f))
                partial_summary.unlink()

        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_path.open("w", encoding="utf-8") as f:
            json.dump(combined_summary, f, ensure_ascii=False, indent=2)
        print(f"Summary metrics saved to {args.summary_path}")


def main() -> None:
    args = parse_args()
    dataset = load_samples(args)

    # Determine corruption rates to evaluate
    if args.corruption_rates:
        corruption_rates = [float(x.strip()) for x in args.corruption_rates.split(",") if x.strip()]
    else:
        corruption_rates = [args.corruption_rate]

    # Determine span configurations
    range_specs: List[Optional[List[Tuple[int, int]]]] = []
    if args.range_sweep:
        parsed = parse_range_spec(args.range_sweep)
        for span in parsed:
            range_specs.append(None if span == (-1, -1) else [span])
    elif args.corrupt_ranges:
        spans = parse_range_spec(args.corrupt_ranges)
        normalized = [None if span == (-1, -1) else [span] for span in spans]
        range_specs.extend(normalized)
    else:
        range_specs.append(None)

    if args.device_ids:
        run_multi_gpu(args, dataset, corruption_rates, range_specs)
        return

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device,
        device_map=args.device_map,
        torch_dtype_str=args.torch_dtype,
    )

    set_seed(args.seed)

    all_results, summary_rows = run_experiments(
        args,
        dataset,
        model,
        tokenizer,
        corruption_rates,
        range_specs,
    )

    if args.output_path:
        save_results(args.output_path, all_results)
        print(f"Detailed predictions saved to {args.output_path}")

    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        print(f"Summary metrics saved to {args.summary_path}")


if __name__ == "__main__":
    main()
