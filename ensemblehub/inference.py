#!/usr/bin/env python3
"""
Batch inference script for Ensemble-Hub.
Simple version that converts all inputs to strings for the models.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any

import ray
from tqdm import tqdm

from ensemblehub.ensemble_methods.ensemble import EnsembleConfig, EnsembleFramework
from ensemblehub.hparams.parser import get_ensemble_args

logger = logging.getLogger(__name__)


def load_dataset_json(input_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON or JSONL file."""
    dataset = []
    with open(input_path, "r", encoding="utf-8") as f:
        if input_path.endswith('.jsonl'):
            # JSONL format - one JSON object per line
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))
        else:
            # JSON format - single JSON object or array
            data = json.load(f)
            if isinstance(data, list):
                dataset = data
            else:
                dataset = [data]
    return dataset


def prepare_prompt_from_example(example: Dict[str, Any]) -> str:
    """Convert any example format to a simple string prompt."""
    if isinstance(example, str):
        return example
    
    if isinstance(example, dict):
        # Handle different dict formats
        if "messages" in example:
            # Chat format - convert to string
            messages = example["messages"]
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            return "\n".join(prompt_parts)
        
        elif "instruction" in example:
            # Instruction format
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            if input_text:
                return f"{instruction}\n{input_text}"
            else:
                return instruction
        
        elif "prompt" in example:
            return str(example["prompt"])
        
        else:
            # Fallback: convert entire dict to string
            return str(example)
    
    # Fallback for any other type
    return str(example)


def process_batch(
    batch: List[str],
    ensemble_framework: EnsembleFramework,
    ensemble_config: EnsembleConfig,
    original_examples: List[Dict[str, Any]],
    show_attribution: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a batch of string prompts through the ensemble framework.
    """
    try:
        # Get all config parameters
        config_dict = ensemble_config.model_dump()
        
        # Build parameters, excluding model_specs since we pass it separately
        gen_params = {
            k: v for k, v in config_dict.items() 
            if k not in ["model_specs"] and v is not None
        }
        
        # Run ensemble inference with string prompts
        ensemble_results = ensemble_framework.ensemble(
            batch,
            model_specs=ensemble_config.model_specs,
            is_chat=False,  # Always use text completion mode
            **gen_params
        )
        
        # Prepare results
        results = []
        for i, (prompt, ensemble_result, original_example) in enumerate(zip(batch, ensemble_results, original_examples)):
            result = {
                "prompt": prompt,
                "predict": ensemble_result.get("output", ""),
                "label": original_example.get("output", "") or original_example.get("label", ""),
                "selected_models": ensemble_result.get("selected_models", []),
                "method": ensemble_result.get("method", ensemble_config.output_aggregation_method),
                "config": ensemble_result.get("config", {
                    "model_selection_method": ensemble_config.model_selection_method,
                    "output_aggregation_method": ensemble_config.output_aggregation_method,
                    "temperature": ensemble_config.temperature,
                    "max_tokens": ensemble_config.max_tokens,
                })
            }
            
            if show_attribution and "attribution" in ensemble_result:
                result["attribution"] = ensemble_result["attribution"]
            
            results.append(result)
        
        return results
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing batch: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return error results for all examples in batch
        return [{
            "prompt": prompt,
            "predict": "",
            "label": original_example.get("output", "") or original_example.get("label", ""),
            "error": str(e),
            "selected_models": [],
            "method": ensemble_config.output_aggregation_method
        } for prompt, original_example in zip(batch, original_examples)]


def save_results(results: List[Dict[str, Any]], output_path: str, mode: str = "a"):
    """Save results to a JSONL file."""
    dirname = os.path.dirname(output_path)
    if dirname:  # Only create directory if it's not empty
        os.makedirs(dirname, exist_ok=True)
    
    with open(output_path, mode, encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    # Check if first argument is YAML file (like api.py)
    yaml_file = None
    args_to_parse = sys.argv[1:]
    
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml')):
        yaml_file = sys.argv[1]
        args_to_parse = sys.argv[2:]  # Remove YAML from args
    
    parser = argparse.ArgumentParser(description="Ensemble-Hub Batch Inference")
    
    # Config option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (e.g., examples/all_progressive.yaml)"
    )
    
    # Data paths
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSON dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    
    # Processing options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--show_attribution",
        action="store_true",
        help="Include attribution data in results"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file"
    )
    
    # Parse inference-specific arguments
    args = parser.parse_args(args_to_parse)
    
    # Determine which YAML config to use
    config_file = yaml_file or args.config
    
    # Temporarily modify sys.argv for get_ensemble_args
    original_argv = sys.argv.copy()
    if config_file:
        sys.argv = [sys.argv[0], config_file]
    else:
        sys.argv = [sys.argv[0]]
    
    try:
        # Get ensemble arguments from YAML
        ensemble_args, method_args, generator_args = get_ensemble_args()
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if ensemble_args.show_output_details else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Suppress Ray's filelock debug messages
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Build ensemble config (matching api.py structure)
    ensemble_config = EnsembleConfig(
        model_specs=ensemble_args.model_specs,
        model_selection_method=method_args.model_selection_method,
        model_selection_params=method_args.model_selection_params,
        output_aggregation_method=method_args.output_aggregation_method,
        output_aggregation_params=method_args.output_aggregation_params,
        max_rounds=ensemble_args.max_rounds,
        # Generation parameters
        max_tokens=generator_args.max_tokens,
        temperature=generator_args.temperature,
        top_p=generator_args.top_p,
        top_k=generator_args.top_k,
        repetition_penalty=generator_args.repetition_penalty,
        stop_strings=generator_args.stop_strings,
        seed=generator_args.seed,
        # Debug options
        show_output_details=ensemble_args.show_output_details,
        show_input_details=ensemble_args.show_input_details,
        enable_thinking=generator_args.enable_thinking,
        save_results=ensemble_args.save_results
    )
    
    # Create ensemble framework
    ensemble_framework = EnsembleFramework(ensemble_config)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.input_path}")
    dataset = load_dataset_json(args.input_path)
    
    if args.max_examples:
        dataset = dataset[:args.max_examples]
    
    # Check resume
    processed_count = 0
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            processed_count = sum(1 for _ in f)
        logger.info(f"Resuming from {processed_count} processed examples")
        dataset = dataset[processed_count:]
    
    # Convert all examples to string prompts
    logger.info("Converting examples to prompts...")
    prompts = []
    for example in dataset:
        prompt = prepare_prompt_from_example(example)
        prompts.append(prompt)
    
    # Process in batches
    total_examples = len(prompts)
    logger.info(f"Processing {total_examples} examples in batches of {args.batch_size}")
    
    all_results = []
    start_time = time.time()
    
    with tqdm(total=total_examples, initial=processed_count, desc="Processing") as pbar:
        for i in range(0, total_examples, args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_examples = dataset[i:i + args.batch_size]
            
            # Process batch
            results = process_batch(
                batch_prompts,
                ensemble_framework,
                ensemble_config,
                batch_examples,
                args.show_attribution
            )
            
            # Save results
            save_results(results, args.output_path, mode="a" if i > 0 or args.resume else "w")
            all_results.extend(results)
            
            pbar.update(len(batch_prompts))
    
    # Summary statistics
    elapsed_time = time.time() - start_time
    successful_results = [r for r in all_results if "error" not in r]
    
    logger.info(f"Inference completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total examples: {len(all_results)}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(all_results) - len(successful_results)}")
    logger.info(f"Results saved to: {args.output_path}")
    
    # Model usage statistics
    if successful_results:
        model_usage = {}
        for result in successful_results:
            for model in result.get("selected_models", []):
                model_usage[model] = model_usage.get(model, 0) + 1
        
        logger.info("Model usage statistics:")
        for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {model}: {count} times")


if __name__ == "__main__":
    main()