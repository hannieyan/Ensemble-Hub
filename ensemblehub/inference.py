#!/usr/bin/env python3
"""
Batch inference script for Ensemble-Hub.
Rewritten to follow api.py patterns with YAML configuration support.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Union, Any

import ray
from tqdm import tqdm

from ensemblehub.ensemble_methods.ensemble import EnsembleConfig, EnsembleFramework
from ensemblehub.hparams.parser import get_ensemble_args

logger = logging.getLogger(__name__)


def load_dataset_json(input_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_example_for_inference(
    example: Union[str, Dict[str, Any]],
    input_format: str = "prompt"
) -> Dict[str, Any]:
    """
    Prepare a single example for inference based on input format.
    
    Args:
        example: Raw example from dataset
        input_format: Format type - "prompt", "dict", or "chat"
        
    Returns:
        Formatted example ready for inference
    """
    if input_format == "prompt":
        # Direct string prompt
        if isinstance(example, str):
            prompt = example
        elif isinstance(example, dict):
            # Concatenate instruction and input if available
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            prompt = f"{instruction}\n{input_text}".strip() if instruction else input_text
        else:
            raise ValueError(f"Invalid example type for prompt format: {type(example)}")
        
        return {
            "prompt": prompt,
            "is_completion": True,
            "label": example.get("output", "") if isinstance(example, dict) else ""
        }
    
    elif input_format == "dict":
        # Dictionary format with instruction/input/output
        if not isinstance(example, dict):
            raise ValueError(f"Expected dict for dict format, got {type(example)}")
        
        return {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "output": "",
            "label": example.get("output", ""),
            "is_completion": False
        }
    
    elif input_format == "chat":
        # Chat format with messages
        if isinstance(example, dict) and "messages" in example:
            messages = example["messages"]
        elif isinstance(example, list):
            messages = example
        else:
            raise ValueError(f"Invalid example type for chat format: {type(example)}")
        
        return {
            "messages": messages,
            "output": "",
            "label": example.get("output", "") if isinstance(example, dict) else "",
            "is_completion": False
        }
    
    else:
        raise ValueError(f"Unknown input format: {input_format}")


def process_batch(
    batch: List[Dict[str, Any]],
    ensemble_framework: EnsembleFramework,
    ensemble_config: EnsembleConfig,
    show_attribution: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a batch of examples through the ensemble framework.
    
    Args:
        batch: List of prepared examples
        ensemble_framework: Initialized ensemble framework
        ensemble_config: Configuration for ensemble
        show_attribution: Whether to include attribution data
        
    Returns:
        List of results with predictions
    """
    try:
        # Extract inputs based on format
        if "messages" in batch[0]:
            inputs = [ex["messages"] for ex in batch]
        elif "instruction" in batch[0]:
            inputs = batch
        else:
            inputs = [ex["prompt"] for ex in batch]
        
        # Run ensemble inference
        outputs, selected_models, attribution = ensemble_framework.ensemble(
            inputs,
            model_specs=ensemble_config.model_specs,
            is_chat=("messages" in batch[0])
        )
        
        # Prepare results
        results = []
        for i, (example, output) in enumerate(zip(batch, outputs)):
            result = {
                "prompt": example.get("prompt") or example.get("messages") or 
                         f"{example.get('instruction', '')}\n{example.get('input', '')}".strip(),
                "predict": output,
                "label": example.get("label", ""),
                "selected_models": selected_models,
                "method": ensemble_config.output_aggregation_method,
                "config": {
                    "model_selection_method": ensemble_config.model_selection_method,
                    "output_aggregation_method": ensemble_config.output_aggregation_method,
                    "temperature": ensemble_config.temperature,
                    "max_tokens": ensemble_config.max_tokens,
                }
            }
            
            if show_attribution and attribution:
                try:
                    result["attribution"] = attribution[i]
                except (IndexError, TypeError):
                    logger.warning(f"Attribution data not available for example {i}")
            
            results.append(result)
        
        return results
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing batch: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return error results for all examples in batch
        return [{
            "prompt": ex.get("prompt", str(ex)),
            "predict": "",
            "label": ex.get("label", ""),
            "error": str(e),
            "selected_models": [],
            "method": ensemble_config.output_aggregation_method
        } for ex in batch]


def save_results(results: List[Dict[str, Any]], output_path: str, mode: str = "a"):
    """
    Save results to a JSONL file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output file
        mode: File open mode ("a" for append, "w" for write)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
    
    # Data paths (these are specific to inference, not in the YAML)
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
    
    # Processing options (these are specific to inference, not in the YAML)
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
        "--input_format",
        type=str,
        default="prompt",
        choices=["prompt", "dict", "chat"],
        help="Input format type"
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
    
    # Process in batches
    total_examples = len(dataset)
    logger.info(f"Processing {total_examples} examples in batches of {args.batch_size}")
    
    all_results = []
    start_time = time.time()
    
    with tqdm(total=total_examples, initial=processed_count, desc="Processing") as pbar:
        for i in range(0, total_examples, args.batch_size):
            batch_data = dataset[i:i + args.batch_size]
            
            # Prepare batch
            batch = []
            for example in batch_data:
                try:
                    prepared = prepare_example_for_inference(example, args.input_format)
                    batch.append(prepared)
                except Exception as e:
                    logger.error(f"Error preparing example: {e}")
                    batch.append({
                        "prompt": str(example),
                        "label": "",
                        "error": str(e)
                    })
            
            # Process batch
            results = process_batch(
                batch,
                ensemble_framework,
                ensemble_config,
                args.show_attribution
            )
            
            # Save results
            save_results(results, args.output_path, mode="a" if i > 0 or args.resume else "w")
            all_results.extend(results)
            
            pbar.update(len(batch_data))
    
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