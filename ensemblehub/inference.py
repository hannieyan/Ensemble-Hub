#!/usr/bin/env python3
"""
Batch inference script for Ensemble-Hub framework.
Supports unified ensemble methods with batch processing and attribution tracking.
"""

import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Union

from ensemblehub.ensemble_methods.ensemble import run_ensemble

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset_json(input_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions(predictions: List[Dict], output_path: str):
    """Save predictions to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_prediction_to_file(item: Dict, output_path: str):
    """Append a single prediction to JSONL file."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_examples_batch(batch: List[Dict], input_format: str = "prompt") -> List[Union[str, Dict]]:
    """
    Prepare examples batch for ensemble processing.
    
    Args:
        batch: List of examples from dataset
        input_format: Format of input data ("prompt", "dict", "chat")
    
    Returns:
        List of formatted examples ready for ensemble processing
    """
    examples = []
    
    for example in batch:
        if input_format == "prompt":
            # Direct prompt format: concatenate instruction + input
            instruction = example.get("instruction", "").strip()
            input_text = example.get("input", "").strip()
            
            if instruction and input_text:
                # Combine instruction and input
                prompt = f"{instruction}\n{input_text}"
            elif instruction:
                prompt = instruction
            elif input_text:
                prompt = input_text
            else:
                # Fallback to any text field
                prompt = example.get("text", "").strip()
            
            examples.append(prompt)
            
        elif input_format == "dict":
            # Dict format with template support: keep original structure
            formatted_example = {
                "instruction": example.get("instruction", "").strip(),
                "input": example.get("input", "").strip(),
                "output": ""  # Will be filled by generation
            }
            examples.append(formatted_example)
            
        elif input_format == "chat":
            # Chat format: messages list
            if "messages" in example:
                examples.append(example["messages"])
            else:
                # Convert to chat format
                instruction = example.get("instruction", "").strip()
                input_text = example.get("input", "").strip()
                
                messages = []
                if instruction:
                    messages.append({"role": "system", "content": instruction})
                if input_text:
                    messages.append({"role": "user", "content": input_text})
                
                examples.append(messages)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
    
    return examples


def run_batch_inference(
    input_path: str,
    output_path: str,
    model_specs: List[Dict],
    reward_spec: List[Dict] = None,
    max_examples: int = None,
    batch_size: int = 4,
    output_aggregation_method: str = "loop",
    model_selection_method: str = "all",
    max_tokens: int = 2048,
    max_rounds: int = 500,
    score_threshold: float = -2.0,
    progressive_mode: str = "length",
    length_thresholds: List[int] = None,
    special_tokens: List[str] = None,
    input_format: str = "prompt",
    is_chat: bool = False,
    show_attribution: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = None
):
    """
    Run batch inference using the unified ensemble framework.
    
    Args:
        input_path: Path to input dataset JSON file
        output_path: Path to save predictions (JSONL format)
        model_specs: List of model specifications
        reward_spec: List of reward model specifications
        max_examples: Maximum number of examples to process
        batch_size: Number of examples to process in each batch
        output_aggregation_method: Method for aggregating outputs
        model_selection_method: Method for selecting models
        max_tokens: Maximum tokens to generate per example
        max_rounds: Maximum rounds for iterative methods
        score_threshold: Score threshold for early stopping
        progressive_mode: Mode for progressive selection
        length_thresholds: Length thresholds for progressive mode
        special_tokens: Special tokens for progressive mode
        input_format: Format of input data ("prompt", "dict", "chat")
        is_chat: Whether to use chat format for generation
        show_attribution: Whether to include attribution information
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        seed: Random seed for reproducibility
    """
    # Validate inputs
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")
    
    if not model_specs:
        raise ValueError("At least one model specification is required")
    
    # Setup reward specs
    if reward_spec is None:
        reward_spec = []
    
    logger.info(f"Starting batch inference with:")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Output aggregation: {output_aggregation_method}")
    logger.info(f"  Model selection: {model_selection_method}")
    logger.info(f"  Max tokens: {max_tokens}")
    logger.info(f"  Max rounds: {max_rounds}")
    logger.info(f"  Models: {len(model_specs)}")
    logger.info(f"  Reward models: {len(reward_spec)}")
    logger.info(f"  Input format: {input_format}")
    logger.info(f"  Chat mode: {is_chat}")
    
    # Load dataset
    dataset = load_dataset_json(input_path)
    logger.info(f"Loaded {len(dataset)} examples from dataset")
    
    # Clear output file
    Path(output_path).write_text("", encoding="utf-8")
    logger.info(f"Cleared output file: {output_path}")
    
    # Limit examples if specified
    if max_examples:
        dataset = dataset[:max_examples]
        logger.info(f"Limited to {len(dataset)} examples")
    
    # Process data in batches
    total_processed = 0
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[batch_idx:batch_idx + batch_size]
        
        try:
            # Prepare examples for ensemble processing
            examples = prepare_examples_batch(batch, input_format)
            
            logger.debug(f"Processing batch {batch_idx // batch_size + 1}/{total_batches} with {len(examples)} examples")
            
            # Run ensemble inference
            results = run_ensemble(
                examples=examples,
                model_specs=model_specs,
                reward_spec=reward_spec,
                output_aggregation_method=output_aggregation_method,
                model_selection_method=model_selection_method,
                max_tokens=max_tokens,
                max_rounds=max_rounds,
                score_threshold=score_threshold,
                progressive_mode=progressive_mode,
                length_thresholds=length_thresholds,
                special_tokens=special_tokens,
                is_chat=is_chat,
                temperature=temperature,
                top_p=top_p,
                seed=seed
            )
            
            # Process results
            for i, (original_example, result) in enumerate(zip(batch, results)):
                example_idx = batch_idx + i + 1
                
                try:
                    prediction_text = result["output"].strip() if result["output"] else ""
                    selected_models = result.get("selected_models", [])
                    attribution_data = result.get("attribution", {})
                    method_info = result.get("method", "unknown")
                    config_info = result.get("config", {})
                    
                    # Log attribution summary if available
                    if attribution_data and show_attribution:
                        if hasattr(attribution_data, 'get_summary'):
                            summary = attribution_data.get_summary()
                            logger.info(f"Example {example_idx} attribution: {summary}")
                    
                    # Create prompt for output
                    instruction = original_example.get("instruction", "").strip()
                    input_text = original_example.get("input", "").strip()
                    if instruction and input_text:
                        prompt = f"{instruction}\n{input_text}"
                    elif instruction:
                        prompt = instruction
                    else:
                        prompt = input_text
                        
                    # Prepare output entry
                    result_entry = {
                        "prompt": prompt,
                        "predict": prediction_text,
                        "label": original_example.get("output", "").strip(),
                        "selected_models": selected_models,
                        "method": method_info,
                        "config": config_info
                    }
                    
                    # Add attribution data if available and requested
                    if attribution_data and show_attribution:
                        result_entry["attribution"] = attribution_data
                    
                    # Save result
                    append_prediction_to_file(result_entry, output_path)
                    total_processed += 1
                    
                    logger.debug(f"Successfully processed example {example_idx}, prediction length: {len(prediction_text)}")
                
                except Exception as e:
                    logger.error(f"Error processing result for example {example_idx}: {e}", exc_info=True)
                    # Save error entry
                    error_entry = {
                        "prompt": original_example.get("instruction", "") + "\n" + original_example.get("input", ""),
                        "predict": "",
                        "label": original_example.get("output", ""),
                        "selected_models": [],
                        "method": "error",
                        "config": {},
                        "error": str(e)
                    }
                    append_prediction_to_file(error_entry, output_path)
                    total_processed += 1
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx // batch_size + 1}: {e}", exc_info=True)
            # Save error entries for the entire batch
            for i, original_example in enumerate(batch):
                error_entry = {
                    "prompt": original_example.get("instruction", "") + "\n" + original_example.get("input", ""),
                    "predict": "",
                    "label": original_example.get("output", ""),
                    "selected_models": [],
                    "method": "batch_error",
                    "config": {},
                    "error": str(e)
                }
                append_prediction_to_file(error_entry, output_path)
                total_processed += 1
    
    logger.info(f"Batch inference completed. Processed {total_processed} examples.")
    logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Inference for Ensemble-Hub")

    # Input/Output
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input dataset JSON file"
    )
    parser.add_argument(
        "--output_path", type=str, default="ensemble-predictions.jsonl",
        help="Path to save generated predictions (JSONL format)"
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Maximum number of examples to process (default: process all)"
    )
    
    # Batch processing
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for processing examples (default: 4)"
    )
    
    # Ensemble configuration
    parser.add_argument(
        "--output_aggregation_method", type=str, default="loop",
        choices=["reward_based", "random", "loop", "progressive", "gac", "distribution"],
        help="Output aggregation method (default: loop)"
    )
    parser.add_argument(
        "--model_selection_method", type=str, default="all",
        choices=["zscore", "all", "random", "llm_blender"],
        help="Model selection method (default: all)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Maximum tokens to generate per example (default: 2048)"
    )
    parser.add_argument(
        "--max_rounds", type=int, default=500,
        help="Maximum rounds for iterative methods (default: 500)"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=-2.0,
        help="Score threshold for early stopping (default: -2.0)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    
    # Progressive method parameters
    parser.add_argument(
        "--progressive_mode", type=str, default="length",
        choices=["length", "token"],
        help="Progressive selector mode (default: length)"
    )
    parser.add_argument(
        "--length_thresholds", type=str, default="1000,2000,3000",
        help="Comma-separated length thresholds for progressive mode (default: 1000,2000,3000)"
    )
    parser.add_argument(
        "--special_tokens", type=str, default="<think>",
        help="Comma-separated special tokens for progressive mode (default: <think>)"
    )
    
    # Input format
    parser.add_argument(
        "--input_format", type=str, default="prompt",
        choices=["prompt", "dict", "chat"],
        help="Input data format - prompt: direct string, dict: template dict, chat: messages (default: prompt)"
    )
    parser.add_argument(
        "--is_chat", action="store_true",
        help="Use chat format for generation"
    )
    
    # Output options
    parser.add_argument(
        "--show_attribution", action="store_true",
        help="Include detailed model attribution information in output"
    )
    
    # Model configuration
    parser.add_argument(
        "--models_config", type=str, default=None,
        help="Path to JSON file with model specifications (overrides default models)"
    )

    args = parser.parse_args()

    # Parse progressive parameters
    length_thresholds = [int(x.strip()) for x in args.length_thresholds.split(",")] if args.length_thresholds else None
    special_tokens = [x.strip() for x in args.special_tokens.split(",")] if args.special_tokens else None

    # Load model configuration
    if args.models_config and Path(args.models_config).exists():
        with open(args.models_config, 'r') as f:
            config = json.load(f)
            model_specs = config.get("model_specs", [])
            reward_spec = config.get("reward_spec", [])
        logger.info(f"Loaded model configuration from {args.models_config}")
    else:
        # Default model configuration - lightweight models for demo
        model_specs = [
            {"path": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "hf", "device": "auto"},
            {"path": "Qwen/Qwen2.5-1.5B-Instruct", "engine": "hf", "device": "auto"},
            # Add more models as needed:
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf", "device": "cuda:0"},
            # {"path": "Qwen/Qwen2.5-Math-7B-Instruct", "engine": "hf", "device": "cuda:1"},
        ]
        
        reward_spec = [
            # Example reward model configurations:
            # {"path": "Qwen/Qwen2.5-Math-PRM-7B", "engine": "hf_rm", "device": "cuda:2"},
            # {"path": "http://localhost:8000/v1/score", "engine": "api", "weight": 1.0},
        ]
        
        logger.info("Using default model configuration")

    # Validate model specs
    if not model_specs:
        logger.error("No model specifications provided. Please provide models via --models_config or modify the default configuration.")
        return

    logger.info(f"Model specs: {model_specs}")
    logger.info(f"Reward specs: {reward_spec}")

    # Run batch inference
    run_batch_inference(
        input_path=args.input_path,
        output_path=args.output_path,
        model_specs=model_specs,
        reward_spec=reward_spec,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        output_aggregation_method=args.output_aggregation_method,
        model_selection_method=args.model_selection_method,
        max_tokens=args.max_tokens,
        max_rounds=args.max_rounds,
        score_threshold=args.score_threshold,
        progressive_mode=args.progressive_mode,
        length_thresholds=length_thresholds,
        special_tokens=special_tokens,
        input_format=args.input_format,
        is_chat=args.is_chat,
        show_attribution=args.show_attribution,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed
    )


if __name__ == "__main__":
    main()