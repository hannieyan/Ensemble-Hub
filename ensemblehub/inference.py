import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from ensemblehub.utils import run_zscore_ensemble, ModelStatStore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_dataset_json(input_path: str) -> list:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions(predictions: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")



def append_prediction_to_file(item: dict, output_path: str):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_batch_inference(
    input_path: str,
    output_path: str,
    model_specs: list,
    reward_spec: list,
    math_problem_stats: list,
    max_examples: int = None,
    batch_size: int = 1,
    ensemble_method: str = "simple",
    max_rounds: int = 500,
    score_threshold: float = -2.0
):
    # Validate inputs
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")
    
    if not model_specs:
        raise ValueError("At least one model specification is required")
    
    logger.info(f"Starting batch inference with:")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Ensemble method: {ensemble_method}")
    logger.info(f"  Max rounds: {max_rounds}")
    logger.info(f"  Score threshold: {score_threshold}")
    logger.info(f"  Models: {len(model_specs)}")
    logger.info(f"  Reward models: {len(reward_spec)}")
    
    dataset = load_dataset_json(input_path)
    logger.info(f"Loaded {len(dataset)} examples from dataset")
    
    stat_store = ModelStatStore()

    # 清空已有文件内容
    Path(output_path).write_text("", encoding="utf-8")
    logger.info(f"Cleared output file: {output_path}")

    if max_examples:
        dataset = dataset[:max_examples]
        logger.info(f"Limited to {len(dataset)} examples")

    # Process data in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[i:i + batch_size]
        batch_results = []

        for example in batch:
            instruction = example["instruction"].strip()
            question = example["input"].strip()
            answer = example["output"].strip()

            prompt = f"<｜User｜>{instruction}\n{question}<｜Assistant｜>"

            try:
                logger.debug(f"Processing example {i * batch_size + len(batch_results) + 1}")
                result = run_zscore_ensemble(
                    example=example,
                    dataset_problems=math_problem_stats,
                    model_specs=model_specs,
                    reward_spec=reward_spec,
                    stat_store=stat_store,
                    ensemble_method=ensemble_method,
                    max_rounds=max_rounds,
                    score_threshold=score_threshold
                )

                prediction_text = result["output"].strip() if result["output"] else ""
                selected_models = result.get("selected_models", [])
                
                logger.debug(f"Successfully processed example, prediction length: {len(prediction_text)}")

            except Exception as e:
                logger.error(f"Error processing example {i * batch_size + len(batch_results) + 1}: {e}", exc_info=True)
                logger.error(f"Question preview: {question[:80]}...")
                prediction_text = ""
                selected_models = []

            batch_results.append({
                "prompt": prompt,
                "predict": prediction_text,
                "label": answer.strip(),
                "selected_models": selected_models
            })

        # Write batch results
        for result in batch_results:
            append_prediction_to_file(result, output_path)


def main():
    parser = argparse.ArgumentParser(description="Batch Inference for Math Tasks")

    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input dataset JSON file"
    )
    parser.add_argument(
        "--output_path", type=str, default="ensemble-generated-predictions.jsonl",
        help="Path to save generated predictions"
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Maximum number of examples to process (default: process all)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for processing examples (default: 1, increase for better throughput)"
    )
    parser.add_argument(
        "--ensemble_method", type=str, default="simple",
        choices=["simple", "random", "loop"],
        help="Ensemble method to use (default: simple)"
    )
    parser.add_argument(
        "--max_rounds", type=int, default=500,
        help="Maximum rounds for ensemble reasoning (default: 500)"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=-2.0,
        help="Score threshold for ensemble reasoning (default: -2.0)"
    )

    args = parser.parse_args()

    # 加载 MATH-500 的问题文本作为参考
    math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_problems = [x["problem"] for x in math_dataset]

    # 模型配置
    model_specs = [
        {"path": "Qwen/Qwen2.5-0.5B-Instruct",           "engine": "hf", "device": "mps"},
        # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf", "device": "cuda:1"},
        # {"path": "Qwen/Qwen3-4B",                             "engine": "hf", "device": "cuda:2"},
        # {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf", "device": "cuda:6"},
        # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "engine": "hf", "device": "cuda:4"},
        # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  "engine": "hf", "device": "cuda:5"},
    ]

    reward_spec = [
        # {"path": "Qwen/Qwen2.5-Math-PRM-7B",                  "engine": "hf_rm",  "device": "cuda:0", "weight": 0.2},
        # {"path": "http://localhost:8000/v1/score/evaluation", "engine": "api",                        "weight": 0.4},
        # {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf_gen", "device": "cuda:0", "weight": 1.0},
    ]

    run_batch_inference(
        input_path=args.input_path,
        output_path=args.output_path,
        model_specs=model_specs,
        reward_spec=reward_spec,
        math_problem_stats=math_problems,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        ensemble_method=args.ensemble_method,
        max_rounds=args.max_rounds,
        score_threshold=args.score_threshold
    )


if __name__ == "__main__":
    main()
