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
    max_examples: int = None
):
    dataset = load_dataset_json(input_path)
    stat_store = ModelStatStore()

    # 清空已有文件内容
    Path(output_path).write_text("", encoding="utf-8")

    for example in tqdm(dataset[:max_examples] if max_examples else dataset):
        instruction = example["instruction"].strip()
        question = example["input"].strip()
        answer = example["output"].strip()

        prompt = f"<｜User｜>{instruction}\n{question}<｜Assistant｜>"

        try:
            result = run_zscore_ensemble(
                example=example,
                dataset_problems=math_problem_stats,
                model_specs=model_specs,
                reward_spec=reward_spec,
                stat_store=stat_store
            )

            prediction_text = result["output"].strip()
            selected_models = result["selected_models"]

        except Exception as e:
            print(f"⚠️ Error on question: {question[:80]}... -> {e}")
            prediction_text = ""
            selected_models = []

        append_prediction_to_file({
            "prompt": prompt,
            "predict": prediction_text,
            "label": answer.strip(),
            "selected_models": selected_models
        }, output_path)


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
        help="Maximum number of examples to process"
    )

    args = parser.parse_args()

    # 加载 MATH-500 的问题文本作为参考
    math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_problems = [x["problem"] for x in math_dataset]

    # 模型配置
    model_specs = [
        {"path": "Qwen/Qwen2.5-Math-1.5B-Instruct",           "engine": "hf", "device": "cuda:0"},
        {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf", "device": "cuda:1"},
        {"path": "Qwen/Qwen3-4B",                             "engine": "hf", "device": "cuda:2"},
        {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf", "device": "cuda:6"},
        {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "engine": "hf", "device": "cuda:4"},
        {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  "engine": "hf", "device": "cuda:5"},
    ]

    reward_spec = [
        {"path": "Qwen/Qwen2.5-Math-PRM-7B",                  "engine": "hf_rm",  "device": "cuda:0", "weight": 0.2},
        {"path": "http://localhost:8000/v1/score/evaluation", "engine": "api",                        "weight": 0.4},
        {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf_gen", "device": "cuda:0", "weight": 1.0},
    ]

    run_batch_inference(
        input_path=args.input_path,
        output_path=args.output_path,
        model_specs=model_specs,
        reward_spec=reward_spec,
        math_problem_stats=math_problems,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()
