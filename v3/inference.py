# inference.py

import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from utils import run_zscore_ensemble, ModelStatStore

# 加载 MATH-500 的问题文本作为参考数据集
math_dataset = load_dataset(
    "/mnt/data/zichuanfu/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/ff5b20257d8185524591543f8ff5993951537bb8",
    split="test"
)
math_problems = [x["problem"] for x in math_dataset]

# 模型配置
model_specs = [
    {"path": "/mnt/data/zichuanfu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35", "engine": "hf", "device": "cuda:0"},
    {"path": "/mnt/data/zichuanfu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562", "engine": "hf", "device": "cuda:1"},
    {"path": "/mnt/data/zichuanfu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d", "engine": "hf", "device": "cuda:2"},
    {"path": "/mnt/data/zichuanfu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60", "engine": "hf", "device": "cuda:3"},
    {"path": "/mnt/data/zichuanfu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/1df8507178afcc1bef68cd8c393f61a886323761", "engine": "hf", "device": "cuda:4"},
]

reward_spec = {
    "path": "/mnt/data/zichuanfu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-PRM-7B/snapshots/0610740060112df12585d00a1c5f4624d2f59051",
    "device": "cuda:5"
}

def load_dataset_json(input_path: str) -> list:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_predictions(predictions: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def run_batch_inference(
    input_path: str,
    output_path: str,
    model_specs: list,
    reward_spec: dict,
    math_problem_stats: list,
    max_examples: int = None
):
    dataset = load_dataset_json(input_path)
    stat_store = ModelStatStore()

    predictions = []
    for example in tqdm(dataset[:max_examples] if max_examples else dataset):
        instruction = example["instruction"].strip()
        question = example["input"].strip()
        answer = example["output"].strip()

        # 构造完整 prompt
        prompt = f"system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\n{instruction}\n{question}\nassistant\n"

        try:
            result = run_zscore_ensemble(
                question=question,
                dataset_problems=math_problem_stats,
                model_specs=model_specs,
                reward_spec=reward_spec,
                stat_store=stat_store
            )
        except Exception as e:
            print(f"⚠️ Error on question: {question[:80]}... -> {e}")
            result = ""

        predictions.append({
            "prompt": prompt,
            "predict": result.strip(),
            "label": answer.strip()
        })

    save_predictions(predictions, output_path)
    print(f"✅ Saved {len(predictions)} predictions to {output_path}")


# 启动批量推理
if __name__ == "__main__":
    run_batch_inference(
        input_path="/mnt/data/zichuanfu/LLaMA-Factory/data/hendrycks_math/train.json",
        output_path="ensemble-generated-predictions.jsonl",
        model_specs=model_specs,
        reward_spec=reward_spec,
        math_problem_stats=math_problems,
        max_examples=100
    )
