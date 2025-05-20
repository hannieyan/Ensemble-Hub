import json
import logging
import time
import re
import fire
import os
from datasets import load_dataset

try:
    import jieba  # type: ignore
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from rouge_chinese import Rouge  # type: ignore

    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()
except ImportError:
    print("Please install llamafactory with `pip install -e .[metrics]`.")
    raise

import sys
sys.path.append(os.path.dirname(__file__))
from grader import grade_answer  # 用官方grader判分！

def extract_boxed_content(text):
    """Extract content inside \boxed{} command, handling nested braces."""
    start = text.find(r'\boxed{')
    if start == -1:
        return ""
    i = start + len(r'\boxed{')
    depth = 1
    content = ""
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth > 0:
            content += text[i]
        i += 1
    return content.strip()

def compute_metrics(sample):
    hypothesis = list(jieba.cut(sample["predict"]))
    reference = list(jieba.cut(sample["label"]))

    predicted_boxed = extract_boxed_content(sample["predict"])
    reference_boxed = extract_boxed_content(sample["label"])
    
    accuracy = 1.0 if grade_answer(predicted_boxed, reference_boxed) else 0.0

    bleu_score = sentence_bleu(
        [list(sample["label"])],
        list(sample["predict"]),
        smoothing_function=SmoothingFunction().method3,
    )

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    metric_result = {}
    for k, v in result.items():
        metric_result[k] = round(v["f"] * 100, 4)

    metric_result["bleu-4"] = float(round(bleu_score * 100, 4))
    metric_result["accuracy"] = float(round(accuracy * 100, 4))

    metric_result["predicted_boxed"] = predicted_boxed
    metric_result["reference_boxed"] = reference_boxed

    return metric_result

def generate_output_paths(input_path):
    base, ext = os.path.splitext(input_path)
    detailed_path = f"{base}-detailed-results.jsonl"
    score_path = f"{base}-predictions-score.json"
    return detailed_path, score_path

def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")
    dataset = dataset.map(compute_metrics, num_proc=64)
    
    metrics = ["rouge-1", "rouge-2", "rouge-l", "bleu-4", "accuracy"]
    average_score = {metric: 0.0 for metric in metrics}
    total_samples = len(dataset)
    
    for sample in dataset:
        for metric in metrics:
            average_score[metric] += sample[metric]
    
    for metric in metrics:
        average_score[metric] = round(average_score[metric] / total_samples, 4)
        print(f"{metric}: {average_score[metric]:.4f}")
    
    detailed_results = []
    for i, sample in enumerate(dataset):
        detailed_results.append({
            "index": i,
            "predicted_boxed": sample["predicted_boxed"],
            "reference_boxed": sample["reference_boxed"],
            "accuracy": sample["accuracy"],
            "metrics": {metric: sample[metric] for metric in metrics if metric != "accuracy"}
        })

    detailed_path, score_path = generate_output_paths(filename)

    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    with open(detailed_path, "w", encoding="utf-8") as f:
        for record in detailed_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone in {time.time() - start_time:.3f}s.")
    print(f"Score file saved to {score_path}")
    print(f"Detailed results saved to {detailed_path}")

if __name__ == "__main__":
    fire.Fire(main)
