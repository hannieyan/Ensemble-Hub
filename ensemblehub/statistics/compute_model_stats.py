from typing import Dict, List

import math
import torch
import torch.nn.functional as F

class ModelStatStore:
    def __init__(self):
        self._stats: Dict[str, Dict[str, float]] = {}

    def has(self, model_path: str) -> bool:
        return model_path in self._stats

    def get(self, model_path: str) -> Dict[str, float]:
        return self._stats[model_path]

    def set(self, model_path: str, stats: Dict[str, float]):
        self._stats[model_path] = stats

    def compute(self, model_path: str, model, tokenizer, device, dataset: List[str]):
        if not self.has(model_path):
            stats = compute_model_stats_on_dataset(model, tokenizer, device, dataset)
            self.set(model_path, stats)
        return self.get(model_path)

def compute_model_stats_on_dataset(model, tokenizer, device, dataset: List[str]) -> Dict[str, float]:
    all_ppls, all_confs = [], []
    for problem in dataset:
        inputs = tokenizer(problem, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            mask = labels != tokenizer.pad_token_id
            token_log_probs = token_log_probs[mask]
            avg_nll = -token_log_probs.mean().item()
            perplexity = math.exp(avg_nll)

            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values.squeeze(0)
            mask_flat = mask.squeeze(0)
            confidence = max_probs[mask_flat].mean().item()

            all_ppls.append(perplexity)
            all_confs.append(confidence)

    return {
        "ppl_mean": float(torch.tensor(all_ppls).mean()),
        "ppl_std": float(torch.tensor(all_ppls).std()),
        "conf_mean": float(torch.tensor(all_confs).mean()),
        "conf_std": float(torch.tensor(all_confs).std()),
    }