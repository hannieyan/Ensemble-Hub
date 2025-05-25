"""
Statistical-based model selection methods.
These use statistical measures like perplexity, confidence, and z-scores.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Callable
import torch
import torch.nn.functional as F

from .base import BaseModelSelector
from ...generator import GeneratorPool

logger = logging.getLogger(__name__)


class StatisticalSelector(BaseModelSelector):
    """
    Base class for statistical model selection methods.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.generator_pool = GeneratorPool()


class ZScoreSelector(StatisticalSelector):
    """
    Select models based on z-score ranking of perplexity and confidence.
    This is the method currently used in utils.py.
    """
    
    def __init__(self, model_count: int = -1, name: str = None):
        super().__init__(name or "ZScoreSelector")
        self.model_count = model_count  # -1 means auto-determine
    
    def select_models(
        self,
        example: Dict[str, Any],
        model_specs: List[Dict[str, Any]],
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        prompt_builder: Optional[Callable] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Select models using z-score ranking based on perplexity and confidence.
        """
        if not model_stats:
            logger.warning("No model stats provided, selecting all models")
            return model_specs
        
        if not prompt_builder:
            # Default prompt builder
            prompt_builder = lambda q: q
        
        question = example.get("input", "")
        if not question:
            logger.warning("No input found in example, selecting all models")
            return model_specs
        
        logger.info(f"Computing z-scores for {len(model_specs)} models")
        
        results = []
        question_scores = []
        
        for spec in model_specs:
            model_path = spec["path"]
            if model_path not in model_stats:
                logger.warning(f"No stats for model {model_path}, skipping")
                continue
                
            try:
                # Get generator and compute scores
                generator = self.generator_pool.get_generator(
                    spec["path"], 
                    spec.get("engine", "hf"), 
                    spec.get("device")
                )
                
                score = self._score_question_for_model(
                    question, 
                    generator.model, 
                    generator.tokenizer, 
                    generator.device, 
                    prompt_builder
                )
                
                stats = model_stats[model_path]
                
                # Calculate z-scores
                z_ppl = (stats["ppl_mean"] - score["ppl"]) / stats["ppl_std"]
                z_conf = (score["conf"] - stats["conf_mean"]) / stats["conf_std"]
                total_score = z_ppl + z_conf
                
                results.append((total_score, spec))
                question_scores.append(score)
                
                logger.debug(f"Model {model_path}: ppl={score['ppl']:.2f}, conf={score['conf']:.2f}, z_score={total_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error scoring model {model_path}: {e}")
                continue
        
        if not results:
            logger.error("No models could be scored, returning all models")
            return model_specs
        
        # Determine model count
        model_count = self.model_count
        if model_count == -1:
            model_count = self._determine_model_count(question_scores, model_stats)
        
        # Sort by z-score and select top models
        results.sort(key=lambda x: x[0], reverse=True)
        selected_specs = [spec for _, spec in results[:model_count]]
        
        logger.info(f"Selected {len(selected_specs)} models: {[s['path'] for s in selected_specs]}")
        return selected_specs
    
    def _score_question_for_model(
        self, 
        question: str, 
        model, 
        tokenizer, 
        device: str, 
        prompt_builder: Callable
    ) -> Dict[str, float]:
        """
        Score a question using perplexity and confidence.
        Migrated from utils.py.
        """
        prompt = prompt_builder(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]
            
            # Calculate perplexity
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            mask = labels != tokenizer.pad_token_id
            token_log_probs = token_log_probs[mask]
            avg_nll = -token_log_probs.mean().item()
            ppl = math.exp(avg_nll)
            
            # Calculate confidence
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values.squeeze(0)
            mask_flat = mask.squeeze(0)
            conf = max_probs[mask_flat].mean().item()
        
        return {"ppl": ppl, "conf": conf}
    
    def _determine_model_count(
        self, 
        question_scores: List[Dict[str, float]], 
        model_stats: Dict[str, Dict[str, float]]
    ) -> int:
        """
        Automatically determine how many models to select.
        Migrated from utils.py.
        """
        over_threshold = 0
        for score, (model_path, stats) in zip(question_scores, model_stats.items()):
            if score["ppl"] > stats["ppl_mean"]:
                over_threshold += 1
        
        if over_threshold >= len(question_scores) * 0.90:
            return 3
        else:
            return 2


class AllModelsSelector(StatisticalSelector):
    """
    Simple selector that returns all models.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name or "AllModelsSelector")
    
    def select_models(
        self,
        example: Dict[str, Any],
        model_specs: List[Dict[str, Any]],
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Return all provided models.
        """
        logger.info(f"Selecting all {len(model_specs)} models")
        return model_specs


class RandomSelector(StatisticalSelector):
    """
    Randomly select a subset of models.
    """
    
    def __init__(self, k: int = 2, name: str = None):
        super().__init__(name or f"RandomSelector(k={k})")
        self.k = k
    
    def select_models(
        self,
        example: Dict[str, Any],
        model_specs: List[Dict[str, Any]],
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Randomly select k models.
        """
        import random
        
        k = min(self.k, len(model_specs))
        selected = random.sample(model_specs, k)
        
        logger.info(f"Randomly selected {len(selected)} models: {[s['path'] for s in selected]}")
        return selected