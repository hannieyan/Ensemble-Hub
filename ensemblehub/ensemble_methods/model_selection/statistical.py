"""
Statistical-based model selection methods.
These use statistical measures like perplexity, confidence, and z-scores.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Callable
import torch
import torch.nn.functional as F
import ray

from .base import BaseModelSelector
from ...generators import GeneratorPool

logger = logging.getLogger(__name__)


class StatisticalSelector(BaseModelSelector):
    """
    Base class for statistical model selection methods.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)


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



class JudgmentSelector(StatisticalSelector):
    """
    Select models based on judgment scores using Ray generators.
    """

    def __init__(self, name: str = None):
        super().__init__(name or "JudgmentSelector")

    def select_models(
        self,
        example: List[str],
        model_specs: List[Dict[str, Any]],
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        generators = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Select one model based on self-judgment confidence (P_yes - P_no).
        Simplified for single sample input.
        
        Args:
            example: List of problem strings (assumes single sample)
            model_specs: List of model specifications
            model_stats: Optional model statistics for normalization
            generators: Dict of Ray remote generators
            **kwargs: Additional arguments
            
        Returns:
            List with single selected model spec
        """
        if generators is None:
            raise ValueError("generators parameter is required for JudgmentSelector")
            
        # Use only the first example for single sample case
        problem = example[0]
        
        # Create judgment prompt
        judgment_prompt = f"""Please carefully assess whether you can solve this problem. Be cautious and avoid saying yes if you're not confident.

        Problem: {problem}
        
        Can you solve this problem? Answer only with 'yes' or 'no': """
        
        # Collect confidence scores for each model
        model_confidences = []
        
        for spec in model_specs:
            model_path = spec["path"]
            
            # Get token probabilities for yes/no using Ray
            raw_confidence = ray.get(generators[model_path].get_token_confidence.remote(judgment_prompt))
            
            # Normalize confidence using model stats if available
            normalized_confidence = raw_confidence
            if model_stats and model_path in model_stats:
                stats = model_stats[model_path]
                if "conf_mean" in stats:
                    # Simple bias correction: raw - mean
                    normalized_confidence = raw_confidence - stats["conf_mean"]
                    logger.debug(f"Model {model_path}: raw_conf={raw_confidence:.3f}, normalized_conf={normalized_confidence:.3f}")
            
            model_confidences.append((spec, normalized_confidence))
        
        # Sort by confidence (P_yes - P_no) descending and select best
        model_confidences.sort(key=lambda x: x[1], reverse=True)
        selected_spec = model_confidences[0][0]
        
        logger.info(f"Selected model: {selected_spec['path']} with confidence: {model_confidences[0][1]:.3f}")
        
        return [selected_spec]
    
    def _get_token_confidence(self, gen, prompt):
        """Get P_yes - P_no confidence score by extracting token probabilities."""
        # Tokenize prompt
        inputs = gen.tokenizer(prompt, return_tensors="pt").to(gen.device)
        
        # Generate with model to get logits
        with torch.no_grad():
            outputs = gen.model.generate(
                **inputs,
                max_new_tokens=1,  # Only need first token
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False
            )
            
            # Get logits for the generated token (first new token)
            logits = outputs.scores[0]  # Shape: [1, vocab_size]
        
        # Get token IDs for "yes" and "no" 
        yes_tokens = gen.tokenizer.encode("yes", add_special_tokens=False)
        no_tokens = gen.tokenizer.encode("no", add_special_tokens=False)
        yes_tokens_cap = gen.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens_cap = gen.tokenizer.encode("No", add_special_tokens=False)
        
        # Get all possible IDs
        yes_ids = list(set(yes_tokens + yes_tokens_cap))
        no_ids = list(set(no_tokens + no_tokens_cap))
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sum probabilities for all yes/no variants
        p_yes = sum(probs[0, token_id].item() for token_id in yes_ids if token_id < probs.shape[1])
        p_no = sum(probs[0, token_id].item() for token_id in no_ids if token_id < probs.shape[1])
        
        # Normalize and compute confidence
        total = p_yes + p_no
        if total > 0:
            p_yes_norm = p_yes / total
            p_no_norm = p_no / total
            confidence = p_yes_norm - p_no_norm
        else:
            confidence = 0.0
        
        return confidence

