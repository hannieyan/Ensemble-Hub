"""
Distribution-based token aggregation methods.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import logging

from .base import BaseTokenAggregator

logger = logging.getLogger(__name__)


class DistributionAggregator(BaseTokenAggregator):
    """
    Base class for distribution-based aggregation methods.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)


class WeightedAverageAggregator(DistributionAggregator):
    """
    Simple weighted average of probability distributions.
    """
    
    def __init__(self, temperature: float = 1.0, name: str = None):
        super().__init__(name or "WeightedAverageAggregator")
        self.temperature = temperature
    
    def aggregate_distributions(
        self,
        distributions: List[torch.Tensor],
        model_weights: List[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate distributions using weighted average.
        """
        if not distributions:
            raise ValueError("No distributions provided")
        
        # Normalize weights
        if model_weights is None:
            model_weights = [1.0 / len(distributions)] * len(distributions)
        else:
            total_weight = sum(model_weights)
            model_weights = [w / total_weight for w in model_weights]
        
        # Apply temperature scaling and aggregate
        aggregated = torch.zeros_like(distributions[0])
        for dist, weight in zip(distributions, model_weights):
            # Apply temperature scaling
            scaled_dist = F.softmax(torch.log(dist + 1e-10) / self.temperature, dim=-1)
            aggregated += weight * scaled_dist
        
        return aggregated
    
    def generate_sequence(
        self,
        generators: List,
        prompt: str,
        max_length: int = 512,
        model_weights: List[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate sequence using weighted average aggregation.
        
        Note: This is a simplified implementation. In practice, you'd need
        to handle tokenization, model loading, and generation carefully.
        """
        logger.warning("Token-level generation not fully implemented yet")
        
        # Placeholder implementation
        # In practice, you would:
        # 1. Tokenize the prompt
        # 2. For each token position:
        #    a. Get logits from all models
        #    b. Convert to probabilities
        #    c. Aggregate using weighted average
        #    d. Sample next token
        # 3. Repeat until EOS or max_length
        
        # For now, just use the first generator
        if generators:
            result = generators[0].generate(prompt, max_tokens=max_length)
            return result.text, {"method": "weighted_average", "models": len(generators)}
        
        return "", {"method": "weighted_average", "models": 0}


class GeometricMeanAggregator(DistributionAggregator):
    """
    Geometric mean of probability distributions.
    """
    
    def __init__(self, epsilon: float = 1e-10, name: str = None):
        super().__init__(name or "GeometricMeanAggregator")
        self.epsilon = epsilon
    
    def aggregate_distributions(
        self,
        distributions: List[torch.Tensor],
        model_weights: List[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate distributions using geometric mean.
        """
        if not distributions:
            raise ValueError("No distributions provided")
        
        # Add small epsilon to avoid log(0)
        log_dists = [torch.log(dist + self.epsilon) for dist in distributions]
        
        if model_weights is None:
            # Unweighted geometric mean
            mean_log_dist = torch.stack(log_dists).mean(dim=0)
        else:
            # Weighted geometric mean
            total_weight = sum(model_weights)
            normalized_weights = [w / total_weight for w in model_weights]
            
            weighted_log_dist = torch.zeros_like(log_dists[0])
            for log_dist, weight in zip(log_dists, normalized_weights):
                weighted_log_dist += weight * log_dist
            mean_log_dist = weighted_log_dist
        
        # Convert back to probabilities
        aggregated = torch.exp(mean_log_dist)
        
        # Renormalize to ensure it's a valid probability distribution
        aggregated = F.softmax(torch.log(aggregated + self.epsilon), dim=-1)
        
        return aggregated
    
    def generate_sequence(
        self,
        generators: List,
        prompt: str,
        max_length: int = 512,
        model_weights: List[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate sequence using geometric mean aggregation.
        """
        logger.warning("Token-level generation not fully implemented yet")
        
        # Placeholder - same as weighted average for now
        if generators:
            result = generators[0].generate(prompt, max_tokens=max_length)
            return result.text, {"method": "geometric_mean", "models": len(generators)}
        
        return "", {"method": "geometric_mean", "models": 0}


class MaxAggregator(DistributionAggregator):
    """
    Element-wise maximum of probability distributions.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name or "MaxAggregator")
    
    def aggregate_distributions(
        self,
        distributions: List[torch.Tensor],
        model_weights: List[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate distributions using element-wise maximum.
        """
        if not distributions:
            raise ValueError("No distributions provided")
        
        # Take element-wise maximum
        aggregated = distributions[0]
        for dist in distributions[1:]:
            aggregated = torch.maximum(aggregated, dist)
        
        # Renormalize
        aggregated = F.softmax(torch.log(aggregated + 1e-10), dim=-1)
        
        return aggregated
    
    def generate_sequence(
        self,
        generators: List,
        prompt: str,
        max_length: int = 512,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate sequence using max aggregation.
        """
        logger.warning("Token-level generation not fully implemented yet")
        
        if generators:
            result = generators[0].generate(prompt, max_tokens=max_length)
            return result.text, {"method": "max", "models": len(generators)}
        
        return "", {"method": "max", "models": 0}