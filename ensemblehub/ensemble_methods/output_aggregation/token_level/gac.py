"""
Generative Aggregate Classification (GaC) token aggregation.

Implementation of methods from:
"Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import logging

from .base import BaseTokenAggregator

logger = logging.getLogger(__name__)


class GaCTokenAggregator(BaseTokenAggregator):
    """
    Generative Aggregate Classification (GaC) aggregator.
    
    Treats token generation as a classification problem and uses
    ensemble classification techniques for aggregation.
    """
    
    def __init__(self, 
                 method: str = "avg",
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 name: str = None):
        super().__init__(name or f"GaCTokenAggregator({method})")
        self.method = method  # 'avg', 'max', 'weighted', 'stacking'
        self.temperature = temperature
        self.top_k = top_k
        self.meta_classifier = None
        
        if method == "stacking":
            self._init_meta_classifier()
    
    def _init_meta_classifier(self):
        """
        Initialize meta-classifier for stacking ensemble.
        """
        # Simple MLP meta-classifier
        # Input: concatenated predictions from base models
        # Output: final probability distribution
        self.meta_classifier = nn.Sequential(
            nn.Linear(512 * 6, 1024),  # Assuming 6 models, vocab_size=512 (placeholder)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.Softmax(dim=-1)
        )
        logger.info("Initialized meta-classifier for GaC stacking")
    
    def aggregate_distributions(
        self,
        distributions: List[torch.Tensor],
        model_weights: List[float] = None,
        confidence_scores: List[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate distributions using GaC method.
        """
        if not distributions:
            raise ValueError("No distributions provided")
        
        if self.method == "avg":
            return self._average_aggregation(distributions, model_weights)
        elif self.method == "max":
            return self._max_aggregation(distributions)
        elif self.method == "weighted":
            return self._confidence_weighted_aggregation(distributions, confidence_scores)
        elif self.method == "stacking":
            return self._stacking_aggregation(distributions)
        else:
            raise ValueError(f"Unknown GaC method: {self.method}")
    
    def _average_aggregation(
        self, 
        distributions: List[torch.Tensor], 
        model_weights: List[float] = None
    ) -> torch.Tensor:
        """
        Simple average aggregation with optional model weights.
        """
        if model_weights is None:
            model_weights = [1.0 / len(distributions)] * len(distributions)
        else:
            total_weight = sum(model_weights)
            model_weights = [w / total_weight for w in model_weights]
        
        # Apply temperature scaling
        scaled_distributions = []
        for dist in distributions:
            logits = torch.log(dist + 1e-10) / self.temperature
            scaled_dist = F.softmax(logits, dim=-1)
            scaled_distributions.append(scaled_dist)
        
        # Weighted average
        aggregated = torch.zeros_like(scaled_distributions[0])
        for dist, weight in zip(scaled_distributions, model_weights):
            aggregated += weight * dist
        
        return aggregated
    
    def _max_aggregation(self, distributions: List[torch.Tensor]) -> torch.Tensor:
        """
        Max probability aggregation - take the maximum probability for each token.
        """
        # Stack distributions and take max across models
        stacked = torch.stack(distributions, dim=0)  # (num_models, vocab_size)
        max_probs, _ = torch.max(stacked, dim=0)
        
        # Renormalize to ensure valid probability distribution
        aggregated = F.softmax(torch.log(max_probs + 1e-10), dim=-1)
        
        return aggregated
    
    def _confidence_weighted_aggregation(
        self, 
        distributions: List[torch.Tensor],
        confidence_scores: List[float] = None
    ) -> torch.Tensor:
        """
        Confidence-weighted aggregation using model confidence scores.
        """
        if confidence_scores is None:
            # Use entropy as confidence measure (lower entropy = higher confidence)
            confidence_scores = []
            for dist in distributions:
                entropy = -torch.sum(dist * torch.log(dist + 1e-10))
                confidence = torch.exp(-entropy)  # Convert to confidence
                confidence_scores.append(confidence.item())
        
        # Normalize confidence scores
        total_confidence = sum(confidence_scores)
        if total_confidence > 0:
            normalized_confidences = [c / total_confidence for c in confidence_scores]
        else:
            normalized_confidences = [1.0 / len(distributions)] * len(distributions)
        
        # Weighted aggregation
        aggregated = torch.zeros_like(distributions[0])
        for dist, conf in zip(distributions, normalized_confidences):
            aggregated += conf * dist
        
        return aggregated
    
    def _stacking_aggregation(self, distributions: List[torch.Tensor]) -> torch.Tensor:
        """
        Stacking ensemble using meta-classifier.
        """
        if self.meta_classifier is None:
            logger.warning("Meta-classifier not initialized, falling back to average")
            return self._average_aggregation(distributions)
        
        # Concatenate all distributions as input to meta-classifier
        concatenated = torch.cat(distributions, dim=-1)
        
        # Pass through meta-classifier
        with torch.no_grad():
            aggregated = self.meta_classifier(concatenated)
        
        return aggregated
    
    def apply_top_k_filtering(self, distribution: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k filtering to the distribution.
        """
        if self.top_k is None:
            return distribution
        
        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(distribution, self.top_k, dim=-1)
        
        # Create filtered distribution
        filtered_dist = torch.zeros_like(distribution)
        filtered_dist.scatter_(-1, top_k_indices, top_k_values)
        
        # Renormalize
        filtered_dist = F.softmax(torch.log(filtered_dist + 1e-10), dim=-1)
        
        return filtered_dist
    
    def generate_sequence(
        self,
        generators: List,
        prompt: str,
        max_length: int = 512,
        model_weights: List[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate sequence using GaC token aggregation.
        
        This is a placeholder implementation. Full implementation would require:
        1. Synchronized generation across models
        2. Real-time distribution aggregation
        3. Proper handling of different tokenizers
        """
        logger.warning("GaC token-level generation not fully implemented yet")
        
        metadata = {
            "method": f"gac_{self.method}",
            "models": len(generators),
            "temperature": self.temperature,
            "top_k": self.top_k
        }
        
        # Placeholder - use first generator for now
        if generators:
            result = generators[0].generate(prompt, max_tokens=max_length)
            return result.text, metadata
        
        return "", metadata
    
    def train_meta_classifier(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 10,
        learning_rate: float = 0.001
    ):
        """
        Train the meta-classifier for stacking ensemble.
        
        Args:
            training_data: List of training examples with model predictions and ground truth
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        if self.meta_classifier is None:
            self._init_meta_classifier()
        
        optimizer = torch.optim.Adam(self.meta_classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Training GaC meta-classifier for {epochs} epochs")
        
        for epoch in range(epochs):
            total_loss = 0
            for example in training_data:
                # Get model predictions and ground truth
                model_predictions = example["model_predictions"]  # List[torch.Tensor]
                ground_truth = example["ground_truth"]  # torch.Tensor (token indices)
                
                # Forward pass
                concatenated = torch.cat(model_predictions, dim=-1)
                predictions = self.meta_classifier(concatenated)
                
                # Compute loss
                loss = criterion(predictions, ground_truth)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(training_data)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("Meta-classifier training completed")