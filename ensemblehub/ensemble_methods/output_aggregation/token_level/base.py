"""
Base class for token-level aggregation methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class BaseTokenAggregator(ABC):
    """
    Abstract base class for token-level output aggregation.
    
    Token-level aggregation combines probability distributions from multiple models
    at each token position to generate a single output sequence.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def aggregate_distributions(
        self,
        distributions: List[torch.Tensor],
        model_weights: List[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate token probability distributions from multiple models.
        
        Args:
            distributions: List of probability distributions (vocab_size,) or (seq_len, vocab_size)
            model_weights: Optional weights for each model
            **kwargs: Additional arguments
            
        Returns:
            Aggregated probability distribution
        """
        pass
    
    @abstractmethod
    def generate_sequence(
        self,
        generators: List,
        prompt: str,
        max_length: int = 512,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a sequence using token-level aggregation.
        
        Args:
            generators: List of generator models
            prompt: Input prompt
            max_length: Maximum sequence length
            **kwargs: Additional generation arguments
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"