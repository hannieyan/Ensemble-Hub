"""
Base class for sentence-level aggregation methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseSentenceAggregator(ABC):
    """
    Abstract base class for sentence-level output aggregation.
    
    Sentence-level aggregation selects the best sentence/segment from multiple
    model outputs at each generation step.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def select_best_sentence(
        self,
        sentences: List[str],
        generators: List,
        prompt: str,
        scorers = None,
        **kwargs
    ) -> Tuple[int, str, float]:
        """
        Select the best sentence from multiple candidates.
        
        Args:
            sentences: List of candidate sentences from different models
            generators: List of generator models
            prompt: Current prompt/context
            scorers: Scorer pool for evaluation
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (best_index, best_sentence, score)
        """
        pass
    
    @abstractmethod
    def aggregate_generation(
        self,
        generators: List,
        scorers,
        example: Dict[str, Any],
        max_rounds: int = 500,
        score_threshold: float = -2.0,
        **kwargs
    ) -> str:
        """
        Run iterative sentence-level aggregation generation.
        
        Args:
            generators: List of generator models
            scorers: Scorer pool for evaluation
            example: Input example
            max_rounds: Maximum generation rounds
            score_threshold: Score threshold for early stopping
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"