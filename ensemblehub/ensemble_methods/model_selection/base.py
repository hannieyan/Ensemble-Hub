"""
Base classes for model selection strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModelSelector(ABC):
    """
    Abstract base class for model selection strategies.
    
    Model selection happens before inference and determines which models
    to use for a given input example.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def select_models(
        self, 
        example: Dict[str, Any],
        model_specs: List[Dict[str, Any]],
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Select models for the given example.
        
        Args:
            example: Input example containing "instruction", "input", "output"
            model_specs: List of model specifications
            model_stats: Optional precomputed model statistics
            **kwargs: Additional arguments
            
        Returns:
            List of selected model specifications
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"