"""
Base classes for response-level output aggregation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseResponseAggregator(ABC):
    """
    Abstract base class for response-level aggregation methods.
    These methods work with complete responses and aggregate them.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def aggregate_responses(
        self,
        responses: List[str],
        scores: List[float] = None,
        **kwargs
    ) -> str:
        """
        Aggregate multiple complete responses into a single response.
        
        Args:
            responses: List of complete response strings
            scores: Optional scores for each response
            **kwargs: Additional aggregation parameters
            
        Returns:
            Aggregated response string
        """
        pass