"""
Output Aggregation Module for Ensemble-Hub

This module contains methods for aggregating outputs from multiple models
during or after inference. Three main granularities:

1. Token-level: Aggregate token distributions (e.g., GaC, weighted averaging)
2. Sentence-level: Aggregate sentence/segment outputs (e.g., reward-based selection)  
3. Response-level: Aggregate complete responses (e.g., voting, summarization)
"""

# Token-level aggregation
from .token_level.base import BaseTokenAggregator
from .token_level.distribution import DistributionAggregator, WeightedAverageAggregator
from .token_level.gac import GaCTokenAggregator

# Sentence-level aggregation  
from .sentence_level.base import BaseSentenceAggregator
from .sentence_level.reward_based import RewardBasedSelector
from .sentence_level.random_selector import RandomSentenceSelector
from .sentence_level.round_robin import RoundRobinSelector

# Response-level aggregation
from .response_level.base import BaseResponseAggregator
# TODO: Implement voting and generative fusion
# from .response_level.voting import MajorityVoting, WeightedVoting
# from .response_level.generative import GenerativeFusion

__all__ = [
    # Token-level
    "BaseTokenAggregator",
    "DistributionAggregator", 
    "WeightedAverageAggregator",
    "GaCTokenAggregator",
    
    # Sentence-level
    "BaseSentenceAggregator",
    "RewardBasedSelector",
    "RandomSentenceSelector", 
    "RoundRobinSelector",
    
    # Response-level
    "BaseResponseAggregator",
    # TODO: Add when implemented
    # "MajorityVoting",
    # "WeightedVoting", 
    # "GenerativeFusion"
]