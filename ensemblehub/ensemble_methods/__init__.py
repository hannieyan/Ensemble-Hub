"""
Ensemble Methods Module for Ensemble-Hub

This module contains all ensemble methods including:
1. Model Selection: Choose which models to use for inference
2. Output Aggregation: Combine outputs from multiple models
3. Unified Framework: Flexible combination of selection and aggregation

The new unified framework (ensemble.py) replaces the old separate methods
(loop.py, random.py, simple.py) with a more flexible and extensible approach.
"""

# Import the new unified framework
from .ensemble import (
    EnsembleFramework,
    EnsembleConfig,
)

# Import model selection methods
from .model_selection.statistical import ZScoreSelector, AllModelsSelector, RandomSelector
from .model_selection.learned import LLMBlenderSelector, MetaLearningSelector

# Import output aggregation methods
from .output_aggregation.sentence_level.reward_based import RewardBasedSelector
from .output_aggregation.sentence_level.random_selector import RandomSentenceSelector
from .output_aggregation.sentence_level.loop_selector import LoopSelector

# Import token-level aggregation
from .output_aggregation.token_level.gac import GaCTokenAggregator
from .output_aggregation.token_level.distribution import DistributionAggregator, WeightedAverageAggregator


__all__ = [
    # New unified framework
    "EnsembleFramework",
    "EnsembleConfig",

    # Model Selection
    "ZScoreSelector",
    "AllModelsSelector", 
    "RandomSelector",
    "LLMBlenderSelector",
    "MetaLearningSelector",
    
    # Output Aggregation - Sentence Level
    "RewardBasedSelector",
    "RandomSentenceSelector",
    "LoopSelector",
    
    # Output Aggregation - Token Level
    "GaCTokenAggregator",
    "DistributionAggregator", 
    "WeightedAverageAggregator",
]