"""
Sentence-level output aggregation methods.
These aggregate sentence/segment outputs from multiple models.
"""

from .base import BaseSentenceAggregator
from .reward_based import RewardBasedSelector
from .random_selector import RandomSentenceSelector
from .loop_selector import LoopSelector
from .progressive_selector import ProgressiveSelector
from .switch import Switch

__all__ = [
    "BaseSentenceAggregator",
    "RewardBasedSelector",
    "RandomSentenceSelector",
    "LoopSelector",
    "ProgressiveSelector",
    "Switch",
]