"""
Arguments for ensemble method configuration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MethodArguments:
    """Arguments pertaining to model selection and output aggregation methods."""
    
    # Model selection
    model_selection_method: str = field(
        default="all",
        metadata={"help": "Model selection method: all, zscore, model_judgment, random."}
    )
    
    model_selection_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Parameters for model selection method."}
    )
    
    # Output aggregation
    output_aggregation_method: str = field(
        default="loop",
        metadata={"help": "Output aggregation method: loop, progressive, random, reward_based, gac, distribution."}
    )
    
    output_aggregation_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Parameters for output aggregation method."}
    )