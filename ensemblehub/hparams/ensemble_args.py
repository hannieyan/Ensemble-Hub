"""
Arguments for general ensemble configuration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EnsembleArguments:
    """Arguments pertaining to general ensemble behavior."""
    
    model_specs: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={"help": "List of model specifications with path, engine, and device settings."}
    )
    
    max_rounds: int = field(
        default=500,
        metadata={"help": "Maximum number of generation rounds for iterative methods."}
    )
    
    show_output_details: bool = field(
        default=False,
        metadata={"help": "Show detailed output results in logs."}
    )
    
    show_input_details: bool = field(
        default=False,
        metadata={"help": "Show raw HTTP request body in logs."}
    )
    
    show_attribution: bool = field(
        default=False,
        metadata={"help": "Include detailed model attribution information in output."}
    )
    
    save_results: bool = field(
        default=False,
        metadata={"help": "Save results to saves/logs directory for debugging and analysis."}
    )