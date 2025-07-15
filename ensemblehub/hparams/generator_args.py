"""
Arguments for text generation and generator configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GeneratorArguments:
    """Arguments pertaining to generation parameters and generator configuration."""
    
    # Generation parameters
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens to generate. None means auto-detect."}
    )
    
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for generation."}
    )
    
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling parameter."}
    )
    
    top_k: int = field(
        default=50,
        metadata={"help": "Top-k sampling parameter."}
    )
    
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    
    
    stop_strings: List[str] = field(
        default_factory=list,
        metadata={"help": "List of strings that stop generation."}
    )
    
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed for reproducible generation."}
    )
    
    diversity_penalty: float = field(
        default=0.0,
        metadata={"help": "This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time."}
    )
    
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences."}
    )

    hf_use_eager_attention: bool = field(
        default=True,
        metadata={"help": "Use eager attention implementation for HuggingFace models."}
    )
    
    hf_low_cpu_mem: bool = field(
        default=True,
        metadata={"help": "Use low CPU memory loading for HuggingFace models."}
    )