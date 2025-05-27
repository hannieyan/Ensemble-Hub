"""
Generator module - backward compatibility wrapper

This module is deprecated. Please use:
    from ensemblehub.generators import BaseGenerator, GenOutput, HFGenerator, VLLMGenerator, GeneratorPool

Instead of:
    from ensemblehub.generator import BaseGenerator, GenOutput, HFGenerator, VLLMGenerator, GeneratorPool
"""
import warnings

# Import everything from the new location
from .generators import (
    BaseGenerator,
    GenOutput,
    HFGenerator,
    VLLMGenerator,
    GeneratorPool,
)

# Optional Ray-based vLLM
try:
    from .generators import VLLMRayGenerator
    __all__ = [
        "BaseGenerator",
        "GenOutput",
        "HFGenerator", 
        "VLLMGenerator",
        "VLLMRayGenerator",
        "GeneratorPool",
    ]
except ImportError:
    __all__ = [
        "BaseGenerator",
        "GenOutput",
        "HFGenerator", 
        "VLLMGenerator",
        "GeneratorPool",
    ]

# Emit deprecation warning
warnings.warn(
    "The 'ensemblehub.generator' module is deprecated. "
    "Please use 'ensemblehub.generators' instead.",
    DeprecationWarning,
    stacklevel=2
)