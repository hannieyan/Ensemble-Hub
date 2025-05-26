"""
Generator modules for different inference backends
"""
from .base import BaseGenerator, GenOutput
from .hf import HFGenerator
from .vllm import VLLMGenerator

# Optional Ray-based vLLM
try:
    from .vllm_ray import VLLMRayGenerator
    __all__ = [
        "BaseGenerator",
        "GenOutput", 
        "HFGenerator",
        "VLLMGenerator",
        "VLLMRayGenerator",
    ]
except ImportError:
    __all__ = [
        "BaseGenerator",
        "GenOutput", 
        "HFGenerator",
        "VLLMGenerator",
    ]