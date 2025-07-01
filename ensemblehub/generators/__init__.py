"""
Generator modules for different inference backends
"""
from .base import BaseGenerator, GenOutput
from .hf_engine import HFGenerator

# Optional Ray-based vLLM
try:
    __all__ = [
        "BaseGenerator",
        "GenOutput", 
        "HFGenerator",
    ]
except ImportError:
    __all__ = [
        "BaseGenerator",
        "GenOutput", 
        "HFGenerator",
    ]