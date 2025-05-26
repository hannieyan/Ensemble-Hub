"""
Generator module - backward compatibility wrapper
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from .generators.base import BaseGenerator, GenOutput
from .generators.hf import HFGenerator
from .generators.vllm import VLLMGenerator

logger = logging.getLogger("ensemble_inference")


# Re-export for backward compatibility
__all__ = [
    "BaseGenerator",
    "GenOutput",
    "HFGenerator", 
    "VLLMGenerator",
    "GeneratorPool",
]


class GeneratorPool:
    """Caches all loaded generators and reward models"""
    _gen_cache: Dict[Tuple[str, str, str, str], BaseGenerator] = {}  # (engine, path, quantization, device)
    _reward_cache: Dict[str, str] = {}
    _vllm_lock = threading.Lock()  # Lock for vLLM initialization
    _vllm_instances = {}  # Track active vLLM instances by device
    _initialized_devices = set()  # Track which devices have been initialized

    @classmethod
    def get_generator(cls, path: str, engine: str = "hf", device: Optional[str] = None, quantization: str = "none") -> BaseGenerator:
        """
        Load a generator model (e.g., HF or vLLM) to a specified device (e.g., 'cuda:0', 'cpu').
        """
        resolved_device = device or "auto"
        key = (engine, path, quantization, resolved_device)  # Include device in cache key
        
        if key not in cls._gen_cache:
            logger.info("[Pool] loading %s (%s) with quantization=%s on device=%s", path, engine, quantization, resolved_device)

            if engine == "hf":
                cls._gen_cache[key] = HFGenerator(path, device=resolved_device, quantization=quantization)
            elif engine == "vllm":
                # Use lock to prevent concurrent vLLM initialization
                with cls._vllm_lock:
                    # Double-check if another thread already initialized it
                    if key not in cls._gen_cache:
                        # Check if there's already a vLLM instance on this device
                        if resolved_device in cls._vllm_instances:
                            logger.warning(f"vLLM instance already exists on {resolved_device}, cleaning up...")
                            old_instance = cls._vllm_instances[resolved_device]
                            if hasattr(old_instance, 'cleanup'):
                                old_instance.cleanup()
                            del cls._vllm_instances[resolved_device]
                            # Wait for cleanup to complete
                            time.sleep(1.0)
                        
                        # Create new instance
                        instance = VLLMGenerator(path, device=resolved_device)
                        cls._gen_cache[key] = instance
                        cls._vllm_instances[resolved_device] = instance
            else:
                raise ValueError(f"Unknown engine: {engine}")
        return cls._gen_cache[key]