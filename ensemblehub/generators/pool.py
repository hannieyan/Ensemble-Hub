"""
Generator Pool - manages caching and initialization of generators
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Tuple, Any

from .hf_engine import HFGenerator
from .vllm import VLLMGenerator

# Optional Ray-based vLLM
try:
    from .vllm_ray import VLLMRayGenerator
    _VLLM_RAY_AVAILABLE = True
except ImportError:
    _VLLM_RAY_AVAILABLE = False

logger = logging.getLogger("ensemble_inference")


class GeneratorPool:
    """Caches all loaded generators and reward models"""
    _gen_cache: Dict[Tuple[str, str, str, str, bool]] = {}  # (engine, path, quantization, device, enable_thinking)
    _reward_cache: Dict[str, str] = {}
    _vllm_lock = threading.Lock()  # Lock for vLLM initialization
    _hf_lock = threading.Lock()  # Lock for HF model initialization
    _vllm_instances = {}  # Track active vLLM instances by device
    _initialized_devices = set()  # Track which devices have been initialized

    @classmethod
    def get_generator(cls, path: str, engine: str = "hf", device: Optional[str] = None, quantization: str = "none", enable_thinking: bool = True, **kwargs) -> Any:
        """
        Load a generator model (e.g., HF or vLLM) to a specified device (e.g., 'cuda:0', 'cpu').
        """
        resolved_device = device or "auto"
        key = (engine, path, quantization, resolved_device, enable_thinking)  # Include device and enable_thinking in cache key
        
        if key not in cls._gen_cache:
            logger.info("[Pool] loading %s (%s) with quantization=%s on device=%s", path, engine, quantization, resolved_device)

            if engine == "hf":
                # Use lock to prevent concurrent HF model initialization
                with cls._hf_lock:
                    # Double-check if another thread already initialized it
                    if key not in cls._gen_cache:
                        cls._gen_cache[key] = HFGenerator(path, quantization=quantization, enable_thinking=enable_thinking)
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
            elif engine == "vllm_ray":
                # Ray-based vLLM for better multi-GPU support
                if not _VLLM_RAY_AVAILABLE:
                    raise RuntimeError("VLLMRayGenerator not available. Please install ray: pip install ray")
                cls._gen_cache[key] = VLLMRayGenerator(path, device=resolved_device)
            else:
                raise ValueError(f"Unknown engine: {engine}")
        return cls._gen_cache[key]