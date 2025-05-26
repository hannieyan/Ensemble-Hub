"""
Ray-based vLLM Generator for multi-GPU support
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Dict, Any

import torch

from .base import BaseGenerator, GenOutput, EOS_TEXT, STOP_TOKENS_TEXT, trim_text

logger = logging.getLogger("ensemble_inference")

# Optional Ray and vLLM imports
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def get_remote_vllm_generator_class(num_gpus: int = 1):
    """Dynamically register VLLMRayActor as a Ray remote class with specified GPUs"""
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not installed. Please install with: pip install ray")
    return ray.remote(num_gpus=num_gpus)(VLLMRayActor)


@ray.remote
class VLLMRayActor:
    """Ray actor for running vLLM on a specific GPU"""
    
    def __init__(self, model_path: str, device_id: int = 0, **vllm_kwargs):
        """Initialize vLLM in Ray actor with proper GPU isolation"""
        # Set CUDA device for this actor
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        from vllm import LLM, SamplingParams
        
        # Base vLLM configuration
        engine_args = {
            "model": model_path,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "disable_log_stats": True,
            "enforce_eager": True,
        }
        
        # Update with user kwargs
        engine_args.update(vllm_kwargs)
        
        # Initialize vLLM
        self.llm = LLM(**engine_args)
        self.model_path = model_path
        
        # Setup tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info(f"Initialized vLLM actor for {model_path} on GPU {device_id}")
    
    def generate(self, prompts: List[str], sampling_params: dict) -> List[dict]:
        """Generate text using vLLM"""
        from vllm import SamplingParams
        
        sp = SamplingParams(**sampling_params)
        outputs = self.llm.generate(prompts, sp)
        
        # Convert to serializable format
        results = []
        for output in outputs:
            results.append({
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason
            })
        return results
    
    def get_tokenizer_info(self) -> dict:
        """Get tokenizer information"""
        return {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }


class VLLMRayGenerator(BaseGenerator):
    """vLLM generator using Ray for proper multi-GPU support"""
    
    def __init__(self, path: str, device: str = "cuda:0", use_ray: bool = True, **vllm_kwargs):
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Please install with: pip install vllm")
            
        self.path = path
        self.name = path
        self.use_ray = use_ray and RAY_AVAILABLE
        
        if self.use_ray:
            # Initialize Ray if needed
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Extract device ID
            if device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
            else:
                device_id = 0
                
            # Create Ray actor on specific GPU
            actor_class = get_remote_vllm_generator_class(num_gpus=1)
            self.actor = actor_class.remote(path, device_id, **vllm_kwargs)
            
            # Get tokenizer info from actor
            tokenizer_info = ray.get(self.actor.get_tokenizer_info.remote())
            self._eos_token_id = tokenizer_info["eos_token_id"]
            
            logger.info(f"Created Ray-based vLLM generator for {path} on {device}")
        else:
            # Fallback to regular vLLM (with limitations)
            logger.warning("Ray not available, falling back to regular vLLM with limited multi-GPU support")
            from .vllm import VLLMGenerator
            self._fallback_generator = VLLMGenerator(path, device, **vllm_kwargs)
    
    def generate(self, dicts, *, max_tokens=256, temperature=0.95, top_p=0.7, **kwargs) -> GenOutput:
        """Generate text using Ray-based vLLM"""
        if not self.use_ray:
            return self._fallback_generator.generate(dicts, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **kwargs)
        
        # Convert input to prompt
        if isinstance(dicts, str):
            prompt = dicts
        else:
            # Simple conversion for dict input
            instruction = dicts.get("instruction", "")
            input_text = dicts.get("input", "")
            prompt = f"{instruction}\n{input_text}".strip()
        
        # Prepare sampling parameters
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": list(STOP_TOKENS_TEXT),
            "skip_special_tokens": True
        }
        
        # Generate using Ray actor
        results = ray.get(self.actor.generate.remote([prompt], sampling_params))
        
        if results:
            result = results[0]
            txt = result["text"]
            ended = result["finish_reason"] == "stop"
            return GenOutput(trim_text(txt) if not ended else txt, ended)
        else:
            return GenOutput("", False)
    
    def batch_generate(self, dicts_list: List[dict], *, max_tokens=256, temperature=0.95, top_p=0.7, **kwargs) -> List[GenOutput]:
        """Batch generation using Ray-based vLLM"""
        if not self.use_ray:
            return self._fallback_generator.batch_generate(dicts_list, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **kwargs)
        
        # Convert all inputs to prompts
        prompts = []
        for dicts in dicts_list:
            if isinstance(dicts, str):
                prompts.append(dicts)
            else:
                instruction = dicts.get("instruction", "")
                input_text = dicts.get("input", "")
                prompts.append(f"{instruction}\n{input_text}".strip())
        
        # Prepare sampling parameters
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": list(STOP_TOKENS_TEXT),
            "skip_special_tokens": True
        }
        
        # Generate using Ray actor
        results = ray.get(self.actor.generate.remote(prompts, sampling_params))
        
        outputs = []
        for result in results:
            txt = result["text"]
            ended = result["finish_reason"] == "stop"
            outputs.append(GenOutput(trim_text(txt) if not ended else txt, ended))
        
        return outputs
    
    def cleanup(self):
        """Clean up Ray actor"""
        if self.use_ray and hasattr(self, 'actor'):
            try:
                ray.kill(self.actor)
                logger.info(f"Cleaned up Ray actor for {self.path}")
            except Exception as e:
                logger.warning(f"Failed to clean up Ray actor: {e}")
        elif hasattr(self, '_fallback_generator'):
            self._fallback_generator.cleanup()