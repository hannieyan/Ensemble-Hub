"""
vLLM-based Generator
"""
from __future__ import annotations

import gc
import logging
import os
import time
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.converter import AlpacaDatasetConverter
from llamafactory.data.parser import DatasetAttr

from .base import BaseGenerator, GenOutput, EOS_TEXT, STOP_TOKENS_TEXT, trim_text

logger = logging.getLogger("ensemble_inference")

# Optional vLLM imports
try:
    from vllm import LLM, SamplingParams
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:
        try:
            from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        except ImportError:
            destroy_model_parallel = None
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    destroy_model_parallel = None


class VLLMGenerator(BaseGenerator):
    """vLLM-based text generator for high-throughput inference"""
    
    def __init__(self, path: str, device: str = "cuda:0", **vllm_kwargs):
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Please install with: pip install vllm")
        
        # Store original device
        self._original_device = device
        
        # Base vLLM configuration
        engine_args = {
            "model": path,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "disable_log_stats": True,
            "enforce_eager": True,  # Disable CUDA graphs to avoid caching allocator issues
        }
        
        # Handle GPU placement for vLLM
        if device == "auto" or device.startswith("cuda"):
            if device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
                # For vLLM, we need to set CUDA_VISIBLE_DEVICES before initialization
                # Save original for cleanup
                self._original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                
                # Set CUDA_VISIBLE_DEVICES to only show the target GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                logger.info(f"Set CUDA_VISIBLE_DEVICES={device_id} for vLLM on {path}")
                
                # vLLM will use the only visible device
                engine_args["tensor_parallel_size"] = 1
            else:
                # Auto device selection
                engine_args["tensor_parallel_size"] = 1
        
        # Update with user-provided arguments
        engine_args.update(vllm_kwargs)
        
        try:
            self._llm = LLM(**engine_args)
        except RuntimeError as e:
            if "CUDA" in str(e) or "captures_underway" in str(e):
                logger.error(f"CUDA initialization error for vLLM. This often happens with concurrent initialization. Error: {e}")
                # Try to clean up CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                raise RuntimeError(f"Failed to initialize vLLM for {path}. Try initializing models sequentially.") from e
            else:
                raise
                
        self._sp = SamplingParams(
            max_tokens=256,
            temperature=0.95, 
            top_p=0.7,
            stop=list(STOP_TOKENS_TEXT),
            skip_special_tokens=True
        )
        self.name = path
        self._eos_text = EOS_TEXT

        # Setup tokenizer and template like HFGenerator
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        if "qwen3" in path.lower():
            data_args = DataArguments(template="qwen")
            self.indent = 2
        elif "qwen2.5" in path.lower():
            data_args = DataArguments(template="qwen")
            self.indent = 2
        elif "deepseek-r1" in path.lower():
            data_args = DataArguments(template="deepseek3")
            self.indent = 1
        else:
            # Default template
            data_args = DataArguments(template="default")
            self.indent = 1

        dataset_attr = DatasetAttr(
            prompt="instruction",
            query="input", 
            response="output",
            load_from="file",
            formatting="alpaca",
            dataset_name="",
        )

        self.converter = AlpacaDatasetConverter(dataset_attr=dataset_attr, data_args=data_args)
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
    
    def cleanup(self):
        """Clean up vLLM resources and restore environment"""
        if hasattr(self, '_llm'):
            # Destroy model parallel if available
            if destroy_model_parallel is not None:
                try:
                    destroy_model_parallel()
                except Exception as e:
                    logger.warning(f"Failed to destroy model parallel: {e}")
            
            # Delete the LLM instance
            del self._llm
            
            # Clean up CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Restore original CUDA_VISIBLE_DEVICES
        if hasattr(self, '_original_cuda_devices'):
            if self._original_cuda_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_devices
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            logger.info(f"Restored CUDA_VISIBLE_DEVICES to: {self._original_cuda_devices}")

    def _dict_to_prompt(self, example_dict: dict) -> str:
        """Convert dict format to prompt string."""
        try:
            converted = self.converter(example_dict)
            prompt_msgs = converted["_prompt"]
            response_msgs = converted["_response"]
            messages = prompt_msgs + response_msgs
            prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages)
            
            ids = prompt_ids + response_ids[:-self.indent]
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            return text
        except Exception as e:
            logger.warning(f"Failed to convert dict to prompt for vLLM: {e}")
            # Fallback to simple format
            instruction = example_dict.get("instruction", "")
            input_text = example_dict.get("input", "")
            return f"{instruction}\n{input_text}"

    @torch.inference_mode()
    def generate(self, dicts, *, max_tokens=256, temperature=0.95, top_p=0.7, seed: Optional[int] = None, stop_strings: Optional[List[str]] = None, **kwargs) -> GenOutput:
        """Generate for single dict input."""
        if isinstance(dicts, str):
            # Handle direct string prompt
            prompt = dicts
        else:
            # Handle dict format like HFGenerator
            prompt = self._dict_to_prompt(dicts)
        
        # Update sampling parameters
        # Use provided stop_strings or default to STOP_TOKENS_TEXT
        stop_sequences = stop_strings if stop_strings is not None else list(STOP_TOKENS_TEXT)
        
        self._sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0,  # vLLM uses 0 for greedy
            top_p=top_p,
            seed=seed,  # vLLM supports seed directly
            stop=stop_sequences,
            skip_special_tokens=True
        )
        
        try:
            output = self._llm.generate([prompt], self._sp)[0]
            txt = output.outputs[0].text
            ended = txt.endswith(self._eos_text) or output.outputs[0].finish_reason == "stop"
            return GenOutput(trim_text(txt) if not ended else txt, ended)
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return GenOutput("", False)

    @torch.inference_mode()
    def batch_generate(self, dicts_list: List[dict], *, max_tokens=256, temperature=0.95, top_p=0.7, seed: Optional[int] = None, stop_strings: Optional[List[str]] = None, **kwargs) -> List[GenOutput]:
        """Batch generation for vLLM."""
        # Convert all dicts to prompts
        prompts = []
        for example_dict in dicts_list:
            if isinstance(example_dict, str):
                prompts.append(example_dict)
            else:
                prompts.append(self._dict_to_prompt(example_dict))
        
        # Update sampling parameters
        # Use provided stop_strings or default to STOP_TOKENS_TEXT
        stop_sequences = stop_strings if stop_strings is not None else list(STOP_TOKENS_TEXT)
        
        self._sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0,  # vLLM uses 0 for greedy
            top_p=top_p,
            seed=seed,  # vLLM supports seed directly
            stop=stop_sequences,
            skip_special_tokens=True
        )
        
        try:
            outputs = self._llm.generate(prompts, self._sp)
            results = []
            for output in outputs:
                txt = output.outputs[0].text
                ended = txt.endswith(self._eos_text) or output.outputs[0].finish_reason == "stop"
                results.append(GenOutput(trim_text(txt) if not ended else txt, ended))
            return results
        except Exception as e:
            logger.error(f"vLLM batch generation failed: {e}")
            return [GenOutput("", False) for _ in dicts_list]