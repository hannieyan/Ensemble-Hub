from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch


from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.converter import AlpacaDatasetConverter
from llamafactory.data.parser import DatasetAttr

# Optional vLLM backend -----------------------------------------------------
try:
    from vllm import LLM, SamplingParams  # type: ignore
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:
        # Fallback for older vLLM versions
        try:
            from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        except ImportError:
            destroy_model_parallel = None

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False
    destroy_model_parallel = None

EOS_TEXT = ""  # Most Qwen / Llama models use empty string as EOS
STOP_TOKENS_TEXT = {"\n"}  # Stop decoding after these tokens

logger = logging.getLogger("ensemble_inference")

# ---------------------------------------------------------------------------
# Utility: trim text at the last occurrence of stop tokens
# ---------------------------------------------------------------------------

def _trim_text(txt: str) -> str:
    """Truncate the text after the last known stop token for cleaner outputs."""
    best_pos = -1
    best_tok = None
    for tok in STOP_TOKENS_TEXT:
        pos = txt.rfind(tok)
        if pos > best_pos:
            best_pos = pos
            best_tok = tok
    if best_pos != -1:
        return txt[: best_pos + len(best_tok)]
    return txt





# ---------------------------------------------------------------------------
# Output container for model generation
# ---------------------------------------------------------------------------

@dataclass
class GenOutput:
    text: str
    ended_with_eos: bool  # Whether EOS token was generated


# ---------------------------------------------------------------------------
# Abstract base class for any generator (HF or vLLM)
# ---------------------------------------------------------------------------

class BaseGenerator:
    name: str

    def generate(self, prompt, **kw) -> GenOutput:
        """Abstract method for generating model outputs."""
        raise NotImplementedError

    def batch_generate(self, prompts: List, **kw) -> List[GenOutput]:
        """Abstract method for batch generation."""
        raise NotImplementedError

    def calculate_ppl(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        logger.warning(f"PPL calculation not implemented for {self.name} or called on BaseGenerator base class.")
        return None

    def calculate_confidence(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        logger.warning(f"Confidence calculation not implemented for {self.name} or called on BaseGenerator base class.")
        return None

# ---------------------------------------------------------------------------
# HuggingFace Transformers-based Generator
# ---------------------------------------------------------------------------

class HFGenerator(BaseGenerator):
    def __init__(self, path: str, *, device: str = "auto", dtype: torch.dtype = torch.bfloat16, quantization: str = "none", **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        # Memory optimization parameters
        memory_optimization = {
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Use eager attention to avoid potential issues
        }
        
        # Add quantization based on user parameter
        if quantization == "8bit":
            try:
                import bitsandbytes as bnb
                memory_optimization["load_in_8bit"] = True
                logger.info(f"Using 8-bit quantization for model {path}")
            except ImportError:
                logger.warning(f"bitsandbytes not available, loading {path} without quantization. Consider installing: pip install bitsandbytes")
        elif quantization == "4bit":
            try:
                import bitsandbytes as bnb
                memory_optimization["load_in_4bit"] = True
                logger.info(f"Using 4-bit quantization for model {path}")
            except ImportError:
                logger.warning(f"bitsandbytes not available, loading {path} without quantization. Consider installing: pip install bitsandbytes")
        
        memory_optimization.update(model_kwargs)
        
        # Handle meta tensor issue with proper loading strategy
        if device == "auto":
            # Use device_map for auto device assignment
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                **memory_optimization
            ).eval()
            self.device = next(self.model.parameters()).device
        else:
            # For specific device, avoid device_map and load directly to target device
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                **memory_optimization
            ).eval().to(device)
            self.device = torch.device(device)
            
        self.name = path
        self.__post_init__()
    

    def __post_init__(self):
        """Initialize template and converter after model loading"""
        # Optional stop string list
        self.stop_strings = list(STOP_TOKENS_TEXT) + [
            self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        ]

        if "Qwen3" in self.name:
            data_args = DataArguments(template="qwen3")
            self.indent = 2
        elif "Qwen2.5" in self.name:
            data_args = DataArguments(template="qwen")
            self.indent = 2
        elif "DeepSeek-R1" in self.name:
            data_args = DataArguments(template="deepseekr1")
            self.indent = 1
        else:
            logger.warning(f"Unknown model template for {self.name}, using default")
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

        # self.suppress_tokens = self.tokenizer.encode("</think>", add_special_tokens=False) + \
        #                        self.tokenizer.encode("<think>", add_special_tokens=False)
        #
        # self.begin_suppress_tokens = self.suppress_tokens  # 如果你要一开始也禁止

    @torch.inference_mode()
    def generate(
        self,
        dicts,
        *,
        enable_thinking: bool = False,
        max_tokens=256,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.0,
        stop_strings: Optional[Union[str, List[str]]] = None,
    ) -> Union[GenOutput, List[GenOutput]]:

        # Handle string input by converting to dict format
        if isinstance(dicts, str):
            dicts = {"instruction": "", "input": dicts, "output": ""}
        
        converted = self.converter(dicts)
        prompt_msgs = converted["_prompt"]
        response_msgs = converted["_response"]
        messages = prompt_msgs + response_msgs
        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, enable_thinking=enable_thinking)

        ids = prompt_ids + response_ids[:-self.indent]

        text = self.tokenizer.decode(ids, skip_special_tokens=False)

        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        cfg = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,  # 加上这个
            repetition_penalty=repetition_penalty,  # 加上这个
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_strings=["Question:", "</s>", "<|im_end|>"] if stop_strings is None else stop_strings,
            # suppress_tokens=self.suppress_tokens,
            # begin_suppress_tokens=self.begin_suppress_tokens,
        )
        out = self.model.generate(**ids, generation_config=cfg, tokenizer=self.tokenizer)[0]

        # ─── Check if the output contains the EOS token and stop_strings ────────────
        generated_ids = out[len(ids["input_ids"][0]):]
        ended = self.tokenizer.eos_token_id in generated_ids.tolist()

        txt = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        if stop_strings:
            if isinstance(stop_strings, str):
                stop_strings = [stop_strings]
            for s in stop_strings:
                if s in txt:
                    txt = txt.split(s)[0]
                    ended = True
                    break

        return GenOutput(_trim_text(txt) if not ended else txt, ended)

    @torch.inference_mode()
    def batch_generate(
        self,
        dicts_list: List[Union[dict, str]],
        *,
        enable_thinking: bool = False,
        max_tokens=256,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.0,
        stop_strings: Optional[Union[str, List[str]]] = None,
    ) -> List[GenOutput]:
        """
        Batch generation for multiple inputs.
        """
        # Process all inputs to get prompts
        all_prompt_texts = []
        for single_dict in dicts_list:
            # Handle string input by converting to dict format
            if isinstance(single_dict, str):
                single_dict = {"instruction": "", "input": single_dict, "output": ""}
                
            converted = self.converter(single_dict)
            prompt_msgs = converted["_prompt"]
            response_msgs = converted["_response"]
            messages = prompt_msgs + response_msgs
            prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, enable_thinking=enable_thinking)
            
            ids = prompt_ids + response_ids[:-self.indent]
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            all_prompt_texts.append(text)

        # Tokenize all prompts with padding
        batch_inputs = self.tokenizer(
            all_prompt_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(self.device)

        cfg = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_strings=["Question:", "</s>", "<|im_end|>"] if stop_strings is None else stop_strings,
        )
        
        # Generate for all inputs at once
        outputs = self.model.generate(**batch_inputs, generation_config=cfg, tokenizer=self.tokenizer)

        # Process outputs
        results = []
        for i, output in enumerate(outputs):
            input_length = len(batch_inputs["input_ids"][i])
            generated_ids = output[input_length:]
            ended = self.tokenizer.eos_token_id in generated_ids.tolist()

            txt = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            if stop_strings:
                if isinstance(stop_strings, str):
                    stop_strings = [stop_strings]
                for s in stop_strings:
                    if s in txt:
                        txt = txt.split(s)[0]
                        ended = True
                        break

            results.append(GenOutput(_trim_text(txt) if not ended else txt, ended))

        return results


    @torch.inference_mode()
    def calculate_ppl(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        if not completion_text.strip():
            logger.debug(f"PPL calculation for {self.name}: empty completion text.")
            return None

        try:
            prompt_token_ids = self.tokenizer(prompt_context_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
            # For completion, typically we don't add special tokens if it's a continuation
            completion_token_ids = self.tokenizer(completion_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

            if completion_token_ids.shape[1] == 0:
                logger.debug(f"PPL calculation for {self.name}: completion tokenized to empty.")
                return None

            full_sequence_ids = torch.cat((prompt_token_ids, completion_token_ids), dim=1)
            attention_mask = torch.ones_like(full_sequence_ids)

            outputs = self.model(
                input_ids=full_sequence_ids,
                attention_mask=attention_mask,
            )
            all_logits = outputs.logits

            shift_logits = all_logits[..., :-1, :].contiguous()
            shift_labels = full_sequence_ids[..., 1:].contiguous()

            start_index_for_loss = prompt_token_ids.shape[1]

            loss_logits = shift_logits[:, start_index_for_loss - 1:, :]
            loss_labels = shift_labels[:, start_index_for_loss - 1:]

            if loss_logits.shape[1] == 0 or loss_labels.shape[1] == 0 or loss_logits.shape[1] != loss_labels.shape[1]:
                logger.debug(f"PPL calculation for {self.name}: no valid tokens for loss or shape mismatch. Logits shape {loss_logits.shape}, Labels shape {loss_labels.shape}")
                return None

            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            neg_log_likelihood = loss_fct(loss_logits.view(-1, loss_logits.size(-1)), loss_labels.view(-1))

            if torch.isnan(neg_log_likelihood) or torch.isinf(neg_log_likelihood):
                logger.warning(f"PPL calculation for {self.name} resulted in NaN/Inf NLL.")
                return float('inf')

            ppl = math.exp(neg_log_likelihood.item())
            return ppl
        except Exception as e:
            logger.error(f"Error in calculate_ppl for {self.name} with completion '{completion_text[:50]}...': {e}", exc_info=True)
            return None

    @torch.inference_mode()
    def calculate_confidence(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        if not completion_text.strip():
            logger.debug(f"Confidence calculation for {self.name}: empty completion text.")
            return None
        try:
            prompt_token_ids = self.tokenizer(prompt_context_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
            completion_token_ids = self.tokenizer(completion_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

            if completion_token_ids.shape[1] == 0:
                logger.debug(f"Confidence calculation for {self.name}: completion tokenized to empty.")
                return None

            full_sequence_ids = torch.cat((prompt_token_ids, completion_token_ids), dim=1)
            attention_mask = torch.ones_like(full_sequence_ids)

            outputs = self.model(input_ids=full_sequence_ids, attention_mask=attention_mask)
            all_logits = outputs.logits

            # Logits corresponding to the prediction of completion tokens
            # These are logits at positions [L_prompt-1, ..., L_prompt + L_completion - 2]
            logits_for_completion_steps = all_logits[:, prompt_token_ids.shape[1] - 1: -1, :]

            if logits_for_completion_steps.shape[1] == 0:
                logger.debug(f"Confidence calculation for {self.name}: no logit steps for completion.")
                return None

            probs_for_completion_steps = torch.softmax(logits_for_completion_steps, dim=-1)
            max_prob_at_each_step, _ = torch.max(probs_for_completion_steps, dim=-1)  # Shape: (batch_size, num_completion_tokens)

            confidence = max_prob_at_each_step.mean().item()
            return confidence
        except Exception as e:
            logger.error(f"Error in calculate_confidence for {self.name} with completion '{completion_text[:50]}...': {e}", exc_info=True)
            return None

# ---------------------------------------------------------------------------
# vLLM-based Generator
# ---------------------------------------------------------------------------

class VLLMGenerator(BaseGenerator):
    def __init__(self, path: str, device: str = "cuda:0", **vllm_kwargs):
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Please install with: pip install vllm")
        
        # Store original device
        self._original_device = device
        
        # 借鉴 LlamaFactory 的成功配置，适配单卡大模型
        engine_args = {
            "model": path,
            "trust_remote_code": True,
            "dtype": "bfloat16",  # 使用 bfloat16 节省内存
            "max_model_len": 4096,  # 适中的context长度，避免OOM
            "disable_log_stats": True,
            "enforce_eager": True,  # Disable CUDA graphs to avoid caching allocator issues
        }
        
        # Handle GPU placement for vLLM
        if device == "auto" or device.startswith("cuda"):
            if device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
                # For vLLM, we need to set tensor_parallel_size and use environment variable
                # The key is to set this BEFORE vLLM initialization
                import os
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
        
        # 更新用户自定义参数
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
        from transformers import AutoTokenizer
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
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Restore original CUDA_VISIBLE_DEVICES
        if hasattr(self, '_original_cuda_devices'):
            import os
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
    def generate(self, dicts, *, max_tokens=256, temperature=0.95, top_p=0.7, **kwargs) -> GenOutput:
        """Generate for single dict input."""
        if isinstance(dicts, str):
            # Handle direct string prompt
            prompt = dicts
        else:
            # Handle dict format like HFGenerator
            prompt = self._dict_to_prompt(dicts)
        
        self._sp.max_tokens = max_tokens
        self._sp.temperature = temperature
        self._sp.top_p = top_p
        
        try:
            output = self._llm.generate([prompt], self._sp)[0]
            txt = output.outputs[0].text
            ended = txt.endswith(self._eos_text) or output.outputs[0].finish_reason == "stop"
            return GenOutput(_trim_text(txt) if not ended else txt, ended)
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return GenOutput("", False)

    @torch.inference_mode()
    def batch_generate(self, dicts_list: List[dict], *, max_tokens=256, temperature=0.95, top_p=0.7, **kwargs) -> List[GenOutput]:
        """Batch generation for vLLM."""
        # Convert all dicts to prompts
        prompts = []
        for example_dict in dicts_list:
            if isinstance(example_dict, str):
                prompts.append(example_dict)
            else:
                prompts.append(self._dict_to_prompt(example_dict))
        
        self._sp.max_tokens = max_tokens
        self._sp.temperature = temperature
        self._sp.top_p = top_p
        
        try:
            outputs = self._llm.generate(prompts, self._sp)
            results = []
            for output in outputs:
                txt = output.outputs[0].text
                ended = txt.endswith(self._eos_text) or output.outputs[0].finish_reason == "stop"
                results.append(GenOutput(_trim_text(txt) if not ended else txt, ended))
            return results
        except Exception as e:
            logger.error(f"vLLM batch generation failed: {e}")
            return [GenOutput("", False) for _ in dicts_list]


# ---------------------------------------------------------------------------
# GeneratorPool: caches all loaded generators and reward models
# ---------------------------------------------------------------------------

class GeneratorPool:
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