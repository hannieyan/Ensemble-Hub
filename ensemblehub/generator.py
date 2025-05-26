from __future__ import annotations

import logging
import math
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

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False

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
    def __init__(self, path: str, *, device: str = "auto", dtype: torch.dtype = torch.bfloat16, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        # Memory optimization parameters
        memory_optimization = {
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Use eager attention to avoid potential issues
        }
        
        # Add optional quantization for large models to save memory
        model_size_gb = self._estimate_model_size(path)
        if model_size_gb > 20:  # For models > 20GB, use 8-bit quantization
            try:
                import bitsandbytes as bnb
                memory_optimization["load_in_8bit"] = True
                logger.info(f"Using 8-bit quantization for large model {path} ({model_size_gb:.1f}GB)")
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
    
    def _estimate_model_size(self, path: str) -> float:
        """Estimate model size in GB based on model name"""
        path_lower = path.lower()
        if "32b" in path_lower:
            return 64.0  # ~64GB for 32B model
        elif "14b" in path_lower:
            return 28.0  # ~28GB for 14B model  
        elif "7b" in path_lower:
            return 14.0  # ~14GB for 7B model
        elif "1.5b" in path_lower:
            return 3.0   # ~3GB for 1.5B model
        else:
            return 10.0  # Default estimate

    def __post_init__(self):
        """Initialize template and converter after model loading"""
        # Optional stop string list
        self.stop_strings = list(STOP_TOKENS_TEXT) + [
            self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        ]

        if "qwen3" in self.name.lower():
            data_args = DataArguments(template="qwen")
            self.indent = 2
        elif "qwen2.5" in self.name.lower():
            data_args = DataArguments(template="qwen")
            self.indent = 2
        elif "qwen" in self.name.lower():
            data_args = DataArguments(template="qwen")
            self.indent = 2
        elif "deepseek-r1" in self.name.lower():
            data_args = DataArguments(template="deepseekr1")
            self.indent = 1
        elif "deepseek" in self.name.lower():
            data_args = DataArguments(template="deepseek3")
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
        dicts_list: List[dict],
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
        
        # 借鉴 LlamaFactory 的成功配置，适配单卡大模型
        engine_args = {
            "model": path,
            "trust_remote_code": True,
            "dtype": "bfloat16",  # 使用 bfloat16 节省内存
            "max_model_len": 4096,  # 适中的context长度，避免OOM
            "tensor_parallel_size": 1,  # 单卡推理
            "disable_log_stats": True,
            # 移除有问题的参数，使用 vLLM 默认设置
        }
        
        # 如果指定了特定设备，设置 GPU 可见性
        if device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
            # 通过 tensor_parallel_size 和 GPU 选择来控制设备
            import os
            current_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not current_cuda_devices:
                # 只在没有设置的情况下设置
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        # 更新用户自定义参数
        engine_args.update(vllm_kwargs)
        
        self._llm = LLM(**engine_args)
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
    _gen_cache: Dict[Tuple[str, str], BaseGenerator] = {}
    _reward_cache: Dict[str, str] = {}

    @classmethod
    def get_generator(cls, path: str, engine: str = "hf", device: Optional[str] = None) -> BaseGenerator:
        """
        Load a generator model (e.g., HF or vLLM) to a specified device (e.g., 'cuda:0', 'cpu').
        """
        key = (engine, path)
        if key not in cls._gen_cache:
            logger.info("[Pool] loading %s (%s)", path, engine)

            resolved_device = device or "auto"
            logger.info(f"→ Assigned to device: {resolved_device}")

            if engine == "hf":
                cls._gen_cache[key] = HFGenerator(path, device=resolved_device)
            elif engine == "vllm":
                cls._gen_cache[key] = VLLMGenerator(path, device=resolved_device)
            else:
                raise ValueError(f"Unknown engine: {engine}")
        return cls._gen_cache[key]