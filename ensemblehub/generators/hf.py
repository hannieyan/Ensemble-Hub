"""
HuggingFace Transformers-based Generator
"""
from __future__ import annotations

import logging
import math
import threading
from typing import List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.converter import AlpacaDatasetConverter
from llamafactory.data.parser import DatasetAttr

from .base import BaseGenerator, GenOutput, STOP_TOKENS_TEXT, trim_text

logger = logging.getLogger("ensemble_inference")


class HFGenerator(BaseGenerator):
    """HuggingFace Transformers-based text generator"""
    
    def __init__(self, path: str, *, device: str = "auto", dtype: torch.dtype = torch.bfloat16, quantization: str = "none", enable_thinking: bool = True, **model_kwargs):
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
            # For specific device, use device_map to handle large models properly
            # This avoids meta tensor issues
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map={"": device},  # Map all layers to the specified device
                **memory_optimization
            ).eval()
            self.device = torch.device(device)
            
        self.name = path
        self.enable_thinking = enable_thinking
        self._lock = threading.Lock()  # Thread safety for concurrent access
        self.__post_init__()
    

    def __post_init__(self):
        """Initialize template and converter after model loading"""
        # Optional stop string list
        self.stop_strings = list(STOP_TOKENS_TEXT) + [
            self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        ]

        # Store data_args as instance variable for later use
        if "Qwen3" in self.name:
            self.data_args = DataArguments(template="qwen3", enable_thinking=self.enable_thinking)
            self.indent = 2
        elif "Qwen2.5" in self.name:
            self.data_args = DataArguments(template="qwen", enable_thinking=self.enable_thinking)
            self.indent = 2
        elif "DeepSeek-R1" in self.name:
            self.data_args = DataArguments(template="deepseekr1", enable_thinking=self.enable_thinking)
            self.indent = 1
        else:
            logger.warning(f"Unknown model template for {self.name}, using default")
            self.data_args = DataArguments(template="default", enable_thinking=self.enable_thinking)
            self.indent = 1

        dataset_attr = DatasetAttr(
            prompt="instruction",
            query="input",
            response="output",
            load_from="file",
            formatting="alpaca",
            dataset_name="",
        )

        self.converter = AlpacaDatasetConverter(dataset_attr=dataset_attr, data_args=self.data_args)
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)

    def _build_generation_config(
        self, 
        temperature: float,
        max_tokens: int,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[Union[str, List[str]]] = None
    ) -> GenerationConfig:
        """Build generation config based on temperature and other parameters"""
        base_config = {
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stop_strings": ["</s>", "<|im_end|>"] if stop_strings is None else stop_strings,
        }
        
        if temperature > 0:
            # Sampling mode
            base_config.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        else:
            # Deterministic mode (temperature=0)
            base_config["do_sample"] = False
            
        return GenerationConfig(**base_config)

    @torch.inference_mode()
    def generate(
        self,
        dicts,
        *,
        max_tokens=2048,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.0,
        stop_strings: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
    ) -> Union[GenOutput, List[GenOutput]]:
        # Use lock to prevent concurrent access issues
        with self._lock:
            # Handle string input by converting to dict format
            if isinstance(dicts, str):
                dicts = {"instruction": "", "input": dicts, "output": ""}
            
            converted = self.converter(dicts)
            prompt_msgs = converted["_prompt"]
            response_msgs = converted["_response"]
            messages = prompt_msgs + response_msgs
            system = converted.get("_system", None)
            prompt_ids, response_ids = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages, system=system)

            ids = prompt_ids + response_ids[:-self.indent]

            text = self.tokenizer.decode(ids, skip_special_tokens=False)

            ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.data_args.cutoff_len).to(self.device)
            
            # Set seed for reproducibility if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            # Build generation config using shared method
            cfg = self._build_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_strings=stop_strings
            )
            out = self.model.generate(**ids, generation_config=cfg, tokenizer=self.tokenizer)[0]

            # Check if the output contains the EOS token and stop_strings
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

            return GenOutput(trim_text(txt) if not ended else txt, ended)

    @torch.inference_mode()
    def batch_generate(
        self,
        dicts_list: List[Union[dict, str]],
        *,
        max_tokens=2048,
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
            system = converted.get("_system", None)
            prompt_ids, response_ids = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages, system=system)
            
            ids = prompt_ids + response_ids[:-self.indent]
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            all_prompt_texts.append(text)

        # Tokenize all prompts with padding
        batch_inputs = self.tokenizer(
            all_prompt_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.data_args.cutoff_len
        ).to(self.device)

        # Build generation config using shared method
        cfg = self._build_generation_config(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings
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

            results.append(GenOutput(trim_text(txt) if not ended else txt, ended))

        return results


    @torch.inference_mode()
    def calculate_ppl(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        if not completion_text.strip():
            logger.debug(f"PPL calculation for {self.name}: empty completion text.")
            return None

        try:
            prompt_token_ids = self.tokenizer(prompt_context_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
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
            logits_for_completion_steps = all_logits[:, prompt_token_ids.shape[1] - 1: -1, :]

            if logits_for_completion_steps.shape[1] == 0:
                logger.debug(f"Confidence calculation for {self.name}: no logit steps for completion.")
                return None

            probs_for_completion_steps = torch.softmax(logits_for_completion_steps, dim=-1)
            max_prob_at_each_step, _ = torch.max(probs_for_completion_steps, dim=-1)

            confidence = max_prob_at_each_step.mean().item()
            return confidence
        except Exception as e:
            logger.error(f"Error in calculate_confidence for {self.name} with completion '{completion_text[:50]}...': {e}", exc_info=True)
            return None