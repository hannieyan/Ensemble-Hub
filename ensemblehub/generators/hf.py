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
)

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
        """Initialize after model loading"""
        # Optional stop string list
        self.stop_strings = list(STOP_TOKENS_TEXT) + [
            self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        ]
        
        logger.info(f"🏗️  HFGenerator {self.name} initialized")


    @torch.inference_mode()
    def generate(
        self,
        inputs: List,
        is_chat,
        *,
        max_tokens=256,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.0,
        stop_strings: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
    ) -> Union[GenOutput, List[GenOutput]]:
        
        # Use lock to prevent concurrent access issues
        with self._lock:
            # Auto-detect format: if input has "prompt" field, it's text completion
            if not is_chat:
                # Text completion mode - use raw prompt without template
                logger.info(f"  Raw prompt: {inputs[0]}")
                ids = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
            else:
                # Chat completion mode - use apply_chat_template
                logger.info(f"  Messages: {inputs[0]}")
                # Check if tokenizer supports enable_thinking
                ids = self.tokenizer.apply_chat_template(
                    inputs,
                    tokenize=False,
                    add_generation_prompt=True,
                    padding=True,
                    enable_thinking=self.enable_thinking,
                    return_tensors="pt",
                    continue_final_message=True if inputs[0][-1].get("role") == "assistant" else False
                ).to(self.device)
            
            # Set seed for reproducibility if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            # Build generation config using shared method
            # Generate with direct parameters
            generate_kwargs = {
                "max_newqqqq_tokens": max_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
                "tokenizer": self.tokenizer
            }
            
            if temperature > 0:
                generate_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                })
            else:
                generate_kwargs["do_sample"] = False
                
            outputs = self.model.generate(**ids, **generate_kwargs)

            # Process each output in the batch
            results = []
            batch_size = outputs.shape[0]

            for i in range(batch_size):
                # Get input length for this sample
                input_length = ids["input_ids"][i].shape[0]

                # Extract generated tokens for this sample
                output_ids = outputs[i]  # Shape: [seq_len]
                generated_ids = output_ids[input_length:]

                # Check if generation ended with EOS
                ended = self.tokenizer.eos_token_id in generated_ids.tolist()

                # Decode the generated text
                txt = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

                # Handle stop strings
                if stop_strings:
                    s_list = [stop_strings] if isinstance(stop_strings, str) else stop_strings
                    for s in s_list:
                        if s in txt:
                            txt = txt.split(s)[0]
                            ended = True
                            break

                # Add result
                results.append(GenOutput(trim_text(txt) if not ended else txt, ended))

            # 永远返回列表
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