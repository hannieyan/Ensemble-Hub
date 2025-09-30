"""
HuggingFace Transformers-based Generator
"""
from __future__ import annotations

import inspect
import logging
import math
import re
from typing import List, Optional, Union

import ray
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, BitsAndBytesConfig,
)
from accelerate import dispatch_model, infer_auto_device_map

from .base import GenOutput

logger = logging.getLogger("ensemble_inference")

def get_remote_hf_generator_class(num_gpus):
    # Dynamically register the ModelGenerator class as a Ray remote class, specifying the required number of GPUs
    return ray.remote(num_gpus=num_gpus)(HFGenerator)


class HFGenerator:
    """HuggingFace Transformers-based text generator"""

    def __init__(
        self,
        model_path: str,
        max_memory=None,
        dtype: torch.dtype = torch.bfloat16,
        quantization: str = "none",
        enable_thinking: bool = True,
        padding_side: str = "left",
    ):

        # Memory optimization parameters
        memory_optimization = {
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Use eager attention to avoid potential issues
        }

        quantization_options = {
            "8bit": BitsAndBytesConfig(load_in_8bit=True),
            "4bit": BitsAndBytesConfig(load_in_4bit=True),
            "none": None,
        }

        # Retrieve the appropriate quantization_config
        quantization_config = quantization_options.get(quantization)

        # Raise an error if an invalid quantization option is provided
        if quantization_config is None and quantization != "none":
            raise ValueError(
                f"Invalid quantization value '{quantization}'. Allowed values are: 'none', '8bit', '4bit'."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            **memory_optimization
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=model._get_no_split_modules("auto"),
        )

        # https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/modeling_utils.py#L3773
        device_map_kwargs = {"device_map": device_map}
        if "skip_keys" in inspect.signature(dispatch_model).parameters:
            device_map_kwargs["skip_keys"] = model._skip_keys_device_placement

        # Load model to GPU
        self.model = dispatch_model(model, **device_map_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side=padding_side,
            trust_remote_code=True
        )
        
        # Ensure pad_token is properly set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = next(self.model.parameters()).device

        self.name = model_path
        self.enable_thinking = enable_thinking

        # Optional stop string list
        self.stop_strings = [
            "<|endoftext|>",  # Common end token for many models
            "<|im_end|>",  # Qwen models use this as end token
            "<｜end▁of▁sentence｜>",  # Deepseek models use this as end token
        ]

        logger.info(f"🏗️  HFGenerator {self.name} initialized")


    def generate(
        self,
        inputs: List,
        is_chat,
        continue_final_message,
        max_tokens=65536,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.1,
        stop_strings: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = 1234,
    ) -> Union[GenOutput, List[GenOutput]]:

        # print(inputs)

        # stop_strings
        stop_strings = stop_strings + self.stop_strings if stop_strings else self.stop_strings

        # Auto-detect format: if input has "prompt" field, it's text completion
        if is_chat or (not is_chat and self.enable_thinking):
            chat_inputs = []
            for text_input in inputs:
                if is_chat:
                    conversation = text_input
                else:
                    conversation = [
                        {"role": "system", "content": "Think step-by-step inside <think></think>, then provide the final answer after </think> and include any required delimiters."},
                        {"role": "user", "content": text_input}
                    ]
                chat_input = self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True if not continue_final_message else False,
                    enable_thinking=self.enable_thinking,
                    continue_final_message=continue_final_message,
                    tokenize = False,
                )

                chat_inputs.append(chat_input)
        else:
            chat_inputs = inputs

        # Text completion mode - use raw prompt without template
        # logger.info(f"  Raw prompt: {inputs[0]}")
        ids = self.tokenizer(
            chat_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Set seed for reproducibility if provided
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Build generation config using shared method
        # Generate with direct parameters
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "tokenizer": self.tokenizer,
            "stop_strings": stop_strings,
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

        outputs = self.model.generate(
            input_ids=ids["input_ids"],
            attention_mask=ids["attention_mask"],
            **generate_kwargs
        )

        # Extract generated tokens (all inputs have same length due to padding)
        input_length = ids["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        # Batch decode all sequences with special tokens removed
        texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        if not is_chat and self.enable_thinking:
            texts = [self._strip_reasoning_tokens(txt) for txt in texts]

        # Determine ended status based on token count
        # Tokenize texts to count tokens (add_special_tokens=False to get accurate count)
        token_counts = [len(self.tokenizer(txt, add_special_tokens=False)["input_ids"]) for txt in texts]
        ended_status = [count < max_tokens - 1 for count in token_counts]

        return [GenOutput(txt, ended) for txt, ended in zip(texts, ended_status)]

    @staticmethod
    def _strip_reasoning_tokens(text: str) -> str:
        pattern = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
        cleaned = re.sub(pattern, "", text)
        return cleaned.lstrip()


    def apply_chat_template(self,
        conversation: List[List[dict]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        continue_final_message: bool = False,
        tokenize: bool = True,
    ) -> List[str]:
        """Apply chat template to a conversation."""
        chat_texts = []
        for conversation in conversation:
            chat_text = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                continue_final_message=continue_final_message,
                tokenize=tokenize,
            )
            chat_texts.append(chat_text)

        return chat_texts



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

    def get_model_name(self):
        return self.name

    def get_tokenizer(self):
        return self.tokenizer
    
    def count_tokens(self, texts: List[str]) -> List[int]:
        """Count tokens for a list of texts"""
        token_counts = []
        for text in texts:
            if text:
                tokens = self.tokenizer.encode(text)
                token_counts.append(len(tokens))
            else:
                token_counts.append(0)
        return token_counts

    def get_model_size(self) -> float:
        """Get model size in billions of parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params / 1e9

    def default_continue_final_message(self) -> bool:
        """HF models can safely continue the last assistant message when requested."""
        return True

    # ================================================================================
    # ========================= CUSTOM METHODS (NON-STANDARD) ========================
    # ================================================================================
    # The following methods are custom implementations specific to model_selection and
    # output_aggregation modules, not part of the standard HFGenerator methods.
    #
    # IMPORTANT: When coding, minimize the use of these custom methods.
    # ================================================================================
    
    def get_token_confidence(self, prompt: str) -> float:
        """Get P_yes - P_no confidence score by extracting token probabilities."""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with model to get logits
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,  # Only need first token
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False
            )
            
            # Get logits for the generated token (first new token)
            logits = outputs.scores[0]  # Shape: [1, vocab_size]
        
        # Get token IDs for "yes" and "no" 
        yes_tokens = self.tokenizer.encode("yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("no", add_special_tokens=False)
        yes_tokens_cap = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens_cap = self.tokenizer.encode("No", add_special_tokens=False)
        
        # Get all possible IDs
        yes_ids = list(set(yes_tokens + yes_tokens_cap))
        no_ids = list(set(no_tokens + no_tokens_cap))
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sum probabilities for all yes/no variants
        p_yes = sum(probs[0, token_id].item() for token_id in yes_ids if token_id < probs.shape[1])
        p_no = sum(probs[0, token_id].item() for token_id in no_ids if token_id < probs.shape[1])
        
        # Normalize and compute confidence
        total = p_yes + p_no
        if total > 0:
            p_yes_norm = p_yes / total
            p_no_norm = p_no / total
            confidence = p_yes_norm - p_no_norm
        else:
            confidence = 0.0
        
        return confidence
