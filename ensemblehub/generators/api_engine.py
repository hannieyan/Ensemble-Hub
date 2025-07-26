"""
API-based Generator for external model APIs (OpenAI-compatible)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

from openai import AsyncOpenAI

from .base import GenOutput

logger = logging.getLogger("ensemble_inference")


class APIGenerator:
    """API-based text generator for OpenAI-compatible endpoints"""

    def __init__(
        self,
        model_path: str,  # This will be just the model name
        max_memory=None,  # Unused for API
        dtype=None,  # Unused for API
        quantization: str = "none",  # Unused for API
        enable_thinking: bool = True,
        padding_side: str = "left",  # Unused for API
        max_concurrent_requests: int = 5,  # Limit concurrent requests
    ):
        self.model_name = model_path
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize AsyncOpenAI client with optimized settings for concurrent requests
        import httpx
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=50,  # Reduced for stability
                    max_keepalive_connections=10
                ),
                timeout=httpx.Timeout(120.0)  # Increased timeout
            )
        )
        
        self.name = self.model_name
        self.enable_thinking = enable_thinking
        
        # Optional stop string list
        self.stop_strings = [
            # "<|endoftext|>",  # Common end token for many models
            # "<|im_end|>",  # Qwen models use this as end token
            # "<｜end▁of▁sentence｜>",  # Deepseek models use this as end token
        ]
        
        logger.info(f"<🏗️ APIGenerator {self.name} initialized with max_concurrent_requests={max_concurrent_requests}")

    async def _process_single_input(
        self,
        input_data,
        is_chat,
        continue_final_message,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        stop_strings,
        seed,
        input_index=None,
        semaphore=None,
    ):
        """Process a single input with semaphore control and return GenOutput."""
        async with semaphore:
            # Use the shared async client
            client = self.client
            
            if is_chat:
                # Chat completion mode
                messages = input_data

                # Chat completion API parameters
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0,
                    "top_p": top_p if temperature > 0 else 1.0,
                    "frequency_penalty": max(-2.0, min(2.0, (repetition_penalty - 1.0) * 2)),
                    "stop": stop_strings if stop_strings else None,
                    "seed": seed,
                }
                
                completion = await client.chat.completions.create(**api_params)
                message = completion.choices[0].message
                response_text = message.content or ""
                
                # Check if there's reasoning (thinking process) and combine with response
                if hasattr(message, 'reasoning') and message.reasoning:
                    if response_text.strip():  # 如果有最终答案
                        response_text = f"<think>\n{message.reasoning}\n</think>\n{response_text}"
                    else:  # 如果没有最终答案，只有思考过程，不加</think>让模型继续
                        response_text = f"<think>\n{message.reasoning}"
                
                logger.info(f"[Input {input_index}] Chat completion response length: {len(response_text)}, preview: '{response_text[:100]}'")
                
            else:
                # Text completion mode - use raw prompt
                prompt = input_data

                # Text completion API parameters
                api_params = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0,
                    "top_p": top_p if temperature > 0 else 1.0,
                    "frequency_penalty": max(-2.0, min(2.0, (repetition_penalty - 1.0) * 2)),
                    "stop": stop_strings if stop_strings else None,
                    "seed": seed,
                }
                
                completion = await client.completions.create(**api_params)
                response_text = completion.choices[0].text or ""

            # Check if response was truncated
            finish_reason = completion.choices[0].finish_reason
            ended = finish_reason == "stop"
            
            # Extract token count from usage
            token_count = completion.usage.completion_tokens if completion.usage else 0
            
            # Create output with token_count attribute
            output = GenOutput(response_text, ended)
            output.token_count = token_count
            
            return output

    def generate(
        self,
        inputs: List,
        is_chat,
        continue_final_message,
        max_tokens,
        temperature=0.95,
        top_p=0.7,
        top_k=50,  # Note: OpenAI doesn't support top_k
        repetition_penalty=1.1,  # Note: OpenAI uses frequency_penalty/presence_penalty
        stop_strings: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = 1234,
    ) -> Union[GenOutput, List[GenOutput]]:
        
        # Combine stop strings
        stop_strings = stop_strings + self.stop_strings if stop_strings else self.stop_strings
        
        logger.info(f"Processing {len(inputs)} inputs concurrently with asyncio")
        
        # Run async code in a separate thread to avoid event loop conflicts
        def run_async_in_thread():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create semaphore for limiting concurrent requests
                semaphore = asyncio.Semaphore(self.max_concurrent_requests)
                
                # Create tasks for all inputs with semaphore control
                tasks = [
                    self._process_single_input(
                        input_data, is_chat, continue_final_message, max_tokens,
                        temperature, top_p, repetition_penalty, stop_strings, seed, i, semaphore
                    )
                    for i, input_data in enumerate(inputs)
                ]
                
                # Run all tasks concurrently
                results = loop.run_until_complete(asyncio.gather(*tasks))
                logger.info(f"Generated {len(results)} results: {[f'({len(r.text)} chars)' for r in results]}")
                return results
            finally:
                loop.close()
        
        # Execute in a separate thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_in_thread)
            results = future.result()
        
        return results

    def apply_chat_template(self,
        conversation: List[List[dict]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        continue_final_message: bool = False,
        tokenize: bool = True,
    ) -> List[str]:
        """Apply chat template using a fixed HuggingFace tokenizer for consistency."""
        from transformers import AutoTokenizer
        
        # Use a fixed tokenizer for template processing
        self._template_tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            trust_remote_code=True
        )

        # Apply chat template to each conversation
        text_results = []
        for conv in conversation:
            text = self._template_tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                continue_final_message=continue_final_message,
                tokenize=tokenize,
            )
            text_results.append(text)
        return text_results

    def calculate_ppl(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        """Calculate perplexity - not available for API models."""
        logger.debug(f"PPL calculation not available for API model {self.name}")
        return None

    def calculate_confidence(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        """Calculate confidence - not available for most API models."""
        logger.debug(f"Confidence calculation not available for API model {self.name}")
        return None

    def get_model_name(self):
        return self.name

    def get_tokenizer(self):
        """API models don't expose tokenizers."""
        return None

    def get_model_size(self) -> float:
        """Extract model size from model name for API models."""
        import re
        
        model_name = self.model_name.lower()
        
        # Common patterns for extracting model size
        patterns = [
            r'(\d+(?:\.\d+)?)b',           # e.g., "7b", "13b", "0.5b"
            r'(\d+(?:\.\d+)?)B',           # e.g., "7B", "13B", "0.5B"
            r'-(\d+(?:\.\d+)?)b-',         # e.g., "-7b-", "-13b-"
            r'-(\d+(?:\.\d+)?)B-',         # e.g., "-7B-", "-13B-"
            r'(\d+(?:\.\d+)?)billion',     # e.g., "7billion"
            r'(\d+(?:\.\d+)?)B-',          # e.g., "7B-"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_name)
            if match:
                size = float(match.group(1))
                logger.info(f"Extracted model size from '{self.model_name}': {size}B parameters")
                return size
        
        # Fallback: try to guess based on common model names
        size_mappings = {
            'gpt-4': 1760.0,     # GPT-4 estimated size
            'gpt-3.5': 175.0,    # GPT-3.5 estimated size
            'claude': 175.0,     # Claude estimated size
            'o1': 175.0,         # o1 estimated size
            'o3': 175.0,         # o3 estimated size
            'o4': 175.0,         # o4 estimated size
            'codex': 175.0,      # Codex estimated size
            'qwen': 7.0,         # Default Qwen size if not specified
            'deepseek': 7.0,     # Default DeepSeek size if not specified
            'llama': 7.0,        # Default Llama size if not specified
        }
        
        for key, size in size_mappings.items():
            if key in model_name:
                logger.info(f"Using fallback size for '{self.model_name}' based on '{key}': {size}B parameters")
                return size
        
        # Final fallback: return a default size for unknown API models
        logger.warning(f"Could not determine size for API model '{self.model_name}', using default 7.0B")
        return 7.0

    # ================================================================================
    # ========================= CUSTOM METHODS (NON-STANDARD) ========================
    # ================================================================================
    
    def get_token_confidence(self, prompt: str) -> float:
        """Get P_yes - P_no confidence score - not available for API models."""
        logger.debug(f"Token confidence not available for API model {self.name}")
        return 0.0