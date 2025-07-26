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
    ):
        self.model_name = model_path
        
        # Initialize AsyncOpenAI client with optimized settings for concurrent requests
        import httpx
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100,  # Allow more concurrent connections
                    max_keepalive_connections=20
                ),
                timeout=httpx.Timeout(60.0)  # 60 second timeout
            )
        )
        
        self.name = self.model_name
        self.enable_thinking = enable_thinking
        
        # Optional stop string list
        self.stop_strings = [
            "<|endoftext|>",  # Common end token for many models
            "<|im_end|>",  # Qwen models use this as end token
            "<｜end▁of▁sentence｜>",  # Deepseek models use this as end token
        ]

        # Create a semaphore to limit concurrent requests (avoid overwhelming the API)
        self.semaphore = None  # Will be created in the async context
        
        logger.info(f"<🏗️ APIGenerator {self.name} initialized with base_url: https://aihubmix.com/v1")

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
    ):
        """Process a single input asynchronously and return GenOutput."""
        import time
        start_time = time.time()
        
        # Use the shared async client
        client = self.client
        
        logger.info(f"[Input {input_index}] Starting API request")
        logger.debug(f"Processing input (is_chat={is_chat}): {input_data}")
        
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
            
            # Add reasoning parameter if thinking is enabled for chat
            if self.enable_thinking:
                api_params["reasoning"] = {
                    "effort": "medium",
                    "summary": "auto"
                }
            
            completion = await client.chat.completions.create(**api_params)
            response_text = completion.choices[0].message.content
            
        else:
            # Text completion mode - use raw prompt
            prompt = input_data if isinstance(input_data, str) else str(input_data)
            
            # Handle continue_final_message for text completion
            if continue_final_message:
                # For text completion, we assume the prompt already contains partial text
                pass  # No special handling needed
            
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
            
            # Note: reasoning parameter is typically not available for text completions
            # but we can try to add it if thinking is enabled
            if self.enable_thinking:
                api_params["reasoning"] = {
                    "effort": "medium",
                    "summary": "auto"
                }
            
            completion = await client.completions.create(**api_params)
            response_text = completion.choices[0].text

        # Check if response was truncated
        finish_reason = completion.choices[0].finish_reason
        ended = finish_reason == "stop"
        
        # Extract token count from usage
        token_count = completion.usage.completion_tokens
        
        # Log completion time
        end_time = time.time()
        logger.info(f"[Input {input_index}] API request completed in {end_time - start_time:.2f}s")
        
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
                # Create tasks for all inputs with index for logging
                tasks = [
                    self._process_single_input(
                        input_data, is_chat, continue_final_message, max_tokens,
                        temperature, top_p, repetition_penalty, stop_strings, seed, i
                    )
                    for i, input_data in enumerate(inputs)
                ]
                
                # Run all tasks concurrently
                results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                return results
            finally:
                loop.close()
        
        # Execute in a separate thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_in_thread)
            results = future.result()
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing input {i}: {result}")
                processed_results.append(GenOutput(f"Error: {str(result)}", True))
            else:
                processed_results.append(result)
        
        return processed_results

    def apply_chat_template(self,
        conversation: List[List[dict]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        continue_final_message: bool = False,
        tokenize: bool = True,
    ) -> List[str]:
        """Apply chat template to a conversation - for API we just format as messages."""
        # For API-based models, we don't need to apply templates
        # Just return the conversation as-is
        return conversation

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
    
    def count_tokens(self, texts: List[str]) -> List[int]:
        """Estimate token count for API models."""
        # Rough estimation: 1 token H 4 characters
        token_counts = []
        for text in texts:
            if text:
                # Very rough approximation
                estimated_tokens = len(text) // 4
                token_counts.append(estimated_tokens)
            else:
                token_counts.append(0)
        return token_counts

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