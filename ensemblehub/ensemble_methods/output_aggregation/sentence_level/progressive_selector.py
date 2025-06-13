"""
Progressive model selection for sentence-level aggregation.
Uses a two-stage approach: first using a large model to generate an outline,
then using a smaller model to generate the final response based on the outline.
"""

import logging
from typing import List, Tuple, Optional, Union, Dict, Any
import ray
from copy import deepcopy

from .base import BaseSentenceAggregator, ModelAttribution

logger = logging.getLogger(__name__)


class ProgressiveSelector(BaseSentenceAggregator):
    """
    Progressive model selector that uses exactly 2 models:
    1. Large model: Generates a solution outline (500 tokens)
    2. Small model: Generates the final answer based on the outline
    
    If more than 2 models are provided, selects the 2 with the largest parameters.
    """
    
    def __init__(
        self,
        outline_max_tokens: int = 500,
        outline_prompt_template: Optional[str] = None,
        final_prompt_template: Optional[str] = None,
        template_language: str = "zh",
        name: str = None
    ):
        """
        Initialize progressive selector.
        
        Args:
            outline_max_tokens: Maximum tokens for the outline generation (default: 500)
            outline_prompt_template: Template for the outline prompt. Use {question} as placeholder.
                                   If None, uses default template based on template_language.
            final_prompt_template: Template for the final answer prompt. Use {outline} as placeholder.
                                 If None, uses default template based on template_language.
            template_language: Language for default prompts ("zh" for Chinese, "en" for English)
            name: Optional name for the selector
        """
        super().__init__(name or "ProgressiveSelector")
        self.outline_max_tokens = outline_max_tokens
        self.template_language = template_language
        
        # Default templates based on language
        if template_language == "en":
            self.outline_prompt_template = outline_prompt_template or (
                "Please carefully analyze the following question and provide a brief solution outline. "
                "Focus on the key steps and approach without detailed solutions.\n\n"
                "Question: {question}\n\n"
                "Solution Outline:"
            )
            self.final_prompt_template = final_prompt_template or (
                "Based on the following solution outline, please provide a detailed answer:\n\n"
                "{outline}\n\n"
                "Complete Solution:"
            )
        else:  # Default to Chinese
            self.outline_prompt_template = outline_prompt_template or (
                "请仔细分析以下问题，并列出一个简要的解答思路大纲。"
                "只需要梳理解题思路和关键步骤，不需要详细解答。\n\n"
                "问题：{question}\n\n"
                "解答思路大纲："
            )
            self.final_prompt_template = final_prompt_template or (
                "基于以下解题思路大纲，请详细解答问题：\n\n"
                "{outline}\n\n"
                "请给出完整的解答："
            )
        
        self._model_sizes = {}  # Cache for model sizes
    
    def _select_two_largest_models(self, generators: List) -> Tuple[Any, Any]:
        """
        Select the two models with the largest parameters.
        Returns (large_model, small_model).
        """
        if len(generators) < 2:
            raise ValueError(f"ProgressiveSelector requires at least 2 models, got {len(generators)}")
        
        # Get sizes for all generators
        gen_sizes = []
        for gen in generators:
            size = ray.get(gen.get_model_size.remote())
            gen_sizes.append((gen, size))
            logger.info(f"Model size detected: {ray.get(gen.get_model_name.remote()) if hasattr(gen, 'get_model_name') else 'unknown'} = {size}B params")
        
        # Sort by size (descending) and get top 2
        gen_sizes.sort(key=lambda x: x[1], reverse=True)
        large_model = gen_sizes[0][0]
        small_model = gen_sizes[1][0]
        
        large_name = ray.get(large_model.get_model_name.remote()) if hasattr(large_model, 'get_model_name') else 'unknown'
        small_name = ray.get(small_model.get_model_name.remote()) if hasattr(small_model, 'get_model_name') else 'unknown'
        
        logger.info(f"Selected models: Large={large_name} ({gen_sizes[0][1]}B), Small={small_name} ({gen_sizes[1][1]}B)")
        
        return large_model, small_model
    
    def aggregate_generation(
        self,
        generators: List,
        examples: List[Union[str, List[Dict]]],
        max_rounds: int = 500,
        max_tokens: int = 16384,
        max_new_tokens_per_round: int = 256,
        is_chat: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Run progressive model selection for batch generation.
        """
        if not generators or not examples:
            return [""] * len(examples)
        
        # Select the two largest models
        large_model, small_model = self._select_two_largest_models(generators)
        
        results = []
        
        # Process each example
        for example in examples:
            result = self._process_single_example(
                large_model, small_model, example, max_tokens, is_chat, **kwargs
            )
            results.append(result)
        
        return results
    
    def _process_single_example(
        self,
        large_model,
        small_model,
        example: Union[str, List[Dict]],
        max_tokens: int,
        is_chat: bool,
        **kwargs
    ) -> str:
        """Process a single example with two-stage generation."""
        self.attribution = ModelAttribution()  # Reset attribution
        
        # Extract the question from the example
        if is_chat:
            # For chat format, get the last user message as the question
            question = ""
            if isinstance(example, list):
                for msg in reversed(example):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg.get("content", "")
                        break
            if not question and example:
                # Fallback: use the entire conversation
                question = str(example)
        else:
            # For text format, use the entire input as the question
            question = example if isinstance(example, str) else str(example)
        
        # Stage 1: Generate outline with large model
        logger.info("📝 Stage 1: Generating outline with large model")
        outline = self._generate_outline(large_model, example, question, is_chat, **kwargs)
        
        if not outline:
            logger.warning("Failed to generate outline, returning empty result")
            return ""
        
        # Stage 2: Generate final answer with small model
        logger.info("🔧 Stage 2: Generating final answer with small model")
        final_answer = self._generate_final_answer(
            small_model, example, outline, is_chat, max_tokens, **kwargs
        )
        
        return final_answer
    
    def _generate_outline(
        self,
        large_model,
        example: Union[str, List[Dict]],
        question: str,
        is_chat: bool,
        **kwargs
    ) -> str:
        """Generate outline using the large model."""
        # Prepare the outline prompt
        outline_prompt = self.outline_prompt_template.format(question=question)
        
        # Prepare the input for the large model
        if is_chat:
            # For chat format, append the outline prompt as a new user message
            if isinstance(example, list):
                outline_conversation = deepcopy(example)
                # Remove any existing assistant messages to ensure clean context
                outline_conversation = [msg for msg in outline_conversation if msg.get("role") != "assistant"]
                # Add our outline prompt
                outline_conversation.append({
                    "role": "user",
                    "content": outline_prompt
                })
            else:
                outline_conversation = [{
                    "role": "user",
                    "content": outline_prompt
                }]
        else:
            # For text format, append the outline prompt
            outline_conversation = f"{example}\n\n{outline_prompt}"
        
        # Generate outline
        gen_kwargs = {
            "max_tokens": self.outline_max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "is_chat": is_chat,
        }
        
        # Add optional parameters
        for key in ['seed', 'stop_strings', 'frequency_penalty', 'presence_penalty']:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]
        
        try:
            output = ray.get(large_model.generate.remote(outline_conversation, **gen_kwargs))
            
            # Extract text from output
            if hasattr(output, 'text'):
                outline_text = output.text
            else:
                outline_text = str(output)
            
            # Track attribution
            model_name = ray.get(large_model.get_model_name.remote())
            self.attribution.add_segment(outline_text, model_name, 1)
            
            logger.info(f"Generated outline ({len(outline_text)} chars)")
            return outline_text
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return ""
    
    def _generate_final_answer(
        self,
        small_model,
        original_example: Union[str, List[Dict]],
        outline: str,
        is_chat: bool,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate final answer using the small model based on the outline."""
        # Prepare the input for the small model
        final_prompt = self.final_prompt_template.format(outline=outline)
        
        if is_chat:
            # For chat format, create a conversation with the outline context
            if isinstance(original_example, list):
                final_conversation = deepcopy(original_example)
                # Remove any existing assistant messages
                final_conversation = [msg for msg in final_conversation if msg.get("role") != "assistant"]
                # Add the outline as context
                final_conversation.append({
                    "role": "user",
                    "content": final_prompt
                })
            else:
                final_conversation = [{
                    "role": "user",
                    "content": final_prompt
                }]
        else:
            # For text format, combine original question with outline
            final_conversation = f"{original_example}\n\n{final_prompt}"
        
        # Generate final answer
        # The small model should generate the remaining tokens after outline
        # But ensure it has enough space to generate a meaningful response
        remaining_tokens = max(1000, max_tokens - self.outline_max_tokens)
        gen_kwargs = {
            "max_tokens": remaining_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "is_chat": is_chat,
        }
        
        # Add optional parameters
        for key in ['seed', 'stop_strings', 'frequency_penalty', 'presence_penalty']:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]
        
        try:
            output = ray.get(small_model.generate.remote(final_conversation, **gen_kwargs))
            
            # Extract text from output
            if hasattr(output, 'text'):
                final_text = output.text
            else:
                final_text = str(output)
            
            # Track attribution
            model_name = ray.get(small_model.get_model_name.remote())
            self.attribution.add_segment(final_text, model_name, 2)
            
            logger.info(f"Generated final answer ({len(final_text)} chars)")
            
            # Combine outline and final answer
            return outline + "\n\n" + final_text
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return outline  # Return at least the outline if final generation fails
    
    def __repr__(self):
        return f"ProgressiveSelector(outline_tokens={self.outline_max_tokens})"