"""
Seamless model switching for sentence-level aggregation.
Switches from large model to small model after a fixed number of tokens.
"""

import logging
from typing import List, Tuple, Optional, Union, Dict, Any
import ray

from .base import BaseSentenceAggregator, ModelAttribution

logger = logging.getLogger(__name__)


def select_two_largest_models(generators: List) -> Tuple[Any, Any]:
    """
    Select the two models with the largest parameters.
    Returns (large_model, small_model).
    """
    if len(generators) < 2:
        raise ValueError(f"Switch requires at least 2 models, got {len(generators)}")

    # Get sizes and names for all generators
    gen_info = []
    for gen in generators:
        size = ray.get(gen.get_model_size.remote())
        name = ray.get(gen.get_model_name.remote()) if hasattr(gen, 'get_model_name') else 'unknown'
        gen_info.append((gen, size, name))
        logger.info(f"Model size detected: {name} = {size}B params")

    # Sort by size (descending) and get top 2
    gen_info.sort(key=lambda x: x[1], reverse=True)
    large_model, large_size, large_name = gen_info[0]
    small_model, small_size, small_name = gen_info[1]

    logger.info(f"Selected models: Large={large_name} ({large_size}B), Small={small_name} ({small_size}B)")

    return large_model, small_model


class Switch(BaseSentenceAggregator):
    """
    Model switch selector that seamlessly switches from large to small model:
    1. Large model: Generates the first part (switch_after_tokens)
    2. Small model: Continues generation until completion
    
    No prompts are used - just direct model switching.
    """
    
    def __init__(
        self,
        switch_after_tokens: int = 500,
        name: str = None
    ):
        """
        Initialize model switch selector.
        
        Args:
            switch_after_tokens: Number of tokens to generate with large model before switching (default: 500)
            name: Optional name for the selector
        """
        super().__init__(name or "Switch")
        self.switch_after_tokens = switch_after_tokens

    def aggregate_generation(
        self,
        generators: List,
        examples: List[Union[str, List[Dict]]],
        max_tokens: int = 16384,
        is_chat: bool = True,
        **kwargs
    ) -> List[str]:
        """Run seamless model switching for batch generation."""
        if not generators or not examples:
            return [""] * len(examples)

        # Select the two largest models
        large_model, small_model = select_two_largest_models(generators)
        
        # Get model names and create attributions
        large_model_name = ray.get(large_model.get_model_name.remote())
        small_model_name = ray.get(small_model.get_model_name.remote())
        attributions = [ModelAttribution() for _ in examples]

        # Stage 1: Generate initial part with large model
        logger.info(f"📝 Stage 1: Generating with large model")
        logger.info(f"  Initial generation tokens: {self.switch_after_tokens}")
        
        # Convert text prompts to chat format for first stage
        first_stage_examples = []
        for ex in examples:
            first_stage_examples.append([
                {"role": "user", "content": ex}
            ])
        
        first_results = self._batch_generate(large_model, first_stage_examples, self.switch_after_tokens, is_chat=True, continue_final_message=False, **kwargs)

        # Stage 2: Continue generation with small model
        logger.info(f"📝 Stage 2: Switching to small model for continuation")
        
        # Prepare continuations - create chat format with assistant message for continuation
        continued_examples = []
        
        for ex, (first_text, _) in zip(examples, first_results):
            # For HF models: use standard chat format
            continued_examples.append([
                {"role": "user", "content": ex},
                {"role": "assistant", "content": first_text}
            ])
            # # For API models: prepend the first_text to the user content
            # api_continued_examples.append([
            #     {"role": "user", "content": ex + "(please continue thinking and end with think token)" + first_text}
            # ])
        
        # Apply chat template to convert chat format to text
        text_inputs = ray.get(small_model.apply_chat_template.remote(
            continued_examples,
            add_generation_prompt=True,
            enable_thinking=True,
            continue_final_message=False,
            tokenize=False,
        ))
        
        # For API models, apply_chat_template returns the original conversation
        # We need to use the conversation directly for API generation
        if isinstance(text_inputs[0], list):
            # API model case - use api_continued_examples with prepended text
            text_inputs = continued_examples
        else:
            # HF model case - clean special tokens and use text mode
            special_tokens_to_remove = ["<｜end▁of▁sentence｜><｜Assistant｜><think>", "<｜end▁of▁sentence｜><｜Assistant｜>"]
            cleaned_text_inputs = []
            for text in text_inputs:
                cleaned_text = text
                # Check if the text ends with any of the special tokens and remove from the end
                for token in special_tokens_to_remove:
                    if cleaned_text.endswith(token):
                        cleaned_text = cleaned_text[:-len(token)]
                        break  # Only remove the first match to avoid over-removing
                cleaned_text_inputs.append(cleaned_text)
            text_inputs = cleaned_text_inputs
        
        # Calculate remaining tokens
        remaining_tokens = max_tokens - self.switch_after_tokens
        logger.info(f"  Remaining tokens for continuation: {remaining_tokens}")

        # Generate continuations with small model
        continuation_results = self._batch_generate(small_model, text_inputs, remaining_tokens, is_chat=False, **kwargs)
        
        # Combine results and record attribution
        results = []
        for i, ((first_text, first_tokens), (continuation_text, continuation_tokens)) in enumerate(
            zip(first_results, continuation_results)
        ):
            attributions[i].add_segment(first_text, large_model_name, first_tokens, 0)
            attributions[i].add_segment(continuation_text, small_model_name, continuation_tokens, 1)
            results.append(first_text + continuation_text)
        
        # Store batch attribution data
        self.batch_attributions = attributions

        return results
    
    def _batch_generate(self, model, conversations: List, max_tokens: int, is_chat: bool, **kwargs) -> List[Tuple[str, int]]:
        """Batch generate with model and return text with token counts.
        
        Returns:
            List of tuples (text, token_count) for each generated output
        """
        if not conversations:
            return []
        
        gen_kwargs = {
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "is_chat": is_chat,
            "continue_final_message": kwargs.get("continue_final_message", False),
            "seed": kwargs.get("seed", 1234),
            "stop_strings": kwargs.get("stop_strings", None),
        }

        outputs = ray.get(model.generate.remote(conversations, **gen_kwargs))
        tokenizer = ray.get(model.get_tokenizer.remote())

        results = []
        for output in outputs:
            text = output.text if hasattr(output, 'text') else str(output)
            
            # Check if output already has token_count (from API)
            if hasattr(output, 'token_count') and output.token_count is not None:
                token_count = output.token_count
            elif tokenizer is not None:
                # Use tokenizer to count tokens
                token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            else:
                # Fallback estimation: 1 token ≈ 4 characters
                token_count = len(text) // 4 if text else 0
            
            results.append((text, token_count))

        return results

    def get_attribution_data(self) -> List[Dict[str, Any]]:
        """Get model attribution data for the last generation.
        
        Returns:
            List of attribution data, one dict per example in the batch
        """
        return [{
            "summary": attr.get_attribution_summary(),
            "detailed": attr.get_detailed_attribution(),
        } for attr in self.batch_attributions]

    def __repr__(self):
        return f"Switch(switch_after={self.switch_after_tokens})"