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
        
        first_results = self._batch_generate(large_model, examples, self.switch_after_tokens, is_chat, **kwargs)
        
        # Stage 2: Continue generation with small model
        logger.info(f"📝 Stage 2: Switching to small model for continuation")
        
        # Prepare continuations
        continued_examples = [self._prepare_continuation(ex, first_text, is_chat) if first_text else None 
                             for ex, (first_text, _) in zip(examples, first_results)]
        
        # Calculate remaining tokens
        remaining_tokens = max_tokens - self.switch_after_tokens
        logger.info(f"  Remaining tokens for continuation: {remaining_tokens}")
        
        # Generate continuations with small model
        valid_continuations = [(i, ex) for i, ex in enumerate(continued_examples) if ex is not None]
        continuation_map = {}
        
        if valid_continuations:
            valid_indices, valid_examples = zip(*valid_continuations)
            continuation_results = self._batch_generate(small_model, valid_examples, remaining_tokens, is_chat, **kwargs)
            continuation_map = {idx: result for idx, result in zip(valid_indices, continuation_results)}
        
        # Combine results and record attribution
        results = []
        for i, (first_text, first_tokens) in enumerate(first_results):
            if first_text:
                attributions[i].add_segment(first_text, large_model_name, first_tokens, 0)
                
                if i in continuation_map:
                    continuation_text, continuation_tokens = continuation_map[i]
                    attributions[i].add_segment(continuation_text, small_model_name, continuation_tokens, 1)
                    results.append(first_text + continuation_text)
                else:
                    results.append(first_text)
            else:
                results.append("")
        
        # Store batch attribution data
        self.batch_attributions = attributions

        return results
    
    def _prepare_continuation(self, example: Union[str, List[Dict]], generated_text: str, is_chat: bool) -> Union[str, List[Dict]]:
        """Prepare continuation by appending generated text to the conversation."""
        if not is_chat:
            # For text completion, just concatenate
            return str(example) + generated_text
        
        if isinstance(example, list):
            # For chat mode, add the generated text as an assistant message
            # and keep the conversation going
            conv = example.copy()
            conv.append({"role": "assistant", "content": generated_text})
            # The model will continue from where it left off
            return conv
        
        # Fallback: treat as single user message
        return [
            {"role": "user", "content": str(example)},
            {"role": "assistant", "content": generated_text}
        ]
    
    def _batch_generate(self, model, conversations: List, max_tokens: int, is_chat: bool, **kwargs) -> List[Tuple[str, int]]:
        """Batch generate with model and return text with token counts.
        
        Returns:
            List of tuples (text, token_count) for each generated output
        """
        if not conversations:
            return []
        
        # Filter out None values
        valid_conversations = [c for c in conversations if c is not None]
        if not valid_conversations:
            return []
        
        gen_kwargs = {
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "is_chat": is_chat,
            "seed": kwargs.get("seed", 1234),
            "stop_strings": kwargs.get("stop_strings", None),
        }

        outputs = ray.get(model.generate.remote(valid_conversations, **gen_kwargs))
        tokenizer = ray.get(model.get_tokenizer.remote())

        results = []
        conv_idx = 0
        for conv in conversations:
            if conv is None:
                results.append(("", 0))
            else:
                output = outputs[conv_idx]
                text = output.text if hasattr(output, 'text') else str(output)
                token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                results.append((text, token_count))
                conv_idx += 1

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