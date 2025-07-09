"""
Progressive model selection for sentence-level aggregation.
Uses a two-stage approach: first using a large model to generate an outline,
then using a smaller model to generate the final response based on the outline.
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
        else:  # Default to Chinese
            self.outline_prompt_template = outline_prompt_template or (
                "请仔细分析以下问题，并列出一个简要的解答思路大纲。"
                "只需要梳理解题思路和关键步骤，不需要详细解答。\n\n"
                "问题：{question}\n\n"
                "解答思路大纲："
            )
        
        self._model_sizes = {}  # Cache for model sizes

    def aggregate_generation(
        self,
        generators: List,
        examples: List[Union[str, List[Dict]]],
        max_rounds: int = 500,
        max_tokens: int = 16384,
        is_chat: bool = True,
        **kwargs
    ) -> List[str]:
        """Run progressive model selection for batch generation."""
        if not generators or not examples:
            return [""] * len(examples)
        
        # Select the two largest models
        large_model, small_model = select_two_largest_models(generators)
        
        # Create separate attribution for each example in the batch
        attributions = [ModelAttribution() for _ in examples]

        # Get model names for attribution
        large_model_name = ray.get(large_model.get_model_name.remote())
        small_model_name = ray.get(small_model.get_model_name.remote())

        # Extract questions and prepare conversations
        questions = [self._extract_question(ex, is_chat) for ex in examples]
        outline_convs = [self._prepare_conversation(ex, self.outline_prompt_template.format(question=q), is_chat) 
                        for ex, q in zip(examples, questions)]
        final_convs = []
        
        # Stage 1: Batch generate outlines
        logger.info(f"📝 Stage 1: Generating outlines")
        logger.info(f"  Outline max tokens: {self.outline_max_tokens}")
        outline_results = self._batch_generate(large_model, outline_convs, self.outline_max_tokens, is_chat, "outline", **kwargs)
        
        # Stage 2: Prepare and generate final answers
        logger.info(f"📝 Stage 2: Generating final answers")
        for ex, (outline_text, _), q in zip(examples, outline_results, questions):
            if outline_text:
                conv = self._prepare_final_conversation(q, outline_text, is_chat)
                final_convs.append(conv)
            else:
                final_convs.append(None)
        
        # Calculate remaining tokens for final answers
        final_max_tokens = max_tokens - self.outline_max_tokens
        logger.info(f"  Final max tokens for answers: {final_max_tokens}")
        
        final_results = self._batch_generate(small_model, [c for c in final_convs if c], final_max_tokens, is_chat, "final", **kwargs)
        
        # Combine results and record attribution for each example separately
        results, j = [], 0
        for i, ((outline_text, outline_tokens), conv) in enumerate(zip(outline_results, final_convs)):
            if outline_text:
                # Add outline segment to this example's attribution (round 0)
                attributions[i].add_segment(outline_text, large_model_name, outline_tokens, 0)
            
            if conv and j < len(final_results):
                # Add final answer segment to this example's attribution (round 1)
                final_text, final_tokens = final_results[j]
                attributions[i].add_segment(final_text, small_model_name, final_tokens, 1)
                results.append(final_text)  # Only append final_text to results
                j += 1
            else:
                results.append("")  # Return empty string if no final answer
        
        # Store batch attribution data
        self.batch_attributions = attributions

        return results
    
    def _extract_question(self, example: Union[str, List[Dict]], is_chat: bool) -> str:
        """Extract question from example."""
        if not is_chat:
            return str(example)
        if isinstance(example, list):
            for msg in reversed(example):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return str(example)
    
    def _prepare_conversation(self, example: Union[str, List[Dict]], prompt: str, is_chat: bool) -> Union[str, List[Dict]]:
        """Prepare conversation with prompt."""
        if not is_chat:
            # For text completion, just return the prompt
            # The example is already included in the prompt from _extract_question
            return prompt
        
        if isinstance(example, list):
            conv = [msg for msg in example if msg.get("role") != "assistant"]
            conv.append({"role": "user", "content": prompt})
            return conv
        return [{"role": "user", "content": prompt}]
    
    def _prepare_final_conversation(self, question: str, outline: str, is_chat: bool) -> Union[str, List[Dict]]:
        """Prepare final conversation with special format for stage 2."""
        if not is_chat:
            # For text completion, include outline in the prompt
            return f"Based on the following examples and solution outline, please provide the answer with target format. {question}<think>{outline}</think>The best answer is"
        
        # For chat mode, use system message and user message with <think> tag
        return [
            {"role": "system", "content": "Based on the following examples and solution outline, please provide the answer with target format."},
            {"role": "user", "content": f"{question}<think>{outline}</think>"}
        ]
    
    def _batch_generate(self, model, conversations: List, max_tokens: int, is_chat: bool, stage: str, **kwargs) -> List[Tuple[str, int]]:
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
            "seed": kwargs.get("seed", 1234),
            "stop_strings": kwargs.get("stop_strings", None) if stage == "final" else ["</think>"],
        }

        outputs = ray.get(model.generate.remote(conversations, **gen_kwargs))

        # Get tokenizer to count tokens
        tokenizer = ray.get(model.get_tokenizer.remote())

        results = []
        for output in outputs:
            text = output.text if hasattr(output, 'text') else str(output)
            # Count tokens for the generated text
            token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            results.append((text, token_count))

        # logger.info(f"{stage}: Generated {len(results)} outputs with total {sum(tc for _, tc in results)} tokens")
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
        return f"ProgressiveSelector(outline_tokens={self.outline_max_tokens})"