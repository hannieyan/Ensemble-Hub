"""
Progressive model selection for sentence-level aggregation.
Switches between models based on their parameter size, starting with larger models
and transitioning to smaller ones as generation progresses.
"""

import logging
from typing import List, Tuple, Optional
import re

from .base import BaseSentenceAggregator, ModelAttribution

logger = logging.getLogger(__name__)


class ProgressiveSelector(BaseSentenceAggregator):
    """
    Progressive model selector that switches models based on parameter size.
    
    The selector starts with the largest model and progressively switches to smaller
    models based on generation length thresholds. This allows using powerful models
    for the critical initial reasoning phase while using efficient models for 
    continuation.
    
    Models are automatically sorted by parameter size (descending).
    """
    
    def __init__(
        self,
        length_thresholds: Optional[List[int]] = None,
        size_threshold: Optional[float] = None,
        max_repeat: int = 3,
        name: str = None
    ):
        """
        Initialize progressive selector.
        
        Args:
            length_thresholds: Token counts for switching models [1000, 2000, 3000]
                              If None, uses size_threshold to auto-compute
            size_threshold: Alternative to length_thresholds - fraction of generation
                          to allocate to each model size tier (e.g., 0.3 = 30% for largest)
            max_repeat: Maximum number of repeated outputs before stopping
            name: Optional name for the selector
        """
        super().__init__(name or "ProgressiveSelector")
        self.length_thresholds = length_thresholds
        self.size_threshold = size_threshold or 0.3
        self.max_repeat = max_repeat
        self._model_sizes = {}  # Cache for model sizes
    
    def _get_model_size(self, generator) -> float:
        """
        Estimate model size from generator.
        Returns size in billions of parameters.
        """
        if hasattr(generator, 'model_path'):
            model_name = generator.model_path.lower()
            # Extract size from common naming patterns
            import re
            size_match = re.search(r'(\d+\.?\d*)b', model_name)
            if size_match:
                return float(size_match.group(1))
            # Common size mappings
            if '70b' in model_name or '72b' in model_name:
                return 70.0
            elif '30b' in model_name or '33b' in model_name:
                return 30.0
            elif '13b' in model_name:
                return 13.0
            elif '7b' in model_name:
                return 7.0
            elif '3b' in model_name:
                return 3.0
            elif '1b' in model_name or '1.5b' in model_name:
                return 1.5
            elif '0.5b' in model_name or '500m' in model_name:
                return 0.5
        
        # Default size if cannot determine
        return 1.0
    
    def _sort_generators_by_size(self, generators: List) -> List:
        """Sort generators by model size in descending order."""
        # Get sizes for all generators
        gen_sizes = []
        for gen in generators:
            if gen.name not in self._model_sizes:
                self._model_sizes[gen.name] = self._get_model_size(gen)
            gen_sizes.append((gen, self._model_sizes[gen.name]))
        
        # Sort by size (descending)
        gen_sizes.sort(key=lambda x: x[1], reverse=True)
        return [gen for gen, _ in gen_sizes]
    
    def _get_token_count(self, text: str, tokenizer) -> int:
        """Get token count for text using the tokenizer."""
        if tokenizer is None:
            return len(text.split())
        
        try:
            return len(tokenizer(text, return_tensors="pt").input_ids[0])
        except Exception:
            return len(text.split())
    
    def _compute_dynamic_thresholds(self, total_expected_tokens: int, num_models: int) -> List[int]:
        """Compute length thresholds based on model count and size threshold."""
        if num_models == 1:
            return [total_expected_tokens]
        
        # Allocate tokens based on size_threshold
        # Largest model gets size_threshold portion, rest distributed among others
        thresholds = []
        first_portion = int(total_expected_tokens * self.size_threshold)
        remaining = total_expected_tokens - first_portion
        
        if num_models > 1:
            # Distribute remaining tokens among other models
            per_model = remaining // (num_models - 1)
            cumulative = first_portion
            thresholds.append(cumulative)
            
            for i in range(1, num_models - 1):
                cumulative += per_model
                thresholds.append(cumulative)
        
        return thresholds
    
    def _determine_current_model_index(
        self,
        current_text: str,
        sorted_generators: List,
        total_expected_tokens: int = 4096
    ) -> int:
        """Determine which model should be used based on current text length."""
        # Get current token count
        tokenizer = None
        for gen in sorted_generators:
            if hasattr(gen, 'tokenizer') and gen.tokenizer is not None:
                tokenizer = gen.tokenizer
                break
        
        token_count = self._get_token_count(current_text, tokenizer)
        
        # Use provided thresholds or compute dynamic ones
        if self.length_thresholds:
            thresholds = self.length_thresholds
        else:
            thresholds = self._compute_dynamic_thresholds(
                total_expected_tokens, 
                len(sorted_generators)
            )
        
        # Find appropriate model index
        for i, threshold in enumerate(thresholds):
            if token_count < threshold:
                return min(i, len(sorted_generators) - 1)
        
        # If exceeded all thresholds, use the smallest model
        return len(sorted_generators) - 1
    
    def select_best_sentence(
        self,
        sentences: List[str],
        generators: List,
        prompt: str,
        current_text: str = "",
        round_num: int = 0,
        scorers = None,
        **kwargs
    ) -> Tuple[int, str, float]:
        """
        Select sentence based on model size progression.
        This method is kept for compatibility but the main logic is in aggregate_generation.
        """
        if not sentences:
            return 0, "", 0.0
        
        # Sort generators if not already sorted
        sorted_gens = self._sort_generators_by_size(generators)
        
        # Determine model index based on current text length
        model_idx = self._determine_current_model_index(
            current_text, 
            sorted_gens,
            kwargs.get('max_tokens', 4096)
        )
        
        # Map back to original generator order
        if model_idx < len(sorted_gens):
            selected_gen = sorted_gens[model_idx]
            for i, gen in enumerate(generators):
                if gen.name == selected_gen.name:
                    return i, sentences[i], 1.0
        
        # Fallback to first
        return 0, sentences[0], 1.0
    
    def aggregate_generation(
        self,
        generators: List,
        scorers,
        examples: List,
        max_rounds: int = 500,
        max_new_tokens_per_round: int = 256,
        is_chat: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Run progressive model selection for batch generation.
        """
        if not generators or not examples:
            return [""] * len(examples)
        
        # Sort generators by size once
        sorted_gens = self._sort_generators_by_size(generators)
        logger.info(f"Models sorted by size: {[g.name for g in sorted_gens]}")
        
        results = []
        
        # Process each example
        for example in examples:
            result = self._process_single_example(
                sorted_gens, scorers, example, max_rounds, 
                max_new_tokens_per_round, is_chat, **kwargs
            )
            results.append(result)
        
        return results
    
    def _process_single_example(
        self,
        sorted_generators: List,
        scorers,
        example,
        max_rounds: int,
        max_new_tokens_per_round: int,
        is_chat: bool,
        **kwargs
    ) -> str:
        """Process a single example with progressive model switching."""
        # Initialize conversation
        if is_chat:
            conv = example if isinstance(example, list) else []
        else:
            conv = example if isinstance(example, str) else ""
        
        last_output = None
        repeat_count = 0
        current_generated_text = ""
        self.attribution = ModelAttribution()  # Reset attribution
        
        for rnd in range(1, max_rounds + 1):
            # Prepare prompt
            if is_chat:
                prompt = conv
            else:
                prompt = conv
            
            # Filter available generators by length
            available_gens = self._filter_by_length(sorted_generators, prompt, is_chat)
            if not available_gens:
                logger.warning("No generators available for current prompt length")
                break
            
            # Determine which model to use
            model_idx = self._determine_current_model_index(
                current_generated_text, 
                available_gens,
                kwargs.get('max_tokens', 4096)
            )
            selected_gen = available_gens[model_idx]
            
            logger.info(f"Round {rnd}: Using {selected_gen.name} (size: {self._model_sizes.get(selected_gen.name, 'unknown')}B)")
            
            # Generate
            gen_kwargs = self._prepare_gen_kwargs(kwargs, max_new_tokens_per_round)
            output = selected_gen.generate(prompt, **gen_kwargs)
            
            # Handle output
            output_text = output.text if hasattr(output, 'text') else str(output)
            if not output_text:
                break
            
            # Track attribution
            model_name = getattr(selected_gen, 'model_path', selected_gen.name)
            self.attribution.add_segment(output_text, model_name, rnd, len(output_text))
            
            # Check repetition
            if output_text == last_output:
                repeat_count += 1
                if repeat_count >= self.max_repeat:
                    logger.info(f"Early stop: repeated {repeat_count} times")
                    break
            else:
                repeat_count = 0
                last_output = output_text
            
            # Update conversation
            current_generated_text += output_text
            if is_chat:
                if conv and isinstance(conv[-1], dict) and conv[-1].get('role') == 'assistant':
                    conv[-1]['content'] += output_text
                else:
                    conv.append({'role': 'assistant', 'content': output_text})
            else:
                conv += output_text
            
            # Check EOS
            if hasattr(output, 'ended_with_eos') and output.ended_with_eos:
                logger.info("Early stop: EOS token")
                break
        
        return current_generated_text
    
    def _filter_by_length(self, generators: List, prompt, is_chat: bool) -> List:
        """Filter generators that can handle the current prompt length."""
        available = []
        for g in generators:
            tok = getattr(g, 'tokenizer', None)
            if tok is None:
                available.append(g)
                continue
            
            try:
                if is_chat:
                    length = len(tok.apply_chat_template(
                        prompt, tokenize=True, add_generation_prompt=True
                    ))
                else:
                    length = len(tok(prompt, return_tensors="pt").input_ids[0])
                
                max_length = getattr(tok, 'model_max_length', 32768)
                if length <= max_length:
                    available.append(g)
                else:
                    logger.debug(f"Skip {g.name}: length {length} > {max_length}")
            except Exception as e:
                logger.warning(f"Error checking length for {g.name}: {e}")
                available.append(g)
        
        return available
    
    def _prepare_gen_kwargs(self, kwargs: dict, default_max_tokens: int) -> dict:
        """Prepare generation kwargs."""
        gen_kwargs = {
            'max_tokens': kwargs.get('max_tokens', default_max_tokens),
            'temperature': kwargs.get('temperature', 0.95),
            'top_p': kwargs.get('top_p', 0.7),
        }
        
        # Add optional parameters
        for key in ['seed', 'stop_strings', 'frequency_penalty', 'presence_penalty']:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]
        
        return gen_kwargs
    
    def __repr__(self):
        return f"ProgressiveSelector(thresholds={self.length_thresholds}, size_threshold={self.size_threshold})"