"""
Reward-based sentence selection using external reward models.
Selects the best output from multiple models based on reward scores.
"""

import logging
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import torch

from .base import BaseSentenceAggregator, ModelAttribution

logger = logging.getLogger(__name__)


class RewardBasedSelector(BaseSentenceAggregator):
    """
    Select sentences based on reward scores from external scorer models.
    
    This selector generates outputs from all available models in parallel,
    then uses reward models to score each output and selects the best one.
    Supports excluding self-scoring to avoid bias.
    """
    
    def __init__(
        self, 
        exclude_self_scoring: bool = True,
        score_threshold: float = -2.0,
        max_repeat: int = 3,
        name: str = None
    ):
        """
        Initialize reward-based selector.
        
        Args:
            exclude_self_scoring: If True, exclude a model's own scorer when evaluating its output
            score_threshold: Minimum score threshold to accept an output
            max_repeat: Maximum number of repeated outputs before stopping
            name: Optional name for the selector
        """
        super().__init__(name or "RewardBasedSelector")
        self.exclude_self_scoring = exclude_self_scoring
        self.score_threshold = score_threshold
        self.max_repeat = max_repeat
    
    def select_best_sentence(
        self,
        sentences: List[str],
        generators: List,
        prompt: str,
        scorers = None,
        **kwargs
    ) -> Tuple[int, str, float]:
        """
        Select the best sentence based on reward scores.
        
        Args:
            sentences: List of generated sentences
            generators: List of generator objects that produced the sentences
            prompt: The prompt used for generation
            scorers: Scorer object for evaluating outputs
            
        Returns:
            Tuple of (best_index, best_sentence, best_score)
        """
        if not sentences:
            return 0, "", 0.0
        
        if len(sentences) == 1:
            return 0, sentences[0], 0.0
        
        # Handle identical outputs
        unique_outputs = set(s.strip() for s in sentences)
        if len(unique_outputs) == 1:
            logger.debug("All models produced identical outputs")
            return 0, sentences[0], 0.0
        
        # Score each sentence
        if not scorers:
            logger.warning("No scorers available, selecting first output")
            return 0, sentences[0], 0.0
        
        def score_one(g_s_pair):
            g, s = g_s_pair
            try:
                if self.exclude_self_scoring:
                    # Get scorer keys that don't include this generator's name
                    all_keys = list(scorers._scorer_cache.keys())
                    other_keys = [k for k in all_keys if g.name not in k]
                    if other_keys:
                        score = scorers.score(prompt, [s], keys=other_keys)[0]
                    else:
                        # If no other scorers, use all
                        score = scorers.score(prompt, [s])[0]
                else:
                    score = scorers.score(prompt, [s])[0]
                return score
            except Exception as e:
                logger.error(f"Error scoring output from {g.name}: {e}")
                return 0.0
        
        # Use thread pool for parallel scoring
        with ThreadPoolExecutor(max_workers=len(generators)) as executor:
            scores = list(executor.map(score_one, zip(generators, sentences)))
        
        # Log results
        for g, s, score in zip(generators, sentences, scores):
            log_text = s.replace("\n", "\\n")[:100] + "..." if len(s) > 100 else s.replace("\n", "\\n")
            logger.info(f"[{g.name}] Score: {score:.3f} | Text: {log_text}")
        
        # Select best
        best_idx = int(torch.tensor(scores).argmax())
        return best_idx, sentences[best_idx], scores[best_idx]
    
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
        Run iterative reward-based sentence aggregation for batch generation.
        
        Args:
            generators: List of generator objects
            scorers: Scorer object for evaluating outputs
            examples: List of input examples
            max_rounds: Maximum number of generation rounds
            max_new_tokens_per_round: Max tokens per round
            is_chat: Whether using chat format
            
        Returns:
            List of generated texts, one per example
        """
        if not generators or not examples:
            return [""] * len(examples)
        
        # Log available scorers
        if scorers:
            try:
                logger.info("Available scorers:")
                for key, (scorer, weight) in scorers._scorer_cache.items():
                    logger.info(f"  → {key} | type: {type(scorer).__name__} | weight: {weight}")
            except Exception as e:
                logger.debug(f"Could not log scorers: {e}")
        
        results = []
        
        # Process each example
        for example in examples:
            result = self._process_single_example(
                generators, scorers, example, max_rounds,
                max_new_tokens_per_round, is_chat, **kwargs
            )
            results.append(result)
        
        return results
    
    def _process_single_example(
        self,
        generators: List,
        scorers,
        example,
        max_rounds: int,
        max_new_tokens_per_round: int,
        is_chat: bool,
        **kwargs
    ) -> str:
        """Process a single example with reward-based selection."""
        # Initialize conversation
        if is_chat:
            conv = deepcopy(example) if isinstance(example, list) else []
        else:
            conv = example if isinstance(example, str) else ""
        
        available_gens = list(generators)
        last_output = None
        repeat_count = 0
        generated_text = ""
        self.attribution = ModelAttribution()  # Reset attribution
        
        for rnd in range(1, max_rounds + 1):
            # Prepare prompt
            if is_chat:
                prompt = conv
            else:
                prompt = conv
            
            # Filter generators by length
            available_gens = self._filter_by_length(available_gens, prompt, is_chat)
            if not available_gens:
                logger.warning("No generators available for current prompt length")
                break
            
            # Special handling for single model
            if len(available_gens) == 1:
                gen_kwargs = self._prepare_gen_kwargs(
                    kwargs, 
                    kwargs.get('max_tokens', 16384)  # Use larger default for single model
                )
            else:
                gen_kwargs = self._prepare_gen_kwargs(kwargs, max_new_tokens_per_round)
            
            # Generate from all models in parallel
            with ThreadPoolExecutor(max_workers=len(available_gens)) as executor:
                outputs = list(
                    executor.map(
                        lambda g: g.generate(prompt, **gen_kwargs),
                        available_gens
                    )
                )
            
            # Extract text from outputs
            sentences = []
            for output in outputs:
                text = output.text if hasattr(output, 'text') else str(output)
                sentences.append(text)
            
            # Select best sentence
            best_idx, best_sentence, best_score = self.select_best_sentence(
                sentences, available_gens, prompt, scorers
            )
            
            # Check score threshold
            if best_score < self.score_threshold:
                logger.info(f"Stop: best score {best_score:.3f} < threshold {self.score_threshold}")
                continue
            
            # Track attribution
            best_gen = available_gens[best_idx]
            model_name = getattr(best_gen, 'model_path', best_gen.name)
            self.attribution.add_segment(best_sentence, model_name, rnd, len(best_sentence))
            
            # Check repetition
            if best_sentence == last_output:
                repeat_count += 1
                if repeat_count >= self.max_repeat:
                    logger.info(f"Early stop: repeated {repeat_count} times")
                    break
            else:
                repeat_count = 0
                last_output = best_sentence
            
            # Update conversation
            generated_text += best_sentence
            if is_chat:
                if conv and isinstance(conv[-1], dict) and conv[-1].get('role') == 'assistant':
                    conv[-1]['content'] += best_sentence
                else:
                    conv.append({'role': 'assistant', 'content': best_sentence})
            else:
                conv += best_sentence
            
            # Check EOS
            best_output = outputs[best_idx]
            if hasattr(best_output, 'ended_with_eos') and best_output.ended_with_eos:
                logger.info("Early stop: EOS token from best model")
                break
        
        return generated_text
    
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
        return f"RewardBasedSelector(exclude_self={self.exclude_self_scoring}, threshold={self.score_threshold})"