"""
Progressive model selection for sentence-level aggregation.
Supports switching models based on length or special tokens.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import re

from .base import BaseSentenceAggregator, ModelAttribution
from ....conversation import ConversationTemplate

logger = logging.getLogger(__name__)


class ProgressiveSelector(BaseSentenceAggregator):
    """
    Progressive model selector that switches models based on:
    1. Fixed length thresholds (e.g., first 1000 tokens with largest model, next 1000 with second largest)
    2. Special tokens (e.g., switch when encountering <\think>)
    
    Models should be provided in descending order of capability/size.
    """
    
    def __init__(
        self,
        switch_mode: str = "length",  # "length" or "token"
        length_thresholds: Optional[List[int]] = None,
        special_tokens: Optional[List[str]] = None,
        max_repeat: int = 3,
        name: str = None
    ):
        """
        Initialize progressive selector.
        
        Args:
            switch_mode: "length" for fixed length switching, "token" for special token switching
            length_thresholds: List of cumulative token counts for switching [1000, 2000, 3000]
            special_tokens: List of special tokens for switching ["<\think>", "<\analyze>"]
            max_repeat: Maximum number of repeated outputs before stopping
            name: Optional name for the selector
        """
        super().__init__(name or "ProgressiveSelector")
        self.switch_mode = switch_mode
        self.length_thresholds = length_thresholds or [1000, 2000, 3000]
        self.special_tokens = special_tokens or [r"<\think>"]
        self.max_repeat = max_repeat
        
        if switch_mode not in ["length", "token"]:
            raise ValueError("switch_mode must be 'length' or 'token'")
    
    def _get_token_count(self, text: str, tokenizer) -> int:
        """Get token count for text using the tokenizer."""
        if tokenizer is None:
            # Fallback: rough estimate
            return len(text.split())
        
        try:
            return tokenizer(text, return_tensors="pt").input_ids.size(1)
        except Exception:
            return len(text.split())
    
    def _find_special_token_positions(self, text: str) -> List[Tuple[str, int]]:
        """Find positions of special tokens in the text."""
        positions = []
        for token in self.special_tokens:
            # Escape special regex characters and find all occurrences
            escaped_token = re.escape(token)
            for match in re.finditer(escaped_token, text):
                positions.append((token, match.start()))
        
        # Sort by position
        positions.sort(key=lambda x: x[1])
        return positions
    
    def _determine_current_model_index(
        self,
        current_text: str,
        generators: List,
        round_num: int
    ) -> int:
        """Determine which model should be used based on current state."""
        if self.switch_mode == "length":
            # Use first available tokenizer to count tokens
            tokenizer = None
            for gen in generators:
                if hasattr(gen, 'tokenizer') and gen.tokenizer is not None:
                    tokenizer = gen.tokenizer
                    break
            
            token_count = self._get_token_count(current_text, tokenizer)
            
            # Find which threshold we've crossed
            model_idx = 0
            for i, threshold in enumerate(self.length_thresholds):
                if token_count < threshold:
                    model_idx = i
                    break
            else:
                # If we've exceeded all thresholds, use the last model
                model_idx = len(self.length_thresholds)
            
            # Ensure we don't exceed available models
            return min(model_idx, len(generators) - 1)
        
        elif self.switch_mode == "token":
            # Find special token positions
            token_positions = self._find_special_token_positions(current_text)
            
            # Count how many different special tokens we've encountered
            encountered_tokens = set()
            for token, _ in token_positions:
                encountered_tokens.add(token)
            
            # Determine model index based on number of unique tokens encountered
            model_idx = len(encountered_tokens)
            
            # Ensure we don't exceed available models
            return min(model_idx, len(generators) - 1)
        
        return 0
    
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
        Select the best sentence based on progressive model selection.
        """
        if not sentences:
            return 0, "", 0.0
        
        # Determine which model should be used
        model_idx = self._determine_current_model_index(current_text, generators, round_num)
        
        # If we have fewer sentences than the determined model index,
        # use the last available sentence
        best_idx = min(model_idx, len(sentences) - 1)
        
        logger.debug(f"Round {round_num}: Using model {best_idx} (mode: {self.switch_mode})")
        
        return best_idx, sentences[best_idx], 1.0
    
    def aggregate_generation(
        self,
        generators: List,
        scorers,
        example: Dict[str, Any],
        max_rounds: int = 500,
        max_new_tokens_per_round: int = 256,
        **kwargs
    ) -> str:
        """
        Run iterative progressive model selection.
        """
        if not generators:
            return ""
        
        convo = ConversationTemplate(example.get("instruction", ""), example.get("input", ""))
        
        last_output = None
        repeat_count = 0
        current_generated_text = ""
        self.attribution = ModelAttribution()  # Reset attribution for new generation
        
        for rnd in range(1, max_rounds + 1):
            prompt = convo.render()
            
            # Check total length limits
            tok = getattr(generators[0], "tokenizer", None)
            if tok is not None:
                total_length = tok(prompt, return_tensors="pt").input_ids.size(1)
                if total_length > 32768:
                    logger.warning(f"Early stop: total prompt length {total_length} > 32768")
                    break
            
            # Filter generators by length constraints
            available_gens = []
            for g in generators:
                tok = getattr(g, "tokenizer", None)
                if tok is not None:
                    length = tok(prompt, return_tensors="pt").input_ids.size(1)
                    if length <= getattr(tok, 'model_max_length', 32768):
                        available_gens.append(g)
                else:
                    available_gens.append(g)
            
            if not available_gens:
                logger.error("No generators available for current prompt length")
                break
            
            # Determine which model to use based on current progress
            model_idx = self._determine_current_model_index(current_generated_text, available_gens, rnd)
            
            # Ensure model index is valid
            model_idx = min(model_idx, len(available_gens) - 1)
            selected_generator = available_gens[model_idx]
            
            logger.debug(f"Round {rnd}: Using {selected_generator.name} (index {model_idx})")
            
            # Generate with selected model
            dicts = convo.render_dict()
            
            # Prepare generation parameters
            gen_kwargs = {
                "max_tokens": kwargs.get("max_tokens", max_new_tokens_per_round),
                "temperature": kwargs.get("temperature", 0.95),
                "top_p": kwargs.get("top_p", 0.7),
            }
            if "seed" in kwargs:
                gen_kwargs["seed"] = kwargs["seed"]
            if "stop_strings" in kwargs:
                gen_kwargs["stop_strings"] = kwargs["stop_strings"]
            if "enable_thinking" in kwargs:
                gen_kwargs["enable_thinking"] = kwargs["enable_thinking"]
            
            best_output = selected_generator.generate(dicts, **gen_kwargs)
            
            # Record model attribution
            model_name = getattr(selected_generator, 'model_path', selected_generator.name)
            self.attribution.add_segment(best_output.text, model_name, rnd)
            
            # Check for repetition
            if best_output.text == last_output:
                repeat_count += 1
                if repeat_count >= self.max_repeat:
                    logger.info(f"Early stop: same output repeated {repeat_count} times")
                    break
            else:
                repeat_count = 1
                last_output = best_output.text
            
            # Add to conversation and update generated text
            convo.add_assistant(best_output.text)
            current_generated_text += best_output.text
            
            # Check for EOS
            if best_output.ended_with_eos:
                logger.info("Early stop: EOS token emitted")
                break
            
            # For token mode, check if we should stop based on special tokens
            if self.switch_mode == "token":
                token_positions = self._find_special_token_positions(current_generated_text)
                unique_tokens = len(set(token[0] for token in token_positions))
                
                # If we've encountered all special tokens and are using the last model
                if unique_tokens >= len(self.special_tokens) and model_idx == len(available_gens) - 1:
                    # Continue with the current model until natural stopping
                    pass
        
        return "".join(convo.assistant_parts)
    
    def __repr__(self):
        if self.switch_mode == "length":
            return f"ProgressiveSelector(mode=length, thresholds={self.length_thresholds})"
        else:
            return f"ProgressiveSelector(mode=token, tokens={self.special_tokens})"