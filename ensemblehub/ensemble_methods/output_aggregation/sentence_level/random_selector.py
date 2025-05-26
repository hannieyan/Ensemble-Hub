"""
Random sentence selection (migrated from random.py).
"""

import random
import logging
from typing import List, Dict, Any, Tuple

from .base import BaseSentenceAggregator, ModelAttribution
from ....conversation import ConversationTemplate

logger = logging.getLogger(__name__)


class RandomSentenceSelector(BaseSentenceAggregator):
    """
    Randomly select sentences from available generators.
    Migrated from ensemble_methods/random.py.
    """
    
    def __init__(self, max_repeat: int = 3, name: str = None):
        super().__init__(name or "RandomSentenceSelector")
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
        Randomly select a sentence.
        """
        if not sentences:
            return 0, "", 0.0
        
        best_idx = random.randint(0, len(sentences) - 1)
        return best_idx, sentences[best_idx], 0.0
    
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
        Run iterative random sentence selection.
        """
        available_gens = [g for g in generators]
        convo = ConversationTemplate(example.get("instruction", ""), example.get("input", ""))
        
        last_output = None
        repeat_count = 0
        self.attribution = ModelAttribution()  # Reset attribution for new generation
        
        for rnd in range(1, max_rounds + 1):
            prompt = convo.render()
            
            # Check length limits
            tok = getattr(available_gens[0], "tokenizer", None)
            if tok is not None:
                total_length = tok(prompt, return_tensors="pt").input_ids.size(1)
                if total_length > 32768:
                    logger.warning(f"Early stop: total prompt length {total_length} > 32768")
                    break
            
            # Filter generators by length
            for g in available_gens[:]:
                tok = getattr(g, "tokenizer", None)
                if tok is not None:
                    length = tok(prompt, return_tensors="pt").input_ids.size(1)
                    if length > getattr(tok, 'model_max_length', 32768):
                        logger.info(f"Skip {g.name}: prompt length {length} > max")
                        available_gens.remove(g)
            
            if not available_gens:
                logger.error("No generators available for current prompt length")
                break
            
            # Randomly select a generator
            selected_generator = random.choice(available_gens)
            dicts = convo.render_dict()
            best_output = selected_generator.generate(dicts, max_tokens=max_new_tokens_per_round)
            
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
            
            # Add to conversation
            convo.add_assistant(best_output.text)
            
            # Check for EOS
            if best_output.ended_with_eos:
                logger.info("Early stop: EOS token emitted")
                break
        
        return "".join(convo.assistant_parts)