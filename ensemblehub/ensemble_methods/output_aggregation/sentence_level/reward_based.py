"""
Reward-based sentence selection (migrated from simple.py).
"""

import logging
from typing import List, Dict, Any, Tuple
import torch
from concurrent.futures import ThreadPoolExecutor

from .base import BaseSentenceAggregator
from ....conversation import ConversationTemplate

logger = logging.getLogger(__name__)


class RewardBasedSelector(BaseSentenceAggregator):
    """
    Select sentences based on reward scores from scorer models.
    This is the main method migrated from ensemble_methods/simple.py.
    """
    
    def __init__(self, 
                 exclude_self_scoring: bool = True,
                 max_repeat: int = 3,
                 name: str = None):
        super().__init__(name or "RewardBasedSelector")
        self.exclude_self_scoring = exclude_self_scoring
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
        """
        if len(sentences) == 1:
            return 0, sentences[0], 0.0
        
        # Handle identical outputs
        unique_outputs = set(s.strip() for s in sentences)
        if len(unique_outputs) == 1:
            logger.warning("All models produced identical outputs")
            return 0, sentences[0], 0.0
        
        # Score each sentence
        def score_one(g_s_pair):
            g, s = g_s_pair
            try:
                if self.exclude_self_scoring and scorers:
                    # Exclude the generator's own scorer
                    other_keys = [k for k in scorers._scorer_cache if g.name not in k]
                    score = scorers.score(prompt, [s], keys=other_keys)[0]
                else:
                    score = 0.0 if not scorers else scorers.score(prompt, [s])[0]
                return score
            except Exception as e:
                logger.error(f"Error scoring sentence from {g.name}: {e}")
                return 0.0
        
        # Use thread pool for parallel scoring
        with ThreadPoolExecutor(max_workers=len(generators)) as executor:
            scores = list(executor.map(score_one, zip(generators, sentences)))
        
        # Log results
        for g, s, score in zip(generators, sentences, scores):
            log_text = s.replace("\n", "\\n").strip()
            logger.info(f"[{g.name}] Score: {score:.2f} | Text: {log_text}")
        
        # Select best
        best_idx = int(torch.tensor(scores).argmax())
        return best_idx, sentences[best_idx], scores[best_idx]
    
    def aggregate_generation(
        self,
        generators: List,
        scorers,
        example: Dict[str, Any],
        max_rounds: int = 500,
        score_threshold: float = -2.0,
        **kwargs
    ) -> str:
        """
        Run iterative reward-based sentence aggregation.
        This is the main logic migrated from simple_ensemble.
        """
        available_gens = [g for g in generators]
        convo = ConversationTemplate(example.get("instruction", ""), example.get("input", ""))
        
        last_output = None
        repeat_count = 0
        
        # Log scorers
        try:
            logger.info("Currently registered scorers:")
            for key, (scorer, weight) in scorers._scorer_cache.items():
                logger.info(f"  → {key} | type: {type(scorer).__name__} | weight: {weight}")
        except Exception as e:
            logger.warning(f"Could not print registered scorers: {e}")
        
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
            
            # Generate from all models
            dicts = convo.render_dict()
            
            with ThreadPoolExecutor(max_workers=len(available_gens)) as executor:
                outputs = list(
                    executor.map(
                        lambda g: g.generate(
                            dicts,
                            max_tokens=(16384 if len(available_gens) == 1 else 256),
                        ),
                        available_gens
                    )
                )
            
            # Select best sentence
            sentences = [o.text for o in outputs]
            best_idx, best_sentence, best_score = self.select_best_sentence(
                sentences, available_gens, prompt, scorers
            )
            
            # Check score threshold
            if best_score <= score_threshold:
                logger.info(f"Stop: best score {best_score:.2f} < threshold {score_threshold}")
                continue
            
            # Check for repetition
            if best_sentence == last_output:
                repeat_count += 1
                if repeat_count >= self.max_repeat:
                    logger.info(f"Early stop: same output repeated {repeat_count} times")
                    break
            else:
                repeat_count = 1
                last_output = best_sentence
            
            # Add to conversation
            convo.add_assistant(best_sentence)
            
            # Check for EOS
            best_output = outputs[best_idx]
            if best_output.ended_with_eos:
                logger.info("Early stop: EOS token emitted by best model")
                break
        
        return "".join(convo.assistant_parts)