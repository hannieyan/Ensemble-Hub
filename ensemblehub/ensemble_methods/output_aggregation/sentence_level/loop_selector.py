"""
Loop sentence selection (migrated from loop.py).
"""

import logging
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union
import ray

from .base import BaseSentenceAggregator, ModelAttribution

logger = logging.getLogger(__name__)


class LoopSelector(BaseSentenceAggregator):
    """
    Loop selection of sentences from generators.
    Migrated from ensemble_methods/loop.py.
    """
    
    # Class-level round counter shared across all instances
    _global_round = 0
    
    def __init__(self, max_repeat: int = 3, name: str = None):
        super().__init__(name or "LoopSelector")
        self.max_repeat = max_repeat

    def aggregate_generation(
        self,
        generators: List,
        examples: List[Union[str, List[Dict]]],  # 批处理输入
        max_rounds: int = 500,
        max_tokens: int = 16384,
        max_new_tokens_per_round: int = 500,
        is_chat: bool = False,
        **kwargs
    ) -> List[str]:  # 返回列表

        if len(generators) == 1:
            max_rounds = 1  # If only one generator, no need for multiple rounds
            max_new_tokens_per_round = max_tokens  # Use all tokens in one go

        batch_size = len(examples)
        available_gens = [g for g in generators]

        # Initialize tracking for each example
        last_outputs = [None] * batch_size
        repeat_counts = [0] * batch_size
        finished = [False] * batch_size
        results = [""] * batch_size

        # Initialize conversations for each example
        if is_chat:
            conversations = [deepcopy(example) for example in examples]
        else:
            conversations = examples.copy()

        attributions = [ModelAttribution() for _ in range(batch_size)]

        for rnd in range(1, max_rounds + 1):
            # Skip if all examples are finished
            if all(finished):
                logger.info("All examples finished generation")
                break

            # Prepare active examples (not finished)
            active_indices = [i for i in range(batch_size) if not finished[i]]

            # Check if any sample exceeds max_tokens and mark as finished
            if available_gens:
                # Get token counts for all active results at once
                active_results = [results[idx] for idx in active_indices]
                token_counts = ray.get(available_gens[0].count_tokens.remote(active_results))
                
                # Check each result and mark finished if exceeds max_tokens
                for i, (active_idx, token_count) in enumerate(zip(active_indices[:], token_counts)):
                    if token_count >= max_tokens:
                        logger.info(f"Example {active_idx}: Reached max_tokens ({token_count}/{max_tokens})")
                        finished[active_idx] = True
                        active_indices.remove(active_idx)

            # Update active conversations after filtering
            active_conversations = [conversations[i] for i in active_indices]
            
            if not active_indices:
                logger.info("All examples finished due to length limits")
                break

            # Round-robin selection using global counter
            LoopSelector._global_round += 1
            gen_idx = (LoopSelector._global_round - 1) % len(available_gens)
            selected_generator = available_gens[gen_idx]
            model_short = ray.get(selected_generator.get_model_name.remote()).split('/')[-1]
            logger.info(f"🔄 Round {rnd} (Global: {LoopSelector._global_round}): Using {model_short} (index {gen_idx + 1}/{len(available_gens)})")

            # Prepare generation parameters
            requested_continue = kwargs.get("continue_final_message", None)
            if requested_continue is None:
                continue_flag = False
                if is_chat and hasattr(selected_generator, "default_continue_final_message"):
                    method = selected_generator.default_continue_final_message
                    if hasattr(method, "remote"):
                        continue_flag = ray.get(method.remote())
                    else:
                        continue_flag = method()
            else:
                continue_flag = requested_continue

            gen_kwargs = {
                "max_tokens": max_new_tokens_per_round,
                "temperature": kwargs.get("temperature", 0.95),
                "top_p": kwargs.get("top_p", 0.7),
                "is_chat": is_chat,  # We've already applied chat template
                "continue_final_message": continue_flag,
            }
            if "seed" in kwargs:
                gen_kwargs["seed"] = kwargs["seed"]
            if "stop_strings" in kwargs:
                gen_kwargs["stop_strings"] = kwargs["stop_strings"]

            # Generate for all active examples
            outputs = ray.get(selected_generator.generate.remote(active_conversations, **gen_kwargs))

            # Ensure outputs is a list
            if not isinstance(outputs, list):
                outputs = [outputs]

            # Extract text from outputs
            output_texts = []
            ended_with_eos = []
            for output in outputs:
                if hasattr(output, 'text'):
                    output_texts.append(output.text)
                    ended_with_eos.append(getattr(output, 'ended_with_eos', False))
                else:
                    output_texts.append(output)
                    ended_with_eos.append(False)

            # Process results for each active example
            for idx, (active_idx, output_text, eos) in enumerate(zip(active_indices, output_texts, ended_with_eos)):
                # Record attribution
                model_name = ray.get(selected_generator.get_model_name.remote())
                attributions[active_idx].add_segment(output_text, model_name, rnd)

                # Check repetition
                if output_text == last_outputs[active_idx]:
                    repeat_counts[active_idx] += 1
                    if repeat_counts[active_idx] >= self.max_repeat:
                        logger.info(f"Example {active_idx}: Early stop - repeated {repeat_counts[active_idx]} times")
                        finished[active_idx] = True
                        continue
                else:
                    repeat_counts[active_idx] = 0
                    last_outputs[active_idx] = output_text

                # Update conversation/text
                if is_chat:
                    # Add assistant response to conversation
                    conversations[active_idx].append({
                        "role": "assistant",
                        "content": output_text
                    })
                else:
                    # Append to text
                    conversations[active_idx] += output_text

                # Accumulate results
                results[active_idx] += output_text

                # Check EOS
                if eos:
                    logger.info(f"Example {active_idx}: Early stop - EOS token")
                    finished[active_idx] = True

        return results

    # Helper method to extract all assistant responses from a conversation
    def extract_assistant_responses(self, conversation: List[Dict]) -> str:
        """Extract and concatenate all assistant responses from a conversation."""
        assistant_parts = []
        for message in conversation:
            if message.get("role") == "assistant":
                assistant_parts.append(message.get("content", ""))
        return "".join(assistant_parts)
