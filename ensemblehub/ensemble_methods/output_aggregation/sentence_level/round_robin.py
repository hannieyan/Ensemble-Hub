"""
Round-robin sentence selection (migrated from loop.py).
"""

import logging
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from .base import BaseSentenceAggregator, ModelAttribution

logger = logging.getLogger(__name__)


class RoundRobinSelector(BaseSentenceAggregator):
    """
    Round-robin selection of sentences from generators.
    Migrated from ensemble_methods/loop.py.
    """
    
    def __init__(self, max_repeat: int = 3, name: str = None):
        super().__init__(name or "RoundRobinSelector")
        self.max_repeat = max_repeat

    def aggregate_generation(
        self,
        generators: List,
        scorers,
        examples: List[Union[str, List[Dict]]],  # 批处理输入
        max_rounds: int = 500,
        max_new_tokens_per_round: int = 256,
        is_chat: bool = False,
        **kwargs
    ) -> List[str]:  # 返回列表
        """
        Run iterative round-robin sentence selection with batch support.
        Args:
            generators: List,
            scorers: Scorers for evaluation (not used in round-robin)
            examples: List of inputs (strings for completion, list of dicts for chat)
            max_rounds: Maximum number of rounds to run
            max_new_tokens_per_round: Maximum tokens to generate in each round
            is_chat: Whether the input is a chat conversation (list of dicts)
        Returns:
            List of generated strings, one for each input
        """


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
            active_conversations = [conversations[i] for i in active_indices]

            # Check length limits and filter generators
            available_gens_copy = available_gens.copy()
            for g in available_gens_copy[:]:
                tok = getattr(g, "tokenizer", None)
                if tok is not None:
                    # Check max length for any active conversation
                    max_length = 0
                    for conv in active_conversations:
                        if is_chat:
                            prompt = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                        else:
                            prompt = conv
                        length = len(tok.encode(prompt))
                        max_length = max(max_length, length)

                    if max_length > getattr(tok, 'model_max_length', 32768):
                        logger.info(f"Skip {g.name}: prompt length {max_length} > max")
                        available_gens_copy.remove(g)

            if not available_gens_copy:
                logger.error("No generators available for current prompt length")
                break

            # Round-robin selection
            gen_idx = (rnd - 1) % len(available_gens_copy)
            selected_generator = available_gens_copy[gen_idx]
            model_short = getattr(selected_generator, 'model_path', selected_generator.name).split('/')[-1]
            logger.info(f"🔄 Round {rnd}: Using {model_short} (index {gen_idx + 1}/{len(available_gens_copy)})")

            # Prepare generation parameters
            gen_kwargs = {
                "max_tokens": kwargs.get("max_tokens", max_new_tokens_per_round),
                "temperature": kwargs.get("temperature", 0.95),
                "top_p": kwargs.get("top_p", 0.7),
                "is_chat": is_chat,  # We've already applied chat template
            }
            if "seed" in kwargs:
                gen_kwargs["seed"] = kwargs["seed"]
            if "stop_strings" in kwargs:
                gen_kwargs["stop_strings"] = kwargs["stop_strings"]

            # Generate for all active examples
            outputs = selected_generator.generate(conversations, **gen_kwargs)

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
                model_name = getattr(selected_generator, 'model_path', selected_generator.name)
                attributions[active_idx].add_segment(output_text, model_name, rnd)

                # Check repetition
                if output_text == last_outputs[active_idx]:
                    repeat_counts[active_idx] += 1
                    if repeat_counts[active_idx] >= self.max_repeat:
                        logger.info(f"Example {active_idx}: Early stop - repeated {repeat_counts[active_idx]} times")
                        finished[active_idx] = True
                        continue
                else:
                    repeat_counts[active_idx] = 1
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