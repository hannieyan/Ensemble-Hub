from __future__ import annotations

import logging
from typing import Dict, List, Optional
import random
import torch
from concurrent.futures import ThreadPoolExecutor

from ensemblehub.generator import BaseGenerator


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation Template
# ---------------------------------------------------------------------------

class ConversationTemplate:
    """
    A conversation template for constructing dialogue prompts.
    It includes an optional system prompt, a single user question, and accumulated assistant responses.
    """

    def __init__(self, initial_question: str, system_prompt: Optional[str] = None):
        self.system = system_prompt
        self.question = initial_question
        self.assistant_parts: List[str] = []  # Collected assistant responses

    def add_assistant(self, content: str):
        """Append a new assistant response to the prompt context."""
        self.assistant_parts.append(content)

    def render(self) -> str:
        """
        Render the full prompt as a raw string. Includes system prompt if provided.
        """
        lines = []
        if self.system:
            lines.append(f"[SYSTEM] {self.system} [/SYSTEM]")
        lines.append(f"<user>\n{self.question.strip()}\n</user>")
        if self.assistant_parts:
            lines.append(f"<assistant>\n{''.join(self.assistant_parts)}")
        return "".join(lines)

    def render_dict(self) -> Dict[str, str]:
        """
        Render the prompt as a dictionary. Omits 'instruction' if system prompt is None.
        """
        output_dict = {
            "input": self.question.strip(),
            "output": "".join(self.assistant_parts)
        }
        if self.system:
            output_dict["instruction"] = self.system
        return output_dict

    def render_list(self) -> List[Dict[str, str]]:
        """
        Render the prompt as a list of role-based messages.
        """
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.question.strip()})
        if self.assistant_parts:
            messages.append({"role": "assistant", "content": "".join(self.assistant_parts)})
        return messages


# ---------------------------------------------------------------------------
# EnsembleReasoner: multi-model decoding loop with step-wise reward scoring
# ---------------------------------------------------------------------------
def random_ensemble(
    generators: List[BaseGenerator],
    scorers,
    example,
    max_rounds: int = 500,
    score_threshold: float = -2.0,
    max_new_tokens_per_round: int = 256,
) -> str:
    """
    Iteratively decode using multiple generators.
    In each round, the best candidate (with highest reward) is selected and appended.
    Generation stops early if
    1) reward is low,
    2) the chosen candidate emits EOS,
    3) every generator has emitted EOS at least once across rounds, or
    4) the chosen segment has appeared `max_repeat` times (to avoid infinite loops).
    """

    model_reward_sum = {g.name: 0.0 for g in generators}       # 累积得分
    available_gens = [g for g in generators]  # 可用的生成器列表

    convo = ConversationTemplate(example["instruction"], example["input"])

    last_output = None
    repeat_count = 0
    max_repeat = 3  # 连续重复 3 次就停止，你可以调

    # ─── 打印当前已注册的 scorers ──────────────────────────────
    try:
        logger.info("Currently registered scorers:")
        for key, (scorer, weight) in scorers._scorer_cache.items():
            logger.info(f"  → {key} | type: {type(scorer).__name__} | weight: {weight}")
    except Exception as e:
        logger.warning(f"Could not print registered scorers: {e}")


    for rnd in range(1, max_rounds + 1):
        prompt = convo.render()

        # ─── 提前终止：总长度超过32768 ───────────────────────────────────────
        tok = getattr(available_gens[0], "tokenizer", None)
        if tok is not None:
            total_length = tok(convo.render(), return_tensors="pt").input_ids.size(1)
            if total_length > 32768:
                logger.warning(f"Early stop: total prompt length {total_length} > 32768")
                break

        # ─── 过滤长度超限的 generator ───────────────────────────────────────
        for g in available_gens:
            tok = getattr(g, "tokenizer", None)
            if tok is not None:
                length = tok(prompt, return_tensors="pt").input_ids.size(1)
                if length > tok.model_max_length:
                    logger.info("Skip %s: prompt length %d > max %d",
                                g.name, length, tok.model_max_length)
                    available_gens.remove(g)
                    model_reward_sum.pop(g.name, None)

        if not available_gens:
            logger.error("No generators available for current prompt length; stopping early.")
            break

        dicts = convo.render_dict()

        # ─── 随机选择一个生成器进行输出 ───────────────────────────────────────────
        selected_generator = random.choice(available_gens)
        best_out = selected_generator.generate(dicts, max_tokens=max_new_tokens_per_round)

        # ─── 重复文本检测 ─────────────────────────────────────────────────
        if best_out.text == last_output:
            repeat_count += 1
            if repeat_count >= max_repeat:
                logger.info("Early stop: same output repeated %d times consecutively → \"%s\"",
                            repeat_count, best_out.text.strip())
                break
        else:
            repeat_count = 1  # 当前这个算第一次
            last_output = best_out.text


        convo.add_assistant(best_out.text)
        # logger.info("Updated conversation:\n%s", convo.render())

        # 若本轮最佳候选已输出 EOS，也可直接终止
        if best_out.ended_with_eos:
            logger.info("Early stop: EOS token emitted by best model")
            break

    return "".join(convo.assistant_parts)

