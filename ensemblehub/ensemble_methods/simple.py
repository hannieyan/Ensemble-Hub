from __future__ import annotations

import logging
from typing import Dict, List
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
    It includes a system prompt, a single user question, and accumulated assistant responses.
    """

    def __init__(self, system_prompt: str, initial_question: str):
        self.system = system_prompt
        self.question = initial_question
        self.assistant_parts: List[str] = []  # Collected assistant responses

    def add_assistant(self, content: str):
        """Append a new assistant response to the prompt context."""
        self.assistant_parts.append(content)

    def render(self) -> str:
        """
        Render the full prompt to be fed into a language model.
        It includes the system message, user input, and accumulated assistant responses.
        """
        lines = [
            f"[SYSTEM] {self.system} [/SYSTEM]",
            f"<user>\n{self.question.strip()}\n</user>",
            f"<assistant>\n" + "\n".join(self.assistant_parts)
        ]
        return "".join(lines)

    def render_dict(self) -> Dict[str, str]:
        """
        Render the full prompt to be fed into a language model.
        It includes the system message, user input, and accumulated assistant responses.
        """
        dicts = {
            "instruction": self.system,
            "input": self.question.strip(),
            "output": "".join(self.assistant_parts)
        }
        return dicts

    def render_list(self) -> list[dict[str, str]]:
        """
        Render the full prompt to be fed into a language model.
        It includes the system message, user input, and accumulated assistant responses.
        """
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.question.strip()},
            {"role": "assistant", "content": "".join(self.assistant_parts)},
        ]

        return messages


# ---------------------------------------------------------------------------
# EnsembleReasoner: multi-model decoding loop with step-wise reward scoring
# ---------------------------------------------------------------------------
def simple_ensemble(
    generators: List[BaseGenerator],
    scorers,
    example,
    max_rounds: int = 500,
    score_threshold: float = -2.0,
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

        # ─── 并行生成 ───────────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=len(available_gens)) as executor:
            outs = list(
                executor.map(
                    lambda g: g.generate(
                        dicts,
                        max_tokens=(16384 if len(available_gens) == 1 else 256),
                    ),
                    available_gens
                )
            )


        # ─── 计算奖励分数（排除自评分）───────────────────────────────────────────
        # 1. 如果多个模型输出结果一样，则全部0分，直接使用第一个模型的输出
        unique_outputs = set(o.text.strip() for o in outs)
        if len(unique_outputs) == 1:
            logger.warning("All generators produced identical outputs. Assigning uniform score of 1.0 to all.")
            scores = [0.0 for _ in outs]
        else:
        # 2. 计算每个模型的分数
            # 准备打分函数（每个模型）
            def score_one(g_o_pair):
                g, o = g_o_pair
                try:
                    other_keys = [k for k in scorers._scorer_cache if g.name not in k]
                    score = scorers.score(prompt, [o.text], keys=other_keys)[0]
                    return g.name, score
                except Exception as e:
                    logger.exception(f"[Score Error] {g.name} failed to score output: {o.text[:80]!r}")
                    return g.name, 0.0  # 出错 fallback

            # 使用线程池并行计算
            with ThreadPoolExecutor(max_workers=len(available_gens)) as executor:
                results = list(executor.map(score_one, zip(available_gens, outs)))

            # 整理结果
            scores = []
            for g_name, score in results:
                scores.append(score)
                model_reward_sum[g_name] += score

        # ─── 打印当前轮次的生成结果 ───────────────────────────────────────────────
        for g, o, s in zip(available_gens, outs, scores):
            other_keys = [k for k in scorers._scorer_cache if g.name not in k]
            log_text = o.text.replace("\n", "\\n").strip()
            logger.info(
                f"[GENERATOR: {g.name}] | [SCORE: {s:.2f}] | [SCORERS: {other_keys}] | TEXT: {log_text}"
            )


        # ─── 选择最佳候选 ───────────────────────────────────────────────────────
        best_idx = int(torch.tensor(scores).argmax())
        best_out = outs[best_idx]
        best_score = scores[best_idx]

        if best_score <= score_threshold:
            logger.info("Stop: best score %.2f < threshold", best_score)
            continue


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

