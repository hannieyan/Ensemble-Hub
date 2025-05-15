from __future__ import annotations

import logging
from typing import Dict, List
import torch

from ensemblehub.generator import BaseGenerator


# ---------------------------------------------------------------------------
# Logging / constants
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ensemble_inference")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = "Solve the following math problem step by step. Write your reasoning clearly using LaTeX. Box the final answer using \\boxed{}."


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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.question.strip()},
            {"role": "assistant", "content": "".join(self.assistant_parts)},
        ]

        return messages






# ---------------------------------------------------------------------------
# EnsembleReasoner: multi-model decoding loop with step-wise reward scoring
# ---------------------------------------------------------------------------

class EnsembleReasoner:
    def __init__(self, generators: List[BaseGenerator], scorers, max_rounds: int = 500,
                 score_threshold: float = -2.0, accumulate_context: bool = True):
        self.generators = generators
        self.scorers = scorers
        self.max_rounds = max_rounds
        self.score_threshold = score_threshold
        self.accumulate_context = accumulate_context


    def __call__(self, example) -> str:
        """
        Iteratively decode using multiple generators.
        In each round, the best candidate (with highest reward) is selected and appended.
        Generation stops early if
        1) reward is low,
        2) the chosen candidate emits EOS,
        3) every generator has emitted EOS at least once across rounds, or
        4) the chosen segment has appeared `max_repeat` times (to avoid infinite loops).
        """

        model_reward_sum = {g.name: 0.0 for g in self.generators}       # 累积得分
        available_gens = [g for g in self.generators]  # 可用的生成器列表

        convo = ConversationTemplate(SYSTEM_PROMPT, example["input"])

        segment_counter: Dict[str, int] = {}                  # 统计已选片段出现次数
        max_repeat = 5                                        # 片段允许重复次数阈值


        # ─── 打印当前已注册的 scorers ──────────────────────────────
        try:
            logger.info("Currently registered scorers:")
            for key, (scorer, weight) in self.scorers._scorer_cache.items():
                logger.info(f"  → {key} | type: {type(scorer).__name__} | weight: {weight}")
        except Exception as e:
            logger.warning(f"Could not print registered scorers: {e}")


        for rnd in range(1, self.max_rounds + 1):
            prompt = convo.render()

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
            # ───────────────────────────────────────────────────────────────────

            dicts = convo.render_dict()

            # ─── 并行生成 ───────────────────────────────────────────────────────
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(available_gens)) as executor:
                outs = list(
                    executor.map(
                        lambda g: g.generate(
                            dicts,
                            max_tokens=(16384 if len(available_gens) == 1 else 32),
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
                scores = []
                for g, o in zip(available_gens, outs):
                    other_keys = [k for k in self.scorers._scorer_cache if g.name not in k]
                    one_score = self.scorers.score(prompt, [o.text], keys=other_keys)[0]
                    scores.append(one_score)
                    model_reward_sum[g.name] += one_score


            # ─── 打印当前轮次的生成结果 ───────────────────────────────────────────────
            for g, o, s in zip(available_gens, outs, scores):
                other_keys = [k for k in self.scorers._scorer_cache if g.name not in k]
                log_text = o.text.replace("\n", "\\n").strip()
                logger.info(
                    f"[GENERATOR: {g.name}] | [SCORE: {s:.2f}] | [SCORERS: {other_keys}] | TEXT: {log_text}"
                )


            # ─── 选择最佳候选 ───────────────────────────────────────────────────────
            best_idx = int(torch.tensor(scores).argmax())
            best_out = outs[best_idx]
            best_score = scores[best_idx]

            if best_score <= self.score_threshold:
                logger.info("Stop: best score %.2f < threshold", best_score)
                continue


            # ─── 重复文本检测 ─────────────────────────────────────────────────
            segment_counter[best_out.text] = segment_counter.get(best_out.text, 0) + 1
            if segment_counter[best_out.text] >= max_repeat:
                logger.info("Early stop: segment repeated %d times → \"%s\"",
                            segment_counter[best_out.text], best_out.text)
                break
            # ──────────────────────────────────────────────────────────────────


            convo.add_assistant(best_out.text)
            # logger.info("Updated conversation:\n%s", convo.render())

            # 若本轮最佳候选已输出 EOS，也可直接终止
            if best_out.ended_with_eos:
                logger.info("Early stop: EOS token emitted by best model")
                break

            # # ─── 通过 model_reward_sum 判断是否禁用该模型 ──────
            # # 1. 直接禁用 reward 累计 < 0 的模型
            # to_disable = []
            # for g in available_gens:
            #     if model_reward_sum[g.name] < 0:
            #         logger.info(f"[DISABLE] {g.name} has negative cumulative reward ({model_reward_sum[g.name]:.2f}); removing.")
            #         to_disable.append(g)
            #
            # # 从 available_gens 中移除这批
            # for g in to_disable:
            #     available_gens.remove(g)
            #     model_reward_sum.pop(g.name, None)
            #
            # # 2. 在剩下模型中判断 reward 占比
            # positive_total = sum(r for r in model_reward_sum.values() if r > 0) + 1e-8
            # to_disable = []
            # for g in available_gens:
            #     r = model_reward_sum[g.name]
            #     if r > 0 and (r / positive_total) >= 0.20:
            #         logger.info(f"[DISABLE] {g.name} reached 20% of total reward; no longer used.")
            #         to_disable.append(g)
            #
            # # 再次禁用
            # for g in to_disable:
            #     available_gens.remove(g)
            #     model_reward_sum.pop(g.name, None)
            # # ────────────────────────────────────────────────────────────────

        return "".join(convo.assistant_parts)


def is_valid_segment(text: str, min_len: int = 5) -> bool:
    import string
    import re

    # 清除特殊 token
    text = text.replace("<｜end▁of▁sentence｜>", "").replace("<|im_end|>", "")

    stripped = text.strip()
    if len(stripped) < min_len:
        return False
    if not stripped or stripped.isspace():
        return False
    if all(c in string.punctuation + string.whitespace for c in stripped):
        return False
    cleaned = re.sub(r"\s+", "", stripped)
    if len(set(cleaned)) <= 2:
        return False
    if re.search(r"(.)\1{4,}", cleaned):
        return False
    return True

