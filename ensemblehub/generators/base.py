"""
Base generator class for all model generators
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

logger = logging.getLogger("ensemble_inference")


@dataclass
class GenOutput:
    """Output container for model generation"""
    text: str
    ended_with_eos: bool  # Whether EOS token was generated


class BaseGenerator:
    """Abstract base class for any generator (HF, vLLM, etc.)"""
    name: str

    def generate(self, prompt, **kw) -> GenOutput:
        """Abstract method for generating model outputs."""
        raise NotImplementedError

    def batch_generate(self, prompts: List, **kw) -> List[GenOutput]:
        """Abstract method for batch generation."""
        raise NotImplementedError

    def calculate_ppl(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        """Calculate perplexity for the completion given the prompt."""
        logger.warning(f"PPL calculation not implemented for {self.name} or called on BaseGenerator base class.")
        return None

    def calculate_confidence(self, prompt_context_text: str, completion_text: str) -> Optional[float]:
        """Calculate confidence score for the completion given the prompt."""
        logger.warning(f"Confidence calculation not implemented for {self.name} or called on BaseGenerator base class.")
        return None


# Common constants
EOS_TEXT = ""  # Most Qwen / Llama models use empty string as EOS
STOP_TOKENS_TEXT = {"\n"}  # Stop decoding after these tokens


def trim_text(txt: str) -> str:
    """Truncate the text after the last known stop token for cleaner outputs."""
    best_pos = -1
    best_tok = None
    for tok in STOP_TOKENS_TEXT:
        pos = txt.rfind(tok)
        if pos > best_pos:
            best_pos = pos
            best_tok = tok
    if best_pos != -1:
        return txt[: best_pos + len(best_tok)]
    return txt