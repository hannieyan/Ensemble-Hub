from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import indent

import torch


from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from types import SimpleNamespace

from v6.data.template import get_template_and_fix_tokenizer
from v6.hparams import DataArguments
from v6.data.converter import AlpacaDatasetConverter
from v6.data.parser import DatasetAttr

# Optional vLLM backend -----------------------------------------------------
try:
    from vllm import LLM, SamplingParams  # type: ignore

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False

EOS_TEXT = ""  # Most Qwen / Llama models use empty string as EOS
STOP_TOKENS_TEXT = {"\n"}  # Stop decoding after these tokens

logger = logging.getLogger("ensemble_inference")

# ---------------------------------------------------------------------------
# Utility: trim text at the last occurrence of stop tokens
# ---------------------------------------------------------------------------

def _trim_text(txt: str) -> str:
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





# ---------------------------------------------------------------------------
# Output container for model generation
# ---------------------------------------------------------------------------

@dataclass
class GenOutput:
    text: str
    ended_with_eos: bool  # Whether EOS token was generated


# ---------------------------------------------------------------------------
# Abstract base class for any generator (HF or vLLM)
# ---------------------------------------------------------------------------

class BaseGenerator:
    name: str

    def generate(self, prompt: str, **kw) -> GenOutput:
        """Abstract method for generating model outputs."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HuggingFace Transformers-based Generator
# ---------------------------------------------------------------------------

class HFGenerator(BaseGenerator):
    def __init__(self, path: str, *, device: str = "auto", dtype: torch.dtype = torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        ).eval()
        self.name = path
        self.device = next(self.model.parameters()).device if device == "auto" else torch.device(device)

        # Optional stop string list
        self.stop_strings = list(STOP_TOKENS_TEXT) + [
            self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        ]

        if "qwen3" in path.lower():
            data_args = DataArguments(template="qwen3")
            self.indent = 2
        elif "qwen2.5" in path.lower():
            data_args = DataArguments(template="qwen")
            self.indent = 2
        elif "deepseek-r1" in path.lower():
            data_args = DataArguments(template="deepseek3")
            self.indent = 1
        else:
            raise NotImplementedError(f"Cannot find model template: {path}")

        dataset_attr = DatasetAttr(
            prompt="instruction",
            query="input",
            response="output",
            load_from="file",
            formatting="alpaca",
            dataset_name="",
        )

        self.converter = AlpacaDatasetConverter(dataset_attr=dataset_attr, data_args=data_args)

        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)

    @torch.inference_mode()
    def generate(self, dicts, *, max_tokens=128, temperature=0.95, top_p=0.7) -> GenOutput:

        converted = self.converter(dicts)
        prompt_msgs = converted["_prompt"]
        response_msgs = converted["_response"]
        messages = prompt_msgs + response_msgs
        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages)

        ids = prompt_ids + response_ids[:-self.indent]

        text = self.tokenizer.decode(ids, skip_special_tokens=False)

        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        cfg = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out = self.model.generate(**ids, generation_config=cfg, tokenizer=self.tokenizer)[0]
        ended = bool(self.tokenizer.eos_token_id in out[len(ids["input_ids"][0]):])
        txt = self.tokenizer.decode(out[len(ids["input_ids"][0]):], skip_special_tokens=False)

        return GenOutput(_trim_text(txt) if not ended else txt, ended)


# ---------------------------------------------------------------------------
# vLLM-based Generator
# ---------------------------------------------------------------------------

class VLLMGenerator(BaseGenerator):
    def __init__(self, path: str):
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed.")
        self._llm = LLM(model=path)
        self._sp = SamplingParams(max_tokens=128, temperature=0.95, top_p=0.7, stop=list(STOP_TOKENS_TEXT))
        self.name = path
        self._eos_text = EOS_TEXT

    @torch.inference_mode()
    def generate(self, prompt: str, *, max_tokens=30, temperature=0.95, top_p=0.7) -> GenOutput:
        self._sp.max_tokens, self._sp.temperature, self._sp.top_p = max_tokens, temperature, top_p
        txt = self._llm.generate([prompt], self._sp)[0].outputs[0].text
        ended = txt.endswith(self._eos_text)
        return GenOutput(_trim_text(txt), ended)


# ---------------------------------------------------------------------------
# GeneratorPool: caches all loaded generators and reward models
# ---------------------------------------------------------------------------

class GeneratorPool:
    _gen_cache: Dict[Tuple[str, str], BaseGenerator] = {}
    _reward_cache: Dict[str, str] = {}

    @classmethod
    def get_generator(cls, path: str, engine: str = "hf", device: Optional[str] = None) -> BaseGenerator:
        """
        Load a generator model (e.g., HF or vLLM) to a specified device (e.g., 'cuda:0', 'cpu').
        """
        key = (engine, path)
        if key not in cls._gen_cache:
            logger.info("[Pool] loading %s (%s)", path, engine)

            resolved_device = device or "auto"
            logger.info(f"→ Assigned to device: {resolved_device}")

            if engine == "hf":
                cls._gen_cache[key] = HFGenerator(path, device=resolved_device)
            elif engine == "vllm":
                cls._gen_cache[key] = VLLMGenerator(path)  # vLLM usually uses global config
            else:
                raise ValueError(f"Unknown engine: {engine}")
        return cls._gen_cache[key]