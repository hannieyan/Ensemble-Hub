"""
Ensemble inference – cached model pool & clean conversation template
===================================================================

🔑 **What’s new**
1. **ModelPool (singleton)** caches every HF/vLLM generator and the reward model
   the first time they are requested → later calls reuse them, so repeated
   `run_ensemble()` invocations are *instant*.
2. **ConversationTemplate** – every round is rendered with a fixed, minimal
   format to avoid stray text like “600 words”. History turns are stored in
   `[TURN i]` blocks.
3. Added optional `accumulate_context` flag (default **True**) to control
   whether previously chosen segments are fed back into the next prompt.
4. Still supports unlimited models & EOS‑based early stopping.

Quick usage
-----------
```python
from ensemble_inference import run_ensemble
models = [
    {"path": ".../DeepSeek-R1-Distill-Qwen-1.5B",  "engine": "hf"},
    {"path": ".../DeepSeek-R1-Distill-Qwen-7B",   "engine": "hf"},
    {"path": ".../DeepSeek-R1-Distill-Qwen-14B",  "engine": "vllm"},
]
ans1 = run_ensemble("Explain gradient accumulation.", model_specs=models)
ans2 = run_ensemble("What is RLHF?",                model_specs=models)  # ⬅ reused, no reload
```

Implementation notes
--------------------
* **ModelPool.get_generator() / get_reward()** return already‑loaded instances.
* Conversation prompt:
  ```
  [SYSTEM] You are a helpful assistant. [/SYSTEM]
  [TURN 1]
  <user>
  ...
  </user>
  <assistant>
  ...
  </assistant>
  ...
  [TURN n]
  <user>
  Current question…
  </user>
  <assistant>
  ```
  Last `<assistant>` tag is left open for the model to fill.
* `_clean()` strips trivial length meta like “\d+ words” at line starts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
)

# Optional vLLM backend ------------------------------------------------------
try:
    from vllm import LLM, SamplingParams  # type: ignore

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging / constants
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ensemble_inference")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EOS_TEXT = ""  # most Qwen / Llama models use empty string as EOS
STEP_TOKEN = "<extra_0>"
SYSTEM_PROMPT = "You are a helpful assistant."
STOP_TOKENS_TEXT = {".", "\n"}  # trimming convenience

# ---------------------------------------------------------------------------
# Conversation template & sanitisation
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip trivial meta lines like "600 words"."""
    return re.sub(r"^\s*\d+\s*words\b.*(?:\n|$)", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()


class ConversationTemplate:
    def __init__(self, system_prompt: str, initial_question: str):
        self.system = system_prompt
        self.turns: List[Tuple[str, str]] = [("user", initial_question)]  # (role, content)

    def add_assistant(self, content: str):
        self.turns[-1] = (self.turns[-1][0], self.turns[-1][1])  # ensure last user ends
        self.turns.append(("assistant", content))

    def new_turn(self, question: str):
        self.turns.append(("user", question))

    def render(self) -> str:
        prompt_lines = [f"[SYSTEM] {self.system} [/SYSTEM]"]
        for idx, (role, content) in enumerate(self.turns, 1):
            if role == "user":
                prompt_lines.append(f"[TURN {idx}]\n<user>\n{content}\n</user>")
            else:  # assistant
                prompt_lines.append(f"<assistant>\n{content}\n</assistant>")
        prompt_lines.append("<assistant>\n")  # leave open for next gen
        return "\n".join(prompt_lines)

# ---------------------------------------------------------------------------
# Helper: trim at first stop token
# ---------------------------------------------------------------------------

def _trim_text(txt: str) -> str:
    for tok in STOP_TOKENS_TEXT:
        pos = txt.find(tok)
        if pos != -1:
            return txt[: pos + len(tok)]
    return txt

# ---------------------------------------------------------------------------
# Step‑probability helper for reward model
# ---------------------------------------------------------------------------

def _step_rewards(logits: torch.Tensor, mask: torch.Tensor):
    probs = F.softmax(logits, dim=-1) * mask.unsqueeze(-1)
    arr: List[List[float]] = []
    for sample in probs:
        pos = sample[sample != 0].view(-1, 2)[:, 1]
        arr.append(pos.cpu().tolist())
    return arr

# ---------------------------------------------------------------------------
# Generator output container
# ---------------------------------------------------------------------------

@dataclass
class GenOutput:
    text: str
    ended_with_eos: bool

# ---------------------------------------------------------------------------
# Generators (HF & vLLM)
# ---------------------------------------------------------------------------

class BaseGenerator:
    name: str

    def generate(self, prompt: str, **kw) -> GenOutput:  # pragma: no cover abstract
        raise NotImplementedError


@dataclass
class HFGenerator(BaseGenerator):
    tokenizer: PreTrainedTokenizerBase
    model: AutoModelForCausalLM

    @classmethod
    def load(cls, path: str, *, dtype: torch.dtype = torch.float16):
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto", trust_remote_code=True).eval()
        inst = cls(tok, mod)
        inst.name = path
        return inst

    @torch.inference_mode()
    def generate(self, prompt: str, *, max_tokens=128, temperature=0.95, top_p=0.7) -> GenOutput:
        ids = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        cfg = GenerationConfig(do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        out = self.model.generate(**ids, generation_config=cfg)[0]
        ended = bool(out[-1] == self.tokenizer.eos_token_id)
        txt = self.tokenizer.decode(out[len(ids["input_ids"][0]) :], skip_special_tokens=True)
        return GenOutput(_trim_text(_clean(txt)), ended)


class VLLMGenerator(BaseGenerator):
    def __init__(self, path: str):
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vllm not installed")
        self._llm = LLM(model=path)
        self._sp = SamplingParams(max_tokens=128, temperature=0.95, top_p=0.7, stop=list(STOP_TOKENS_TEXT))
        self.name = path
        self._eos_text = EOS_TEXT

    @torch.inference_mode()
    def generate(self, prompt: str, *, max_tokens=128, temperature=0.95, top_p=0.7) -> GenOutput:
        self._sp.max_tokens, self._sp.temperature, self._sp.top_p = max_tokens, temperature, top_p
        txt = self._llm.generate([prompt], self._sp)[0].outputs[0].text
        ended = txt.endswith(self._eos_text)
        return GenOutput(_trim_text(_clean(txt)), ended)

# ---------------------------------------------------------------------------
# ModelPool: caching layer
# ---------------------------------------------------------------------------

class ModelPool:
    _gen_cache: Dict[Tuple[str, str], BaseGenerator] = {}
    _reward_cache: Dict[str, "PRMScorer"] = {}

    @classmethod
    def get_generator(cls, path: str, engine: str = "hf") -> BaseGenerator:
        key = (engine, path)
        if key not in cls._gen_cache:
            logger.info("[Pool] loading %s (%s)", path, engine)
            if engine == "hf":
                cls._gen_cache[key] = HFGenerator.load(path)
            elif engine == "vllm":
                cls._gen_cache[key] = VLLMGenerator(path)
            else:
                raise ValueError(engine)
        return cls._gen_cache[key]

    @classmethod
    def get_reward(cls, path: str) -> "PRMScorer":
        if path not in cls._reward_cache:
            logger.info("[Pool] loading reward model %s", path)
            cls._reward_cache[path] = PRMScorer(path)
        return cls._reward_cache[path]

# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class PRMScorer:
    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.mod = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        self.sep_id = self.tok.encode(STEP_TOKEN)[0]

    @torch.inference_mode()
    def score(self, question: str, answer: str) -> float:
        if not answer.endswith(STEP_TOKEN):
            answer = answer + STEP_TOKEN
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        convo = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = self.tok(convo, return_tensors="pt").input_ids.to(DEVICE)
        mask = ids == self.sep_id
        probs = _step_rewards(self.mod(ids).logits, mask)[0]
        return float(sum(probs) / len(probs) * 10.0) if probs else 0.0

# ---------------------------------------------------------------------------
# Ensemble reasoner
# ---------------------------------------------------------------------------

@dataclass
class EnsembleReasoner:
    generators: List[BaseGenerator]
    scorer: PRMScorer
    max_rounds: int = 5
    score_threshold: float = 0.5
    accumulate_context: bool = True

    def __call__(self, question: str) -> str:
        convo = ConversationTemplate(SYSTEM_PROMPT, question)
        collected: List[str] = []

        for rnd in range(1, self.max_rounds + 1):
            prompt = convo.render()
            outs = [g.generate(prompt) for g in self.generators]
            segs = [o.text for o in outs]
            scores = [self.scorer.score(question, s) for s in segs]

            for g, t, s in zip(self.generators, segs, scores):
                logger.info("→ %s | %.2f | %s", g.name, s, t.replace("\n", "\\n"))

            best_idx = int(torch.tensor(scores).argmax())
            best_out = outs[best_idx]
            best_score = scores[best_idx]

            if best_score < self.score_threshold:
                logger.info("Stop: best score %.2f < threshold", best_score)
                break

            collected.append(best_out.text)
            if self.accumulate_context:
                convo.add_assistant(best_out.text)
            if best_out.ended_with_eos:
                logger.info("Early stop: EOS token emitted")
                break

        return " ".join(collected)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ensemble(
    question: str,
    *,
    model_specs: Optional[List[Dict]] = None,
    max_rounds: int = 5,
    score_threshold: float = 0.5,
    accumulate_context: bool = True,
) -> str:
    if model_specs is None:
        model_specs = [
            {"path": "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf"},
            {"path": "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B",  "engine": "hf"},
        ]

    gens = [ModelPool.get_generator(spec["path"], spec.get("engine", "hf")) for spec in model_specs]
    scorer = ModelPool.get_reward("/root/autodl-tmp/Qwen2.5-Math-PRM-7B")

    reasoner = EnsembleReasoner(gens, scorer, max_rounds, score_threshold, accumulate_context)
    return reasoner(question)

