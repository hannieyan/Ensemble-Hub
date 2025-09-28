#!/usr/bin/env python3
"""Utility script to verify DeepSeek tokenizer availability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the tokenizer for DeepSeek models")
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Hugging Face model id or local directory",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Force Transformers to use local files only (no download)",
    )
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            local_files_only=args.local_only,
        )
    except Exception as exc:  # broad to surface nested JSON errors too
        print("[ERROR] Failed to load tokenizer:", exc, file=sys.stderr)
        if not args.local_only:
            print(
                "Hint: use --local-only to confirm local cache, or download via\n"
                "      huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                file=sys.stderr,
            )
        sys.exit(1)

    vocab_size = getattr(tokenizer, "vocab_size", "unknown")
    tokenizer_dir = Path(tokenizer.name_or_path)

    print("Tokenizer loaded successfully.")
    print(f"- name_or_path: {tokenizer.name_or_path}")
    print(f"- vocab_size: {vocab_size}")
    if tokenizer_dir.exists():
        config_file = tokenizer_dir / "tokenizer_config.json"
        if config_file.exists():
            print(f"- tokenizer_config.json: {config_file} ({config_file.stat().st_size} bytes)")
        merges = tokenizer_dir / "merges.txt"
        if merges.exists():
            print(f"- merges.txt: {merges} ({merges.stat().st_size} bytes)")
        tokenizer_model = tokenizer_dir / "tokenizer.model"
        if tokenizer_model.exists():
            print(f"- tokenizer.model: {tokenizer_model} ({tokenizer_model.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
