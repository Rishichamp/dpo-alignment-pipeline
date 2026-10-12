"""
Dataset preparation for DPO (Direct Preference Optimisation) training.

DPO datasets consist of (prompt, chosen, rejected) triplets:
  - prompt:   the input/instruction
  - chosen:   the preferred response (helpful, empathetic, policy-compliant)
  - rejected: the less-preferred response (curt, unhelpful, or policy-violating)

This module loads, validates, and prepares such datasets for TRL's DPOTrainer.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line: {e}")
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def validate_dpo_example(ex: dict) -> Optional[str]:
    """
    Validate a single DPO example. Returns an error message if invalid, else None.

    Required fields: prompt, chosen, rejected (all non-empty strings)
    Additional checks:
      - chosen != rejected (must be genuinely different)
      - chosen should generally be longer/more substantive than rejected
        (not strictly enforced, but logged as a warning if violated)
    """
    for field in ("prompt", "chosen", "rejected"):
        if field not in ex:
            return f"Missing required field: '{field}'"
        if not isinstance(ex[field], str) or not ex[field].strip():
            return f"Field '{field}' must be a non-empty string"

    if ex["chosen"].strip() == ex["rejected"].strip():
        return "chosen and rejected responses are identical"

    return None


def quality_filter_dpo(
    examples: list[dict],
    min_chosen_words: int = 5,
    max_chosen_words: int = 1000,
) -> list[dict]:
    """
    Filter DPO examples for quality:
      1. Schema validation (prompt/chosen/rejected present and valid)
      2. Length bounds on chosen response
      3. Deduplication by prompt
      4. Warn if chosen is not longer than rejected (weak signal of quality)
    """
    kept = []
    seen_prompts = set()
    length_warnings = 0

    for ex in examples:
        error = validate_dpo_example(ex)
        if error:
            logger.debug(f"Skipping invalid example: {error}")
            continue

        chosen_words = len(ex["chosen"].split())
        if chosen_words < min_chosen_words or chosen_words > max_chosen_words:
            continue

        # Deduplication by prompt
        prompt_key = ex["prompt"].strip()[:150]
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)

        # Weak quality signal: chosen should usually be more substantive
        if len(ex["chosen"]) <= len(ex["rejected"]):
            length_warnings += 1

        kept.append({
            "prompt":   ex["prompt"].strip(),
            "chosen":   ex["chosen"].strip(),
            "rejected": ex["rejected"].strip(),
        })

    pct = 100 * len(kept) / max(len(examples), 1)
    logger.info(f"DPO quality filter: kept {len(kept)}/{len(examples)} ({pct:.1f}%)")
    if length_warnings:
        logger.warning(
            f"{length_warnings}/{len(kept)} examples have chosen <= rejected in length — "
            f"verify these reflect genuine quality differences, not just verbosity."
        )
    return kept


def build_dpo_dataset(
    data_path: str,
    eval_path: Optional[str] = None,
    eval_split: float = 0.1,
    seed: int = 42,
    min_chosen_words: int = 5,
    max_chosen_words: int = 1000,
) -> DatasetDict:
    """
    Full DPO dataset pipeline:
      1. Load from JSONL
      2. Validate and quality filter
      3. Split into train/eval (or use provided eval file)
      4. Return as HuggingFace DatasetDict with prompt/chosen/rejected columns
    """
    raw = load_jsonl(data_path)
    cleaned = quality_filter_dpo(raw, min_chosen_words, max_chosen_words)

    if not cleaned:
        raise ValueError("No examples passed the DPO quality filter. Check your data.")

    if eval_path and Path(eval_path).exists():
        eval_raw     = load_jsonl(eval_path)
        eval_cleaned = quality_filter_dpo(eval_raw, min_chosen_words, max_chosen_words)
        return DatasetDict({
            "train": Dataset.from_list(cleaned),
            "eval":  Dataset.from_list(eval_cleaned),
        })
    else:
        full_ds = Dataset.from_list(cleaned)
        split   = full_ds.train_test_split(test_size=eval_split, seed=seed)
        return DatasetDict({"train": split["train"], "eval": split["test"]})


def print_dpo_dataset_stats(datasets: DatasetDict) -> None:
    """Print dataset statistics for verification before training."""
    print("\n" + "=" * 50)
    print("DPO DATASET STATISTICS")
    print("=" * 50)
    for split, ds in datasets.items():
        chosen_lens   = [len(ex["chosen"].split()) for ex in ds]
        rejected_lens = [len(ex["rejected"].split()) for ex in ds]
        avg_chosen    = sum(chosen_lens) / max(len(chosen_lens), 1)
        avg_rejected  = sum(rejected_lens) / max(len(rejected_lens), 1)
        print(f"{split:10} | {len(ds):6,} pairs | "
              f"avg chosen={avg_chosen:.0f}w, rejected={avg_rejected:.0f}w")
        if len(ds) > 0:
            print(f"           | Sample prompt: {ds[0]['prompt'][:100]}...")
    print("=" * 50 + "\n")
