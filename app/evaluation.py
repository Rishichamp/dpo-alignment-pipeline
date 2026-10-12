"""
Evaluation module for the DPO Alignment Pipeline.

DPO evaluation focuses on PREFERENCE ACCURACY rather than perplexity:
  - Implicit reward margin: log π_θ(chosen)/π_ref(chosen) - log π_θ(rejected)/π_ref(rejected)
    A positive margin means the model now prefers "chosen" more than the reference did.
  - Win rate: fraction of eval examples where the policy assigns higher
    log-probability to chosen than rejected.
  - LLM-as-judge: side-by-side comparison of pre-DPO vs post-DPO generations
    on held-out prompts.

Usage:
    python scripts/evaluate_dpo.py --compare
"""

import json
import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from app.config import settings
from app.data_prep import load_jsonl

logger = logging.getLogger(__name__)


# ── Sequence log-probability ──────────────────────────────────────

def sequence_log_prob(model, tokeniser, prompt: str, response: str) -> float:
    """
    Compute the sum of log-probabilities the model assigns to `response`
    tokens, conditioned on `prompt`. This is the core quantity DPO optimises.
    """
    full_text   = prompt + response
    prompt_ids  = tokeniser(prompt, return_tensors="pt").input_ids
    full_ids    = tokeniser(full_text, return_tensors="pt").input_ids.to(model.device)

    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits  = outputs.logits[:, :-1, :]          # predict token t from tokens < t
        targets = full_ids[:, 1:]                     # shift

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        # Only sum log-probs for the RESPONSE portion (exclude prompt tokens)
        response_log_probs = token_log_probs[:, prompt_len - 1:]
        return response_log_probs.sum().item()


# ── Win rate / reward margin ──────────────────────────────────────

def compute_preference_metrics(
    policy_model, ref_model, tokeniser, eval_examples: list[dict], max_examples: int = 50
) -> dict:
    """
    For each (prompt, chosen, rejected) triplet, compute:
      - policy_margin = logp_policy(chosen) - logp_policy(rejected)
      - ref_margin     = logp_ref(chosen) - logp_ref(rejected)
      - implicit_reward_delta = policy_margin - ref_margin

    Returns aggregate win rate and average implicit reward delta.
    """
    eval_examples = eval_examples[:max_examples]
    policy_wins, total = 0, 0
    reward_deltas = []

    for ex in eval_examples:
        prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

        policy_chosen   = sequence_log_prob(policy_model, tokeniser, prompt, chosen)
        policy_rejected = sequence_log_prob(policy_model, tokeniser, prompt, rejected)
        policy_margin   = policy_chosen - policy_rejected

        if ref_model is not None:
            ref_chosen   = sequence_log_prob(ref_model, tokeniser, prompt, chosen)
            ref_rejected = sequence_log_prob(ref_model, tokeniser, prompt, rejected)
            ref_margin   = ref_chosen - ref_rejected
            reward_deltas.append(policy_margin - ref_margin)

        if policy_margin > 0:
            policy_wins += 1
        total += 1

    win_rate = policy_wins / max(total, 1)
    avg_delta = sum(reward_deltas) / len(reward_deltas) if reward_deltas else None

    return {
        "n_examples": total,
        "win_rate": round(win_rate, 4),
        "policy_wins": policy_wins,
        "avg_implicit_reward_delta": round(avg_delta, 4) if avg_delta is not None else None,
        "interpretation": (
            f"Policy prefers 'chosen' over 'rejected' in {policy_wins}/{total} cases "
            f"({100*win_rate:.1f}%). "
            + (f"Average improvement in preference margin vs reference: {avg_delta:+.4f} "
               f"({'improved' if avg_delta and avg_delta > 0 else 'no improvement'})"
               if avg_delta is not None else "")
        ),
    }


# ── Generation comparison ──────────────────────────────────────────

def generate_response(model, tokeniser, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate a response from the model for a given prompt."""
    inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokeniser.eos_token_id,
        )
    full_text = tokeniser.decode(output_ids[0], skip_special_tokens=True)
    return full_text[len(tokeniser.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


def side_by_side_comparison(
    sft_model, sft_tokeniser, dpo_model, dpo_tokeniser,
    prompts: list[str], max_new_tokens: int = 150,
) -> list[dict]:
    """Generate responses from both SFT and DPO models for qualitative comparison."""
    results = []
    for prompt in prompts:
        sft_response = generate_response(sft_model, sft_tokeniser, prompt, max_new_tokens)
        dpo_response = generate_response(dpo_model, dpo_tokeniser, prompt, max_new_tokens)
        results.append({
            "prompt": prompt,
            "sft_response": sft_response,
            "dpo_response": dpo_response,
        })
    return results


def llm_judge_comparison(prompt: str, response_a: str, response_b: str) -> dict:
    """
    Use a strong LLM (e.g. GPT-4o) to judge which response is better.
    Requires OPENAI_API_KEY in environment. Falls back gracefully if unavailable.
    """
    try:
        from openai import OpenAI
        import os
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        judge_prompt = f"""Compare these two AI assistant responses to the same prompt.
Evaluate on: helpfulness, empathy, clarity, and policy-appropriateness.

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Return JSON only: {{"winner": "A"|"B"|"tie", "reasoning": "brief explanation"}}"""

        raw = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        ).choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM judge unavailable: {e}")
        return {"winner": "unknown", "reasoning": f"LLM judge unavailable: {e}"}


if __name__ == "__main__":
    import argparse
    from app.model_setup import load_sft_model, load_tokeniser, load_for_inference

    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare SFT vs DPO models")
    parser.add_argument("--test-path", default="./data/dpo_eval.jsonl")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.compare:
        print("Loading SFT (reference) model...")
        ref_model = load_sft_model(quantise=True)
        tokeniser = load_tokeniser()

        print("Loading DPO-aligned model...")
        dpo_model, _ = load_for_inference(adapter_path=settings.ADAPTER_PATH, quantise=True)

        eval_examples = load_jsonl(args.test_path)
        metrics = compute_preference_metrics(dpo_model, ref_model, tokeniser, eval_examples, args.max_examples)

        print("\n" + "=" * 60)
        print("DPO PREFERENCE METRICS")
        print("=" * 60)
        print(metrics["interpretation"])
        print(json.dumps(metrics, indent=2))
