"""
Model loading and LoRA configuration for DPO training.

DPO requires TWO copies of the model:
  - policy model (trainable, with LoRA adapters)
  - reference model (frozen copy of the SFT model — the KL anchor)

When using PEFT/LoRA for the policy model, TRL's DPOTrainer can use the
SAME base model for both policy and reference by disabling the adapter
for reference forward passes — this avoids loading the model twice.
"""

import logging
from typing import Optional
import torch

from app.config import settings

logger = logging.getLogger(__name__)


def get_bnb_config():
    """Build BitsAndBytesConfig for 4-bit quantisation (same as M6P1)."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=settings.USE_4BIT,
        bnb_4bit_use_double_quant=settings.USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=settings.QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_sft_model(quantise: bool = True):
    """
    Load the SFT model that DPO will align.

    SFT_MODEL can be:
      - A HuggingFace model id (e.g. "meta-llama/Llama-3.2-3B") — start DPO
        from a base instruction-tuned model
      - A local path to the merged model from M6P1 (recommended for a
        consistent pipeline: SFT via LoRA -> merge -> DPO via LoRA -> merge)
    """
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading SFT model: {settings.SFT_MODEL}")
    kwargs = dict(
        pretrained_model_name_or_path=settings.SFT_MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    if settings.HF_TOKEN:
        kwargs["token"] = settings.HF_TOKEN

    if quantise and settings.USE_4BIT:
        kwargs["quantization_config"] = get_bnb_config()
        logger.info("4-bit NF4 quantisation enabled")
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.config.use_cache = False
    return model


def load_tokeniser():
    """Load and configure the tokeniser for the SFT model."""
    from transformers import AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(
        settings.SFT_MODEL,
        trust_remote_code=True,
        token=settings.HF_TOKEN or None,
    )
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
        tokeniser.pad_token_id = tokeniser.eos_token_id
    tokeniser.padding_side = "right"
    return tokeniser


def apply_lora_for_dpo(model) -> tuple:
    """
    Apply LoRA adapters for DPO training.

    Same structure as SFT LoRA (M6P1) but DPO typically uses a lower
    learning rate and fewer epochs since it's a lighter alignment pass
    on top of an already-capable SFT model.
    """
    from peft import (
        LoraConfig, get_peft_model,
        prepare_model_for_kbit_training, TaskType
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=settings.LORA_R,
        lora_alpha=settings.LORA_ALPHA,
        target_modules=settings.target_modules_list,
        lora_dropout=settings.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    peft_model = get_peft_model(model, lora_config)
    _print_trainable_params(peft_model)
    return peft_model, lora_config


def _print_trainable_params(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / max(total, 1)
    print(f"\n{'=' * 50}")
    print(f"TRAINABLE PARAMETERS (DPO)")
    print(f"{'=' * 50}")
    print(f"  Trainable : {trainable:>15,}  ({pct:.4f}%)")
    print(f"  Frozen    : {total - trainable:>15,}")
    print(f"  Total     : {total:>15,}")
    print(f"{'=' * 50}\n")


def load_for_inference(
    adapter_path: Optional[str] = None,
    merged_path: Optional[str] = None,
    quantise: bool = True,
):
    """Load DPO-aligned model for inference (same pattern as M6P1)."""
    from transformers import AutoModelForCausalLM

    tokeniser = load_tokeniser()

    if merged_path:
        logger.info(f"Loading merged DPO model from {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    elif adapter_path:
        from peft import PeftModel
        logger.info(f"Loading SFT base + DPO adapter from {adapter_path}")
        base  = load_sft_model(quantise=quantise)
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        raise ValueError("Provide either adapter_path or merged_path.")

    model.eval()
    return model, tokeniser


def merge_and_save(adapter_path: str, output_path: str) -> None:
    """Merge DPO LoRA weights into the SFT base model and save."""
    from peft import PeftModel

    logger.info("Loading SFT base model for merging (BF16, no quantisation)...")
    base = load_sft_model(quantise=False)

    logger.info(f"Loading DPO adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging DPO LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged DPO-aligned model to {output_path}")
    model.save_pretrained(output_path)
    load_tokeniser().save_pretrained(output_path)
    logger.info("Merge complete.")
