"""
Training script for the DPO Alignment Pipeline.

Usage:
    python scripts/train_dpo.py

Pipeline:
  1. Load and prepare DPO dataset (prompt/chosen/rejected triplets)
  2. Load SFT model with 4-bit quantisation
  3. Apply LoRA adapters (policy model)
  4. Train with DPOTrainer
     - Reference model: same base model with adapters disabled (PEFT pattern)
     - Loss: DPO loss with KL penalty (beta) toward reference model
  5. Save DPO adapter weights

Run AFTER M6P1's SFT pipeline for a complete SFT -> DPO alignment workflow.
Set SFT_MODEL in .env to M6P1's merged model path for a connected pipeline,
or use a base instruction-tuned model directly.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.data_prep import build_dpo_dataset, print_dpo_dataset_stats
from app.model_setup import load_sft_model, load_tokeniser, apply_lora_for_dpo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 60)
    print("DPO ALIGNMENT PIPELINE")
    print("=" * 60)
    print(f"SFT model      : {settings.SFT_MODEL}")
    print(f"LoRA rank      : {settings.LORA_R} (alpha={settings.LORA_ALPHA})")
    print(f"Beta (KL coef) : {settings.BETA}")
    print(f"Quantisation   : {'4-bit NF4' if settings.USE_4BIT else 'BF16 (full precision)'}")
    print(f"Effective batch: {settings.effective_batch_size} "
          f"({settings.PER_DEVICE_BATCH} x {settings.GRAD_ACCUM} grad accum)")
    print(f"Learning rate  : {settings.LEARNING_RATE}")
    print(f"Epochs         : {settings.NUM_EPOCHS}")
    print("=" * 60 + "\n")

    # ── Step 1: Dataset ──────────────────────────────────────────
    logger.info("Building DPO dataset...")
    datasets = build_dpo_dataset(
        data_path=settings.DATA_PATH,
        eval_path=settings.EVAL_DATA_PATH if Path(settings.EVAL_DATA_PATH).exists() else None,
    )
    print_dpo_dataset_stats(datasets)

    # ── Step 2: Tokeniser ────────────────────────────────────────
    logger.info("Loading tokeniser...")
    tokeniser = load_tokeniser()

    # ── Step 3: Model (policy) ───────────────────────────────────
    logger.info("Loading SFT model (policy)...")
    model = load_sft_model(quantise=settings.USE_4BIT)

    # ── Step 4: Apply LoRA ───────────────────────────────────────
    logger.info("Applying LoRA adapters for DPO...")
    model, lora_config = apply_lora_for_dpo(model)

    # ── Step 5: DPO training arguments ──────────────────────────
    from trl import DPOConfig, DPOTrainer

    dpo_config = DPOConfig(
        output_dir=settings.OUTPUT_DIR,
        beta=settings.BETA,
        per_device_train_batch_size=settings.PER_DEVICE_BATCH,
        per_device_eval_batch_size=settings.PER_DEVICE_BATCH,
        gradient_accumulation_steps=settings.GRAD_ACCUM,
        learning_rate=settings.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=settings.WARMUP_RATIO,
        num_train_epochs=settings.NUM_EPOCHS,
        max_length=settings.MAX_LENGTH,
        max_prompt_length=settings.MAX_PROMPT_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit" if settings.USE_4BIT else "adamw_torch",
        save_strategy="steps",
        save_steps=settings.SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=settings.EVAL_STEPS,
        logging_steps=settings.LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
        save_total_limit=2,
    )

    # ── Step 6: Build trainer ────────────────────────────────────
    # Note: ref_model=None tells DPOTrainer to use the PEFT pattern —
    # the SAME base model is used for the reference, with LoRA adapters
    # disabled during reference forward passes. This avoids loading
    # two full copies of the model into memory.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokeniser,
    )

    # ── Step 7: Train ─────────────────────────────────────────────
    logger.info("Starting DPO training...")
    trainer.train()

    # ── Step 8: Save adapter ─────────────────────────────────────
    adapter_path = settings.ADAPTER_PATH
    logger.info(f"Saving DPO adapter to {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokeniser.save_pretrained(adapter_path)

    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved to: {adapter_path}")
    print(f"Next steps:")
    print(f"  1. Evaluate:  python -m app.evaluation --compare")
    print(f"  2. Merge:     python scripts/merge_model.py")
    print(f"  3. Serve:     python run.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
