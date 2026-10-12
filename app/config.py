"""
Configuration for the DPO Alignment Pipeline.

This project continues from M6P1's domain (IT/tech support) for consistency:
the SFT model is assumed to be the merged output of M6P1, and DPO further
aligns it toward more helpful, empathetic, and policy-compliant responses.
"""
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    # ── Models ────────────────────────────────────────────────────
    # SFT_MODEL: starting point for DPO — either a HF model id or a local
    # path to the merged SFT model produced by M6P1 (scripts/merge_model.py)
    SFT_MODEL: str          = os.getenv("SFT_MODEL", "meta-llama/Llama-3.2-3B")
    HF_TOKEN: str           = os.getenv("HF_TOKEN", "")

    # ── LoRA config (DPO also trains via LoRA for efficiency) ───────
    LORA_R: int             = int(os.getenv("LORA_R", "16"))
    LORA_ALPHA: int         = int(os.getenv("LORA_ALPHA", "32"))
    LORA_DROPOUT: float     = float(os.getenv("LORA_DROPOUT", "0.05"))
    LORA_TARGET_MODULES: str = os.getenv(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )

    # ── Quantisation ──────────────────────────────────────────────
    USE_4BIT: bool          = os.getenv("USE_4BIT", "true").lower() == "true"
    USE_DOUBLE_QUANT: bool  = os.getenv("USE_DOUBLE_QUANT", "true").lower() == "true"
    QUANT_TYPE: str         = os.getenv("QUANT_TYPE", "nf4")

    # ── DPO training ─────────────────────────────────────────────
    OUTPUT_DIR: str         = os.getenv("OUTPUT_DIR", "./output")
    DATA_PATH: str          = os.getenv("DATA_PATH", "./data/dpo_train.jsonl")
    EVAL_DATA_PATH: str     = os.getenv("EVAL_DATA_PATH", "./data/dpo_eval.jsonl")

    BETA: float             = float(os.getenv("BETA", "0.1"))   # KL penalty strength
    PER_DEVICE_BATCH: int   = int(os.getenv("PER_DEVICE_BATCH", "2"))
    GRAD_ACCUM: int         = int(os.getenv("GRAD_ACCUM", "4"))
    LEARNING_RATE: float    = float(os.getenv("LEARNING_RATE", "5e-5"))
    NUM_EPOCHS: int         = int(os.getenv("NUM_EPOCHS", "1"))
    WARMUP_RATIO: float     = float(os.getenv("WARMUP_RATIO", "0.05"))
    MAX_LENGTH: int         = int(os.getenv("MAX_LENGTH", "1024"))
    MAX_PROMPT_LENGTH: int  = int(os.getenv("MAX_PROMPT_LENGTH", "512"))

    SAVE_STEPS: int         = int(os.getenv("SAVE_STEPS", "100"))
    EVAL_STEPS: int         = int(os.getenv("EVAL_STEPS", "100"))
    LOGGING_STEPS: int      = int(os.getenv("LOGGING_STEPS", "10"))

    # ── Inference / API ───────────────────────────────────────────
    ADAPTER_PATH: str       = os.getenv("ADAPTER_PATH", "./dpo_adapter")
    MERGED_PATH: str        = os.getenv("MERGED_PATH", "./merged_dpo_model")
    APP_PORT: int           = int(os.getenv("APP_PORT", "8002"))
    APP_HOST: str           = os.getenv("APP_HOST", "0.0.0.0")

    @property
    def target_modules_list(self) -> list[str]:
        return [m.strip() for m in self.LORA_TARGET_MODULES.split(",")]

    @property
    def effective_batch_size(self) -> int:
        return self.PER_DEVICE_BATCH * self.GRAD_ACCUM

settings = Settings()
