"""
Merge DPO LoRA adapter weights into the SFT base model for production deployment.

Usage:
    python scripts/merge_model.py
    python scripts/merge_model.py --adapter ./dpo_adapter --output ./merged_dpo_model
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.model_setup import merge_and_save

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge DPO LoRA adapter into SFT base model")
    parser.add_argument("--adapter", default=settings.ADAPTER_PATH,
                        help="Path to the trained DPO adapter")
    parser.add_argument("--output", default=settings.MERGED_PATH,
                        help="Output path for the merged DPO-aligned model")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        logger.error(f"Adapter path does not exist: {adapter_path}")
        logger.error("Run 'python scripts/train_dpo.py' first to produce a DPO adapter.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MERGING DPO ADAPTER INTO SFT MODEL")
    print("=" * 60)
    print(f"SFT model  : {settings.SFT_MODEL}")
    print(f"DPO adapter: {args.adapter}")
    print(f"Output     : {args.output}")
    print("=" * 60 + "\n")

    merge_and_save(adapter_path=str(adapter_path), output_path=args.output)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"DPO-aligned model saved to: {args.output}")
    print("\nThis model has been through the full pipeline:")
    print("  Base model -> SFT (LoRA, M6P1) -> merge -> DPO (LoRA, M6P2) -> merge")
    print("\nNext steps:")
    print("  - Serve: set MERGED_PATH in .env and run 'python run.py'")
    print("  - Compare against SFT-only model using scripts/evaluate_dpo.py --compare")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
