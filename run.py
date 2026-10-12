"""Entry point for the DPO inference server. Usage: python run.py"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"\n🎯  DPO-Aligned Inference Server")
    print(f"   SFT model  : {settings.SFT_MODEL}")
    print(f"   Beta (KL)  : {settings.BETA}")
    print(f"   LoRA r/α   : {settings.LORA_R}/{settings.LORA_ALPHA}")
    print(f"   Docs       : http://127.0.0.1:{settings.APP_PORT}/docs")
    print(f"\n   If you haven't trained yet:")
    print(f"   1) python scripts/train_dpo.py")
    print(f"   2) python scripts/merge_model.py  (optional)")
    print(f"   3) python run.py\n")
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, log_level="info")
