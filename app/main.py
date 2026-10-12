"""
FastAPI inference server for the DPO-aligned model.

Endpoints:
  POST /chat       — generate a response for a given prompt
  GET  /info       — model and DPO configuration
  GET  /health     — health check

Supports both:
  - Adapter mode: SFT base model + DPO LoRA adapter loaded separately
  - Merged mode:  pre-merged DPO-aligned model (faster, no PEFT overhead)
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_state = {"model": None, "tokeniser": None, "mode": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the DPO-aligned model once at startup. Degrades gracefully if ML deps are unavailable."""
    merged_path  = Path(settings.MERGED_PATH)
    adapter_path = Path(settings.ADAPTER_PATH)

    try:
        from app.model_setup import load_for_inference

        if merged_path.exists() and any(merged_path.iterdir()):
            logger.info(f"Loading MERGED DPO model from {merged_path}")
            model, tokeniser = load_for_inference(merged_path=str(merged_path), quantise=False)
            _state["mode"] = "merged"
        elif adapter_path.exists() and any(adapter_path.iterdir()):
            logger.info(f"Loading SFT base + DPO adapter from {adapter_path}")
            model, tokeniser = load_for_inference(adapter_path=str(adapter_path), quantise=True)
            _state["mode"] = "adapter"
        else:
            logger.warning(
                "No trained DPO adapter or merged model found. "
                "Run 'python scripts/train_dpo.py' first. "
                "Serving will return 503 until a model is available."
            )
            model, tokeniser = None, None
            _state["mode"] = "none"
    except ImportError as e:
        logger.warning(f"ML dependencies unavailable ({e}). Serving will return 503.")
        model, tokeniser = None, None
        _state["mode"] = "deps_unavailable"

    _state["model"]     = model
    _state["tokeniser"] = tokeniser
    yield
    _state["model"] = None
    _state["tokeniser"] = None


app = FastAPI(
    title="DPO-Aligned Inference Server",
    description=(
        "Serves a DPO-aligned model trained on IT/tech-support preference pairs. "
        "The model is optimised to prefer helpful, empathetic, policy-compliant "
        "responses over curt or unhelpful ones."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The customer support scenario/prompt")
    max_new_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "A customer writes: 'I've been waiting 2 weeks for a refund with no updates.'\n\nWrite a support response.",
                "max_new_tokens": 200,
                "temperature": 0.3,
            }
        }
    }


class ChatResponse(BaseModel):
    prompt: str
    response: str
    model_mode: str
    alignment: str = "dpo"


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return f"""<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>🎯 DPO-Aligned Inference Server</h1>
    <p>SFT model: {settings.SFT_MODEL}</p>
    <p>Mode: <b>{_state.get('mode','not loaded')}</b></p>
    <p>Beta (KL coefficient): {settings.BETA}</p>
    <p><a href="/docs">📖 API Docs</a></p>
    <h3>Alignment focus: IT/tech support</h3>
    <p>Trained to prefer responses that are empathetic, transparent, policy-aware,
    and avoid generic/dismissive language.</p>
    </body></html>"""


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["model"] is not None else "no_model_loaded",
        "mode": _state["mode"],
        "sft_model": settings.SFT_MODEL,
        "beta": settings.BETA,
    }


@app.get("/info")
async def info():
    return {
        "sft_model": settings.SFT_MODEL,
        "mode": _state["mode"],
        "lora_r": settings.LORA_R,
        "lora_alpha": settings.LORA_ALPHA,
        "beta": settings.BETA,
        "target_modules": settings.target_modules_list,
        "quantised": settings.USE_4BIT,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Generate a response using the DPO-aligned model.

    The model was aligned on IT/tech-support preference pairs to prefer
    responses that are empathetic, transparent about limitations, escalate
    appropriately, and decline policy-violating requests gracefully with
    helpful alternatives.
    """
    if _state["model"] is None:
        raise HTTPException(
            503,
            "No DPO model loaded. Run 'python scripts/train_dpo.py' to train an adapter, "
            "then restart the server."
        )

    if not req.prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty.")

    from app.evaluation import generate_response

    try:
        result = generate_response(
            _state["model"], _state["tokeniser"], req.prompt,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(500, f"Generation failed: {str(e)}")

    return ChatResponse(
        prompt=req.prompt,
        response=result,
        model_mode=_state["mode"],
    )
