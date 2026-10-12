"""
Pytest configuration for the DPO alignment pipeline tests.

Injects a stub `torch` module into sys.modules if torch is not installed,
allowing app.evaluation and app.model_setup to be imported in CI environments
without GPU/torch — these tests verify API contracts and data logic.
"""
import sys
import importlib.machinery
from unittest.mock import MagicMock

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        fake_torch = MagicMock()
        fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        fake_torch.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = fake_torch
