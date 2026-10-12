"""
Tests for the DPO Alignment Pipeline.

These tests avoid loading actual models — they test:
  - DPO dataset validation and quality filtering
  - Configuration defaults
  - API contract (with mocked model state)
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.config import settings
from app.data_prep import (
    load_jsonl, validate_dpo_example, quality_filter_dpo, build_dpo_dataset
)


# ── Config tests ──────────────────────────────────────────────────
def test_config_defaults():
    assert settings.BETA == 0.1
    assert settings.LORA_R == 16
    assert settings.NUM_EPOCHS == 1  # DPO typically uses fewer epochs than SFT

def test_target_modules_list():
    modules = settings.target_modules_list
    assert "q_proj" in modules
    assert len(modules) == 7

def test_effective_batch_size():
    assert settings.effective_batch_size == settings.PER_DEVICE_BATCH * settings.GRAD_ACCUM


# ── Validation tests ─────────────────────────────────────────────
def test_validate_dpo_example_valid():
    ex = {"prompt": "Hello", "chosen": "A helpful response here", "rejected": "Nope"}
    assert validate_dpo_example(ex) is None

def test_validate_dpo_example_missing_field():
    ex = {"prompt": "Hello", "chosen": "A response"}  # missing 'rejected'
    error = validate_dpo_example(ex)
    assert error is not None
    assert "rejected" in error

def test_validate_dpo_example_empty_field():
    ex = {"prompt": "Hello", "chosen": "", "rejected": "Something"}
    error = validate_dpo_example(ex)
    assert error is not None

def test_validate_dpo_example_identical_chosen_rejected():
    ex = {"prompt": "Hello", "chosen": "Same text", "rejected": "Same text"}
    error = validate_dpo_example(ex)
    assert error is not None
    assert "identical" in error

def test_validate_dpo_example_non_string_field():
    ex = {"prompt": "Hello", "chosen": 123, "rejected": "Something"}
    error = validate_dpo_example(ex)
    assert error is not None


# ── Quality filter tests ─────────────────────────────────────────
def test_quality_filter_dpo_removes_invalid():
    examples = [
        {"prompt": "Q1", "chosen": "A " * 10, "rejected": "B " * 5},  # valid
        {"prompt": "Q2", "chosen": "Same", "rejected": "Same"},        # identical - invalid
        {"prompt": "Q3", "chosen": "x", "rejected": "y"},              # too short
    ]
    filtered = quality_filter_dpo(examples, min_chosen_words=5, max_chosen_words=1000)
    assert len(filtered) == 1
    assert filtered[0]["prompt"] == "Q1"

def test_quality_filter_dpo_deduplication():
    examples = [
        {"prompt": "Same prompt here", "chosen": "Response A " * 5, "rejected": "Bad A"},
        {"prompt": "Same prompt here", "chosen": "Response B " * 5, "rejected": "Bad B"},
    ]
    filtered = quality_filter_dpo(examples, min_chosen_words=5, max_chosen_words=1000)
    assert len(filtered) == 1  # second is a duplicate prompt

def test_quality_filter_dpo_too_long():
    examples = [
        {"prompt": "Q1", "chosen": "word " * 2000, "rejected": "short"},
    ]
    filtered = quality_filter_dpo(examples, min_chosen_words=5, max_chosen_words=1000)
    assert len(filtered) == 0

def test_quality_filter_returns_clean_dict_keys():
    examples = [
        {"prompt": "  Q1  ", "chosen": "  Response here with enough words now  ", "rejected": "Bad", "extra_field": "ignored"},
    ]
    filtered = quality_filter_dpo(examples, min_chosen_words=5, max_chosen_words=1000)
    assert len(filtered) == 1
    assert set(filtered[0].keys()) == {"prompt", "chosen", "rejected"}
    assert filtered[0]["prompt"] == "Q1"  # stripped


# ── JSONL loading ──────────────────────────────────────────────────
def test_load_jsonl(tmp_path):
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(
        '{"prompt": "Q1", "chosen": "A " * 5, "rejected": "B"}\n'.replace('"A " * 5', '"A A A A A"')
    )
    examples = load_jsonl(str(test_file))
    assert len(examples) == 1

def test_load_jsonl_skips_malformed(tmp_path):
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(
        '{"prompt": "Q1", "chosen": "Valid response here", "rejected": "Bad"}\n'
        'not valid json\n'
        '{"prompt": "Q2", "chosen": "Another valid response", "rejected": "Bad2"}\n'
    )
    examples = load_jsonl(str(test_file))
    assert len(examples) == 2


# ── Dataset building (uses real dpo_train.jsonl) ───────────────────
def test_build_dpo_dataset_from_real_data():
    data_path = Path(__file__).resolve().parent.parent / "data" / "dpo_train.jsonl"
    eval_path = Path(__file__).resolve().parent.parent / "data" / "dpo_eval.jsonl"
    if not data_path.exists():
        pytest.skip("dpo_train.jsonl not found")

    datasets = build_dpo_dataset(
        data_path=str(data_path),
        eval_path=str(eval_path) if eval_path.exists() else None,
    )
    assert "train" in datasets
    assert "eval" in datasets
    assert len(datasets["train"]) > 0
    # Verify schema
    example = datasets["train"][0]
    assert "prompt" in example
    assert "chosen" in example
    assert "rejected" in example
    assert example["chosen"] != example["rejected"]


# ── API tests (mocked model) ────────────────────────────────────────
@pytest.fixture
def client_with_mock_model():
    import app.main as main_module
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        main_module._state["model"] = MagicMock()
        main_module._state["tokeniser"] = MagicMock()
        main_module._state["mode"] = "adapter"
        yield client

    main_module._state["model"] = None
    main_module._state["tokeniser"] = None
    main_module._state["mode"] = "none"


def test_health_no_model():
    import app.main as main_module
    from fastapi.testclient import TestClient

    main_module._state["model"] = None
    main_module._state["mode"] = "none"

    with TestClient(main_module.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "no_model_loaded"
        assert r.json()["beta"] == settings.BETA


def test_info_endpoint():
    import app.main as main_module
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert data["beta"] == settings.BETA
        assert "target_modules" in data


def test_chat_no_model_returns_503():
    import app.main as main_module
    from fastapi.testclient import TestClient

    main_module._state["model"] = None
    main_module._state["mode"] = "none"

    with TestClient(main_module.app) as client:
        r = client.post("/chat", json={"prompt": "Test prompt"})
        assert r.status_code == 503


def test_chat_empty_prompt(client_with_mock_model):
    r = client_with_mock_model.post("/chat", json={"prompt": ""})
    assert r.status_code == 400


@patch("app.evaluation.generate_response", return_value="I understand your frustration, let me help with that right away.")
def test_chat_success(mock_gen, client_with_mock_model):
    r = client_with_mock_model.post("/chat", json={
        "prompt": "A customer writes: 'My order is late.'\n\nWrite a support response.",
        "max_new_tokens": 100,
    })
    assert r.status_code == 200
    data = r.json()
    assert "frustration" in data["response"]
    assert data["alignment"] == "dpo"
    assert data["model_mode"] == "adapter"
