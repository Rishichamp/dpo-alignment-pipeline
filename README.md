# DPO Alignment Pipeline

[![tests](https://github.com/Rishichamp/dpo-alignment-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/dpo-alignment-pipeline/actions/workflows/tests.yml)

Aligns a supervised fine-tuned model towards more helpful, empathetic and policy-compliant answers using **Direct Preference Optimisation (DPO)**, the simpler alternative to RLHF that needs no separate reward model. It continues from [qlora-finetuning-pipeline](https://github.com/Rishichamp/qlora-finetuning-pipeline): that project's merged model is the starting point here.

## How it works

```
(prompt, chosen, rejected) ──► DPOTrainer (TRL)
                                 ├─ policy model:    SFT model + trainable LoRA adapter
                                 └─ reference model: the same weights with the adapter disabled
                                         │
                                         ▼
     evaluation: implicit reward margin, win rate, optional LLM-as-judge ──► merge ──► FastAPI /chat
```

Because the reference model is the policy with its adapter switched off, only **one copy** of the base weights sits in memory.

## Features

- **Preference data:** validation of (prompt, chosen, rejected) triplets, catching missing or empty fields, identical chosen/rejected answers and wrong types, plus de-duplication and length limits.
- **DPO training** with LoRA (r=16), 4-bit NF4 quantisation and β = 0.1 (the strength of the KL anchor to the reference).
- **Evaluation focused on preferences:** implicit reward margin (does the policy prefer the chosen answer more than the reference did?), win rate, side-by-side generations and an optional LLM-as-judge.
- **Adapter merging** and an **inference API** that works in adapter or merged mode.

## Tech stack

Python 3.11 · PyTorch · Hugging Face Transformers · PEFT · TRL · bitsandbytes · Datasets · FastAPI · pytest

## Project structure

```
app/
  data_prep.py     triplet validation, quality filtering, dataset building
  model_setup.py   SFT model loading, LoRA policy, adapter-disabled reference
  evaluation.py    reward margin, win rate, side-by-side, LLM judge
  config.py        all hyperparameters via .env
  main.py          FastAPI inference server
scripts/
  train_dpo.py     run DPO training
  merge_model.py   merge the DPO adapter into the SFT model
data/              sample preference pairs (IT support)
tests/             20 tests
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # set SFT_MODEL (e.g. the merged model from the QLoRA project) and HF_TOKEN
python scripts/train_dpo.py    # needs an NVIDIA GPU
python -m app.evaluation --compare
python scripts/merge_model.py
python run.py                  # http://127.0.0.1:8000/docs
```

The LLM-as-judge comparison is optional and needs `OPENAI_API_KEY`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Generate a response for a prompt |
| GET | `/info` | Model and DPO configuration |
| GET | `/health` | Health check (reports whether a model is loaded) |

## Tests

```bash
pytest tests/ -q
```

20 tests covering configuration, triplet validation edge cases, quality filtering, JSONL loading and the API (503 before a model is loaded), all without downloading a model.

## Notes

The sample preference set is small (15 training and 3 evaluation pairs), enough to exercise the pipeline. Built as part of Module 6 of a Generative AI engineering programme.

## Author

**Rishi Singh** — [GitHub](https://github.com/Rishichamp)
