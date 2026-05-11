# TIME — Training Code and Data

Training pipeline and curriculum data for **TIME** (*Temporally Intelligent Meta-reasoning Engine*), as described in:

> Susmit Das. **TIME: Temporally Intelligent Meta-reasoning Engine for Context-Triggered Explicit Reasoning.** *Findings of ACL 2026*. arXiv:[2601.05300](https://arxiv.org/abs/2601.05300)

This repository contains all four phases of the QLoRA curriculum, the phase datasets, data preparation utilities, and an optional FP8 quantisation script. No model checkpoints are distributed; the training configuration is sufficient to recreate them faithfully.

For the evaluation harness (TIMEBench, LLM-as-judge, statistical comparisons), see the companion repository: [The-Coherence-Initiative/TIMEBench](https://github.com/The-Coherence-Initiative/TIMEBench).

---

## Overview

TIME trains Qwen3 dense models (4B–32B) to invoke explicit `<think>` reasoning *only* when contextual or temporal cues warrant it, rather than always or never. The objective is a reasoning *policy* — brief, placeable reasoning bursts triggered by context — rather than a fixed reasoning *mode*. Training proceeds through four curriculum phases on synthetic dialogue data generated with GPT-4o and Gemini 2.5 Flash via template-guided pipelines:

| Phase | Name | Data size | Purpose |
|---|---|---|---|
| **1** | Structural Seeding | 2,188 train / 387 test | Single-turn prompts pair `<time>` metadata with short `<think>` bursts, priming compact and well-delimited reasoning |
| **2** | Temporal Exposure | 5,291 train / 935 test (+25% Phase 1 replay) | Two-turn dialogues with time gaps and tick events; the model learns to revise assumptions after silence and suppress verbosity when no update is required |
| **3** | Contextual Modulation | 5,878 train / 1,039 test (+25% Phase 1–2 replay) | Multi-turn settings; trains both suppression and re-triggering of `<think>` blocks under changing context; ~33% tick frequency builds reliance on non-temporal cues too |
| **4** | Gradient-Aligned Convergence via Maximal Diversity | 128 conversations (no split, replay disabled) | Full-batch alignment over a maximally-diverse set whose only shared invariant is the target policy; concentrates gradients on context-triggered reasoning while suppressing incidental correlations |

Phases 1–3 share identical QLoRA hyperparameters. Phase 4 uses a high-intensity alignment regime: full-batch updates (effective batch = entire 128-sample dataset), a higher learning rate, and epoch-level checkpointing for automatic loss-band selection.

---

## Hardware and software

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 9 7950X3D |
| RAM | 128 GB DDR5 |
| GPU | NVIDIA RTX Pro 6000 Blackwell (96 GB VRAM) |
| OS | Ubuntu 24.04.3 LTS (WSL2 on Windows 11 Build 26100) |
| CUDA | 13.0 (driver 582.08) |

VRAM guidance:
- **32B**: ≥40 GB recommended
- **4B / 8B / 14B**: trainable on 24 GB

---

## Installation

Python 3.12 or later is required.

```bash
pip install -r requirements.txt
```

---

## Repository layout

```
.
├── scripts/
│   ├── prepare_data.py     Regenerate train/test splits from raw phase files
│   ├── train.py            Unified four-phase curriculum trainer
│   └── convert_to_fp8.py  Optional FP8-Dynamic quantisation (see below)
├── trainer_pkg/
│   ├── __init__.py
│   └── chat_template.py   TIME-modified Qwen3 Jinja2 chat template
├── notebooks/
│   ├── Data_Sampling.ipynb  Reference notebook — data preparation
│   ├── Trainer-4B.ipynb     Reference notebook — 4B training run
│   ├── Trainer-8B.ipynb     Reference notebook — 8B training run
│   ├── Trainer-14B.ipynb    Reference notebook — 14B training run
│   └── Trainer-32B.ipynb    Reference notebook — 32B training run
├── data/
│   ├── phase1.json          Raw Phase 1 conversations
│   ├── phase2.json          Raw Phase 2 conversations
│   ├── phase3.json          Raw Phase 3 conversations
│   ├── phase4.json          Phase 4 alignment set (128 conversations)
│   ├── phase{1,2,3}_train.json   Pre-generated train splits
│   └── phase{1,2,3}_test.json    Pre-generated test splits
└── requirements.txt
```

> **Notebooks** are preserved as reference artefacts. Their cell outputs record the exact training runs from the paper, including per-step loss logs for all four phases across all model sizes. The canonical way to reproduce training is `scripts/train.py`.

---

## Data preparation

Pre-generated train/test splits are included in `data/` and can be used directly — skip this step unless you need to regenerate them from the raw phase files.

```bash
python scripts/prepare_data.py
```

The script applies the following transformations:

- **System prompt injection**: a randomly selected prompt from a pool of 30 variants explaining the `<time>` / `<think>` conventions is prepended to ~5% of conversations (seed 42).
- **85/15 train/test split** (seed 42).
- **Replay augmentation**: Phase 2 receives a 25% sample from Phase 1; Phase 3 receives 25% from Phase 1 (seed 43, distinct from the Phase 2 sample) and 25% from Phase 2.
- **Deduplication** by JSON-serialised conversation content.

---

## Training

### Run the full pipeline

```bash
python scripts/train.py --model 8B
```

This sequentially runs all four phases, saves LoRA adapters and merged checkpoints after each phase, and produces `TIME-8B/` as the final model directory.

### Resume from a specific phase

If earlier phases are already complete (their merged checkpoints exist in `--output-dir`):

```bash
python scripts/train.py --model 32B --start-phase 3
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Model size: `4B`, `8B`, `14B`, `32B` |
| `--data-dir` | `data` | Directory containing phase split JSON files |
| `--output-dir` | `.` | Root directory for adapters, merged checkpoints, and Phase 4 trainer output |
| `--start-phase` | `1` | Phase to start from (prior phase checkpoints must exist) |
| `--phase4-loss-threshold` | `1.05` | Loss threshold for Phase 4 auto-checkpoint selection (see below) |
| `--seed` | `3407` | Random seed for all phases |

### Hyperparameters

**Phases 1–3 (shared)**

| Parameter | Value |
|---|---|
| Per-device batch size | 8 |
| Gradient accumulation steps | 4 (effective batch = 32) |
| Learning rate | 2×10⁻⁵ |
| Warmup steps | 100 |
| Epochs | 3 |
| Optimizer | AdamW 8-bit |
| LR scheduler | Linear |
| LoRA rank / alpha | 32 / 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q, k, v, o, gate, up, down proj |
| Gradient checkpointing | Enabled |

**Phase 4** (Gradient-Aligned Convergence via Maximal Diversity)

| Parameter | Value |
|---|---|
| Per-device batch size | 8 |
| Gradient accumulation steps | 16 (effective batch = 128 = entire dataset) |
| Learning rate | 1.5×10⁻⁴ |
| Warmup steps | 6 |
| Max steps | per model (see table below) |
| Save strategy | every epoch |
| Replay | disabled |

### Phase 4 checkpoint selection

Phase 4 checkpoints are saved after every training step (since effective batch = full dataset, one step = one epoch). After training, `train.py` automatically scans the training log and selects the **first checkpoint whose loss drops below 1.05**, which is the entry point of the empirically identified target loss band **[1.045, 1.050]**.

This band marks the inflection point between two failure modes: above 1.05 the model has not yet reliably acquired the target reasoning policy; below 1.045 degeneracy begins to increase — infinite loops, `<think>` format bleed into responses, and style collapse. The `max_steps` values below are set so that training reliably crosses the upper bound of this window before stopping.

The checkpoint selection is fully automatic. If your hardware produces a different loss trajectory (e.g. due to a different true effective batch size from tensor parallelism or micro-batch rounding), adjust `--phase4-loss-threshold` or the `phase4_max_steps` in `_MODEL_CONFIGS`.

Reference checkpoints from the paper's training runs:

| Model | Max steps | Selected step | Loss |
|---|---|---|---|
| 4B | 46 | 31 | 1.0474 |
| 8B | 40 | 30 | 1.0485 |
| 14B | 36 | 24 | 1.0496 |
| 32B | 35 | 18 | 1.0491 |

---

## FP8 quantisation (optional)

`scripts/convert_to_fp8.py` is provided for users who cannot serve the full BF16 model due to VRAM constraints. It quantises all linear layers (except the LM head) to FP8-Dynamic using llmcompressor, producing a smaller checkpoint suitable for FP8-capable inference engines such as vLLM. This step is **not required** for training or evaluation — the paper's results use the BF16 checkpoints throughout.

```bash
python scripts/convert_to_fp8.py --model TIME-8B
```

Writes `TIME-8B-FP8-Dynamic/` to the current directory.

---

## Reproducibility

- Seeds are fixed to 3407 across all phases.
- The Unsloth warning emitted during LoRA injection is benign — it reflects the dropout trade-off accepted for regularisation and does not affect correctness or convergence.
- Phase 4 uses full-batch updates (effective batch = entire dataset), so each gradient step is deterministic given fixed seeds and fixed data order. This removes sampling variance and makes the loss trajectory stable across hardware, allowing automatic checkpoint selection to reliably land in the target loss band.
- Loss metrics for phases 1–3 match to two decimal places across supported hardware configurations.

---

## Citation

```bibtex
@article{das2026time,
  title     = {{TIME}: Temporally Intelligent Meta-reasoning Engine for
               Context-Triggered Explicit Reasoning},
  author    = {Susmit Das},
  journal   = {arXiv preprint arXiv:2601.05300},
  year      = {2026}
}
```
