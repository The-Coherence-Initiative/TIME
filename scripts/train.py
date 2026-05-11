"""Four-phase QLoRA curriculum trainer for TIME models.

Runs the full TIME alignment curriculum (Phases 1–4) for a given Qwen3 model
size. Each phase fine-tunes the previous phase's merged checkpoint with
QLoRA-style 4-bit quantization via Unsloth, then merges the LoRA adapter back
into the base weights before the next phase begins.

Phases 1–3 share identical trainer hyperparameters and differ only in which
checkpoint they start from and which data split they consume. Phase 4 uses a
high-intensity alignment regime with a tiny, maximally-diverse batch of 128
conversations and epoch-level checkpointing. After training, the script
automatically selects the first checkpoint whose training loss falls below
1.05 — the threshold above which the model is behaviorally under-formed and
below which it risks collapsing into degenerate outputs.

Usage
-----
    # Train all four phases for the 8B model
    python scripts/train.py --model 8B

    # Resume from Phase 3 (phases 1–3 must already be complete)
    python scripts/train.py --model 32B --start-phase 4

    # Override the Phase 4 loss threshold
    python scripts/train.py --model 14B --phase4-loss-threshold 1.048

Notes
-----
- All seeds are fixed to 3407 for reproducibility.
- The Unsloth dropout warning during LoRA injection is benign; see README.
- Phases 1–3 evaluate on their held-out test split after training.
- Phase 4 does not use an eval split — checkpoints are saved per epoch and
  the first one below the loss threshold is automatically selected.
"""

import argparse
import gc
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------

# Only Phase 4 max_steps differs between model sizes. It is tuned so that
# training runs long enough for at least one checkpoint to cross the 1.05
# loss threshold before degeneracy sets in. The checkpoint step is determined
# at runtime from the training log rather than being hardcoded.
_MODEL_CONFIGS: dict[str, dict] = {
    "4B":  {"hf_name": "Qwen/Qwen3-4B",  "phase4_max_steps": 46},
    "8B":  {"hf_name": "Qwen/Qwen3-8B",  "phase4_max_steps": 40},
    "14B": {"hf_name": "Qwen/Qwen3-14B", "phase4_max_steps": 36},
    "32B": {"hf_name": "Qwen/Qwen3-32B", "phase4_max_steps": 35},
}

# Phases 1–3 share these hyperparameters exactly.
_CURRICULUM_TRAINER_CONFIG = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,   # effective batch = 32
    "warmup_steps": 100,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "logging_steps": 10,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "max_grad_norm": 1.0,
    "report_to": "none",
}

# Phase 4 trainer hyperparameters (max_steps is per-model, set in main).
_ALIGNMENT_TRAINER_CONFIG = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 16,  # effective batch = 128
    "warmup_steps": 6,
    "learning_rate": 1.5e-4,
    "logging_steps": 1,                 # log every step for fine-grained loss tracking
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "max_grad_norm": 1.0,
    "report_to": "none",
    "save_strategy": "epoch",
}

# LoRA adapter configuration — identical across all phases and model sizes.
_LORA_CONFIG = {
    "r": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "use_gradient_checkpointing": True,
}

# Default loss threshold for Phase 4 checkpoint auto-selection.
_PHASE4_LOSS_THRESHOLD = 1.05


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Four-phase QLoRA curriculum trainer for TIME models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(_MODEL_CONFIGS.keys()),
        help="Model size to train.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing phase split JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help=(
            "Root directory for all output artefacts (adapters, merged "
            "checkpoints, Phase 4 trainer output). Defaults to the current "
            "working directory."
        ),
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help=(
            "Phase to start from. All prior phase checkpoints must already "
            "exist in --output-dir."
        ),
    )
    parser.add_argument(
        "--phase4-loss-threshold",
        type=float,
        default=_PHASE4_LOSS_THRESHOLD,
        help=(
            "Training loss threshold for automatic Phase 4 checkpoint selection. "
            "The first checkpoint whose loss is strictly below this value is "
            "selected for the final merge. Adjust if your hardware or effective "
            "batch size produces a different loss trajectory than the reference "
            "configuration."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for all phases.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _init_environment(seed: int) -> None:
    """Pre-allocate GPU buffers and fix all random seeds."""
    dtype = torch.float16
    n_gpus = torch.cuda.device_count()
    global _GPU_BUFFERS  # noqa: PLW0603
    _GPU_BUFFERS = tuple(
        torch.empty(2 * 256 * 2048, dtype=dtype, device=f"cuda:{i}")
        for i in range(n_gpus)
    )
    from transformers import set_seed
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _log_gpu_stats() -> None:
    """Print GPU name, total VRAM, and currently reserved memory."""
    props = torch.cuda.get_device_properties(0)
    reserved = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    total = round(props.total_memory / 1024 ** 3, 3)
    print(f"  GPU: {props.name}  |  Total: {total} GB  |  Reserved: {reserved} GB")


def _cleanup(*objects) -> None:
    """Delete objects, run the garbage collector, and empty the CUDA cache."""
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_split(train_path: str, test_path: str) -> tuple[list, list]:
    """Load train and test splits from JSON files."""
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test = json.load(f)
    return train, test


def _load_phase4(data_path: str) -> list:
    """Load the Phase 4 alignment data (no test split)."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_datasets(tokenizer, train_convs: list, eval_convs: list | None):
    """Apply the chat template and wrap conversations as HuggingFace Datasets."""
    from datasets import Dataset

    train_texts = tokenizer.apply_chat_template(train_convs, tokenize=False)
    train_lengths = [
        len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in train_texts
    ]
    max_seq_len = max(train_lengths)
    p90 = sorted(train_lengths)[int(0.9 * len(train_lengths))]
    print(
        f"  Train: {len(train_texts):,} examples  |  "
        f"max={max_seq_len}  "
        f"mean={sum(train_lengths)/len(train_lengths):.0f}  "
        f"p90={p90}"
    )

    train_dataset = Dataset.from_list([{"text": t} for t in train_texts])
    eval_dataset = None

    if eval_convs is not None:
        eval_texts = tokenizer.apply_chat_template(eval_convs, tokenize=False)
        eval_lengths = [
            len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in eval_texts
        ]
        print(
            f"  Eval:  {len(eval_texts):,} examples  |  "
            f"max={max(eval_lengths)}  "
            f"mean={sum(eval_lengths)/len(eval_lengths):.0f}"
        )
        eval_dataset = Dataset.from_list([{"text": t} for t in eval_texts])

    return train_dataset, eval_dataset, max_seq_len


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _load_model_with_lora(model_path: str, max_seq_len: int, seed: int):
    """Load a model via Unsloth and attach LoRA adapters."""
    from unsloth import FastLanguageModel
    from trainer_pkg.chat_template import CHAT_TEMPLATE

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    tokenizer.chat_template = CHAT_TEMPLATE

    model = FastLanguageModel.get_peft_model(
        model,
        r=_LORA_CONFIG["r"],
        target_modules=_LORA_CONFIG["target_modules"],
        lora_alpha=_LORA_CONFIG["lora_alpha"],
        lora_dropout=_LORA_CONFIG["lora_dropout"],
        bias=_LORA_CONFIG["bias"],
        random_state=seed,
        use_gradient_checkpointing=_LORA_CONFIG["use_gradient_checkpointing"],
    )
    return model, tokenizer


def _merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """Merge a LoRA adapter into its base model and save the result."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Merging adapter '{adapter_path}' into '{base_model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"  Merged model saved to '{output_path}'.")
    _cleanup(model, tokenizer)


# ---------------------------------------------------------------------------
# Phase 4 checkpoint auto-selection
# ---------------------------------------------------------------------------


def _find_phase4_checkpoint(
    log_history: list[dict],
    trainer_output_dir: str,
    loss_threshold: float,
) -> int:
    """Return the step of the first checkpoint whose training loss is below *loss_threshold*.

    The paper uses a target loss band of **[1.045, 1.050]**: above 1.050 the
    model has not yet acquired the target reasoning policy; below 1.045
    degeneracy (format bleed, infinite loops, style collapse) begins to
    increase. The default threshold of 1.05 selects the first checkpoint that
    enters this band from above, which is the correct trigger in the reference
    configuration. Adjust *--phase4-loss-threshold* if your hardware produces
    a different trajectory.

    Scans ``trainer.state.log_history`` in ascending step order, filters to
    steps that have a saved checkpoint directory on disk, and returns the
    first whose loss is strictly below *loss_threshold*.

    Falls back to the checkpoint with the lowest recorded loss and emits a
    warning if no checkpoint reaches the threshold — this can happen when the
    true effective batch size differs from the reference configuration (e.g.
    on machines with a different number of GPUs or different micro-batch
    rounding). In that case, lower *--phase4-loss-threshold* slightly or
    increase the model's ``phase4_max_steps``.

    Parameters
    ----------
    log_history:
        ``trainer.state.log_history`` from a completed ``SFTTrainer`` run.
    trainer_output_dir:
        Directory where epoch checkpoints were saved.
    loss_threshold:
        Loss value below which a checkpoint is considered aligned.
    """
    # Collect training-loss entries (exclude eval_loss entries)
    step_losses = sorted(
        [
            (int(entry["step"]), float(entry["loss"]))
            for entry in log_history
            if "loss" in entry and "eval_loss" not in entry
        ],
        key=lambda x: x[0],
    )

    if not step_losses:
        raise RuntimeError(
            "No training loss entries found in trainer log history. "
            "Ensure logging_steps=1 is set for Phase 4."
        )

    # Find checkpoint directories that actually exist on disk
    output_path = Path(trainer_output_dir)
    existing_steps: set[int] = set()
    if output_path.exists():
        for p in output_path.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint-"):
                try:
                    existing_steps.add(int(p.name.split("-")[1]))
                except (IndexError, ValueError):
                    pass

    if not existing_steps:
        raise RuntimeError(
            f"No checkpoints found in '{trainer_output_dir}'. "
            "Check that save_strategy='epoch' produced checkpoint directories."
        )

    # Only consider steps that have a corresponding saved checkpoint
    valid = [(s, l) for s, l in step_losses if s in existing_steps]

    if not valid:
        raise RuntimeError(
            f"Log history steps {[s for s, _ in step_losses]} do not overlap "
            f"with saved checkpoint steps {sorted(existing_steps)}. "
            "Cannot auto-select checkpoint."
        )

    # Return the first checkpoint strictly below the threshold
    for step, loss in valid:
        if loss < loss_threshold:
            print(
                f"  Auto-selected checkpoint-{step}  "
                f"(loss={loss:.4f} < threshold={loss_threshold})"
            )
            return step

    # Fallback: lowest-loss checkpoint
    best_step, best_loss = min(valid, key=lambda x: x[1])
    print(
        f"\n  Warning: no checkpoint reached loss < {loss_threshold}. "
        f"Using checkpoint-{best_step} (loss={best_loss:.4f}) as fallback.\n"
        f"  Hint: lower --phase4-loss-threshold or increase phase4_max_steps "
        f"in _MODEL_CONFIGS for this model size."
    )
    return best_step


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------


def run_curriculum_phase(
    phase: int,
    model_path: str,
    train_data: list,
    eval_data: list,
    adapter_output: str,
    merged_output: str,
    seed: int,
) -> None:
    """Run a single curriculum phase (1, 2, or 3).

    Trains from *model_path* on *train_data*, evaluates on *eval_data*, saves
    the LoRA adapter to *adapter_output*, and merges it into *merged_output*.
    """
    from trl import SFTTrainer, SFTConfig
    from trainer_pkg.chat_template import CHAT_TEMPLATE
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Phase {phase} — Curriculum Training")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = CHAT_TEMPLATE

    train_dataset, eval_dataset, max_seq_len = _prepare_datasets(
        tokenizer, train_data, eval_data
    )
    model, tokenizer = _load_model_with_lora(model_path, max_seq_len, seed)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            seed=seed,
            **_CURRICULUM_TRAINER_CONFIG,
        ),
    )

    print("\n  GPU state before training:")
    _log_gpu_stats()
    print("\n  Training...")
    trainer_stats = trainer.train()
    peak_vram = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"  Peak VRAM: {peak_vram} GB  |  Final loss: {trainer_stats.training_loss:.4f}")

    trainer.args.gradient_accumulation_steps = 1
    eval_metrics = trainer.evaluate()
    print(f"  Eval metrics: {eval_metrics}")

    model.save_pretrained(adapter_output)
    tokenizer.save_pretrained(adapter_output)
    print(f"  Adapter saved to '{adapter_output}'.")
    _cleanup(model, tokenizer, trainer)

    _merge_adapter(model_path, adapter_output, merged_output)


def run_alignment_phase(
    model_path: str,
    train_data: list,
    trainer_output_dir: str,
    max_steps: int,
    loss_threshold: float,
    seed: int,
) -> int:
    """Run Phase 4 — maximal-diversity full-batch alignment.

    Trains from *model_path* on the 128-conversation alignment set, saves
    per-epoch checkpoints, then automatically selects the first checkpoint
    whose training loss is below *loss_threshold*.

    Returns
    -------
    int
        Step number of the selected checkpoint.
    """
    from trl import SFTTrainer, SFTConfig
    from trainer_pkg.chat_template import CHAT_TEMPLATE
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print("  Phase 4 — Full-Batch Alignment")
    print(f"{'='*60}")
    print(
        f"  max_steps={max_steps}  |  effective batch=128  |  "
        f"loss threshold={loss_threshold}"
    )
    print(f"  Checkpoints → '{trainer_output_dir}'")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = CHAT_TEMPLATE

    train_dataset, _, max_seq_len = _prepare_datasets(tokenizer, train_data, None)
    model, tokenizer = _load_model_with_lora(model_path, max_seq_len, seed)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=trainer_output_dir,
            max_steps=max_steps,
            seed=seed,
            **_ALIGNMENT_TRAINER_CONFIG,
        ),
    )

    print("\n  GPU state before training:")
    _log_gpu_stats()
    print("\n  Training...")
    trainer.train()
    peak_vram = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"  Peak VRAM: {peak_vram} GB")

    # Capture log history before cleanup
    log_history = trainer.state.log_history
    _cleanup(model, tokenizer, trainer)

    print("\n  Scanning checkpoints for first loss below threshold...")
    return _find_phase4_checkpoint(log_history, trainer_output_dir, loss_threshold)


# ---------------------------------------------------------------------------
# Sanity-check inference
# ---------------------------------------------------------------------------


def run_inference_check(model_path: str) -> None:
    """Generate one response to confirm the chat template is intact."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from trainer_pkg.chat_template import CHAT_TEMPLATE

    print("\n  Sanity-check inference ('How many days till Christmas?')...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = model.to("cuda")

    messages = [{
        "role": "user",
        "content": "How many days till Christmas?",
        "timestamp": datetime.now().isoformat()[:19],
    }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    _cleanup(model, tokenizer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the four-phase curriculum training pipeline."""
    args = get_args()

    cfg = _MODEL_CONFIGS[args.model]
    hf_name = cfg["hf_name"]
    size = args.model
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)

    _init_environment(args.seed)

    # Resolve checkpoint paths — mirrors the notebook naming convention
    paths = {
        "base":         hf_name,
        "phase1_adapt": str(out / f"{hf_name}phase1_adapter"),
        "phase1":       str(out / f"{hf_name}phase1"),
        "phase2_adapt": str(out / f"{hf_name}phase2_adapter"),
        "phase2":       str(out / f"{hf_name}phase2"),
        "phase3_adapt": str(out / f"{hf_name}phase3_adapter"),
        "phase3":       str(out / f"{hf_name}phase3"),
        "trainer_out":  str(out / f"trainer_output{size}"),
        "final":        str(out / f"TIME-{size}"),
    }

    print("\nTIME Training Pipeline")
    print(f"  Model      : {hf_name}")
    print(f"  Data       : {data_dir}")
    print(f"  Outputs    : {out}")
    print(f"  Seed       : {args.seed}")
    print(f"  Start phase: {args.start_phase}")

    # ------------------------------------------------------------------ #
    # Phase 1 — Single-turn structural seeding
    # ------------------------------------------------------------------ #
    if args.start_phase <= 1:
        train, test = _load_split(
            str(data_dir / "phase1_train.json"),
            str(data_dir / "phase1_test.json"),
        )
        print(f"\nPhase 1 data: {len(train):,} train  {len(test):,} test")
        run_curriculum_phase(
            phase=1,
            model_path=paths["base"],
            train_data=train,
            eval_data=test,
            adapter_output=paths["phase1_adapt"],
            merged_output=paths["phase1"],
            seed=args.seed,
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — Two-turn temporal scenarios
    # ------------------------------------------------------------------ #
    if args.start_phase <= 2:
        train, test = _load_split(
            str(data_dir / "phase2_train.json"),
            str(data_dir / "phase2_test.json"),
        )
        print(f"\nPhase 2 data: {len(train):,} train  {len(test):,} test")
        run_curriculum_phase(
            phase=2,
            model_path=paths["phase1"],
            train_data=train,
            eval_data=test,
            adapter_output=paths["phase2_adapt"],
            merged_output=paths["phase2"],
            seed=args.seed,
        )

    # ------------------------------------------------------------------ #
    # Phase 3 — Multi-turn generalisation
    # ------------------------------------------------------------------ #
    if args.start_phase <= 3:
        train, test = _load_split(
            str(data_dir / "phase3_train.json"),
            str(data_dir / "phase3_test.json"),
        )
        print(f"\nPhase 3 data: {len(train):,} train  {len(test):,} test")
        run_curriculum_phase(
            phase=3,
            model_path=paths["phase2"],
            train_data=train,
            eval_data=test,
            adapter_output=paths["phase3_adapt"],
            merged_output=paths["phase3"],
            seed=args.seed,
        )
        run_inference_check(paths["phase3"])

    # ------------------------------------------------------------------ #
    # Phase 4 — Full-batch alignment + auto checkpoint selection
    # ------------------------------------------------------------------ #
    if args.start_phase <= 4:
        train_data = _load_phase4(str(data_dir / "phase4.json"))
        print(f"\nPhase 4 data: {len(train_data):,} conversations (no test split)")

        selected_step = run_alignment_phase(
            model_path=paths["phase3"],
            train_data=train_data,
            trainer_output_dir=paths["trainer_out"],
            max_steps=cfg["phase4_max_steps"],
            loss_threshold=args.phase4_loss_threshold,
            seed=args.seed,
        )

        checkpoint_path = str(Path(paths["trainer_out"]) / f"checkpoint-{selected_step}")
        print(f"\n  Merging Phase 4 checkpoint-{selected_step} → '{paths['final']}'...")
        _merge_adapter(paths["phase3"], checkpoint_path, paths["final"])

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Final model: {paths['final']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
