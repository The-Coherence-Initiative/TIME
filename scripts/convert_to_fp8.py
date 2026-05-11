"""Convert a merged TIME model checkpoint to FP8-Dynamic quantisation.

Quantises all linear layers (except the language model head) using one-shot
FP8-Dynamic quantisation via llmcompressor, then saves the quantised model
and tokenizer to ``<model_name>-FP8-Dynamic/`` in the current directory.

This step is optional and only needed for FP8 inference deployment (e.g. via
vLLM with FP8 kernels). Core training does not depend on llmcompressor.

Usage
-----
    python scripts/convert_to_fp8.py --model TIME-8B
    python scripts/convert_to_fp8.py --model /path/to/TIME-32B --output-dir /models
"""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantise a TIME model checkpoint to FP8-Dynamic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to quantise.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write the quantised model. "
            "Defaults to '<model_basename>-FP8-Dynamic' in the current directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load, quantise, and save the model."""
    args = get_args()

    print(f"Loading model: {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],   # exclude the unembedding head from quantisation
    )

    print("Applying one-shot FP8-Dynamic quantisation...")
    oneshot(model=model, recipe=recipe)

    if args.output_dir:
        save_dir = args.output_dir
    else:
        basename = args.model.rstrip("/").split("/")[-1]
        save_dir = f"{basename}-FP8-Dynamic"

    print(f"Saving quantised model and tokenizer to '{save_dir}'...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Done. Quantised model saved to '{save_dir}'.")


if __name__ == "__main__":
    main()
