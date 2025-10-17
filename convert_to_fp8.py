import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def main():
    # Set up an argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(
        description="Quantize a Hugging Face model to FP8 and save it."
    )
    # The --model or -m argument is now a required input.
    # The 'required=True' flag ensures the script will not run without a model.
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The required identifier of the model to be quantized from Hugging Face.",
    )
    args = parser.parse_args()
    MODEL_ID = args.model

    # Load the model and tokenizer with automatic data type selection.
    print(f"Loading model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Define the quantization recipe, targeting linear layers with dynamic FP8.
    # The language model head is excluded from quantization.
    recipe = QuantizationModifier(
      targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    # Apply the one-shot quantization to the model.
    print("Applying one-shot FP8 quantization...")
    oneshot(model=model, recipe=recipe)

    # Determine the directory name for saving the quantized model.
    SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
    
    # Save the quantized model and its tokenizer to the specified directory.
    print(f"Saving quantized model and tokenizer to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"Model successfully quantized and saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
