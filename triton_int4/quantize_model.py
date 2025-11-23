import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from triton_int4.quant_layer import replace_linear_with_int4


def parse_args() -> argparse.Namespace:
    """Parses args."""
    parser = argparse.ArgumentParser(
        description="Quantize HF model to int4 packed weights"
    )
    parser.add_argument("model", help="model name or path")
    parser.add_argument("output", help="where to save quantized model")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run quantization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    model = replace_linear_with_int4(model)
    model.save_pretrained(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
