import argparse
import math

import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from triton_int4.quant_layer import replace_linear_with_int4


def parse_args() -> argparse.Namespace:
    """Parses args."""
    parser = argparse.ArgumentParser(description="Evaluate WikiText-2 perplexity")
    parser.add_argument("model", help="model name or path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def chunk_tokens(tokenizer, text: str, seq_len: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    chunks = ids.unfold(0, seq_len, seq_len)
    return chunks


def evaluate(model, tokenizer, device: str, seq_len: int, batch_size: int) -> float:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for sample in dataset:
        text = sample["text"].strip()
        if not text:
            continue
        tokens = chunk_tokens(tokenizer, text, seq_len)
        if tokens.numel() == 0:
            continue
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i : i + batch_size].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            with torch.no_grad():
                outputs = model(input_ids=inputs)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )
            total_loss += loss.item()
            total_tokens += targets.numel()
    return math.exp(total_loss / total_tokens)


def main() -> None:
    args = parse_args()
    device = args.device
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.quantize:
        replace_linear_with_int4(model)
    ppl = evaluate(model, tokenizer, device, args.seq_len, args.batch_size)
    print(f"perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
