import argparse
import math
import time
from tqdm.auto import tqdm

from typing import Iterable, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from triton_int4.quant_layer import replace_linear_with_int4


def parse_args() -> argparse.Namespace:
    """Parses args."""
    parser = argparse.ArgumentParser(description="WikiText-2 perplexity + speed")
    parser.add_argument("model", help="model name or path", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def chunk_tokens(tokenizer, text: str, seq_len: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    if ids.numel() < seq_len:
        return ids.new_empty((0, seq_len))
    return ids.unfold(0, seq_len, seq_len)


def iter_dataset(dataset: Dataset, limit: int) -> Iterable:
    if limit > 0:
        for sample in dataset.select(range(min(limit, len(dataset)))):
            yield sample
    else:
        for sample in dataset:
            yield sample


def evaluate(
    model,
    tokenizer,
    device: str,
    seq_len: int,
    batch_size: int,
    dataset: Dataset,
    limit: int,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_tokens = 0
    total_time = 0.0
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    model.eval()

    total_samples = min(limit, len(dataset)) if limit > 0 else len(dataset)

    for sample in tqdm(iter_dataset(dataset, limit), total=total_samples, desc="Evaluating"):
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
            start_event = torch.cuda.Event(True) if use_cuda else None
            end_event = torch.cuda.Event(True) if use_cuda else None
            start_time = time.perf_counter() if not use_cuda else None
            with torch.no_grad():
                if use_cuda:
                    start_event.record()
                outputs = model(input_ids=inputs)
                if use_cuda:
                    end_event.record()
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )
            if use_cuda:
                end_event.synchronize()
                total_time += start_event.elapsed_time(end_event) / 1000.0
            else:
                total_time += time.perf_counter() - start_time
            total_loss += loss.item()
            total_tokens += targets.numel()

    ppl = math.exp(total_loss / total_tokens)
    tokens_per_s = total_tokens / total_time if total_time > 0 else 0.0
    return ppl, tokens_per_s


def run_eval(args, quantize: bool, dataset: Optional[Dataset] = None) -> Tuple[float, float]:
    device = args.device
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if quantize:
        model = replace_linear_with_int4(model)
    if dataset is None:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return evaluate(
        model,
        tokenizer,
        device,
        args.seq_len,
        args.batch_size,
        dataset,
        args.max_samples,
    )


def main() -> None:
    args = parse_args()
    if args.compare:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        fp16_ppl, fp16_tps = run_eval(args, quantize=False, dataset=dataset)
        int4_ppl, int4_tps = run_eval(args, quantize=True, dataset=dataset)
        print("mode,perplexity,tokens_per_s")
        print(f"fp16,{fp16_ppl:.4f},{fp16_tps:.2f}")
        print(f"int4,{int4_ppl:.4f},{int4_tps:.2f}")
    else:
        ppl, tokens_per_s = run_eval(args, quantize=args.quantize)
        label = "int4" if args.quantize else "fp16"
        print(f"mode={label}, perplexity={ppl:.4f}, tokens_per_s={tokens_per_s:.2f}")


if __name__ == "__main__":
    main()