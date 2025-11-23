import argparse
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoConfig

from triton_int4.triton_kernels.matmul_bf16_i4_pack8 import matmul_bf16_i4
from triton_int4.triton_kernels.quant_i4_pack8 import quantize_i4_pack8


LLAMA_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def load_shapes(model_id: str) -> List[Tuple[int, int]]:
    """Returns llama linear shapes."""
    cfg = AutoConfig.from_pretrained(model_id)
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    return [
        (hidden, hidden),
        (hidden, inter),
        (inter, hidden),
    ]


def benchmark_case(
    x_tokens: int,
    shape: Tuple[int, int],
    device: str,
    iters: int,
    group_size: int,
) -> Tuple[float, float]:
    in_features = shape[1]
    out_features = shape[0]
    x = torch.randn(x_tokens, in_features, dtype=torch.bfloat16, device=device)
    w = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    # if quantize_i4_pack8 has no group_size arg, drop that kwarg
    packed, scales = quantize_i4_pack8(w, group_size=group_size)

    def run_int4() -> torch.Tensor:
        return matmul_bf16_i4(x, packed, scales)

    def run_fp16() -> torch.Tensor:
        return torch.matmul(x.to(torch.float16), w.t()).to(torch.float32)

    # warmup
    for _ in range(5):
        run_int4()
        run_fp16()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start_int4 = torch.cuda.Event(True) if device.startswith("cuda") else None
    end_int4 = torch.cuda.Event(True) if device.startswith("cuda") else None
    start_fp = torch.cuda.Event(True) if device.startswith("cuda") else None
    end_fp = torch.cuda.Event(True) if device.startswith("cuda") else None

    if start_int4 is not None:
        start_int4.record()
        for _ in range(iters):
            run_int4()
        end_int4.record()
        end_int4.synchronize()
        int4_ms = start_int4.elapsed_time(end_int4) / iters

        start_fp.record()
        for _ in range(iters):
            run_fp16()
        end_fp.record()
        end_fp.synchronize()
        fp16_ms = start_fp.elapsed_time(end_fp) / iters
        torch.cuda.synchronize()
        return int4_ms, fp16_ms

    import time

    t0 = time.perf_counter()
    for _ in range(iters):
        run_int4()
    int4_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        run_fp16()
    fp16_time = time.perf_counter() - t0
    return (int4_time / iters) * 1e3, (fp16_time / iters) * 1e3


def parse_tokens(raw: Sequence[int]) -> Iterable[int]:
    seen = []
    for item in raw:
        if item not in seen:
            seen.append(item)
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark X16@W4^T vs X16@W16^T")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=LLAMA_MODEL)
    parser.add_argument("--tokens", nargs="*", type=int, default=[128, 512, 2048])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--group-size",
        type=int,
        default=256,
        help="Group size for int4 quantization (columns per scale group).",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark")
    device = args.device
    shapes = load_shapes(args.model)
    tokens = list(parse_tokens(args.tokens))
    print("tokens,out,in,int4_ms,fp16_ms,speedup")
    for shape in shapes:
        for t in tokens:
            int4_ms, fp16_ms = benchmark_case(t, shape, device, args.iters, args.group_size)
            speedup = fp16_ms / int4_ms if int4_ms > 0 else float("nan")
            print(f"{t},{shape[0]},{shape[1]},{int4_ms:.4f},{fp16_ms:.4f},{speedup:.2f}x")


if __name__ == "__main__":
    main()