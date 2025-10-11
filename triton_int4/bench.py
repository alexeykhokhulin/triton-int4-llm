import argparse
from typing import List, Tuple

import torch
from transformers import AutoConfig

from triton_int4.triton_kernels.matmul_bf16_i4_pack8 import matmul_bf16_i4
from triton_int4.triton_kernels.quant_i4_pack8 import quantize_i4_pack8


LLAMA_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def load_shapes() -> List[Tuple[int, int]]:
    """Returns key linear shapes."""
    cfg = AutoConfig.from_pretrained(LLAMA_MODEL)
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    return [
        (hidden, hidden),
        (hidden, inter),
        (inter, hidden),
    ]


def benchmark_case(x_tokens: int, shape: Tuple[int, int], device: str) -> Tuple[float, float]:
    in_features = shape[1]
    out_features = shape[0]
    x = torch.randn(x_tokens, in_features, dtype=torch.bfloat16, device=device)
    w = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    packed, scales = quantize_i4_pack8(w)

    def run_int4() -> torch.Tensor:
        return matmul_bf16_i4(x, packed, scales)

    def run_fp16() -> torch.Tensor:
        return torch.matmul(x.to(torch.float16), w.t()).to(torch.float32)

    for _ in range(5):
        run_int4()
        run_fp16()
    torch.cuda.synchronize()
    iters = 50
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        run_int4()
    end.record()
    end.synchronize()
    int4_ms = start.elapsed_time(end) / iters

    start_fp = torch.cuda.Event(True)
    end_fp = torch.cuda.Event(True)
    start_fp.record()
    for _ in range(iters):
        run_fp16()
    end_fp.record()
    end_fp.synchronize()
    fp16_ms = start_fp.elapsed_time(end_fp) / iters
    torch.cuda.synchronize()
    return int4_ms, fp16_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark X16@W4^T vs X16@W16^T")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark")
    device = args.device
    shapes = load_shapes()
    tokens = [128, 512, 2048]
    print("tokens, out, in, int4_ms, fp16_ms, speedup")
    for shape in shapes:
        for t in tokens:
            int4_ms, fp16_ms = benchmark_case(t, shape, device)
            speedup = fp16_ms / int4_ms
            print(f"{t},{shape[0]},{shape[1]},{int4_ms:.4f},{fp16_ms:.4f},{speedup:.2f}x")


if __name__ == "__main__":
    main()
