import argparse
import torch
from transformers import AutoConfig

from triton_int4.quant.quant_schemes import (
    quant_i4_sym_rowwise,
    quant_i4_asym_rowwise,
    quant_i4_sym_groupwise,
)


def load_shapes(model_id: str):
    """Return typical LLaMA linear layer shapes."""
    cfg = AutoConfig.from_pretrained(model_id)
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    return [
        (h, h),  # attn / output proj
        (h, inter),  # MLP up
        (inter, h),  # MLP down
    ]


def bytes_fp16(shape):
    """FP16: 2 bytes per weight."""
    M, N = shape
    return 2 * M * N


def bytes_sym_rowwise(shape):
    """Symmetric row-wise int4: 4 bits per weight + 1 fp32 scale per row."""
    M, N = shape
    w_bytes = M * N / 2.0  # 4 bits per weight
    scales = M * 4  # float32 per row
    return w_bytes + scales


def bytes_asym_rowwise(shape):
    """Asymmetric row-wise int4: 4 bits per weight + scale + zero-point per row."""
    M, N = shape
    w_bytes = M * N / 2.0
    scales = M * 4  # scale per row
    zeros = M * 4  # zero-point per row
    return w_bytes + scales + zeros


def bytes_groupwise(shape, group_size: int):
    """Symmetric group-wise int4: 4 bits per weight + one fp32 scale per group."""
    M, N = shape
    assert N % group_size == 0, f"N={N} is not divisible by group_size={group_size}"
    num_groups = N // group_size
    w_bytes = M * N / 2.0  # 4 bits per weight
    scales = M * num_groups * 4  # one scale per group
    return w_bytes + scales


def main():
    parser = argparse.ArgumentParser(
        description="Compare int4 quantization schemes on LLaMA-3.2-1B-style shapes."
    )
    parser.add_argument(
        "--model",
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Model id to read hidden/intermediate sizes from.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generating random weights.",
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="*",
        default=[8, 128, 256, 512, 1024, 2048, 4096],
        help="Group sizes to test for symmetric group-wise quantization.",
    )
    args = parser.parse_args()

    shapes = load_shapes(args.model)
    torch.manual_seed(0)

    print("shape,scheme,mae,memory_reduction")
    for shape in shapes:
        M, N = shape
        w = torch.randn(M, N, dtype=torch.float16, device=args.device)
        w_fp32 = w.to(torch.float32)
        base_bytes = bytes_fp16(shape)

        # symmetric row-wise
        w_hat_sym, _ = quant_i4_sym_rowwise(w)
        mae_sym = (w_fp32 - w_hat_sym.to(torch.float32)).abs().mean().item()
        mem_sym = base_bytes / bytes_sym_rowwise(shape)
        print(f"{M}x{N},sym_row,{mae_sym:.5f},{mem_sym:.2f}x")

        # asymmetric row-wise
        w_hat_asym, _, _ = quant_i4_asym_rowwise(w)
        mae_asym = (w_fp32 - w_hat_asym.to(torch.float32)).abs().mean().item()
        mem_asym = base_bytes / bytes_asym_rowwise(shape)
        print(f"{M}x{N},asym_row,{mae_asym:.5f},{mem_asym:.2f}x")

        # symmetric group-wise for multiple group sizes
        for g in args.group_sizes:
            if N % g != 0:
                # skip incompatible group sizes for this shape
                continue
            w_hat_grp, _ = quant_i4_sym_groupwise(w, g)
            mae_grp = (w_fp32 - w_hat_grp.to(torch.float32)).abs().mean().item()
            mem_grp = base_bytes / bytes_groupwise(shape, g)
            print(f"{M}x{N},sym_group_g{g},{mae_grp:.5f},{mem_grp:.2f}x")


if __name__ == "__main__":
    main()