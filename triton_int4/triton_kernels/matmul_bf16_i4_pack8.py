from typing import Literal

import torch
import triton
import triton.language as tl


DTYPE_TO_PACK = {"int8": 2, "int32": 8}


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bo,
    stride_bp,
    stride_scales,
    stride_cm,
    stride_cn,
    elems_per_pack: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    PACKS_PER_ROW,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scales = tl.load(
        scales_ptr + offs_n * stride_scales,
        mask=offs_n < N,
        other=0.0,
    )
    for k_iter in range(NUM_K):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_block = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        pack_base = (k_start // elems_per_pack) + tl.arange(0, BLOCK_K // elems_per_pack)
        for pack_idx in range(BLOCK_K // elems_per_pack):
            pack_col = pack_base[pack_idx]
            mask_col = pack_col < PACKS_PER_ROW
            b_ptrs = b_ptr + offs_n * stride_bo + pack_col * stride_bp
            mask_pack = (offs_n < N) & mask_col
            pack_vals = tl.load(b_ptrs, mask=mask_pack, other=0).to(tl.int32)
            for lane in range(elems_per_pack):
                k_col = k_start + pack_idx * elems_per_pack + lane
                mask_k = k_col < K
                a_vals = tl.where(mask_k, a_block[:, pack_idx * elems_per_pack + lane], 0.0)
                q_vals = ((pack_vals >> (4 * lane)) & 0xF).to(tl.float32) - 8.0
                b_vals = q_vals * scales
                b_vals = tl.where(mask_k, b_vals, 0.0)
                acc += a_vals[:, None] * b_vals[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_bf16_i4(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    pack_dtype: Literal["int8", "int32"] = "int32",
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Computes X16 @ W4^T."""
    if a.dtype != torch.bfloat16:
        raise TypeError("expected bf16 activations")
    if not a.is_contiguous():
        a = a.contiguous()
    if not b_packed.is_contiguous():
        b_packed = b_packed.contiguous()
    if not scales.is_contiguous():
        scales = scales.contiguous()
    m, k = a.shape
    n = scales.numel()
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    packs_per_row = b_packed.shape[1]
    if packs_per_row * elems_per_pack != k:
        raise ValueError("packed weights do not match K")
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    num_k = (k + block_k - 1) // block_k
    _matmul_kernel[grid](
        a,
        b_packed,
        scales,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        scales.stride(0),
        out.stride(0),
        out.stride(1),
        elems_per_pack=elems_per_pack,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_K=num_k,
        PACKS_PER_ROW=packs_per_row,
    )
    return out