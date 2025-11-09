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
    stride_scales_n,
    stride_scales_p,
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

    for k_iter in range(NUM_K):
        k_start = k_iter * BLOCK_K

        for pack_idx in range(BLOCK_K // elems_per_pack):
            pack_col = (k_start // elems_per_pack) + pack_idx
            mask_col = pack_col < PACKS_PER_ROW

            b_ptrs = b_ptr + offs_n * stride_bo + pack_col * stride_bp
            packs = tl.load(b_ptrs, mask=(offs_n < N) & mask_col, other=0).to(tl.int32)

            s_ptrs = scales_ptr + offs_n * stride_scales_n + pack_col * stride_scales_p
            scales = tl.load(s_ptrs, mask=(offs_n < N) & mask_col, other=0.0).to(tl.float32)

            lanes = tl.arange(0, elems_per_pack)
            k_cols = k_start + pack_idx * elems_per_pack + lanes
            mask_k = k_cols < K

            a_ptrs_l = a_ptr + offs_m[:, None] * stride_am + k_cols[None, :] * stride_ak
            a_lanes = tl.load(
                a_ptrs_l, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0
            ).to(tl.float32)  # [BM, L]

            shifts = (lanes[:, None] * 4).to(tl.int32)
            nibbles = ((packs[None, :] >> shifts) & 0xF).to(tl.float32) - 8.0  # [L, BN]
            b_lanes = tl.where(mask_k[:, None], nibbles * scales[None, :], 0.0)

            acc += tl.sum(a_lanes[:, :, None] * b_lanes[None, :, :], axis=1)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_bf16_i4(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    pack_dtype: Literal["int8", "int32"] = "int32",
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Computes A(bf16) @ (dequantize_i4(B, scales))^T, using per-pack scales."""
    if a.dtype != torch.bfloat16:
        raise TypeError("expected bf16 activations")
    if not a.is_contiguous():
        a = a.contiguous()
    if not b_packed.is_contiguous():
        b_packed = b_packed.contiguous()
    if not scales.is_contiguous():
        scales = scales.contiguous()

    M, K = a.shape
    N, packs_per_row = b_packed.shape
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    if packs_per_row * elems_per_pack != K:
        raise ValueError("packed weights do not match K")
    if scales.shape != (N, packs_per_row):
        raise ValueError("scales must be [N, packs_per_row] to match packed weights")

    out = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    num_k = (K + block_k - 1) // block_k

    _matmul_kernel[grid](
        a,
        b_packed,
        scales,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        scales.stride(0),
        scales.stride(1),
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
