from typing import Literal
import math
import torch
import triton
import triton.language as tl

DTYPE_TO_PACK = {"int8": 2, "int32": 8}


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # lanes & shifts — один раз на kernel, а не в каждом inner-цикле
    lanes = tl.arange(0, elems_per_pack)
    shifts = (lanes[:, None] * 4).to(tl.int32)

    for k_iter in range(NUM_K):
        k_start = k_iter * BLOCK_K

        for pack_idx in range(BLOCK_K // elems_per_pack):
            pack_col = (k_start // elems_per_pack) + pack_idx
            mask_pack = pack_col < PACKS_PER_ROW

            # загружаем packed B
            b_ptrs = b_ptr + offs_n * stride_bo + pack_col * stride_bp
            packs = tl.load(
                b_ptrs,
                mask=mask_n & mask_pack,
                other=0,
            ).to(tl.int32)

            # загружаем scales
            s_ptrs = scales_ptr + offs_n * stride_scales_n + pack_col * stride_scales_p
            scales = tl.load(
                s_ptrs,
                mask=mask_n & mask_pack,
                other=0.0,
            ).to(tl.float32)

            # соответствующие этому pack’у K-колонки
            k_cols = k_start + pack_idx * elems_per_pack + lanes
            mask_k = k_cols < K

            # грузим A по этим колонкам K
            a_ptrs_l = a_ptr + offs_m[:, None] * stride_am + k_cols[None, :] * stride_ak
            a_lanes = tl.load(
                a_ptrs_l,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(
                tl.float32
            )  # [BM, L]

            # распаковываем int4 и применяем scale
            nibbles = ((packs[None, :] >> shifts) & 0xF).to(tl.float32) - 8.0  # [L, BN]
            b_lanes = tl.where(
                mask_k[:, None],
                nibbles * scales[None, :],
                0.0,
            )

            # маленький GEMM по L
            acc += tl.sum(a_lanes[:, :, None] * b_lanes[None, :, :], axis=1)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
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
    """Computes A(bf16) @ (dequantize_i4(B, scales))^T, using per-pack scales."""
    if a.dtype != torch.bfloat16:
        raise TypeError("expected bf16 activations")

    if a.ndim < 2:
        raise ValueError("input tensor must have at least 2 dimensions")
    *batch_dims, K = a.shape
    M_flat = math.prod(batch_dims) if batch_dims else a.shape[0]

    a_2d = a.reshape(M_flat, K).contiguous()

    if not b_packed.is_contiguous():
        b_packed = b_packed.contiguous()
    if not scales.is_contiguous():
        scales = scales.contiguous()

    N, packs_per_row = b_packed.shape
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    if packs_per_row * elems_per_pack != K:
        raise ValueError("packed weights do not match K")
    if scales.shape != (N, packs_per_row):
        raise ValueError("scales must be [N, packs_per_row] to match packed weights")

    out_2d = torch.empty((M_flat, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M_flat, block_m), triton.cdiv(N, block_n))
    num_k = (K + block_k - 1) // block_k

    _matmul_kernel[grid](
        a_2d,
        b_packed,
        scales,
        out_2d,
        M_flat,
        N,
        K,
        a_2d.stride(0),
        a_2d.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        scales.stride(0),
        scales.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        elems_per_pack=elems_per_pack,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_K=num_k,
        PACKS_PER_ROW=packs_per_row,
    )

    if batch_dims:
        out = out_2d.reshape(*batch_dims, N)
    else:
        out = out_2d
    return out
