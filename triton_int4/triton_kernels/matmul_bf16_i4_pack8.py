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
    stride_scales_g,
    stride_cm,
    stride_cn,
    elems_per_pack: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    PACKS_PER_ROW,
    PACKS_PER_GROUP: tl.constexpr,
):
    """
    A: [M, K]  (bf16)
    B_packed: [N, packs_per_row] (int32/int8, 8 int4 per pack)
    scales: [N, num_groups]  (group-wise); num_groups = packs_per_row / PACKS_PER_GROUP
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(NUM_K):
        k_start = k_iter * BLOCK_K

        # iterate over packs within this K block
        for pack_idx in range(BLOCK_K // elems_per_pack):
            pack_col = (k_start // elems_per_pack) + pack_idx
            mask_pack = pack_col < PACKS_PER_ROW

            # load B packs: [BN]
            b_ptrs = b_ptr + offs_n * stride_bo + pack_col * stride_bp
            packs = tl.load(b_ptrs, mask=(offs_n < N) & mask_pack, other=0).to(tl.int32)

            # map pack index -> group index
            group_col = pack_col // PACKS_PER_GROUP

            # load group-wise scales: [BN]
            s_ptrs = scales_ptr + offs_n * stride_scales_n + group_col * stride_scales_g
            scales = tl.load(s_ptrs, mask=(offs_n < N) & mask_pack, other=1.0).to(tl.float32)

            # columns in K covered by this pack
            lanes = tl.arange(0, elems_per_pack)
            k_cols = k_start + pack_idx * elems_per_pack + lanes
            mask_k = k_cols < K

            # load A lanes: [BM, L]
            a_ptrs_l = a_ptr + offs_m[:, None] * stride_am + k_cols[None, :] * stride_ak
            a_lanes = tl.load(
                a_ptrs_l,
                mask=(offs_m[:, None] < M) & (mask_k[None, :]),
                other=0.0,
            ).to(tl.float32)

            # unpack int4 nibbles → [-8..7]
            shifts = (lanes[:, None] * 4).to(tl.int32)
            nibbles = ((packs[None, :] >> shifts) & 0xF).to(tl.float32) - 8.0  # [L, BN]

            # apply group-wise scales, broadcast over lanes: [L, BN]
            b_lanes = tl.where(mask_k[:, None], nibbles * scales[None, :], 0.0)

            # accumulate: [BM, BN]
            acc += tl.sum(a_lanes[:, :, None] * b_lanes[None, :, :], axis=1)

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
    """
    Computes A(bf16) @ dequantize_i4(B, scales)^T.

    - A can be 2D [M, K] or higher-rank [..., K]; we flatten to [M_flat, K].
    - B_packed is [N, packs_per_row].
    - scales is [N, S]:
        * S == packs_per_row → per-pack scales (original layout)
        * S < packs_per_row and packs_per_row % S == 0 → group-wise scales.
    """
    if a.dtype != torch.bfloat16:
        raise TypeError("expected bf16 activations")

    if not a.is_contiguous():
        a = a.contiguous()
    if not b_packed.is_contiguous():
        b_packed = b_packed.contiguous()
    if not scales.is_contiguous():
        scales = scales.contiguous()

    # flatten A to 2D: [M_flat, K]
    if a.ndim < 2:
        raise ValueError("input tensor must have at least 2 dimensions")
    *batch_dims, K = a.shape
    if batch_dims:
        M_flat = 1
        for d in batch_dims:
            M_flat *= d
    else:
        M_flat = a.shape[0]
    a_2d = a.reshape(M_flat, K)

    N, packs_per_row = b_packed.shape
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    if packs_per_row * elems_per_pack != K:
        raise ValueError("packed weights do not match K")

    if scales.ndim != 2 or scales.shape[0] != N:
        raise ValueError("scales must be [N, S]")

    num_groups = scales.shape[1]
    if num_groups <= 0:
        raise ValueError("scales second dim must be > 0")

    if packs_per_row % num_groups != 0:
        raise ValueError(
            f"packed columns (packs_per_row={packs_per_row}) "
            f"do not align with scale groups (num_groups={num_groups})"
        )
    packs_per_group = packs_per_row // num_groups

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
        PACKS_PER_GROUP=packs_per_group,
    )

    if batch_dims:
        return out_2d.reshape(*batch_dims, N)
    return out_2d