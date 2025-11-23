from typing import Literal, Tuple

import torch
import triton
import triton.language as tl

DTYPE_TO_PACK = {"int8": 2, "int32": 8}
EPS = 1e-8


@triton.jit
def _quant_pack(
    x_ptr,
    scales_ptr,
    out_ptr,
    M,
    N,
    stride_x_m,
    stride_x_n,
    stride_scales_m,
    stride_scales_p,
    stride_out_m,
    elems_per_pack: tl.constexpr,
    block_packs: tl.constexpr,
    packs_per_group: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_pack = tl.program_id(1)
    if pid_m >= M:
        return

    # which output packs we handle
    pack_ids = pid_pack * block_packs + tl.arange(0, block_packs)
    num_packs = (N + elems_per_pack - 1) // elems_per_pack
    mask_pack = pack_ids < num_packs

    # lanes within each pack
    lanes = tl.arange(0, elems_per_pack)[None, :]
    base_cols = pack_ids[:, None] * elems_per_pack + lanes
    mask_cols = base_cols < N

    # map pack -> group id (group-wise scales)
    group_ids = pack_ids // packs_per_group

    # per-group scales (broadcast over packs in the same group)
    s_ptrs = scales_ptr + pid_m * stride_scales_m + group_ids * stride_scales_p
    scales = tl.load(s_ptrs, mask=mask_pack, other=1.0).to(tl.float32)
    scales = scales[:, None]

    # load & quantize in fp32
    ptrs = x_ptr + pid_m * stride_x_m + base_cols * stride_x_n
    vals = tl.load(ptrs, mask=mask_cols, other=0.0).to(tl.float32)

    # symmetric int4 quantization with ties-to-even rounding
    x = vals / scales
    f = tl.math.floor(x)
    r = tl.math.floor(x + 0.5)
    is_half = (x - f) == 0.5
    # if it's an exact half and r would be odd, subtract 1 to make it even
    r_adj = tl.where(is_half & ((r.to(tl.int32) & 1) == 1), r - 1.0, r)
    qf = tl.maximum(tl.minimum(r_adj, 7.0), -8.0)
    q = qf.to(tl.int32) + 8  # store as 0..15

    # pack 8 nibbles -> 1 int32 (vectorized)
    shifts = (tl.arange(0, elems_per_pack)[None, :] * 4).to(tl.int32)
    nibbles = (q & 0xF) << shifts
    pack_vals = tl.sum(nibbles, axis=1)
    if elems_per_pack == 2:
        pack_vals = pack_vals.to(tl.int8)

    out_ptrs = out_ptr + pid_m * stride_out_m + pack_ids
    tl.store(out_ptrs, pack_vals, mask=mask_pack)


def quantize_i4_pack8(
    x: torch.Tensor,
    pack_dtype: Literal["int8", "int32"] = "int32",
    group_size: int = 256,
    block_packs: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes fp16 -> packed int4 with group-wise scales.

    - int4 stored as 8 nibbles per int32 (or 2 per int8) = "pack8"
    - scales are symmetric, one per group of `group_size` columns
    - group_size is clamped to at most N and must be a multiple of the pack size
    """
    if x.dtype != torch.float16:
        raise TypeError("expected fp16 input")
    if not x.is_contiguous():
        x = x.contiguous()

    M, N = x.shape
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    if N % elems_per_pack != 0:
        raise ValueError("cols must align to pack size")

    # effective group size: not larger than N
    g = min(group_size, N)
    if g % elems_per_pack != 0:
        raise ValueError("group_size must be a multiple of pack size")
    if N % g != 0:
        raise ValueError("cols must align to group size")

    num_groups = N // g
    num_packs = N // elems_per_pack
    packs_per_group = g // elems_per_pack

    x32 = x.float()
    # [M, num_groups, group_size] -> max abs per group
    scales = (
        x32.abs()
        .view(M, num_groups, g)
        .amax(dim=2)
        .clamp_min(EPS)
        / 7.0
    ).contiguous()  # [M, num_groups]

    if pack_dtype == "int8":
        out = torch.empty((M, num_packs), dtype=torch.int8, device=x.device)
    else:
        out = torch.empty((M, num_packs), dtype=torch.int32, device=x.device)

    grid = (M, triton.cdiv(num_packs, block_packs))
    _quant_pack[grid](
        x,
        scales,
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        elems_per_pack=elems_per_pack,
        block_packs=block_packs,
        packs_per_group=packs_per_group,
    )
    return out, scales


def dequantize_i4_pack8(
    packed: torch.Tensor,
    scales: torch.Tensor,
    pack_dtype: Literal["int8", "int32"] = "int32",
) -> torch.Tensor:
    """
    Dequantizes packed int4 back to fp32 using group-wise scales.

    - `packed`: [M, num_packs], int8/int32 packs of 8 nibbles
    - `scales`: [M, num_groups], one scale per group of columns
      (group size is inferred from K and num_groups)
    """
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    M, num_packs = packed.shape
    M_s, num_groups = scales.shape
    if M_s != M:
        raise ValueError("scales first dim must match packed")

    K = num_packs * elems_per_pack
    if K % num_groups != 0:
        raise ValueError("K must be divisible by number of groups in scales")

    group_size = K // num_groups
    out = torch.empty((M, K), dtype=torch.float32, device=packed.device)

    data = packed.to(torch.int32)  # [M, num_packs]

    # build per-column scales: [M, K]
    device = packed.device
    col_idx = torch.arange(K, device=device)
    group_idx = col_idx // group_size
    scale_cols = scales[:, group_idx]

    pack_idx = torch.arange(num_packs, device=device)

    # decode each lane (nibble) into its column positions
    for lane in range(elems_per_pack):
        shift = lane * 4
        vals = (data >> shift) & 0xF
        q = vals.to(torch.float32) - 8.0
        cols = pack_idx * elems_per_pack + lane
        out[:, cols] = q * scale_cols[:, cols]

    return out
