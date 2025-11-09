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

    # per-pack scales
    s_ptrs = scales_ptr + pid_m * stride_scales_m + pack_ids * stride_scales_p
    scales = tl.load(s_ptrs, mask=mask_pack, other=1.0).to(tl.float32)
    scales = scales[:, None]

    # load & quantize in fp32
    ptrs = x_ptr + pid_m * stride_x_m + base_cols * stride_x_n
    vals = tl.load(ptrs, mask=mask_cols, other=0.0).to(tl.float32)

    # x = vals / scales
    # qf = tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5))
    # qf = tl.maximum(tl.minimum(qf, 7.0), -8.0)
    # q  = qf.to(tl.int32) + 8                                              # 0..15
    x = vals / scales
    f = tl.math.floor(x)
    r = tl.math.floor(x + 0.5)
    is_half = (x - f) == 0.5
    # if it's an exact half and r would be odd, subtract 1 to make it even
    r_adj = tl.where(is_half & ((r.to(tl.int32) & 1) == 1), r - 1.0, r)
    qf = tl.maximum(tl.minimum(r_adj, 7.0), -8.0)
    q = qf.to(tl.int32) + 8

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
    block_packs: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes fp16 -> packed int4 with per-pack scales (one scale per 8 values)."""
    if x.dtype != torch.float16:
        raise TypeError("expected fp16 input")
    if not x.is_contiguous():
        x = x.contiguous()
    M, N = x.shape
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    if N % elems_per_pack != 0:
        raise ValueError("cols must align to pack size")

    num_packs = N // elems_per_pack
    x32 = x.float()
    scales = (
        x32.abs().view(M, num_packs, elems_per_pack).amax(dim=2).clamp_min(EPS) / 7.0
    ).contiguous()

    # output buffer
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
    )
    return out, scales


def dequantize_i4_pack8(
    packed: torch.Tensor,
    scales: torch.Tensor,
    pack_dtype: Literal["int8", "int32"] = "int32",
) -> torch.Tensor:
    """Dequantizes packed int4 back to fp32 using per-pack scales."""
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    M, num_packs = packed.shape
    K = num_packs * elems_per_pack
    out = torch.empty((M, K), dtype=torch.float32, device=packed.device)

    data = packed.to(torch.int32)
    # each lane writes into columns lane, lane+L, lane+2L, ...
    for lane in range(elems_per_pack):
        shift = lane * 4
        vals = (data >> shift) & 0xF
        chunk = vals.to(torch.float32) - 8.0
        out[:, lane::elems_per_pack] = chunk * scales
    return out
