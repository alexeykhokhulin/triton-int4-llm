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
    stride_scales,
    stride_out_m,
    elems_per_pack: tl.constexpr,
    block_packs: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_pack = tl.program_id(1)
    if pid_m >= M:
        return
    scale = tl.load(scales_ptr + pid_m * stride_scales)
    pack_ids = pid_pack * block_packs + tl.arange(0, block_packs)
    num_packs = (N + elems_per_pack - 1) // elems_per_pack
    mask_pack = pack_ids < num_packs
    base_cols = pack_ids[:, None] * elems_per_pack + tl.arange(0, elems_per_pack)[None, :]
    mask_cols = base_cols < N
    ptrs = x_ptr + pid_m * stride_x_m + base_cols * stride_x_n
    vals = tl.load(ptrs, mask=mask_cols, other=0.0)
    q = tl.round(vals / scale)
    q = tl.maximum(tl.minimum(q, 7.0), -8.0).to(tl.int32) + 8
    pack_vals = tl.zeros([block_packs], dtype=tl.int32)
    for i in range(elems_per_pack):
        pack_vals |= (q[:, i] & 0xF) << (4 * i)
    if elems_per_pack == 2:
        pack_vals = pack_vals.to(tl.int8)
    out_ptrs = out_ptr + pid_m * stride_out_m + pack_ids
    tl.store(out_ptrs, pack_vals, mask=mask_pack)


def quantize_i4_pack8(
    x: torch.Tensor,
    pack_dtype: Literal["int8", "int32"] = "int32",
    block_packs: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes fp16 into packed int4."""
    if x.dtype != torch.float16:
        raise TypeError("expected fp16 input")
    if not x.is_contiguous():
        x = x.contiguous()
    m, n = x.shape
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    if n % elems_per_pack != 0:
        raise ValueError("cols must align to pack size")
    scales = x.abs().amax(dim=1).clamp_min(EPS).to(torch.float32) / 7.0
    if pack_dtype == "int8":
        out = torch.empty((m, n // 2), dtype=torch.int8, device=x.device)
    else:
        out = torch.empty((m, n // 8), dtype=torch.int32, device=x.device)
    grid = (m, triton.cdiv(out.shape[1], block_packs))
    _quant_pack[grid](
        x,
        scales,
        out,
        m,
        n,
        x.stride(0),
        x.stride(1),
        scales.stride(0),
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
    """Dequantizes packed int4 back to fp32."""
    elems_per_pack = DTYPE_TO_PACK[pack_dtype]
    m = packed.shape[0]
    k = packed.shape[1] * elems_per_pack
    out = torch.empty((m, k), dtype=torch.float32, device=packed.device)
    if pack_dtype == "int8":
        data = packed.view(torch.int8).to(torch.int32)
    else:
        data = packed.to(torch.int32)
    for lane in range(elems_per_pack):
        shift = lane * 4
        vals = (data >> shift) & 0xF
        vals = vals.view(m, -1)
        chunk = vals.to(torch.float32) - 8.0
        out[:, lane::elems_per_pack] = chunk * scales[:, None]
    return out 
