import math
from typing import Tuple

import torch


def quant_i4_sym_rowwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric row-wise int4: one scale per row."""
    assert w.ndim == 2
    w_fp32 = w.to(torch.float32)
    max_abs = w_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = max_abs / 7.0
    q = torch.round(w_fp32 / scales).clamp(-8, 7)
    q_int = q.to(torch.int8)  # store as int8, each value in [-8, 7]
    w_hat = (q_int.to(torch.float32) * scales).to(w_fp32.dtype)
    return w_hat, scales.squeeze(1)


def quant_i4_asym_rowwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric row-wise int4: scale + zero-point per row, uint4 in [0, 15]."""
    assert w.ndim == 2
    w_fp32 = w.to(torch.float32)
    w_min = w_fp32.amin(dim=1, keepdim=True)
    w_max = w_fp32.amax(dim=1, keepdim=True)
    scales = (w_max - w_min).clamp(min=1e-8) / 15.0
    q = torch.round((w_fp32 - w_min) / scales).clamp(0, 15)  # [0..15]
    q_int = q.to(torch.int8)
    w_hat = (q_int.to(torch.float32) * scales + w_min).to(w_fp32.dtype)
    return w_hat, scales.squeeze(1), w_min.squeeze(1)


def quant_i4_sym_groupwise(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric group-wise int4 along columns (per-row groups)."""
    assert w.ndim == 2
    M, N = w.shape
    assert N % group_size == 0
    w_fp32 = w.to(torch.float32).view(M, N // group_size, group_size)
    max_abs = w_fp32.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # [M, num_groups, 1]
    scales = max_abs / 7.0
    q = torch.round(w_fp32 / scales).clamp(-8, 7)
    q_int = q.to(torch.int8)
    w_hat = (q_int.to(torch.float32) * scales).view(M, N).to(w.dtype)
    return w_hat, scales.squeeze(2)