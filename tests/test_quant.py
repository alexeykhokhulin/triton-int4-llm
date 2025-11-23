import math

import pytest
import torch

from triton_int4.quant_layer import Int4PackedLinear, replace_linear_with_int4
from triton_int4.triton_kernels.quant_i4_pack8 import (
    dequantize_i4_pack8,
    quantize_i4_pack8,
)
from triton_int4.triton_kernels.matmul_bf16_i4_pack8 import matmul_bf16_i4


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"


def test_memory_reduction(device):
    x = torch.randn(64, 512, dtype=torch.float16, device=device)
    packed, _ = quantize_i4_pack8(x)
    orig_bytes = x.numel() * x.element_size()
    packed_bytes = packed.numel() * packed.element_size()
    assert math.isclose(packed_bytes / orig_bytes, 0.25, rel_tol=1e-3)


def test_roundtrip_accuracy(device):
    x = torch.randn(32, 256, dtype=torch.float16, device=device)
    packed, scales = quantize_i4_pack8(x)
    restored = dequantize_i4_pack8(packed, scales)
    x_fp32 = x.to(torch.float32)
    diff = (x_fp32 - restored).abs().mean()
    rel = diff / (x_fp32.abs().mean() + 1e-6)
    assert rel < 0.15


def test_matmul_kernel_matches_dequant(device):
    a = torch.randn(8, 256, dtype=torch.bfloat16, device=device)
    w = torch.randn(128, 256, dtype=torch.float16, device=device)
    packed, scales = quantize_i4_pack8(w)

    w_deq = dequantize_i4_pack8(packed, scales)
    out_ref = torch.matmul(a.to(torch.float32), w_deq.t())

    out_int4 = matmul_bf16_i4(a, packed, scales)

    max_err = (out_int4 - out_ref).abs().max()
    assert max_err < 5e-2


def test_matmul_matches_fp16(device):
    a = torch.randn(8, 256, dtype=torch.bfloat16, device=device)
    w = torch.randn(128, 256, dtype=torch.float16, device=device)
    packed, scales = quantize_i4_pack8(w)

    out_int4 = matmul_bf16_i4(a, packed, scales)
    out_ref = torch.matmul(a.to(torch.float16), w.t()).to(torch.float32)

    err = (out_int4 - out_ref).abs()
    mean_err = err.mean()
    max_err = err.max()

    assert mean_err < 1.8
    assert max_err < 8.0


def test_int4_linear_layer(device):
    linear = torch.nn.Linear(64, 128, bias=True, device=device, dtype=torch.float16)
    x = torch.randn(32, 64, dtype=torch.float16, device=device)
    ref = linear(x).to(torch.float32)
    packed = Int4PackedLinear(linear)
    out = packed(x)
    assert out.shape == ref.shape
    assert (out - ref).abs().max() < 1.0


def test_replace_linear_recursively(device):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(32, 64, bias=False, device=device, dtype=torch.float16),
                torch.nn.Linear(64, 16, bias=False, device=device, dtype=torch.float16),
            )

        def forward(self, x):
            return self.ffn(x)

    model = Toy().to(device)
    replace_linear_with_int4(model)
    for module in model.modules():
        if module is model:
            continue
        assert not isinstance(module, torch.nn.Linear)