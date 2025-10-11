import torch
import torch.nn as nn

from triton_int4.triton_kernels.matmul_bf16_i4_pack8 import matmul_bf16_i4
from triton_int4.triton_kernels.quant_i4_pack8 import quantize_i4_pack8


class Int4PackedLinear(nn.Module):
    """Int4 linear layer."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach().to(torch.float16)
        packed, scales = quantize_i4_pack8(weight)
        self.register_buffer("weight", packed)
        self.register_buffer("scales", scales)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(torch.bfloat16)
        out = matmul_bf16_i4(x_bf16, self.weight, self.scales)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linear_with_int4(module: nn.Module) -> nn.Module:
    """Converts linear layers to int4 packed layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int4PackedLinear(child))
        else:
            replace_linear_with_int4(child)
    return module
