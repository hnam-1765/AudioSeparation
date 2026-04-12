"""
Normalization utilities for MossFormer2.
LayerNorm, CLayerNorm (channel-wise), ScaleNorm,
GlobalLayerNorm, CumulativeLayerNorm, select_norm factory.
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Standard LayerNorm for (B, T, C) tensors."""
    def __init__(self, input_size=None, input_shape=None, eps=1e-05,
                 elementwise_affine=True):
        super().__init__()
        if input_shape is not None:
            input_size = input_shape[-1]
        self.norm = nn.LayerNorm(input_size, eps=eps,
                                 elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.norm(x)


class CLayerNorm(nn.LayerNorm):
    """Channel-wise LayerNorm: (B, C, T) → transpose → LayerNorm → transpose."""
    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(f'{self.__class__.__name__} only accepts 3-D tensor, got {x.dim()}D')
        x = x.transpose(1, 2)        # (B, T, C)
        x = super().forward(x)
        return x.transpose(1, 2)    # (B, C, T)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization over all dims except channel."""
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(dim, 1))
                self.bias = nn.Parameter(torch.zeros(dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Cumulative Layer Normalization."""
    def __init__(self, dim, elementwise_affine=True):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = super().forward(x)
            return x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        return super().forward(x)


def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    return nn.BatchNorm1d(dim)