"""
Convolutional modules for MossFormer2.
Transpose, DepthwiseConv1d, ConvModule.
"""
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConvModule(nn.Module):
    """
    Conformer-style conv module:
    input (B,T,C) → transpose → depthwise conv → transpose → residual add.
    """
    def __init__(self, in_channels, kernel_size=17, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd for 'SAME' padding"
        assert expansion_factor == 2, "Currently only supports expansion_factor=2"

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),                                   # (B,C,T)
            DepthwiseConv1d(in_channels, in_channels, kernel_size,
                            stride=1, padding=(kernel_size - 1) // 2),
        )

    def forward(self, inputs):
        return inputs + self.sequential(inputs).transpose(1, 2)