"""
FSMN (Feedforward Sequential Memory Network) layers with dilation.
UniDeepFsmn: standard causal FIR filter via 2D depthwise conv.
DilatedDenseNet: multi-scale dilated depthwise convolutions.
UniDeepFsmn_dilated: FSMN with DilatedDenseNet backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniDeepFsmn(nn.Module):
    """
    Standard deep FSMN with a causal 2D depthwise conv.
    Acts as an efficient all-pass filter over the sequence.
    """
    def __init__(self, input_dim, output_dim, lorder=20, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lorder = lorder
        self.hidden_size = hidden_size or output_dim

        self.linear = nn.Linear(input_dim, self.hidden_size)
        self.project = nn.Linear(self.hidden_size, output_dim, bias=False)
        # Causal conv over (2*lorder-1) × 1 patches — depthwise
        self.conv1 = nn.Conv2d(
            output_dim, output_dim,
            [lorder + lorder - 1, 1], [1, 1],
            groups=output_dim, bias=False)

    def forward(self, x):
        # x: (B, T, D_in)
        f1 = F.relu(self.linear(x))                         # (B, T, H)
        p1 = self.project(f1)                               # (B, T, D_out)
        x_per = p1.unsqueeze(1).permute(0, 3, 2, 1)         # (B, D_out, T, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1).squeeze()            # (B, T, D_out)
        return x + out1                                     # residual


class DilatedDenseNet(nn.Module):
    """
    Dense connections of dilated depthwise 2D convolutions.
    Dilation factors: 1, 2, 4, 8 (powers of 2).
    Each layer doubles channels via concatenation.
    """
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.twidth = lorder * 2 - 1
        self.kernel_size = (self.twidth, 1)

        for i in range(depth):
            dil = 2 ** i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
            setattr(self, f'pad{i + 1}',
                    nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))
            setattr(self, f'conv{i + 1}',
                    nn.Conv2d(in_channels * (i + 1), in_channels,
                              kernel_size=self.kernel_size,
                              dilation=(dil, 1), groups=in_channels, bias=False))
            setattr(self, f'norm{i + 1}',
                    nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, f'prelu{i + 1}',
                    nn.PReLU(in_channels))

    def forward(self, x):
        # x: (B, C, T, 1)
        skip = x
        for i in range(self.depth):
            out = getattr(self, f'pad{i + 1}')(skip)
            out = getattr(self, f'conv{i + 1}')(out)
            out = getattr(self, f'norm{i + 1}')(out)
            out = getattr(self, f'prelu{i + 1}')(out)
            skip = torch.cat([out, skip], dim=1)
        return skip


class UniDeepFsmn_dilated(nn.Module):
    """
    Dilated FSMN: uses DilatedDenseNet for multi-scale temporal context.
    """
    def __init__(self, input_dim, output_dim, lorder=20, hidden_size=None):
        super().__init__()
        self.lorder = lorder
        self.hidden_size = hidden_size or output_dim

        self.linear = nn.Linear(input_dim, self.hidden_size)
        self.project = nn.Linear(self.hidden_size, output_dim, bias=False)
        self.conv = DilatedDenseNet(depth=2, lorder=lorder, in_channels=output_dim)

    def forward(self, x):
        # x: (B, T, D_in)
        f1 = F.relu(self.linear(x))                         # (B, T, H)
        p1 = self.project(f1)                              # (B, T, D_out)
        x_per = p1.unsqueeze(1).permute(0, 3, 2, 1)        # (B, D_out, T, 1)
        out = self.conv(x_per)
        out1 = out.permute(0, 3, 2, 1).squeeze()           # (B, T, D_out)
        return x + out1                                      # residual