"""
MossFormer2 core model: Encoder, Decoder, Dual_Path_Model, ScaledSinuEmbedding,
SBFLASHBlock_DualA, Dual_Computation_Block.

The model architecture:
  Raw waveform (B,T) → Encoder → (B, N=512, T/8)
      ↓ (stack as K copies × mask)
    → Dual_Path_Model (intra-only FLASH+FSMN) → (K, B, N, T/8)
      ↓ apply mask
    → Decoder → (B, T, K)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .normalization import LayerNorm, select_norm
from .Transformer import SBFLASHBlock_DualA


# ──────────────────────────────────────────────────────────────────────────────
# Scaled Sinusoidal Positional Embedding
# ──────────────────────────────────────────────────────────────────────────────
class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[1]
        device = x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


# ──────────────────────────────────────────────────────────────────────────────
# Linear projection
# ──────────────────────────────────────────────────────────────────────────────
class Linear(nn.Module):
    def __init__(self, n_neurons, input_size=None, bias=True):
        super().__init__()
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        return self.w(x)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder — strided Conv1d
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, kernel_size=16, out_channels=512, in_channels=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=kernel_size // 2,
            groups=1, bias=False)
        self.in_channels = in_channels

    def forward(self, x):
        # (B, L) → (B, 1, L) → (B, N, T_out)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.in_channels == 1 and x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1d(x)
        return F.relu(x)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder — transposed Conv1d
# ──────────────────────────────────────────────────────────────────────────────
class Decoder(nn.ConvTranspose1d):
    def __init__(self, in_channels=512, out_channels=1, kernel_size=16, stride=8, bias=False):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError(f"{self.__class__.__name__} expects 2/3D input")
        x = super().forward(x if x.dim() == 3 else x.unsqueeze(1))
        return x.squeeze(1) if x.squeeze().dim() == 1 else x.squeeze()


# ──────────────────────────────────────────────────────────────────────────────
# Dual_Computation_Block — one dual-path computation pass
# ──────────────────────────────────────────────────────────────────────────────
class Dual_Computation_Block(nn.Module):
    def __init__(self, intra_mdl, out_channels, norm="ln",
                 skip_around_intra=True, linear_layer_after_inter_intra=True):
        super().__init__()
        self.intra_mdl = intra_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)
        if linear_layer_after_inter_intra:
            self.intra_linear = Linear(out_channels, input_size=out_channels)

    def forward(self, x):
        # x: (B, N, S)  — S = number of segments after segmentation
        B, N, S = x.shape
        # Permute: (B, N, S) → (B, S, N) — sequence over S
        intra = x.permute(0, 2, 1).contiguous()
        intra = self.intra_mdl(intra)
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)
        if self.skip_around_intra:
            intra = intra + x
        return intra


# ──────────────────────────────────────────────────────────────────────────────
# Dual_Path_Model — the core separation network (intra-only, no inter)
# ──────────────────────────────────────────────────────────────────────────────
class Dual_Path_Model(nn.Module):
    """
    Dual-path model used as masknet in MossFormer2.
    Only uses intra-block (24-layer FLASH+FSMN), no inter-block.
    Segmentation: overlap-add with hop = K//2.
    """
    def __init__(self, in_channels, out_channels, intra_model,
                 num_layers=1, norm="ln", K=250, num_spks=2,
                 skip_around_intra=True, linear_layer_after_inter_intra=True,
                 use_global_pos_enc=True, max_length=20000):
        super().__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.dual_mdl = nn.ModuleList()
        for _ in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model, out_channels, norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra)))

        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1d_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()

        # Gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def _padding(self, input, K):
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.zeros(B, N, gap, device=input.device, dtype=input.dtype)
            input = torch.cat([input, pad], dim=2)
        _pad = torch.zeros(B, N, P, device=input.device, dtype=input.dtype)
        return torch.cat([_pad, input, _pad], dim=2), gap

    def _Segmentation(self, input, K):
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # Split into overlapping chunks
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input_cat = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        return input_cat.contiguous(), gap

    def _over_add(self, input, gap):
        B, N, K, S = input.shape
        P = K // 2
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)
        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        out = input1 + input2
        if gap > 0:
            out = out[:, :, :-gap]
        return out

    def forward(self, x):
        # x: (B, N, L)
        x = self.norm(x)
        x = self.conv1d_encoder(x)

        if self.use_global_pos_enc:
            base = x
            x_T = x.transpose(1, -1)                          # (B, L, N)
            emb = self.pos_enc(x_T)                            # (B, L, N)
            emb = emb.transpose(0, -1)                         # (B, N, L)
            x = base + emb

        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)

        x = self.prelu(x)

        # Output: (B, N*spks, S)
        x = self.conv1d_out(x)
        B, _, S = x.shape
        x = x.view(B * self.num_spks, -1, S)  # (BK, N, S)

        # Gated output
        x = self.output(x) * self.output_gate(x)  # (BK, N, S)

        # Decode: (BK, N, S) → (BK, in_channels, S) → reshape
        x = self.conv1d_decoder(x)                 # (BK, in_channels, S)

        _, N_in, S = x.shape
        x = x.view(B, self.num_spks, N_in, S)
        x = self.activation(x)

        # (spks, B, N_in, S)
        x = x.transpose(0, 1)
        return x
