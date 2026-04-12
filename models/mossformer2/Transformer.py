"""
FLASH Attention + Gated Dilated FSMN Transformer blocks for MossFormer2.

Key components:
- FFConvM: Feed-Forward with ConvModule
- OffsetScale: Learnable per-head scale/bias for Q/K decomposition
- FLASH_ShareA_FFConvM: FLASH attention with dual (quadratic + linear) attention
- Gated_FSMN_dilated: Gated FSMN using dilated convolutions
- Gated_FSMN_Block_Dilated: Full FSMN block
- FLASHTransformer_DualA_FSMN: Stacked FLASH + FSMN layers
- TransformerEncoder_FLASH_DualA_FSMN: SpeechBrain-compatible wrapper
- SBFLASHBlock_DualA: Wrapper class used by MossFormer2
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .fsmn import UniDeepFsmn_dilated
from .normalization import LayerNorm, CLayerNorm, ScaleNorm
from .conv_module import ConvModule

try:
    from rotary_embedding_torch import RotaryEmbedding
    HAS_ROTARY = True
except ImportError:
    HAS_ROTARY = False
    RotaryEmbedding = None


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    r = n % mult
    return 0 if r == 0 else mult - r


# ──────────────────────────────────────────────────────────────────────────────
# FFConvM — Feed-Forward with ConvModule
# ──────────────────────────────────────────────────────────────────────────────
class FFConvM(nn.Module):
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.mdl(x)


# ──────────────────────────────────────────────────────────────────────────────
# OffsetScale — Decompose Q/K into 4 heads with learnable scale + bias
# ──────────────────────────────────────────────────────────────────────────────
class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


# ──────────────────────────────────────────────────────────────────────────────
# FLASH_ShareA_FFConvM — FLASH Attention with Shared A
# ──────────────────────────────────────────────────────────────────────────────
class FLASH_ShareA_FFConvM(nn.Module):
    """
    FLASH attention combining:
    - Quadratic (grouped) attention over local groups
    - Linear (feature-averaged) attention for efficiency
    - Token shifting, rotary pos embeddings, gating
    """
    def __init__(self, dim, group_size=256, query_key_dim=128,
                 expansion_factor=1., causal=False, dropout=0.1,
                 rotary_pos_emb=None, norm_klass=nn.LayerNorm,
                 shift_tokens=True):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = FFConvM(dim_in=dim, dim_out=hidden_dim,
                                 norm_klass=norm_klass, dropout=dropout)
        self.to_qk = FFConvM(dim_in=dim, dim_out=query_key_dim,
                              norm_klass=norm_klass, dropout=dropout)
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(dim_in=dim * 2, dim_out=dim,
                               norm_klass=norm_klass, dropout=dropout)
        self.gateActivate = nn.Sigmoid()

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = mask.unsqueeze(-1)
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # Rotate with rotary embeddings
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys,
                (quad_q, lin_q, quad_k, lin_k))

        # Pad to multiple of group_size
        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            pad_tuple = (0, 0, 0, padding)
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, pad_tuple, value=0.),
                (quad_q, quad_k, lin_q, lin_k, v, u))
            if exists(mask):
                mask = F.pad(mask, (0, padding), value=True)

        # Group sequence into chunks of group_size
        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, 'b (g n) d -> b g n d', n=g),
            (quad_q, quad_k, lin_q, lin_k, v, u))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        # ── Quadratic attention ──
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / math.sqrt(g)
        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)
        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)
        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # ── Linear attention ──
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)
            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # Fold groups back, trim padding
        def fold_and_trim(t):
            t = rearrange(t, 'b g n d -> b (g n) d')[:, :n]
            return t
        return fold_and_trim(quad_out_v + lin_out_v), fold_and_trim(quad_out_u + lin_out_u)

    def forward(self, x, mask=None):
        residual = x
        normed_x = x

        # Token shift
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)

        # Gated output: (att_u * v) * sigmoid(att_v * u)
        out = (att_u * v) * self.gateActivate(att_v * u)
        return x + self.to_out(out)


try:
    from einops import rearrange
except ImportError:
    def rearrange(x, pattern, **kwargs):
        """Fallback: minimal rearrange for common patterns."""
        B, G, N, D = pattern.count('b'), pattern.count('g'), pattern.count('n'), pattern.count('d')
        if pattern == 'b (g n) d -> b g n d':
            b, gn, d = x.shape
            n = N; g = G
            x = x.view(b, g, n, d)
        elif pattern == 'b g n d -> b (g n) d':
            b, g, n, d = x.shape
            x = x.reshape(b, g * n, d)
        elif pattern == 'b (g j) -> b g 1 j':
            b, gj = x.shape
            j = 1; g = gj
            x = x.view(b, g, 1, 1)
        elif pattern == 'b (g n) d -> b g n d':
            b, gn, d = x.shape
            x = x.view(b, G, N, d) if G*N == gn else x
        elif pattern == 'b g n d -> b g n d':
            pass
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Gated_FSMN_dilated — Gated FSMN using DilatedDenseNet backbone
# ──────────────────────────────────────────────────────────────────────────────
class Gated_FSMN_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, lorder=20, hidden_size=None):
        super().__init__()
        self.to_u = FFConvM(dim_in=in_channels, dim_out=hidden_size or out_channels,
                             norm_klass=nn.LayerNorm, dropout=0.1)
        self.to_v = FFConvM(dim_in=in_channels, dim_out=hidden_size or out_channels,
                             norm_klass=nn.LayerNorm, dropout=0.1)
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        input_x = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        return x_v * x_u + input_x


# ──────────────────────────────────────────────────────────────────────────────
# Gated_FSMN_Block_Dilated — Conv1d → Norm → Gated FSMN → Norm → Conv1d
# ──────────────────────────────────────────────────────────────────────────────
class Gated_FSMN_Block_Dilated(nn.Module):
    def __init__(self, dim, inner_channels=256, norm_type='layernorm'):
        super().__init__()
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU())
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN_dilated(
            inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        # input: (B, T, N)
        conv1 = self.conv1(input.transpose(2, 1))                # (B, H, T)
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))        # (B, T, H)
        norm2 = self.norm2(seq_out.transpose(2, 1))              # (B, H, T)
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input                     # (B, T, N)


# ──────────────────────────────────────────────────────────────────────────────
# FLASHTransformer_DualA_FSMN — Stacked FLASH + Gated FSMN blocks
# ──────────────────────────────────────────────────────────────────────────────
class FLASHTransformer_DualA_FSMN(nn.Module):
    def __init__(self, dim, depth, group_size=256, query_key_dim=128,
                 expansion_factor=4., causal=False, attn_dropout=0.1,
                 norm_type='layernorm', shift_tokens=True):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm')
        norm_klass = ScaleNorm if norm_type == 'scalenorm' else nn.LayerNorm

        self.group_size = group_size
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim)) \
            if HAS_ROTARY else None

        self.fsmn = nn.ModuleList([
            Gated_FSMN_Block_Dilated(dim, dim, norm_type=norm_type)
            for _ in range(depth)])

        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim, group_size=group_size, query_key_dim=query_key_dim,
                expansion_factor=expansion_factor, causal=causal,
                dropout=attn_dropout, rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass, shift_tokens=shift_tokens)
            for _ in range(depth)])

    def forward(self, x, mask=None):
        for i, (flash, fsmn) in enumerate(zip(self.layers, self.fsmn)):
            x = flash(x, mask=mask)
            x = fsmn(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# TransformerEncoder_FLASH_DualA_FSMN — SpeechBrain-compatible wrapper
# ──────────────────────────────────────────────────────────────────────────────
class TransformerEncoder_FLASH_DualA_FSMN(nn.Module):
    def __init__(self, num_layers, nhead, d_ffn, input_shape=None,
                 d_model=None, kdim=None, vdim=None, dropout=0.0,
                 activation=nn.ReLU, normalize_before=False,
                 causal=False, attention_type="regularMHA"):
        super().__init__()
        act_cls = nn.ReLU if activation == "relu" else nn.GELU
        self.flashT = FLASHTransformer_DualA_FSMN(
            dim=d_model, depth=num_layers,
            query_key_dim=d_model // 2,
            expansion_factor=4., attn_dropout=dropout,
            norm_type='layernorm', shift_tokens=True)
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos_embs=None):
        output = self.flashT(src)
        return self.norm(output)


# ──────────────────────────────────────────────────────────────────────────────
# SBFLASHBlock_DualA — SpeechBrain wrapper used in MossFormer2
# ──────────────────────────────────────────────────────────────────────────────
class SBFLASHBlock_DualA(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ffn=2048,
                 input_shape=None, kdim=None, vdim=None, dropout=0.1,
                 activation="relu", use_positional_encoding=True,
                 norm_before=True, attention_type="regularMHA"):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.mdl = TransformerEncoder_FLASH_DualA_FSMN(
            num_layers=num_layers, nhead=nhead, d_ffn=d_ffn,
            input_shape=input_shape, d_model=d_model,
            kdim=kdim, vdim=vdim, dropout=dropout,
            activation=activation, normalize_before=norm_before,
            causal=False, attention_type=attention_type)

    def forward(self, x):
        return self.mdl(x)
