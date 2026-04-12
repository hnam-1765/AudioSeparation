import math

import numpy
import torch


class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, layer_scale_init=1.0e-5):
        super().__init__()
        if dims == 1:
            value = torch.ones(input_size) * layer_scale_init
        elif dims == 2:
            value = torch.ones(1, input_size) * layer_scale_init
        else:
            value = torch.ones(1, 1, input_size) * layer_scale_init
        self.layer_scale = torch.nn.Parameter(value, requires_grad=True)

    def forward(self, x):
        return x * self.layer_scale


class Masking(torch.nn.Module):
    def __init__(self, input_dim, Activation_mask="Sigmoid", **options):
        super().__init__()
        self.options = options
        if self.options["concat_opt"]:
            self.pw_conv = torch.nn.Conv1d(input_dim * 2, input_dim, 1, stride=1, padding=0)
        self.gate_act = torch.nn.Sigmoid() if Activation_mask == "Sigmoid" else torch.nn.ReLU()

    def forward(self, x, skip):
        y = torch.cat([x, skip], dim=-2) if self.options["concat_opt"] else x
        if self.options["concat_opt"]:
            y = self.pw_conv(y)
        return self.gate_act(y) * skip


class GCFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, layer_scale_init=1.0e-5):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels * 6),
        )
        self.depthwise = torch.nn.Conv1d(in_channels * 6, in_channels * 6, 3, padding=1, groups=in_channels * 6)
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels * 3, in_channels),
            torch.nn.Dropout(dropout_rate),
        )
        self.layer_scale = LayerScale(dims=3, input_size=in_channels, layer_scale_init=layer_scale_init)

    def forward(self, x):
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + self.layer_scale(y)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head: int, in_channels: int, dropout_rate: float, layer_scale_init=1.0e-5):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear_q = torch.nn.Linear(in_channels, in_channels)
        self.linear_k = torch.nn.Linear(in_channels, in_channels)
        self.linear_v = torch.nn.Linear(in_channels, in_channels)
        self.linear_out = torch.nn.Linear(in_channels, in_channels)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.layer_scale = LayerScale(dims=3, input_size=in_channels, layer_scale_init=layer_scale_init)

    def forward(self, x, pos_k, mask):
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        attention_logits = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0, 1)

        if pos_k is not None:
            position_logits = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            position_logits = position_logits.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (attention_logits + position_logits) / math.sqrt(self.d_k)
        else:
            scores = attention_logits / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.layer_scale(self.dropout(self.linear_out(x)))


class EGA(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict(
            {
                "self_attn": MultiHeadAttention(
                    n_head=num_mha_heads,
                    in_channels=in_channels,
                    dropout_rate=dropout_rate,
                ),
                "linear": torch.nn.Sequential(
                    torch.nn.LayerNorm(normalized_shape=in_channels),
                    torch.nn.Linear(in_features=in_channels, out_features=in_channels),
                    torch.nn.Sigmoid(),
                ),
            }
        )

    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        down_len = pos_k.shape[0]
        x_down = torch.nn.functional.adaptive_avg_pool1d(input=x, output_size=down_len)
        x = x.permute([0, 2, 1])
        x_down = x_down.permute([0, 2, 1])
        x_down = self.block["self_attn"](x_down, pos_k, None)
        x_down = x_down.permute([0, 2, 1])
        x_downup = torch.nn.functional.interpolate(input=x_down, size=x.shape[1], mode="nearest")
        x_downup = x_downup.permute([0, 2, 1])
        return x + self.block["linear"](x) * x_downup


class CLA(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, dropout_rate, layer_scale_init=1.0e-5):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear1 = torch.nn.Linear(in_channels, in_channels * 2)
        self.glu = torch.nn.GLU()
        self.dw_conv_1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding="same", groups=in_channels)
        self.linear2 = torch.nn.Linear(in_channels, 2 * in_channels)
        self.batch_norm = torch.nn.BatchNorm1d(2 * in_channels)
        self.linear3 = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(2 * in_channels, in_channels),
            torch.nn.Dropout(dropout_rate),
        )
        self.layer_scale = LayerScale(dims=3, input_size=in_channels, layer_scale_init=layer_scale_init)

    def forward(self, x):
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.glu(y)
        y = y.permute([0, 2, 1])
        y = self.dw_conv_1d(y)
        y = y.permute(0, 2, 1)
        y = self.linear2(y)
        y = y.permute(0, 2, 1)
        y = self.batch_norm(y)
        y = y.permute(0, 2, 1)
        y = self.linear3(y)
        return x + self.layer_scale(y)


class GlobalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict(
            {
                "ega": EGA(num_mha_heads=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate),
                "gcfn": GCFN(in_channels=in_channels, dropout_rate=dropout_rate),
            }
        )

    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        x = self.block["ega"](x, pos_k)
        x = self.block["gcfn"](x)
        return x.permute([0, 2, 1])


class LocalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict(
            {
                "cla": CLA(in_channels, kernel_size, dropout_rate),
                "gcfn": GCFN(in_channels, dropout_rate),
            }
        )

    def forward(self, x: torch.Tensor):
        x = self.block["cla"](x)
        return self.block["gcfn"](x)


class SpkAttention(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate)
        self.feed_forward = GCFN(in_channels=in_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, num_spk: int):
        batch_size, channels, time_steps = x.shape
        x = x.view(batch_size // num_spk, num_spk, channels, time_steps).contiguous()
        x = x.permute([0, 3, 1, 2]).contiguous()
        x = x.view(-1, num_spk, channels).contiguous()
        x = x + self.self_attn(x, None, None)
        x = x.view(batch_size // num_spk, time_steps, num_spk, channels).contiguous()
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = x.view(batch_size, channels, time_steps).contiguous()
        x = x.permute([0, 2, 1])
        x = self.feed_forward(x)
        return x.permute([0, 2, 1])
