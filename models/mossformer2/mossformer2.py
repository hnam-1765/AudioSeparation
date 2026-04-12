"""
MossFormer2 full model: combines Encoder, Dual_Path_Model (masknet), Decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder, Decoder, Dual_Path_Model
from .Transformer import SBFLASHBlock_DualA


class MossFormer2(nn.Module):
    """
    MossFormer2 model for audio source separation.

    Architecture:
      Raw Waveform (B,T)
        → Encoder (Conv1d 1→512, k=16, s=8) → (B, 512, T/8)
        → Dual_Path_Model (24-layer FLASH+FSMN, K=250)
            → (spks, B, 512, T/8)
        → Stack encoder output × mask → apply element-wise
        → Decoder (ConvTranspose1d 512→1, k=16, s=8) → (B, T, num_spks)

    Config (Libri2Mix 2-speaker):
      encoder_kernel_size=16, encoder_out_nchannels=512
      masknet_chunksize=250, masknet_numlayers=1, masknet_numspks=2
      intra_numlayers=24, intra_nhead=8, intra_dffn=1024
    """
    def __init__(self, num_spks=2,
                 encoder_kernel_size=16, encoder_out_nchannels=512,
                 encoder_in_nchannels=1,
                 masknet_chunksize=250, masknet_numlayers=1,
                 masknet_norm="ln",
                 intra_numlayers=24, intra_nhead=8, intra_dffn=1024,
                 intra_dropout=0, intra_use_positional=True,
                 intra_norm_before=True):
        super().__init__()
        self.num_spks = num_spks

        # ── Encoder ──
        self.encoder = Encoder(
            kernel_size=encoder_kernel_size,
            out_channels=encoder_out_nchannels,
            in_channels=encoder_in_nchannels)

        # ── Intra-block: 24-layer FLASH + Gated FSMN ──
        intra_model = SBFLASHBlock_DualA(
            num_layers=intra_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            dropout=intra_dropout,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before)

        # ── Masknet: Dual-path model ──
        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            num_layers=masknet_numlayers,
            norm=masknet_norm,
            K=masknet_chunksize,
            num_spks=num_spks,
            skip_around_intra=True,
            linear_layer_after_inter_intra=False)

        # ── Decoder ──
        self.decoder = Decoder(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_in_nchannels,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            bias=False)

    def forward(self, mix):
        """
        mix: (B, T) raw waveform
        Returns: (B, T, num_spks) separated waveforms
        """
        # Encode
        mix_w = self.encoder(mix)  # (B, N=512, T/8)

        # Mask estimation
        est_mask = self.masknet(mix_w)  # (spks, B, N, T/8)

        # Apply mask: stack mix_w across speakers
        mix_w_stacked = torch.stack([mix_w] * self.num_spks, dim=0)  # (K, B, N, T/8)
        sep_h = mix_w_stacked * est_mask  # (K, B, N, T/8)

        # Decode each speaker
        est_source_list = []
        for k in range(self.num_spks):
            dec_out = self.decoder(sep_h[k])  # (B, T')
            est_source_list.append(dec_out.unsqueeze(-1))  # (B, T', 1)

        est_source = torch.cat(est_source_list, dim=-1)  # (B, T', num_spks)

        # Restore original time length
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source  # (B, T, num_spks)
