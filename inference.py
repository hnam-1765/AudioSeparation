"""Inference script for audio source separation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.sepformer.model import Model as SepFormer
from models.mossformer2 import MossFormer2
from utils.audio import load_wav, save_wav
from engine import SeparationEngine


def build_model(model_name, num_spks=2, checkpoint=None, device='cuda'):
    if model_name == 'sepformer':
        model = SepFormer(
            num_stages=4, num_spks=num_spks,
            enc_out_channels=256, feature_dim=256,
            encoder_kernel=16, encoder_stride=4,
            mha_heads=8, dropout_rate=0.1,
            cla_kernel=65, samp_kernel=5,
        )
    elif model_name == 'mossformer2':
        model = MossFormer2(
            num_spks=num_spks,
            encoder_kernel_size=16,
            encoder_out_nchannels=512,
            masknet_chunksize=250,
            masknet_numlayers=1,
            intra_numlayers=24, intra_nhead=8, intra_dffn=1024,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint and os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")

    return model.to(device)


def inference(model_name, input_path, output_dir,
              checkpoint=None, num_spks=2, device='cuda'):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(model_name, num_spks, checkpoint, device)

    # Load audio
    mix, sr = load_wav(input_path, target_sr=8000)
    mix_tensor = torch.from_numpy(mix).float()
    print(f"Loaded: {input_path} | sr={sr}Hz | len={len(mix)}")

    # Run model
    engine = SeparationEngine(model, None, {}, device=device)
    spk_audio = engine.inference(mix_tensor)

    # Save outputs
    basename = Path(input_path).stem
    for k, wav in enumerate(spk_audio):
        out_path = output_dir / f"{basename}_spk{k+1}.wav"
        audio_np = wav[0].cpu().numpy() if wav.dim() > 1 else wav.cpu().numpy()
        save_wav(str(out_path), audio_np, sr=8000)
        print(f"  Saved: {out_path}")

    print(f"All outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inference for audio separation")
    parser.add_argument('--model', type=str, required=True,
                        choices=['sepformer', 'mossformer2'],
                        help='Model architecture')
    parser.add_argument('--input', type=str, required=True,
                        help='Input mixture WAV file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint (.pt)')
    parser.add_argument('--num_spks', type=int, default=2,
                        help='Number of speakers')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Use CPU')
    args = parser.parse_args()

    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    inference(args.model, args.input, args.output_dir,
              checkpoint=args.checkpoint,
              num_spks=args.num_spks,
              device=device)


if __name__ == '__main__':
    main()
