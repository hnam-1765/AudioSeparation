"""
Training utilities: checkpoint save/load, LR schedulers.
"""
import os
import torch


def save_checkpoint(epoch, model, optimizer, path, best_loss=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if best_loss is not None:
        ckpt['best_loss'] = best_loss
    torch.save(ckpt, path)
    print(f"  ✓ Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer=None, device='cuda'):
    if not os.path.exists(path):
        return 0, None
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', None)
        print(f"  ✓ Loaded checkpoint from epoch {epoch}: {path}")
        return epoch, best_loss

    # Backward compatibility for plain state_dict checkpoints.
    model.load_state_dict(ckpt)
    print(f"  ✓ Loaded weights-only checkpoint: {path}")
    return 0, None


def get_last_checkpoint_path(log_dir):
    """Find the latest resumable checkpoint in log_dir."""
    if not os.path.exists(log_dir):
        return None

    preferred = [
        os.path.join(log_dir, "latest.pt"),
        os.path.join(log_dir, "best.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path
    return None
