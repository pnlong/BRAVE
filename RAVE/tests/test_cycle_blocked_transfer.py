"""Blocked transfer helper for Max/nn~-like CycleGAN val audio."""

import torch

from rave.canonicalizer.cycle_inference import transfer_waveform_blocked


def test_transfer_waveform_blocked_identity():
    def identity(x):
        return x

    x = torch.randn(1, 1, 2000)
    y = transfer_waveform_blocked(
        identity, x, block_size=512, left_context=256)
    assert y.shape == x.shape
    assert torch.allclose(y, x)


def test_transfer_waveform_blocked_uses_context_tail():
    """Output for each block is the transfer of [context|block], last |block| samples."""

    def scale_by_length(x):
        # Mark the window so we can see which samples were kept.
        return x + x.shape[-1] * 1e-6

    x = torch.arange(20, dtype=torch.float32).view(1, 1, -1)
    y = transfer_waveform_blocked(
        scale_by_length, x, block_size=8, left_context=4)
    assert y.shape == x.shape
    # First block: window = x[0:8], take last 8 → offset by 8e-6
    assert torch.allclose(y[..., :8], x[..., :8] + 8e-6)
    # Second block: window = x[4:16] (len 12), take last 8 → offset by 12e-6
    assert torch.allclose(y[..., 8:16], x[..., 8:16] + 12e-6)
