"""Ckpt-free tests for hop-aligned RMS mute-gate (CycleGAN silence diagnosis)."""

import torch

from rave.canonicalizer.silence_gate import (
    CausalLoudnessSidechain,
    expand_frame_mask,
    frame_rms,
    frame_rms_dbfs,
    hold_quiet_mask,
    latent_hop,
    mute_by_input_rms,
    replace_latent_on_quiet,
    sidechain_by_input_loudness,
)


HOP = 8


def _burst_then_silence(n_burst_frames: int = 4, n_sil_frames: int = 8):
    burst = torch.ones(1, 1, n_burst_frames * HOP)
    sil = torch.zeros(1, 1, n_sil_frames * HOP)
    return torch.cat([burst, sil], dim=-1), n_burst_frames, n_sil_frames


def test_latent_hop_matches_encode_length():
    assert latent_hop(128 * 16, 16) == 128
    assert latent_hop(100, 3) == 33


def test_frame_rms_burst_vs_silence():
    x, n_b, n_s = _burst_then_silence()
    rms = frame_rms(x, HOP)
    db = frame_rms_dbfs(x, HOP)
    assert rms.shape == (1, n_b + n_s)
    assert torch.allclose(rms[0, :n_b], torch.ones(n_b))
    assert torch.all(rms[0, n_b:] <= 1e-12)
    assert torch.all(db[0, :n_b] > -1.0)
    assert torch.all(db[0, n_b:] < -80.0)


def test_hold_quiet_mask_needs_consecutive_frames():
    quiet = torch.tensor([[False, True, True, True, False, True]])
    held = hold_quiet_mask(quiet, hold_frames=2)
    # mute only after two consecutive quiet frames
    expected = torch.tensor([[False, False, True, True, False, False]])
    assert torch.equal(held, expected)


def test_mute_gate_kills_late_silence_keeps_burst():
    x, n_b, n_s = _burst_then_silence()
    leak = 0.25
    y = x + leak  # constant ringing on top of the tap
    muted = mute_by_input_rms(
        y, x, HOP, threshold_db=-40.0, hold_frames=2
    )
    burst = muted[..., : n_b * HOP]
    late = muted[..., (n_b + 2) * HOP :]  # after 2-frame hold
    assert torch.allclose(burst, y[..., : n_b * HOP])
    assert torch.allclose(late, torch.zeros_like(late))
    first_quiet = muted[..., n_b * HOP : (n_b + 1) * HOP]
    # hold_frames=2: first quiet frame stays open
    assert torch.allclose(first_quiet, y[..., n_b * HOP : (n_b + 1) * HOP])


def test_mute_gate_c_t_shape():
    x = torch.cat([torch.ones(1, 16), torch.zeros(1, 32)], dim=-1)
    y = torch.ones_like(x)
    muted = mute_by_input_rms(y, x, hop=8, threshold_db=-40.0, hold_frames=1)
    assert muted.shape == x.shape
    assert torch.allclose(muted[..., :16], y[..., :16])
    assert torch.allclose(muted[..., 16:], torch.zeros_like(muted[..., 16:]))


def test_expand_frame_mask_pads_tail():
    mask = torch.tensor([True, False])
    exp = expand_frame_mask(mask, hop=4, n_samples=10)
    assert exp.shape[-1] == 10
    assert torch.all(exp[:4])
    assert not torch.any(exp[4:8])
    assert not torch.any(exp[8:])


def test_replace_latent_on_quiet_uses_fill():
    z = torch.ones(1, 4, 6)
    fill = torch.zeros(1, 4, 1)
    rms_db = torch.tensor([[0.0, 0.0, -80.0, -80.0, -80.0, 0.0]])
    out = replace_latent_on_quiet(
        z, fill, rms_db, threshold_db=-40.0, hold_frames=2
    )
    # frames 0-1 loud (kept); 2 is first quiet (hold, kept); 3-4 muted; 5 loud
    assert torch.allclose(out[..., 0:3], torch.ones(1, 4, 3))
    assert torch.allclose(out[..., 3:5], torch.zeros(1, 4, 2))
    assert torch.allclose(out[..., 5], torch.ones(1, 4))


def test_sidechain_ducks_gap_without_hard_zero():
    x, n_b, n_s = _burst_then_silence()
    leak = 0.25
    y = torch.full_like(x, leak)
    y[..., : n_b * HOP] = 1.0
    sc = sidechain_by_input_loudness(
        y, x, HOP, threshold_db=-40.0, smooth_frames=1
    )
    burst = sc[..., : n_b * HOP]
    late = sc[..., (n_b + 2) * HOP :]
    assert burst.abs().mean() > 0.5
    assert late.abs().mean() < 0.05
    assert late.abs().max() > 0.0  # not a hard mute


def test_sidechain_c_t_shape():
    x = torch.cat([torch.ones(1, 16), torch.zeros(1, 32)], dim=-1)
    y = torch.ones_like(x)
    sc = sidechain_by_input_loudness(y, x, hop=8, threshold_db=-40.0, smooth_frames=1)
    assert sc.shape == x.shape
    assert sc[..., :16].abs().mean() > sc[..., 16:].abs().mean()


def test_causal_sidechain_state_persists_across_blocks():
    sc = CausalLoudnessSidechain(hop=8, knee_db=-40.0, smooth_frames=4)
    loud = torch.ones(1, 1, 32)
    quiet = torch.zeros(1, 1, 32)
    y = torch.ones(1, 1, 32)
    y1 = sc(loud, y)
    env_after_loud = float(sc.env.item())
    y2 = sc(quiet, y)
    assert float(sc.env.item()) < env_after_loud
    assert y1.abs().mean() > y2.abs().mean()


def test_causal_sidechain_reset():
    sc = CausalLoudnessSidechain(hop=8)
    sc(torch.ones(1, 1, 32), torch.ones(1, 1, 32))
    sc.reset()
    assert float(sc.env.item()) == 0.0

