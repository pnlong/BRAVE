"""Step dumps must use Lightning global_step so SLURM resumes still hit 1.5M."""

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from rave.core import ModelCheckpoint


class _Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)

    def training_step(self, batch, _batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return (self.layer(x) ** 2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loader():
    data = torch.randn(64, 4)
    return DataLoader(TensorDataset(data), batch_size=1)


def _trainer(tmp_path: Path, max_steps: int) -> pl.Trainer:
    cb = ModelCheckpoint(
        dirpath=str(tmp_path),
        filename="epoch-{epoch:04d}",
        step_period=5,
        save_top_k=0,
    )
    return pl.Trainer(
        max_steps=max_steps,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=True,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
    )


def test_step_ckpt_uses_global_step_across_resume(tmp_path):
    model = _Tiny()
    _trainer(tmp_path, max_steps=7).fit(model, _loader())

    period = tmp_path / "epoch_5.ckpt"
    final_job1 = tmp_path / "epoch_7.ckpt"
    assert period.is_file(), "should dump on global_step % step_period"
    assert final_job1.is_file(), "should dump the job's final global_step"

    model2 = _Tiny()
    _trainer(tmp_path, max_steps=12).fit(
        model2, _loader(), ckpt_path=str(final_job1))

    assert (tmp_path / "epoch_10.ckpt").is_file(), (
        "resumed run must name dumps by global_step, not a per-job counter")
    assert (tmp_path / "epoch_12.ckpt").is_file()
    # Old bug: second job's local counter hit 5 at global 12 and wrote epoch_5.ckpt
    # as the "periodic" dump. epoch_5 from job 1 is allowed; a *new* local-5 dump
    # would still be named epoch_5.ckpt. The assertion that matters is epoch_10.
