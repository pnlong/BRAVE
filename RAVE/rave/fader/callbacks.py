"""
Lightning callbacks for Fader RAVE training schedules.

Lambda warmup mirrors neurorave faderave.get_lambda().
"""

import gin
import pytorch_lightning as pl

from ..train_logging import log_train


@gin.configurable
class LambdaWarmupCallback(pl.Callback):
    """
    Ramp latent adversarial weight lambda_factor from 0 to lambda_inf.

    See neurorave faderave.py get_lambda(step, lambda_inf, lambda_delay).
    """

    def __init__(
        self,
        lambda_inf: float = 0.5,
        lambda_delay: int = 15000,
    ) -> None:
        super().__init__()
        self.lambda_inf = lambda_inf
        self.lambda_delay = lambda_delay
        self.state = {"training_steps": 0}

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        if not hasattr(pl_module, "lambda_factor"):
            return
        step = trainer.global_step
        if step < self.lambda_delay:
            pl_module.lambda_factor = 0.0
        else:
            ramp = min(
                self.lambda_inf,
                self.lambda_inf * (step - self.lambda_delay) / self.lambda_delay,
            )
            pl_module.lambda_factor = ramp

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # --- Surface lambda_factor at epoch boundary for W&B ---
        if hasattr(pl_module, "lambda_factor"):
            log_train(
                pl_module,
                "fader/lambda_factor_epoch",
                pl_module.lambda_factor,
                on_step=False,
            )

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
