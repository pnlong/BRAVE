"""
Lightning callbacks for Fader RAVE training schedules.

Lambda warmup mirrors neurorave faderave.get_lambda().
"""

import warnings

import gin
import pytorch_lightning as pl


def _effective_lambda_delay(lambda_delay: int, phase_1_duration: int) -> int:
    """Delay must be < phase 1 or latent adversarial never runs before phase 2."""
    if phase_1_duration <= 0:
        return lambda_delay
    if lambda_delay >= phase_1_duration:
        return max(1, phase_1_duration // 5)
    return lambda_delay


@gin.configurable
class LambdaWarmupCallback(pl.Callback):
    """
    Ramp latent adversarial weight lambda_factor from 0 to lambda_inf.

    lambda_factor stays 0 until ``lambda_delay``, then linearly ramps to
    ``lambda_inf`` over the next ``lambda_delay`` steps. Requires
    ``lambda_delay < phase_1_duration`` so the encoder adversary runs in phase 1.

    See neurorave faderave.py get_lambda(step, lambda_inf, lambda_delay).
    """

    def __init__(
        self,
        lambda_inf: float = 0.5,
        lambda_delay: int = 2000,
    ) -> None:
        super().__init__()
        self.lambda_inf = lambda_inf
        self.lambda_delay = lambda_delay
        self.state = {"training_steps": 0}
        self._lambda_delay_warned = False

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
        phase_1_duration = int(getattr(pl_module, "warmup", 0))
        lambda_delay = _effective_lambda_delay(self.lambda_delay, phase_1_duration)
        if (
            not self._lambda_delay_warned
            and phase_1_duration > 0
            and self.lambda_delay >= phase_1_duration
        ):
            warnings.warn(
                f"lambda_delay ({self.lambda_delay}) >= phase_1_duration "
                f"({phase_1_duration}): latent adversarial would never run in "
                f"phase 1. Using effective delay {lambda_delay}.",
                stacklevel=2,
            )
            self._lambda_delay_warned = True
        if step < lambda_delay:
            pl_module.lambda_factor = 0.0
        else:
            ramp = min(
                self.lambda_inf,
                self.lambda_inf * (step - lambda_delay) / lambda_delay,
            )
            pl_module.lambda_factor = ramp

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
