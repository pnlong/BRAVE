"""Audio discriminator: real in-domain vs OOD-translated reconstructions."""

from __future__ import annotations

from typing import Callable, List, Literal, Optional, Sequence

import gin
import torch
import torch.nn as nn

from .attribute_conditioning import AttributeConditioningEmbed, ConditionKey

ConditionInput = Literal["attr_cls", "attr_norm"]


@gin.configurable
class InDomainAudioDiscriminator(nn.Module):
    """
    Multi-scale audio discriminator for one-way OOD → in-domain transfer.

    When ``num_attributes > 0``, adds a projection term to each scale logit from
    pooled attribute conditioning (``attr_cls`` by default). ``num_attributes=0``
    recovers the plain audio-only MSD (plain BRAVE).
    """

    def __init__(
        self,
        discriminator: Callable[..., nn.Module],
        n_channels: int = 1,
        num_attributes: int = 0,
        num_classes_per_attribute: Optional[Sequence[int]] = None,
        embed_dim: int = 32,
        condition_on: ConditionKey = "attr_cls",
    ) -> None:
        super().__init__()
        self.net = discriminator(n_channels=n_channels)
        self.num_attributes = int(num_attributes)
        self.condition_on: ConditionInput = condition_on

        self.cond_embed: Optional[AttributeConditioningEmbed] = None
        self.projections: Optional[nn.ModuleList] = None

        if self.num_attributes > 0:
            if not num_classes_per_attribute:
                raise ValueError(
                    "num_classes_per_attribute required when num_attributes > 0")
            self.cond_embed = AttributeConditioningEmbed(
                num_attributes=self.num_attributes,
                num_classes_per_attribute=num_classes_per_attribute,
                embed_dim=embed_dim,
                condition_on=condition_on,
            )
            n_scales = len(self.net.layers)
            self.projections = nn.ModuleList([
                nn.Linear(self.cond_embed.out_dim, 1)
                for _ in range(n_scales)
            ])
            for proj in self.projections:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attr_cls: Optional[torch.Tensor] = None,
        attr_norm: Optional[torch.Tensor] = None,
    ) -> List[List[torch.Tensor]]:
        features = self.net(x)
        if self.num_attributes == 0:
            return features

        if self.cond_embed is None or self.projections is None:
            return features

        cond = self.cond_embed(attr_cls=attr_cls, attr_norm=attr_norm)
        out: List[List[torch.Tensor]] = []
        for scale_feats, proj in zip(features, self.projections):
            layers = list(scale_feats)
            logit = layers[-1]
            layers[-1] = logit + proj(cond).view(-1, 1, 1)
            out.append(layers)
        return out

    @staticmethod
    def gan_losses(
        features_real: List[List[torch.Tensor]],
        features_fake: List[List[torch.Tensor]],
        gan_loss_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate multi-scale GAN loss (D step, G step)."""
        loss_d = torch.tensor(0.0, device=features_real[0][-1].device)
        loss_g = torch.tensor(0.0, device=features_real[0][-1].device)
        n_scales = len(features_real)
        for scale_real, scale_fake in zip(features_real, features_fake):
            score_real = scale_real[-1]
            score_fake = scale_fake[-1]
            if score_real.shape[0] != score_fake.shape[0]:
                # Mixed batches often have unequal in-domain vs OOD counts.
                loss_dis = (
                    torch.relu(1 - score_real).mean()
                    + torch.relu(1 + score_fake).mean()
                )
                loss_gen = -score_fake.mean()
            else:
                loss_dis, loss_gen = gan_loss_fn(score_real, score_fake)
            loss_d = loss_d + loss_dis
            loss_g = loss_g + loss_gen
        return loss_d / n_scales, loss_g / n_scales


@gin.configurable
class InDomainLatentDiscriminator(nn.Module):
    """
    Real/fake discriminator on content latent z (B, latent_size, T_lat).

    Returns a single-scale feature pyramid ``[[h1, h2, ..., logit]]`` compatible
    with ``gan_utils`` hinge + feature matching.
    """

    def __init__(
        self,
        latent_size: int = 128,
        hidden_size: int = 256,
        n_layers: int = 3,
        kernel_size: int = 7,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        blocks: List[nn.Module] = []
        in_ch = int(latent_size)
        pad = int(kernel_size) // 2
        for _ in range(int(n_layers)):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, hidden_size, kernel_size, padding=pad),
                    nn.LeakyReLU(negative_slope),
                )
            )
            in_ch = int(hidden_size)
        self.blocks = nn.ModuleList(blocks)
        self.logit = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> List[List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        h = z
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        feats.append(self.logit(h))
        return [feats]

    @staticmethod
    def gan_losses(
        features_real: List[List[torch.Tensor]],
        features_fake: List[List[torch.Tensor]],
        gan_loss_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return InDomainAudioDiscriminator.gan_losses(
            features_real, features_fake, gan_loss_fn)


_MSD_SCOPE = "discriminator.MultiScaleDiscriminator"


def require_gin_binding(param: str) -> None:
    """Raise if ``param`` is not bound (canonicalizer gin was not parsed)."""
    try:
        gin.query_parameter(param)
    except (ValueError, KeyError) as exc:
        preview = gin.config_str()
        raise RuntimeError(
            f"Missing required gin binding {param!r}. "
            "Parse configs/brave_canonicalizer.gin from the configs/ directory "
            f"before building the in-domain discriminator "
            f"(config_str length={len(preview)}).\n"
            f"Preview:\n{preview[:2000]}"
        ) from exc


def build_in_domain_discriminator(
    n_channels: int,
    *,
    num_attributes: int = 0,
    num_classes_per_attribute: Optional[Sequence[int]] = None,
) -> InDomainAudioDiscriminator:
    """Construct MSD from gin bindings in ``brave_canonicalizer.gin``."""
    require_gin_binding(f"{_MSD_SCOPE}.n_discriminators")
    msd = gin.get_configurable(_MSD_SCOPE)(n_channels=n_channels)
    return InDomainAudioDiscriminator(
        discriminator=lambda **_kwargs: msd,
        n_channels=n_channels,
        num_attributes=num_attributes,
        num_classes_per_attribute=num_classes_per_attribute,
    )


def build_latent_discriminator(
    latent_size: int,
    *,
    hidden_size: Optional[int] = None,
    n_layers: Optional[int] = None,
    kernel_size: Optional[int] = None,
) -> InDomainLatentDiscriminator:
    """Construct a latent real/fake D.

    Only ``latent_size`` is passed through by default so gin bindings for
    ``hidden_size``, ``n_layers``, and ``kernel_size`` take effect. Explicit
    kwargs still override gin (for tests / non-gin callers).
    """
    kwargs = {"latent_size": int(latent_size)}
    if hidden_size is not None:
        kwargs["hidden_size"] = hidden_size
    if n_layers is not None:
        kwargs["n_layers"] = n_layers
    if kernel_size is not None:
        kwargs["kernel_size"] = kernel_size
    return InDomainLatentDiscriminator(**kwargs)
