from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DDPMConfig:
    """Configuration for WaveGrad-style DDPM."""

    timesteps_train: int = 1000
    beta_start_train: float = 1e-4
    beta_end_train: float = 2e-2

    timesteps_infer: int = 50
    beta_start_infer: float = 1e-4
    beta_end_infer: float = 0.95


class DDPM:
    """WaveGrad-style DDPM utilities for training and sampling.

    The implementation follows the schedule described in the HiC2MicroC
    paper: a long chain for training (1,000 steps) with a linear beta
    schedule, and a short reverse chain (~50 steps) for inference.
    """

    def __init__(self, config: DDPMConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        # Training schedule.
        beta_train = torch.linspace(
            config.beta_start_train,
            config.beta_end_train,
            config.timesteps_train,
            device=device,
        )
        alpha_train = 1.0 - beta_train
        alpha_bar_train = torch.cumprod(alpha_train, dim=0)

        self.beta_train = beta_train
        self.alpha_train = alpha_train
        self.alpha_bar_train = alpha_bar_train

        # Inference schedule (short chain).
        beta_infer = torch.linspace(
            config.beta_start_infer,
            config.beta_end_infer,
            config.timesteps_infer,
            device=device,
        )
        alpha_infer = 1.0 - beta_infer
        alpha_bar_infer = torch.cumprod(alpha_infer, dim=0)

        self.beta_infer = beta_infer
        self.alpha_infer = alpha_infer
        self.alpha_bar_infer = alpha_bar_infer

    # ------------------------------------------------------------------
    # Training-time utilities
    # ------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random integer timesteps for training."""
        return torch.randint(
            low=1,
            high=self.config.timesteps_train,
            size=(batch_size,),
            device=self.device,
        )

    def sample_noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """Sample continuous noise levels between adjacent timesteps.

        For each integer timestep ``t`` in ``[1, T]``, we define the
        cumulative product of ``alpha`` as::

            l_t = sqrt(prod_{i=1}^t (1 - beta_i)) = sqrt(alpha_bar_t)

        and uniformly sample a noise level in the interval
        ``[l_{t-1}, l_t]`` (with ``l_0 = 1``).
        """
        alpha_bar = self.alpha_bar_train
        # l_t = sqrt(alpha_bar_t)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)

        prev_t = torch.clamp(t - 1, min=0)
        l_prev = sqrt_alpha_bar[prev_t]
        l_curr = sqrt_alpha_bar[t]

        u = torch.rand_like(l_prev)
        noise_level = l_prev + u * (l_curr - l_prev)
        return noise_level

    def q_sample(
        self,
        x0: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t given x_0 and a continuous noise level.

        The forward process is::

            x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps,

        where ``alpha_bar = noise_level**2`` and ``eps ~ N(0, I)``.
        """
        alpha_bar = noise_level**2
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar).view(-1, 1, 1, 1) * x0 + torch.sqrt(
            1.0 - alpha_bar
        ).view(-1, 1, 1, 1) * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # Inference-time utilities
    # ------------------------------------------------------------------

    def sample(
        self,
        model,
        *,
        shape: Tuple[int, int, int, int],
        x_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Run the reverse diffusion process to generate samples.

        Parameters
        ----------
        model:
            A neural network with signature
            ``model(x_noisy, noise_level, x_cond) -> eps_pred``.
        shape:
            Output tensor shape ``[B, C, H, W]``.
        x_cond:
            Conditioning Hi-C tensor of shape ``[B, C, H, W]``.
        """
        device = self.device

        x = torch.randn(shape, device=device)
        alpha = self.alpha_infer
        alpha_bar = self.alpha_bar_infer
        beta = self.beta_infer
        T = self.config.timesteps_infer

        sqrt_alpha_bar = torch.sqrt(alpha_bar)

        for n in reversed(range(T)):
            c1 = 1.0 / torch.sqrt(alpha[n])
            c2 = (1.0 - alpha[n]) / torch.sqrt(1.0 - alpha_bar[n])

            noise_level = sqrt_alpha_bar[n].expand(shape[0])
            eps_pred = model(x, noise_level, x_cond)

            x = c1 * (x - c2 * eps_pred)
            if n > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(
                    (1.0 - alpha_bar[n - 1]) / (1.0 - alpha_bar[n]) * beta[n]
                )
                x = x + sigma * noise

        return torch.clamp(x, -1.0, 1.0)


__all__ = ["DDPM", "DDPMConfig"]

