from __future__ import annotations

import importlib
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Lazily expose the local `mar/diffusion` package as `mar.diffusion`
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAR_DIR = _REPO_ROOT / "mar"
if "mar" not in sys.modules and _MAR_DIR.is_dir():
    mar_pkg = types.ModuleType("mar")
    mar_pkg.__path__ = [str(_MAR_DIR)]
    sys.modules["mar"] = mar_pkg

from mar.diffusion import create_diffusion
import mar.diffusion.gaussian_diffusion as _gd


def _patched_extract_into_tensor(arr, timesteps, broadcast_shape):
    """Float32-friendly version of _extract_into_tensor for MPS/CPU."""
    res = torch.from_numpy(arr).float().to(device=timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


_gd._extract_into_tensor = _patched_extract_into_tensor


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """Residual block with AdaLN modulation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """Final AdaLN projection layer (adapted from DiT)."""

    def __init__(self, model_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """MLP backbone for Diffusion Loss."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList(
            [ResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.final_layer = FinalLayer(model_channels, out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP.
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out AdaLN modulation layers.
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """Apply the model to an input batch."""
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Classifier-free guidance variant."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


@dataclass
class DiffusionLossConfig:
    target_channels: int
    z_channels: int
    depth: int = 3
    width: int = 512
    diffusion_steps: int = 1000
    num_sampling_steps: int | str = "100"
    grad_checkpointing: bool = False


class DiffusionLoss(nn.Module):
    """Per-token Diffusion Loss with cosine noise schedule."""

    def __init__(self, cfg: DiffusionLossConfig) -> None:
        super().__init__()
        self.in_channels = cfg.target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=cfg.target_channels,
            model_channels=cfg.width,
            out_channels=cfg.target_channels * 2,  # for VLB-style loss
            z_channels=cfg.z_channels,
            num_res_blocks=cfg.depth,
            grad_checkpointing=cfg.grad_checkpointing,
        )

        self.train_diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="cosine",
            diffusion_steps=cfg.diffusion_steps,
        )
        self.gen_diffusion = create_diffusion(
            timestep_respacing=cfg.num_sampling_steps,
            noise_schedule="cosine",
            diffusion_steps=cfg.diffusion_steps,
        )

    def forward(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        """Compute diffusion training loss on masked tokens.

        Parameters
        ----------
        target:
            Tensor of shape ``[N, C]`` containing token vectors.
        z:
            Conditioning tensor ``[N, D]`` from the MAR Transformer.
        mask:
            Optional 1D tensor ``[N]`` of {0, 1} weights or boolean mask.
        """
        t = torch.randint(
            0,
            self.train_diffusion.num_timesteps,
            (target.shape[0],),
            device=target.device,
        )
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = mask.float()
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample token vectors conditioned on MAR outputs."""
        if device is None:
            device = z.device

        if cfg_scale != 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg_scale)

            def model_fn(x, t, **kwargs):
                return self.net.forward_with_cfg(x, t, kwargs["c"], kwargs["cfg_scale"])

        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=device)
            model_kwargs = dict(c=z)

            def model_fn(x, t, **kwargs):
                return self.net(x, t, kwargs["c"])

        x = noise
        diffusion = self.gen_diffusion
        for i in reversed(range(diffusion.num_timesteps)):
            t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
            out = diffusion.p_sample(
                model_fn,
                x,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                temperature=temperature,
            )
            x = out["sample"]
        return x


__all__ = ["DiffusionLoss", "DiffusionLossConfig"]
