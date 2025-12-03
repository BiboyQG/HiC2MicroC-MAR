from __future__ import annotations

from functools import partial
from math import log as ln
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def exists(x) -> bool:
    return x is not None


def default(val, d):
    return val if exists(val) else d() if callable(d) else d


class Residual(nn.Module):
    """Simple residual wrapper."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        return self.fn(x, *args, **kwargs) + x


class PositionalEncoding(nn.Module):
    """Noise-level positional encoding as in HiC2MicroC."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


def Upsample(dim: int, dim_out: int | None = None) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim: int, dim_out: int | None = None) -> nn.Sequential:
    # No strided convolutions or pooling; instead, use pixel-unshuffle style.
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class WeightStandardizedConv2d(nn.Conv2d):
    """Weight-standardized conv as used with GroupNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        kernel_size: int = 3,
        groups: int = 8,
        weight_standardized: bool = False,
    ) -> None:
        super().__init__()
        if weight_standardized:
            self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        else:
            self.proj = nn.Conv2d(dim, dim_out, kernel_size, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1.0) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """ResNet block with optional time embedding modulation."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: int | None = None,
        groups: int = 8,
        weight_standardized: bool = False,
    ) -> None:
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(
            dim,
            dim_out,
            groups=groups,
            weight_standardized=weight_standardized,
        )
        self.block2 = Block(
            dim_out,
            dim_out,
            groups=groups,
            weight_standardized=weight_standardized,
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        scale_shift: Tuple[torch.Tensor, torch.Tensor] | None = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    """Linearized self-attention over 2D feature maps."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """Full self-attention over 2D feature maps."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """GroupNorm followed by the wrapped module."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.norm(x)
        return self.fn(x)


class UNet(nn.Module):
    """U-Net backbone used in the HiC2MicroC DDPM baseline.

    The network takes as input a noisy Micro-C patch and a conditioning
    Hi-C patch (concatenated along the channel dimension), together with
    a continuous noise level scalar per sample.
    """

    def __init__(
        self,
        dim: int,
        input_channels: int = 2,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        channels: int = 1,
        resnet_block_groups: int = 4,
        weight_standardized: bool = False,
        use_linear_attention: bool = True,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.use_linear_attention = use_linear_attention

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            weight_standardized=weight_standardized,
        )

        # Time / noise-level embeddings.
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Downsampling path.
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                        if use_linear_attention
                        else nn.Identity(),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # Bottleneck.
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling path.
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                        if use_linear_attention
                        else nn.Identity(),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        x_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_noisy:
            Noisy Micro-C patch, shape ``[B, C, H, W]``.
        noise_level:
            Continuous noise level per sample, shape ``[B]``.
        x_cond:
            Conditioning Hi-C patch, shape ``[B, C, H, W]``.
        """
        # Concatenate conditioning and target channels.
        x = torch.cat((x_cond, x_noisy), dim=1)
        x = self.init_conv(x)
        residual = x.clone()

        t = self.time_mlp(noise_level)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


__all__ = ["UNet"]

