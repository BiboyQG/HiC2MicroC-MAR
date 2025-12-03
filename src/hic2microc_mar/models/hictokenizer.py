from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HiCTokenizerConfig:
    img_size: int = 256
    patch_size: int = 16
    token_dim: int = 256


class TokenType(enum.IntEnum):
    HIC = 0
    MICROC = 1
    MASK = 2
    CLS = 3


class HiCTokenizer(nn.Module):
    """Continuous patch tokenizer for Hi-C / Micro-C patches.

    This module converts a single-channel 2D contact map into a sequence
    of continuous tokens by splitting the image into non-overlapping
    patches and projecting each patch through a linear layer. It also
    provides positional and token-type embeddings, and an inverse mapping
    back to image space.
    """

    def __init__(self, config: HiCTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        img_size = config.img_size
        patch_size = config.patch_size

        assert (
            img_size % patch_size == 0
        ), f"img_size {img_size} must be divisible by patch_size {patch_size}"

        self.grid_size = img_size // patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size
        self.token_dim = config.token_dim

        # Patch projection.
        self.to_token = nn.Linear(self.patch_dim, self.token_dim)
        self.to_pixels = nn.Linear(self.token_dim, self.patch_dim)

        # Embeddings.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.token_dim)
        )
        self.token_type_embed = nn.Embedding(len(TokenType), self.token_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.to_token.weight)
        nn.init.zeros_(self.to_token.bias)
        nn.init.xavier_uniform_(self.to_pixels.weight)
        nn.init.zeros_(self.to_pixels.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    # Patchify / unpatchify
    # ------------------------------------------------------------------

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, 1, H, W] into [B, L, patch_dim] tokens."""
        bsz, c, h, w = x.shape
        p = self.config.patch_size
        assert c == 1, f"Expected single-channel input, got {c} channels"
        assert (
            h == self.config.img_size and w == self.config.img_size
        ), f"Expected {self.config.img_size}x{self.config.img_size} patches"

        h_, w_ = h // p, w // p
        x = x.view(bsz, c, h_, p, w_, p)
        # [B, c, h_, p, w_, p] -> [B, h_, w_, c, p, p]
        x = torch.einsum("bchpwq->bhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p * p)
        return x  # [B, L, patch_dim]

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, L, patch_dim] back to [B, 1, H, W]."""
        bsz, L, d = x.shape
        p = self.config.patch_size
        h_ = w_ = self.grid_size

        assert L == h_ * w_, f"Expected {h_ * w_} tokens, got {L}"
        assert d == self.patch_dim, f"Expected patch_dim={self.patch_dim}, got {d}"

        x = x.view(bsz, h_, w_, 1, p, p)
        # [B, h_, w_, 1, p, p] -> [B, 1, h_, p, w_, p]
        x = torch.einsum("bhwcpq->bchpwq", x)
        x = x.reshape(bsz, 1, h_ * p, w_ * p)
        return x

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _encode(
        self,
        x: torch.Tensor,
        token_type: TokenType,
    ) -> torch.Tensor:
        patches = self.patchify(x)  # [B, L, patch_dim]
        tokens = self.to_token(patches)  # [B, L, token_dim]

        pos = self.pos_embed[:, : tokens.size(1), :]
        type_emb = self.token_type_embed(
            torch.full(
                (tokens.size(0), tokens.size(1)),
                int(token_type),
                device=tokens.device,
                dtype=torch.long,
            )
        )
        return tokens + pos + type_emb

    def encode_hic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode Hi-C patches into token embeddings."""
        return self._encode(x, TokenType.HIC)

    def encode_microc(self, x: torch.Tensor) -> torch.Tensor:
        """Encode Micro-C patches into token embeddings."""
        return self._encode(x, TokenType.MICROC)

    def decode_microc(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode Micro-C tokens back to image space.

        Parameters
        ----------
        tokens:
            Tensor of shape ``[B, L, token_dim]`` containing token
            embeddings corresponding to Micro-C patches.
        """
        patches = self.to_pixels(tokens)  # [B, L, patch_dim]
        return self.unpatchify(patches)


__all__ = ["HiCTokenizer", "HiCTokenizerConfig", "TokenType"]

