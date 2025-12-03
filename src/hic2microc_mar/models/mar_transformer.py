from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


@dataclass
class MARConfig:
    img_size: int = 256
    patch_size: int = 16
    token_dim: int = 256
    encoder_embed_dim: int = 512
    encoder_depth: int = 8
    encoder_num_heads: int = 8
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 8
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    buffer_size: int = 32
    mask_ratio_min: float = 0.7
    grad_checkpointing: bool = False


class HiC2MicroCMAR(nn.Module):
    """Transformer backbone for MAR-style Hi-C → Micro-C modeling.

    The module operates on token sequences produced by :class:`HiCTokenizer`.
    It implements an encoder-decoder Transformer similar in spirit to the
    MAR paper, where the encoder processes known tokens (Hi-C + unmasked
    Micro-C + CLS), and the decoder produces conditioning vectors ``z_i``
    for Micro-C tokens (both masked and unmasked).
    """

    def __init__(self, config: MARConfig) -> None:
        super().__init__()
        self.cfg = config

        img_size = config.img_size
        patch_size = config.patch_size
        assert img_size % patch_size == 0

        self.seq_h = self.seq_w = img_size // patch_size
        self.seq_len = self.seq_h * self.seq_w  # tokens per Hi-C or Micro-C patch
        self.buffer_size = config.buffer_size
        self.grad_checkpointing = config.grad_checkpointing

        self.mask_ratio_generator = stats.truncnorm(
            (config.mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # Token projection to encoder dimension.
        self.token_proj = nn.Linear(config.token_dim, config.encoder_embed_dim, bias=True)
        self.token_ln = nn.LayerNorm(config.encoder_embed_dim, eps=1e-6)

        # CLS / buffer tokens.
        self.cls_tokens = nn.Parameter(
            torch.zeros(1, self.buffer_size, config.encoder_embed_dim)
        )

        # Positional embeddings for encoder / decoder.
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.buffer_size + 2 * self.seq_len,
                config.encoder_embed_dim,
            )
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.buffer_size + 2 * self.seq_len,
                config.decoder_embed_dim,
            )
        )
        self.diffusion_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, config.decoder_embed_dim)
        )

        # Encoder blocks.
        norm_layer = nn.LayerNorm
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    config.encoder_embed_dim,
                    config.encoder_num_heads,
                    config.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=config.proj_dropout,
                    attn_drop=config.attn_dropout,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(config.encoder_embed_dim)

        # Decoder blocks.
        self.decoder_embed = nn.Linear(
            config.encoder_embed_dim, config.decoder_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    config.decoder_embed_dim,
                    config.decoder_num_heads,
                    config.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=config.proj_dropout,
                    attn_drop=config.attn_dropout,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(config.decoder_embed_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.token_proj.weight)
        if self.token_proj.bias is not None:
            nn.init.zeros_(self.token_proj.bias)

        nn.init.normal_(self.cls_tokens, std=0.02)
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.diffusion_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize decoder embed.
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        if self.decoder_embed.bias is not None:
            nn.init.zeros_(self.decoder_embed.bias)

        # Standard initialization for Transformer blocks.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if m.weight is not None:
                    nn.init.ones_(m.weight)

    # ------------------------------------------------------------------
    # Masking utilities
    # ------------------------------------------------------------------

    def sample_orders(self, bsz: int) -> torch.Tensor:
        """Sample a batch of random generation orders for Micro-C tokens."""
        orders = []
        for _ in range(bsz):
            order = np.arange(self.seq_len)
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.as_tensor(np.stack(orders, axis=0), dtype=torch.long)
        return orders

    def random_mask(self, bsz: int, device: torch.device) -> torch.Tensor:
        """Sample a random mask over Micro-C tokens using the MAR schedule.

        Returns a boolean tensor of shape ``[B, L]`` where ``True``
        corresponds to masked tokens.
        """
        orders = self.sample_orders(bsz).to(device)
        mask_rate = float(self.mask_ratio_generator.rvs(1)[0])
        num_masked = int(math.ceil(self.seq_len * mask_rate))

        mask = torch.zeros(bsz, self.seq_len, device=device, dtype=torch.bool)
        indices = orders[:, :num_masked]
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
        return mask

    # ------------------------------------------------------------------
    # Core encoder / decoder
    # ------------------------------------------------------------------

    def forward_encoder(
        self,
        hic_tokens: torch.Tensor,
        micro_tokens: torch.Tensor,
        micro_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode known tokens (CLS + Hi-C + unmasked Micro-C).

        Parameters
        ----------
        hic_tokens:
            Tensor of shape ``[B, L, token_dim]``.
        micro_tokens:
            Tensor of shape ``[B, L, token_dim]``.
        micro_mask:
            Boolean tensor of shape ``[B, L]``, where ``True`` indicates
            that the Micro-C token is masked (unknown).
        """
        B, L, _ = hic_tokens.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"
        assert micro_tokens.shape[:2] == (B, L)
        assert micro_mask.shape == (B, L)

        # Project tokens to encoder dimension.
        hic = self.token_ln(self.token_proj(hic_tokens))
        micro = self.token_ln(self.token_proj(micro_tokens))

        cls_tokens = self.cls_tokens.expand(B, -1, -1)  # [B, buffer, D]
        x_all = torch.cat([cls_tokens, hic, micro], dim=1)  # [B, S, D]
        S = x_all.size(1)

        pos = self.encoder_pos_embed[:, :S, :].to(x_all.device)
        x_all = x_all + pos

        # Build full mask over sequence: 1 for masked/dropped.
        mask_full = torch.zeros(B, S, device=x_all.device, dtype=torch.bool)
        start_micro = self.buffer_size + self.seq_len
        mask_full[:, start_micro : start_micro + self.seq_len] = micro_mask

        # Drop masked tokens before encoder as in MAE.
        keep = ~mask_full
        x_visible = x_all[keep].reshape(B, -1, x_all.size(-1))

        for block in self.encoder_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x_visible = torch.utils.checkpoint.checkpoint(block, x_visible)  # type: ignore[attr-defined]
            else:
                x_visible = block(x_visible)
        x_visible = self.encoder_norm(x_visible)

        return x_visible, mask_full

    def forward_decoder(
        self,
        encoded: torch.Tensor,
        mask_full: torch.Tensor,
    ) -> torch.Tensor:
        """Decode to conditioning vectors for all Micro-C tokens."""
        B, _, D_enc = encoded.shape
        S = mask_full.size(1)

        x = self.decoder_embed(encoded)  # [B, N_vis, D_dec]
        D_dec = x.size(-1)

        # Insert mask tokens at masked positions.
        mask_tokens = self.mask_token.expand(B, S, -1).to(dtype=x.dtype, device=x.device)
        x_after_pad = mask_tokens.clone()
        keep = ~mask_full
        x_after_pad[keep] = x.reshape(-1, D_dec)

        pos = self.decoder_pos_embed[:, :S, :].to(x_after_pad.device)
        x_after = x_after_pad + pos

        for block in self.decoder_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x_after = torch.utils.checkpoint.checkpoint(block, x_after)  # type: ignore[attr-defined]
            else:
                x_after = block(x_after)
        x_after = self.decoder_norm(x_after)

        # Extract Micro-C portion and add diffusion positional embedding.
        start_micro = self.buffer_size + self.seq_len
        micro_dec = x_after[:, start_micro : start_micro + self.seq_len, :]
        micro_dec = micro_dec + self.diffusion_pos_embed.to(micro_dec.device)
        return micro_dec  # [B, L, D_dec]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        hic_tokens: torch.Tensor,
        micro_tokens: torch.Tensor,
        micro_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditioning vectors for Micro-C tokens.

        Parameters
        ----------
        hic_tokens:
            [B, L, token_dim] Hi-C token embeddings.
        micro_tokens:
            [B, L, token_dim] Micro-C token embeddings.
        micro_mask:
            [B, L] boolean mask for Micro-C tokens (True = masked).

        Returns
        -------
        z_micro:
            [B, L, decoder_embed_dim] conditioning vectors for each
            Micro-C token. Training code typically uses only the
            positions where ``micro_mask`` is True.
        """
        encoded, mask_full = self.forward_encoder(hic_tokens, micro_tokens, micro_mask)
        z_micro = self.forward_decoder(encoded, mask_full)
        return z_micro


__all__ = ["HiC2MicroCMAR", "MARConfig"]

