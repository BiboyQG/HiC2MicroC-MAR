from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import HiC2MicroCDataset
from .data.utils import DEFAULT_MAXV_1KB, DEFAULT_MAXV_5KB
from .diffusion.diffusion_loss import DiffusionLoss, DiffusionLossConfig
from .models.hictokenizer import HiCTokenizer, HiCTokenizerConfig
from .models.mar_transformer import HiC2MicroCMAR, MARConfig
from .train_ddpm import resolve_device, set_seed


@dataclass
class OptimConfig:
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 10
    grad_clip: float | None = None


@dataclass
class DataConfig:
    cell_type: str = "HFFc6"
    resolution: int = 5000
    chromosomes_train: Tuple[str, ...] = (
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
    )
    chromosomes_val: Tuple[str, ...] = ("chr17",)
    hic_root: str = "data/cool"
    microc_root: str = "data/cool"


@dataclass
class TokenizerConfig:
    img_size: int = 256
    patch_size: int = 16
    token_dim: int = 256


@dataclass
class DiffHeadConfig:
    width: int = 512
    depth: int = 3
    diffusion_steps: int = 1000
    num_sampling_steps: str = "100"


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "checkpoints/mar"
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    mar: MARConfig = field(default_factory=MARConfig)
    diff_head: DiffHeadConfig = field(default_factory=DiffHeadConfig)


def load_config(path: Path) -> TrainingConfig:
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    optim = OptimConfig(**raw.get("optimizer", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    tok_cfg = TokenizerConfig(**raw.get("tokenizer", {}))
    mar_cfg = MARConfig(**raw.get("mar", {}))
    diff_cfg = DiffHeadConfig(**raw.get("diff_head", {}))

    top: Dict[str, Any] = {
        "seed": raw.get("seed", 42),
        "device": raw.get("device", "auto"),
        "output_dir": raw.get("output_dir", "checkpoints/mar"),
        "optimizer": optim,
        "data": data_cfg,
        "tokenizer": tok_cfg,
        "mar": mar_cfg,
        "diff_head": diff_cfg,
    }
    return TrainingConfig(**top)


def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader, float]:
    data_cfg = cfg.data

    res_label = "5kb" if data_cfg.resolution == 5000 else "1kb"
    hic_path = (
        Path(data_cfg.hic_root)
        / data_cfg.cell_type
        / "HiC"
        / f"{res_label}.cool"
    )
    microc_path = (
        Path(data_cfg.microc_root)
        / data_cfg.cell_type
        / "MicroC"
        / f"{res_label}.cool"
    )

    if data_cfg.resolution == 5000:
        maxv = DEFAULT_MAXV_5KB
    elif data_cfg.resolution == 1000:
        maxv = DEFAULT_MAXV_1KB
    else:
        maxv = DEFAULT_MAXV_5KB

    train_ds = HiC2MicroCDataset(
        hic_path=hic_path,
        microc_path=microc_path,
        resolution=data_cfg.resolution,
        chromosomes=list(data_cfg.chromosomes_train),
        max_hic_value=maxv,
        max_microc_value=maxv,
    )
    val_ds = HiC2MicroCDataset(
        hic_path=hic_path,
        microc_path=microc_path,
        resolution=data_cfg.resolution,
        chromosomes=list(data_cfg.chromosomes_val),
        max_hic_value=maxv,
        max_microc_value=maxv,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.optimizer.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.optimizer.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, val_loader, maxv


def train_epoch(
    mar: HiC2MicroCMAR,
    tokenizer: HiCTokenizer,
    diff_head: DiffusionLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    mar.train()
    diff_head.train()
    total_loss = 0.0
    n_batches = 0

    for hic, microc in tqdm(loader, desc="train-mar", leave=False):
        hic = hic.to(device)
        microc = microc.to(device)

        hic_tokens = tokenizer.encode_hic(hic)  # [B, L, D]
        micro_tokens = tokenizer.encode_microc(microc)  # [B, L, D]

        B, L, D = micro_tokens.shape
        micro_mask = mar.random_mask(B, device=device)  # [B, L] bool

        z_micro = mar(hic_tokens, micro_tokens, micro_mask)  # [B, L, Dz]

        # Select masked positions only.
        mask_flat = micro_mask.view(-1)
        target_flat = micro_tokens.view(B * L, D)[mask_flat]
        z_flat = z_micro.view(B * L, -1)[mask_flat]

        loss = diff_head(target_flat, z_flat)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.optimizer.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(mar.parameters()) + list(tokenizer.parameters()) + list(diff_head.parameters()),
                cfg.optimizer.grad_clip,
            )
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch(
    mar: HiC2MicroCMAR,
    tokenizer: HiCTokenizer,
    diff_head: DiffusionLoss,
    loader: DataLoader,
    device: torch.device,
) -> float:
    mar.eval()
    diff_head.eval()
    total_loss = 0.0
    n_batches = 0

    for hic, microc in tqdm(loader, desc="val-mar", leave=False):
        hic = hic.to(device)
        microc = microc.to(device)

        hic_tokens = tokenizer.encode_hic(hic)
        micro_tokens = tokenizer.encode_microc(microc)

        B, L, D = micro_tokens.shape
        micro_mask = mar.random_mask(B, device=device)

        z_micro = mar(hic_tokens, micro_tokens, micro_mask)

        mask_flat = micro_mask.view(-1)
        target_flat = micro_tokens.view(B * L, D)[mask_flat]
        z_flat = z_micro.view(B * L, -1)[mask_flat]

        loss = diff_head(target_flat, z_flat)
        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(1, n_batches)


def save_checkpoint(
    path: Path,
    mar: HiC2MicroCMAR,
    tokenizer: HiCTokenizer,
    diff_head: DiffusionLoss,
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "mar_state": mar.state_dict(),
            "tokenizer_state": tokenizer.state_dict(),
            "diff_head_state": diff_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MAR + Diffusion Loss model for Hi-C → Micro-C."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to MAR YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    global cfg  # used inside train_epoch for grad clipping

    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    tok_cfg = cfg.tokenizer
    tokenizer = HiCTokenizer(
        HiCTokenizerConfig(
            img_size=tok_cfg.img_size,
            patch_size=tok_cfg.patch_size,
            token_dim=tok_cfg.token_dim,
        )
    ).to(device)

    mar = HiC2MicroCMAR(cfg.mar).to(device)

    diff_cfg = cfg.diff_head
    diff_loss = DiffusionLoss(
        DiffusionLossConfig(
            target_channels=tok_cfg.token_dim,
            z_channels=cfg.mar.decoder_embed_dim,
            depth=diff_cfg.depth,
            width=diff_cfg.width,
            diffusion_steps=diff_cfg.diffusion_steps,
            num_sampling_steps=diff_cfg.num_sampling_steps,
        )
    ).to(device)

    optim_cfg = cfg.optimizer
    params = list(mar.parameters()) + list(tokenizer.parameters()) + list(
        diff_loss.parameters()
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []

    for epoch in range(1, optim_cfg.epochs + 1):
        print(f"[MAR] Epoch {epoch}/{optim_cfg.epochs} (device={device})")
        train_loss = train_epoch(
            mar=mar,
            tokenizer=tokenizer,
            diff_head=diff_loss,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss = eval_epoch(
            mar=mar,
            tokenizer=tokenizer,
            diff_head=diff_loss,
            loader=val_loader,
            device=device,
        )
        print(f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        ckpt_path = output_dir / f"mar_epoch{epoch}.pt"
        save_checkpoint(
            path=ckpt_path,
            mar=mar,
            tokenizer=tokenizer,
            diff_head=diff_loss,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = output_dir / "mar_best.pt"
            save_checkpoint(
                path=best_path,
                mar=mar,
                tokenizer=tokenizer,
                diff_head=diff_loss,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

    hist_path = output_dir / "mar_history.json"
    with hist_path.open("w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
