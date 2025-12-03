from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import HiC2MicroCDataset
from .data.utils import DEFAULT_MAXV_1KB, DEFAULT_MAXV_5KB
from .diffusion.ddpm_wavegrad import DDPM, DDPMConfig
from .models.ddpm_unet import UNet


@dataclass
class OptimConfig:
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 10
    grad_clip: float | None = None


@dataclass
class ModelConfig:
    dim: int = 64
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    resnet_block_groups: int = 4
    weight_standardized: bool = False
    use_linear_attention: bool = True


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
class TrainingConfig:
    seed: int = 42
    device: str = "auto"  # cpu | cuda | mps | auto
    output_dir: str = "checkpoints/ddpm"
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DDPMConfig = field(default_factory=DDPMConfig)
    loss: str = "l1"  # l1 or l2


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> TrainingConfig:
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
        return d.get(key, {}) if isinstance(default, dict) else d.get(key, default)

    optim = OptimConfig(**raw.get("optimizer", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    diff_cfg = DDPMConfig(**raw.get("diffusion", {}))

    top = {
        "seed": raw.get("seed", 42),
        "device": raw.get("device", "auto"),
        "output_dir": raw.get("output_dir", "checkpoints/ddpm"),
        "optimizer": optim,
        "model": model_cfg,
        "data": data_cfg,
        "diffusion": diff_cfg,
        "loss": raw.get("loss", "l1"),
    }
    return TrainingConfig(**top)


def build_dataloaders(
    cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, float]:
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
    model: nn.Module,
    diffusion: DDPM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for hic, microc in tqdm(loader, desc="train", leave=False):
        hic = hic.to(device)
        microc = microc.to(device)

        t = diffusion.sample_timesteps(batch_size=microc.size(0))
        noise_level = diffusion.sample_noise_level(t)
        x_t, eps = diffusion.q_sample(microc, noise_level)

        eps_pred = model(x_t, noise_level, hic)
        if loss_type == "l2":
            loss = F.mse_loss(eps_pred, eps)
        else:
            loss = F.l1_loss(eps_pred, eps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    diffusion: DDPM,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for hic, microc in tqdm(loader, desc="val", leave=False):
        hic = hic.to(device)
        microc = microc.to(device)

        t = diffusion.sample_timesteps(batch_size=microc.size(0))
        noise_level = diffusion.sample_noise_level(t)
        x_t, eps = diffusion.q_sample(microc, noise_level)

        eps_pred = model(x_t, noise_level, hic)
        if loss_type == "l2":
            loss = F.mse_loss(eps_pred, eps)
        else:
            loss = F.l1_loss(eps_pred, eps)

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(1, n_batches)


def save_checkpoint(
    path: Path,
    model: nn.Module,
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
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DDPM baseline for Hi-C → Micro-C."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model_cfg = cfg.model
    model = UNet(
        dim=model_cfg.dim,
        input_channels=2,
        channels=1,
        dim_mults=model_cfg.dim_mults,
        resnet_block_groups=model_cfg.resnet_block_groups,
        weight_standardized=model_cfg.weight_standardized,
        use_linear_attention=model_cfg.use_linear_attention,
    ).to(device)

    diff_engine = DDPM(cfg.diffusion, device=device)

    optim_cfg = cfg.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []

    for epoch in range(1, optim_cfg.epochs + 1):
        print(f"Epoch {epoch}/{optim_cfg.epochs} (device={device})")
        train_loss = train_epoch(
            model=model,
            diffusion=diff_engine,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=cfg.loss,
            grad_clip=optim_cfg.grad_clip,
        )
        val_loss = eval_epoch(
            model=model,
            diffusion=diff_engine,
            loader=val_loader,
            device=device,
            loss_type=cfg.loss,
        )
        print(f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        ckpt_path = output_dir / f"ddpm_epoch{epoch}.pt"
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = output_dir / "ddpm_best.pt"
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

    # Save simple history JSON for quick plotting.
    hist_path = output_dir / "ddpm_history.json"
    with hist_path.open("w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
