from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cooler
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from tqdm import tqdm

from .data.dataset import HiC2MicroCDataset
from .data.utils import DEFAULT_MAXV_1KB, DEFAULT_MAXV_5KB
from .infer_ddpm import infer_chromosome as ddpm_infer_chrom
from .infer_mar import infer_chromosome as mar_infer_chrom
from .train_ddpm import resolve_device


@dataclass
class EvalConfig:
    model: str
    cell_type: str
    resolution: int
    chromosomes: Sequence[str]
    results_dir: Path
    device: torch.device
    ddpm_config: Path | None
    ddpm_checkpoint: Path | None
    mar_config: Path | None
    mar_checkpoint: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hi-C → Micro-C models (DDPM vs MAR)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["ddpm", "mar", "both"],
        required=True,
        help="Which model(s) to evaluate.",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        required=True,
        help="Cell type, e.g. HFFc6 or H1ESC.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Resolution in base pairs (e.g. 5000 or 1000).",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        required=True,
        help="Chromosomes to evaluate on, e.g. chr1 chr2.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory to store evaluation outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: cpu | cuda | mps | auto.",
    )
    parser.add_argument(
        "--ddpm-config",
        type=str,
        default=None,
        help="DDPM YAML config used for training (required if evaluating DDPM).",
    )
    parser.add_argument(
        "--ddpm-checkpoint",
        type=str,
        default=None,
        help="DDPM checkpoint (.pt).",
    )
    parser.add_argument(
        "--mar-config",
        type=str,
        default=None,
        help="MAR YAML config (required if evaluating MAR).",
    )
    parser.add_argument(
        "--mar-checkpoint",
        type=str,
        default=None,
        help="MAR checkpoint (.pt).",
    )
    return parser.parse_args()


def build_eval_config(args: argparse.Namespace) -> EvalConfig:
    device = resolve_device(args.device)
    results_dir = Path(args.results_dir).expanduser().resolve()

    ddpm_config = Path(args.ddpm_config) if args.ddpm_config else None
    ddpm_checkpoint = Path(args.ddpm_checkpoint) if args.ddpm_checkpoint else None
    mar_config = Path(args.mar_config) if args.mar_config else None
    mar_checkpoint = Path(args.mar_checkpoint) if args.mar_checkpoint else None

    if args.model in ("ddpm", "both"):
        if ddpm_config is None or ddpm_checkpoint is None:
            raise ValueError("DDPM evaluation requested but --ddpm-config / --ddpm-checkpoint not provided.")
    if args.model in ("mar", "both"):
        if mar_config is None or mar_checkpoint is None:
            raise ValueError("MAR evaluation requested but --mar-config / --mar-checkpoint not provided.")

    return EvalConfig(
        model=args.model,
        cell_type=args.cell_type,
        resolution=args.resolution,
        chromosomes=args.chromosomes,
        results_dir=results_dir,
        device=device,
        ddpm_config=ddpm_config,
        ddpm_checkpoint=ddpm_checkpoint,
        mar_config=mar_config,
        mar_checkpoint=mar_checkpoint,
    )


def get_maxv(resolution: int) -> float:
    if resolution == 5000:
        return DEFAULT_MAXV_5KB
    if resolution == 1000:
        return DEFAULT_MAXV_1KB
    return DEFAULT_MAXV_5KB


def compute_fid_for_model(
    model_name: str,
    pred_patches: np.ndarray,
    gt_patches: np.ndarray,
    results_dir: Path,
) -> float:
    """Compute a FID-like metric using torch-fidelity."""
    # torch-fidelity expects images on disk; write temporary PNGs.
    import imageio.v2 as imageio  # type: ignore
    import os

    pred_dir = results_dir / f"fid_images_{model_name}_pred"
    gt_dir = results_dir / f"fid_images_{model_name}_gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    def _write_images(arr: np.ndarray, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for i, patch in enumerate(arr):
            img = patch.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            img_rgb = np.stack([img] * 3, axis=-1)
            imageio.imwrite(root / f"{i:06d}.png", img_rgb)

    _write_images(pred_patches, pred_dir)
    _write_images(gt_patches, gt_dir)

    metrics = calculate_metrics(
        input1=str(pred_dir),
        input2=str(gt_dir),
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def collect_patches_from_cool(
    cool_path: Path,
    microc_max: float,
    resolution: int,
    chromosomes: Sequence[str],
    num_patches: int = 10000,
) -> np.ndarray:
    """Extract normalized patches from a Micro-C .cool file."""
    # Use the same sliding-window scheme as the dataset.
    from .data.utils import compute_patch_indices, normalize_contacts

    clr = cooler.Cooler(str(cool_path))
    patches: List[np.ndarray] = []
    for chrom in chromosomes:
        chrom_len = int(clr.chromsizes[chrom])
        idx_list = compute_patch_indices(
            chrom=chrom,
            chrom_length_bp=chrom_len,
            resolution=resolution,
            image_size=256,
        )
        for idx in idx_list:
            row_start_bp = idx.row_start * resolution
            row_end_bp = (idx.row_end - 1) * resolution + resolution
            col_start_bp = idx.col_start * resolution
            col_end_bp = (idx.col_end - 1) * resolution + resolution
            row_end_bp = min(row_end_bp, chrom_len)
            col_end_bp = min(col_end_bp, chrom_len)
            row_region = (chrom, row_start_bp, row_end_bp)
            col_region = (chrom, col_start_bp, col_end_bp)
            mat = clr.matrix(balance=True).fetch(row_region, col_region)
            mat = normalize_contacts(mat, microc_max)
            patches.append(mat.astype("float32"))
            if len(patches) >= num_patches:
                return np.stack(patches, axis=0)
    if not patches:
        raise RuntimeError(f"No patches collected from {cool_path}")
    return np.stack(patches, axis=0)


def evaluate_speed_and_generate(
    cfg: EvalConfig,
    model_name: str,
    infer_fn,
    **infer_kwargs,
) -> Tuple[np.ndarray, float, float]:
    """Run inference and measure per-window and per-chromosome speed."""
    res_label = "5kb" if cfg.resolution == 5000 else "1kb"
    hic_path = (
        Path("data/cool")
        / cfg.cell_type
        / "HiC"
        / f"{res_label}.cool"
    )
    hic_clr = cooler.Cooler(str(hic_path))

    maxv = get_maxv(cfg.resolution)

    all_predictions: List[np.ndarray] = []
    total_time = 0.0
    total_windows = 0

    for chrom in cfg.chromosomes:
        print(f"[eval] {model_name} inferring {chrom}")
        start = time.time()
        pred_arr, patch_indices = infer_fn(
            hic_clr=hic_clr,
            chrom=chrom,
            resolution=cfg.resolution,
            max_value=maxv,
            device=cfg.device,
            **infer_kwargs,
        )
        elapsed = time.time() - start

        all_predictions.append(pred_arr)
        total_time += elapsed
        total_windows += pred_arr.shape[0]

    preds = np.concatenate(all_predictions, axis=0)
    per_window = total_time / max(total_windows, 1)
    return preds, per_window, total_time


def main() -> None:
    args = parse_args()
    cfg = build_eval_config(args)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "cell_type": cfg.cell_type,
        "resolution": cfg.resolution,
        "chromosomes": list(cfg.chromosomes),
    }

    # Experimental Micro-C ground truth for FID and APA.
    res_label = "5kb" if cfg.resolution == 5000 else "1kb"
    microc_exp_path = (
        Path("data/cool")
        / cfg.cell_type
        / "MicroC"
        / f"{res_label}.cool"
    )
    microc_max = get_maxv(cfg.resolution)

    # Collect a reference set of patches for FID.
    print("[eval] Collecting ground-truth Micro-C patches for FID...")
    gt_patches = collect_patches_from_cool(
        microc_exp_path,
        microc_max=microc_max,
        resolution=cfg.resolution,
        chromosomes=cfg.chromosomes,
        num_patches=5000,
    )

    # DDPM evaluation.
    if cfg.model in ("ddpm", "both"):
        from .infer_ddpm import load_model as load_ddpm_model

        print("[eval] Loading DDPM model...")
        ddpm_model, ddpm_engine = load_ddpm_model(cfg.ddpm_checkpoint, cfg.device)  # type: ignore[arg-type]

        def _ddpm_infer_wrapper(
            hic_clr,
            chrom,
            resolution,
            max_value,
            device,
            batch_size: int = 8,
        ):
            return ddpm_infer_chrom(
                model=ddpm_model,
                ddpm=ddpm_engine,
                hic_clr=hic_clr,
                chrom=chrom,
                resolution=resolution,
                max_value=max_value,
                batch_size=batch_size,
                device=device,
            )

        ddpm_preds, ddpm_t_per_window, ddpm_t_total = evaluate_speed_and_generate(
            cfg, "ddpm", _ddpm_infer_wrapper
        )

        fid_ddpm = compute_fid_for_model(
            "ddpm", ddpm_preds, gt_patches, cfg.results_dir
        )

        summary["ddpm"] = {
            "fid": fid_ddpm,
            "time_per_window_sec": ddpm_t_per_window,
            "time_total_sec": ddpm_t_total,
            "num_windows": int(ddpm_preds.shape[0]),
        }

    # MAR evaluation.
    if cfg.model in ("mar", "both"):
        from .infer_mar import load_components as load_mar_components

        print("[eval] Loading MAR model...")
        tokenizer, mar_model, diff_head, _ = load_mar_components(  # type: ignore[arg-type]
            checkpoint_path=cfg.mar_checkpoint,
            device=cfg.device,
            num_sampling_steps_override=None,
        )

        def _mar_infer_wrapper(
            hic_clr,
            chrom,
            resolution,
            max_value,
            device,
            batch_size: int = 4,
            num_iter: int = 64,
            temperature: float = 1.0,
        ):
            return mar_infer_chrom(
                tokenizer=tokenizer,
                mar=mar_model,
                diff_head=diff_head,
                hic_clr=hic_clr,
                chrom=chrom,
                resolution=resolution,
                max_value=max_value,
                batch_size=batch_size,
                num_iter=num_iter,
                temperature=temperature,
                device=device,
            )

        mar_preds, mar_t_per_window, mar_t_total = evaluate_speed_and_generate(
            cfg,
            "mar",
            _mar_infer_wrapper,
        )

        fid_mar = compute_fid_for_model(
            "mar", mar_preds, gt_patches, cfg.results_dir
        )

        summary["mar"] = {
            "fid": fid_mar,
            "time_per_window_sec": mar_t_per_window,
            "time_total_sec": mar_t_total,
            "num_windows": int(mar_preds.shape[0]),
        }

    # NOTE: Loop calling (Mustache/SIP) and APA analyses are intentionally
    # stubbed out here for simplicity. The evaluation scaffold is in place,
    # and these can be added by extending this script to call the respective
    # tools on the experimental and predicted .cool files.

    out_json = cfg.results_dir / "evaluation_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Wrote summary to {out_json}")


if __name__ == "__main__":  # pragma: no cover
    main()
