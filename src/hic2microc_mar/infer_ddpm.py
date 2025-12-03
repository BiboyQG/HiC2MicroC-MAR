from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import cooler
import numpy as np
import torch
from tqdm import tqdm

from .data.utils import (
    DEFAULT_MAXV_1KB,
    DEFAULT_MAXV_5KB,
    PatchIndex,
    add_weight_column,
    compute_patch_indices,
    get_chr_start_indices,
    merge_patch_predictions,
)
from .diffusion.ddpm_wavegrad import DDPM, DDPMConfig
from .models.ddpm_unet import UNet
from .train_ddpm import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DDPM Hi-C → Micro-C inference and write a .cool file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training YAML config used for the DDPM model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained DDPM checkpoint (.pt).",
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
        help="Chromosomes to process, e.g. chr1 chr2.",
    )
    parser.add_argument(
        "--hic-root",
        type=str,
        default="data/cool",
        help="Root directory containing Hi-C .cool files.",
    )
    parser.add_argument(
        "--microc-max",
        type=float,
        default=None,
        help="Maximum Micro-C value (maxV) used for denormalization. "
        "Defaults to 0.05 for 5 kb and 0.08 for 1 kb.",
    )
    parser.add_argument(
        "--chrom-sizes",
        type=str,
        default="HiC2MicroC/data/hg38.sizes",
        help="Path to chromosome sizes file (e.g. hg38.sizes).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for output; `.cool` will be appended.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: cpu | cuda | mps | auto.",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[UNet, DDPM]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    model_cfg = cfg_dict.get("model", {})
    model = UNet(
        dim=model_cfg.get("dim", 64),
        input_channels=2,
        channels=1,
        dim_mults=tuple(model_cfg.get("dim_mults", (1, 2, 4, 8))),
        resnet_block_groups=model_cfg.get("resnet_block_groups", 4),
        weight_standardized=model_cfg.get("weight_standardized", False),
        use_linear_attention=model_cfg.get("use_linear_attention", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diff_cfg = cfg_dict.get("diffusion", {})
    ddpm_cfg = DDPMConfig(
        timesteps_train=diff_cfg.get("timesteps_train", 1000),
        beta_start_train=diff_cfg.get("beta_start_train", 1e-4),
        beta_end_train=diff_cfg.get("beta_end_train", 2e-2),
        timesteps_infer=diff_cfg.get("timesteps_infer", 50),
        beta_start_infer=diff_cfg.get("beta_start_infer", 1e-4),
        beta_end_infer=diff_cfg.get("beta_end_infer", 0.95),
    )
    ddpm = DDPM(ddpm_cfg, device=device)
    return model, ddpm


def infer_chromosome(
    model: UNet,
    ddpm: DDPM,
    hic_clr: cooler.Cooler,
    chrom: str,
    resolution: int,
    max_value: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[PatchIndex]]:
    chrom_len = int(hic_clr.chromsizes[chrom])
    patch_indices = compute_patch_indices(
        chrom=chrom,
        chrom_length_bp=chrom_len,
        resolution=resolution,
        image_size=256,
    )

    hic_patches: List[np.ndarray] = []
    for idx in patch_indices:
        row_region = (chrom, idx.row_start * resolution, idx.row_end * resolution)
        col_region = (chrom, idx.col_start * resolution, idx.col_end * resolution)
        mat = hic_clr.matrix(balance=True).fetch(row_region, col_region)
        mat = np.nan_to_num(mat, copy=True)
        mat[mat > max_value] = max_value
        mat = np.log10((9.0 / max_value) * mat + 1.0)
        mat = 2.0 * mat - 1.0
        hic_patches.append(mat)

    hic_arr = np.stack(hic_patches, axis=0).astype("float32")
    hic_tensor = torch.from_numpy(hic_arr)  # [N, H, W]

    predictions: List[np.ndarray] = []
    with torch.no_grad():
        n_total = hic_tensor.shape[0]
        for start in tqdm(
            range(0, n_total, batch_size), desc=f"infer {chrom}", leave=False
        ):
            end = min(start + batch_size, n_total)
            hic_batch = hic_tensor[start:end].to(device)  # [B, H, W]
            hic_batch = hic_batch.unsqueeze(1)

            samples = ddpm.sample(
                model=model,
                shape=hic_batch.shape,
                x_cond=hic_batch,
            )
            samples = samples.squeeze(1).cpu().numpy()
            predictions.append(samples)

    pred_arr = np.concatenate(predictions, axis=0)
    return pred_arr, patch_indices


def write_cool_from_coo(
    coo_path: Path,
    cool_out: Path,
    chrom_sizes_path: Path,
    resolution: int,
) -> None:
    """Invoke the ``cooler load`` CLI to build a .cool file."""
    cmd = [
        "cooler",
        "load",
        "-f",
        "coo",
        "--count-as-float",
        f"{chrom_sizes_path}:{resolution}",
        str(coo_path),
        str(cool_out),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    res_label = "5kb" if args.resolution == 5000 else "1kb"
    hic_path = (
        Path(args.hic_root)
        / args.cell_type
        / "HiC"
        / f"{res_label}.cool"
    )

    if args.microc_max is not None:
        max_value = args.microc_max
    else:
        if args.resolution == 5000:
            max_value = DEFAULT_MAXV_5KB
        elif args.resolution == 1000:
            max_value = DEFAULT_MAXV_1KB
        else:
            max_value = DEFAULT_MAXV_5KB

    hic_clr = cooler.Cooler(str(hic_path))

    model, ddpm = load_model(Path(args.checkpoint), device=device)

    # Per-chromosome sparse matrices.
    chroms = list(args.chromosomes)
    chromsizes = [int(hic_clr.chromsizes[c]) for c in chroms]
    resolution = args.resolution

    from scipy import sparse as sp  # local import to avoid unused in training

    coo_rows: List[int] = []
    coo_cols: List[int] = []
    coo_vals: List[float] = []

    for chrom in chroms:
        print(f"[infer_ddpm] Processing {chrom}")
        pred_arr, patch_indices = infer_chromosome(
            model=model,
            ddpm=ddpm,
            hic_clr=hic_clr,
            chrom=chrom,
            resolution=resolution,
            max_value=max_value,
            batch_size=8,
            device=device,
        )
        chrom_len = int(hic_clr.chromsizes[chrom])
        coo_mat = merge_patch_predictions(
            predictions=pred_arr,
            max_value=max_value,
            chrom_length_bp=chrom_len,
            patch_indices=patch_indices,
            resolution=resolution,
            max_bin=456,
            image_size=256,
        )

        mask_upper = (coo_mat.row < coo_mat.col) & (
            (coo_mat.col - coo_mat.row) <= 456
        )
        rows = coo_mat.row[mask_upper]
        cols = coo_mat.col[mask_upper]
        vals = coo_mat.data[mask_upper]

        coo_rows.append(rows)
        coo_cols.append(cols)
        coo_vals.append(vals)

    # Stack all chromosomes into a genome-wide COO using start indices.
    chr_start = get_chr_start_indices(chroms, chromsizes, resolution=resolution)
    rows_all: List[int] = []
    cols_all: List[int] = []
    vals_all: List[float] = []

    for chrom, rows, cols, vals in zip(chroms, coo_rows, coo_cols, coo_vals):
        offset = chr_start[chrom]
        rows_all.append(rows + offset)
        cols_all.append(cols + offset)
        vals_all.append(vals)

    rows_concat = np.concatenate(rows_all)
    cols_concat = np.concatenate(cols_all)
    vals_concat = np.concatenate(vals_all)

    coo_tmp = Path(args.out_prefix + "_COO.tsv.gz")
    import pandas as pd  # local import

    df = pd.DataFrame({"i": rows_concat, "j": cols_concat, "value": vals_concat})
    df.to_csv(coo_tmp, index=False, sep="\t", header=False, compression="gzip")

    cool_out = Path(args.out_prefix + ".cool")
    write_cool_from_coo(
        coo_path=coo_tmp,
        cool_out=cool_out,
        chrom_sizes_path=Path(args.chrom_sizes),
        resolution=resolution,
    )
    add_weight_column(cool_out)
    coo_tmp.unlink(missing_ok=True)
    print(f"[infer_ddpm] Wrote {cool_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
