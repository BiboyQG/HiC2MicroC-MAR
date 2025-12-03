from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import cooler
import numpy as np
import pandas as pd
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
from .diffusion.diffusion_loss import DiffusionLoss, DiffusionLossConfig
from .models.hictokenizer import HiCTokenizer, HiCTokenizerConfig
from .models.mar_transformer import HiC2MicroCMAR, MARConfig
from .train_mar import TrainingConfig as MARTrainingConfig  # type: ignore
from .train_ddpm import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAR+DiffusionLoss Hi-C → Micro-C inference and write a .cool file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="MAR YAML config used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MAR checkpoint (.pt).",
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
    parser.add_argument(
        "--num-iter",
        type=int,
        default=64,
        help="Number of MAR decoding iterations.",
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=None,
        help="Number of sampling steps for the DiffusionLoss head "
        "(overrides config if set).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for DiffusionLoss.",
    )
    return parser.parse_args()


def load_components(
    checkpoint_path: Path,
    device: torch.device,
    num_sampling_steps_override: int | None = None,
) -> Tuple[HiCTokenizer, HiC2MicroCMAR, DiffusionLoss, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    tok_cfg_dict = cfg_dict.get("tokenizer", {})
    tok_cfg = HiCTokenizerConfig(
        img_size=tok_cfg_dict.get("img_size", 256),
        patch_size=tok_cfg_dict.get("patch_size", 16),
        token_dim=tok_cfg_dict.get("token_dim", 256),
    )
    tokenizer = HiCTokenizer(tok_cfg).to(device)
    tokenizer.load_state_dict(ckpt["tokenizer_state"])

    mar_cfg_dict = cfg_dict.get("mar", {})
    mar_cfg = MARConfig(**mar_cfg_dict)
    mar = HiC2MicroCMAR(mar_cfg).to(device)
    mar.load_state_dict(ckpt["mar_state"])
    mar.eval()

    diff_head_cfg_dict = cfg_dict.get("diff_head", {})
    num_sampling = diff_head_cfg_dict.get("num_sampling_steps", "100")
    if num_sampling_steps_override is not None:
        num_sampling = str(num_sampling_steps_override)

    diff_cfg = DiffusionLossConfig(
        target_channels=tok_cfg.token_dim,
        z_channels=mar_cfg.decoder_embed_dim,
        depth=diff_head_cfg_dict.get("depth", 3),
        width=diff_head_cfg_dict.get("width", 512),
        diffusion_steps=diff_head_cfg_dict.get("diffusion_steps", 1000),
        num_sampling_steps=num_sampling,
    )
    diff_head = DiffusionLoss(diff_cfg).to(device)
    diff_head.load_state_dict(ckpt["diff_head_state"])
    diff_head.eval()

    return tokenizer, mar, diff_head, cfg_dict


def mask_by_order(
    mask_len: int,
    orders: torch.Tensor,
    bsz: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct a boolean mask from generation orders."""
    mask = torch.zeros(bsz, seq_len, device=device, dtype=torch.bool)
    indices = orders[:, :mask_len]
    src = torch.ones_like(indices, dtype=torch.bool)
    return mask.scatter(dim=-1, index=indices, src=src)


def generate_micro_tokens(
    tokenizer: HiCTokenizer,
    mar: HiC2MicroCMAR,
    diff_head: DiffusionLoss,
    hic_batch: torch.Tensor,
    num_iter: int,
    temperature: float,
) -> torch.Tensor:
    """MAR-style masked autoregressive decoding for a batch of patches."""
    device = hic_batch.device
    hic_tokens = tokenizer.encode_hic(hic_batch)  # [B, L, D]
    B, L, D = hic_tokens.shape

    tokens = torch.zeros(B, L, D, device=device)
    mask = torch.ones(B, L, device=device, dtype=torch.bool)
    orders = mar.sample_orders(B).to(device)  # [B, L]

    for step in range(num_iter):
        cur_tokens = tokens.clone()

        z_micro = mar(hic_tokens, tokens, mask)  # [B, L, Dz]

        # Mask schedule following MAR / MaskGIT.
        mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter)
        mask_len = int(np.floor(L * mask_ratio))

        # Ensure at least one token remains masked for the next iteration.
        cur_masked = mask.sum(dim=-1, keepdim=True).float()
        mask_len_tensor = torch.full(
            (B, 1), float(mask_len), device=device, dtype=torch.float32
        )
        one = torch.tensor(1.0, device=device)
        mask_len_tensor = torch.maximum(
            one, torch.minimum(cur_masked - 1.0, mask_len_tensor)
        )
        mask_len_int = int(mask_len_tensor[0, 0].item())

        mask_next = mask_by_order(mask_len_int, orders, B, L, device)

        if step >= num_iter - 1:
            mask_to_pred = mask.clone()
        else:
            mask_to_pred = torch.logical_xor(mask, mask_next)
        mask = mask_next

        # Sample latents for the tokens predicted at this step.
        z_step = z_micro[mask_to_pred]
        if z_step.numel() == 0:
            continue
        sampled = diff_head.sample(z_step, temperature=temperature, cfg_scale=1.0, device=device)

        cur_tokens[mask_to_pred] = sampled
        tokens = cur_tokens.clone()

    return tokens


def infer_chromosome(
    tokenizer: HiCTokenizer,
    mar: HiC2MicroCMAR,
    diff_head: DiffusionLoss,
    hic_clr: cooler.Cooler,
    chrom: str,
    resolution: int,
    max_value: float,
    batch_size: int,
    num_iter: int,
    temperature: float,
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
            range(0, n_total, batch_size), desc=f"infer MAR {chrom}", leave=False
        ):
            end = min(start + batch_size, n_total)
            hic_batch = hic_tensor[start:end].to(device).unsqueeze(1)  # [B, 1, H, W]

            micro_tokens = generate_micro_tokens(
                tokenizer=tokenizer,
                mar=mar,
                diff_head=diff_head,
                hic_batch=hic_batch,
                num_iter=num_iter,
                temperature=temperature,
            )
            micro_patches = tokenizer.decode_microc(micro_tokens)  # [B, 1, H, W]
            micro_patches = micro_patches.squeeze(1).cpu().numpy()
            predictions.append(micro_patches)

    pred_arr = np.concatenate(predictions, axis=0)
    return pred_arr, patch_indices


def write_cool_from_coo(
    coo_path: Path,
    cool_out: Path,
    chrom_sizes_path: Path,
    resolution: int,
) -> None:
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

    tokenizer, mar, diff_head, cfg_dict = load_components(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        num_sampling_steps_override=args.num_diffusion_steps,
    )

    chroms = list(args.chromosomes)
    chromsizes = [int(hic_clr.chromsizes[c]) for c in chroms]
    resolution = args.resolution

    coo_rows: List[np.ndarray] = []
    coo_cols: List[np.ndarray] = []
    coo_vals: List[np.ndarray] = []

    for chrom in chroms:
        print(f"[infer_mar] Processing {chrom}")
        pred_arr, patch_indices = infer_chromosome(
            tokenizer=tokenizer,
            mar=mar,
            diff_head=diff_head,
            hic_clr=hic_clr,
            chrom=chrom,
            resolution=resolution,
            max_value=max_value,
            batch_size=4,
            num_iter=args.num_iter,
            temperature=args.temperature,
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

    chr_start = get_chr_start_indices(chroms, chromsizes, resolution=resolution)
    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    vals_all: List[np.ndarray] = []

    for chrom, rows, cols, vals in zip(chroms, coo_rows, coo_cols, coo_vals):
        offset = chr_start[chrom]
        rows_all.append(rows + offset)
        cols_all.append(cols + offset)
        vals_all.append(vals)

    rows_concat = np.concatenate(rows_all)
    cols_concat = np.concatenate(cols_all)
    vals_concat = np.concatenate(vals_all)

    coo_tmp = Path(args.out_prefix + "_COO.tsv.gz")
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
    print(f"[infer_mar] Wrote {cool_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
