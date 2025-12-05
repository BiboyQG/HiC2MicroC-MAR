from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cooler
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_fidelity import calculate_metrics
from tqdm import tqdm

from .data.utils import (
    DEFAULT_MAXV_1KB,
    DEFAULT_MAXV_5KB,
    add_weight_column,
    get_chr_start_indices,
    merge_patch_predictions,
)
from .infer_ddpm import infer_chromosome as ddpm_infer_chrom
from .infer_ddpm import write_cool_from_coo as ddpm_write_cool
from .infer_mar import infer_chromosome as mar_infer_chrom
from .infer_mar import write_cool_from_coo as mar_write_cool
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
    parser.add_argument(
        "--chrom-sizes",
        type=str,
        default="HiC2MicroC/data/hg38.sizes",
        help="Chromosome sizes file used when building predicted .cool files.",
    )
    parser.add_argument(
        "--mustache-template",
        type=str,
        default="",
        help=(
            "Shell template to call Mustache. "
            "Placeholders: {cool}, {res}, {chrom}, {out}, {fdr}."
        ),
    )
    parser.add_argument(
        "--sip-template",
        type=str,
        default="",
        help=(
            "Shell template to call SIP. "
            "Placeholders: {cool}, {res}, {chrom}, {out}, {fdr}."
        ),
    )
    parser.add_argument(
        "--loop-tolerance-bins",
        type=int,
        default=2,
        help="Neighborhood tolerance in bins when matching loops.",
    )
    parser.add_argument(
        "--apa-window-bins",
        type=int,
        default=5,
        help="APA window half-size in bins (e.g. 5 → 11×11).",
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


def ensure_predicted_cool_ddpm(
    cfg: EvalConfig,
    chrom_sizes_path: Path,
) -> Path:
    """Run DDPM inference to produce a genome-wide predicted .cool if needed."""
    out_prefix = cfg.results_dir / f"ddpm_pred_{cfg.cell_type}_{cfg.resolution}"
    cool_path = Path(str(out_prefix) + ".cool")
    if cool_path.exists():
        return cool_path

    cmd = [
        sys.executable,
        "-m",
        "hic2microc_mar.infer_ddpm",
        "--config",
        str(cfg.ddpm_config),
        "--checkpoint",
        str(cfg.ddpm_checkpoint),
        "--cell-type",
        cfg.cell_type,
        "--resolution",
        str(cfg.resolution),
        "--chromosomes",
        *cfg.chromosomes,
        "--hic-root",
        "data/cool",
        "--chrom-sizes",
        str(chrom_sizes_path),
        "--out-prefix",
        str(out_prefix),
        "--device",
        "auto",
    ]
    print("[eval] Running DDPM inference to build predicted .cool...")
    subprocess.run(cmd, check=True)
    return cool_path


def ensure_predicted_cool_mar(
    cfg: EvalConfig,
    chrom_sizes_path: Path,
) -> Path:
    """Run MAR inference to produce a genome-wide predicted .cool if needed."""
    out_prefix = cfg.results_dir / f"mar_pred_{cfg.cell_type}_{cfg.resolution}"
    cool_path = Path(str(out_prefix) + ".cool")
    if cool_path.exists():
        return cool_path

    cmd = [
        sys.executable,
        "-m",
        "hic2microc_mar.infer_mar",
        "--config",
        str(cfg.mar_config),
        "--checkpoint",
        str(cfg.mar_checkpoint),
        "--cell-type",
        cfg.cell_type,
        "--resolution",
        str(cfg.resolution),
        "--chromosomes",
        *cfg.chromosomes,
        "--hic-root",
        "data/cool",
        "--chrom-sizes",
        str(chrom_sizes_path),
        "--out-prefix",
        str(out_prefix),
        "--device",
        "auto",
        "--num-iter",
        "64",
        "--num-diffusion-steps",
        "50",
        "--temperature",
        "1.0",
    ]
    print("[eval] Running MAR inference to build predicted .cool...")
    subprocess.run(cmd, check=True)
    return cool_path


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


def run_loop_caller_template(
    template: str,
    label: str,
    cool_path: Path,
    resolution: int,
    chromosomes: Sequence[str],
    fdr_values: Sequence[float],
    out_dir: Path,
    tool_name: str,
) -> Path:
    """Run an external loop caller via a user-provided shell template."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_chr_files: List[Path] = []

    for chrom in chromosomes:
        for fdr in fdr_values:
            out_path = out_dir / f"{label}_{tool_name}_fdr{fdr}_{chrom}.bedpe"
            cmd = template.format(
                cool=str(cool_path),
                res=resolution,
                chrom=chrom,
                out=str(out_path),
                fdr=fdr,
            )
            print(f"[eval] Running {tool_name} on {label} {chrom} (FDR={fdr})...")
            try:
                subprocess.run(cmd, shell=True, check=True)
                per_chr_files.append(out_path)
            except FileNotFoundError:
                print(f"[eval] {tool_name} not found on PATH; skipping all {tool_name} metrics.")
                # Return a nonexistent path so downstream parsing just yields no loops.
                return out_dir / f"{label}_{tool_name}_loops.bedpe"
            except subprocess.CalledProcessError as exc:
                print(f"[eval] {tool_name} failed (skipping all {tool_name} metrics): {exc}")
                return out_dir / f"{label}_{tool_name}_loops.bedpe"

    merged = out_dir / f"{label}_{tool_name}_loops.bedpe"
    with merged.open("w") as fout:
        for path in per_chr_files:
            if not path.exists():
                continue
            with path.open() as fin:
                for line in fin:
                    if not line.strip() or line.startswith("#"):
                        continue
                    fout.write(line)
    return merged


def parse_loops_bedpe(path: Path, resolution: int) -> List[Tuple[str, int, int]]:
    """Parse a simple BEDPE-like loop file into (chrom, bin1, bin2)."""
    loops: List[Tuple[str, int, int]] = []
    if not path.exists() or path.is_dir():
        return loops

    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split()
            if len(parts) < 6:
                continue
            chrom1, start1, end1, chrom2, start2, end2 = parts[:6]
            if chrom1 != chrom2:
                continue
            try:
                s1 = int(start1)
                s2 = int(start2)
            except ValueError:
                continue
            bin1 = s1 // resolution
            bin2 = s2 // resolution
            if bin2 < bin1:
                bin1, bin2 = bin2, bin1
            loops.append((chrom1, bin1, bin2))
    return loops


def loop_overlap_metrics(
    ref_loops: List[Tuple[str, int, int]],
    pred_loops: List[Tuple[str, int, int]],
    tol_bins: int,
) -> Dict[str, float]:
    """Compute simple recall/precision-style metrics for loop overlap."""
    if not ref_loops or not pred_loops:
        return {
            "ref_count": float(len(ref_loops)),
            "pred_count": float(len(pred_loops)),
            "recall": 0.0,
            "precision": 0.0,
        }

    matched_ref = set()
    matched_pred = set()

    for i_ref, (chrom_r, br1, br2) in enumerate(ref_loops):
        for i_pred, (chrom_p, bp1, bp2) in enumerate(pred_loops):
            if chrom_r != chrom_p:
                continue
            if abs(br1 - bp1) <= tol_bins and abs(br2 - bp2) <= tol_bins:
                matched_ref.add(i_ref)
                matched_pred.add(i_pred)
                break

    recall = len(matched_ref) / max(len(ref_loops), 1)
    precision = len(matched_pred) / max(len(pred_loops), 1)
    return {
        "ref_count": float(len(ref_loops)),
        "pred_count": float(len(pred_loops)),
        "recall": float(recall),
        "precision": float(precision),
    }


def compute_apa(
    cool_path: Path,
    loops: List[Tuple[str, int, int]],
    resolution: int,
    window_bins: int,
) -> Tuple[np.ndarray, float]:
    """Compute a simple APA matrix and score for a set of loops."""
    clr = cooler.Cooler(str(cool_path))
    size = 2 * window_bins + 1
    apa_accum = np.zeros((size, size), dtype=float)
    n_used = 0

    for chrom, bin1, bin2 in loops:
        if chrom not in clr.chromsizes:
            continue
        chr_len_bp = int(clr.chromsizes[chrom])
        n_bins = math.ceil(chr_len_bp / resolution)
        if bin1 < window_bins or bin2 < window_bins:
            continue
        if bin1 + window_bins >= n_bins or bin2 + window_bins >= n_bins:
            continue

        start1_bp = (bin1 - window_bins) * resolution
        end1_bp = (bin1 + window_bins + 1) * resolution
        start2_bp = (bin2 - window_bins) * resolution
        end2_bp = (bin2 + window_bins + 1) * resolution
        region1 = (chrom, start1_bp, end1_bp)
        region2 = (chrom, start2_bp, end2_bp)

        mat = clr.matrix(balance=True).fetch(region1, region2)
        if mat.shape != apa_accum.shape:
            continue
        apa_accum += np.nan_to_num(mat)
        n_used += 1

    if n_used == 0:
        return apa_accum, float("nan")

    apa = apa_accum / n_used
    center = apa[window_bins, window_bins]

    c = max(1, window_bins // 2)
    corner_mask = np.zeros_like(apa, dtype=bool)
    corner_mask[:c, :c] = True
    corner_mask[:c, -c:] = True
    corner_mask[-c:, :c] = True
    corner_mask[-c:, -c:] = True
    corner_vals = apa[corner_mask]
    apa_score = float(center / (corner_vals.mean() + 1e-8))
    return apa, apa_score


def save_apa_heatmap(apa: np.ndarray, out_path: Path, title: str) -> None:
    """Save APA matrix as a PNG heatmap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(apa, origin="lower", cmap="coolwarm")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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

    # ------------------------------------------------------------------
    # Loop calling (Mustache/SIP) and APA
    # ------------------------------------------------------------------
    chrom_sizes_path = Path(args.chrom_sizes)
    loops_section: Dict[str, Dict[str, float]] = {}
    apa_section: Dict[str, Dict[str, float]] = {}

    # Prepare experimental coolers.
    hic_exp_path = (
        Path("data/cool")
        / cfg.cell_type
        / "HiC"
        / f"{res_label}.cool"
    )

    ddpm_cool_path: Path | None = None
    mar_cool_path: Path | None = None

    if args.mustache_template or args.sip_template:
        # Build predicted coolers as needed.
        if cfg.model in ("ddpm", "both"):
            ddpm_cool_path = ensure_predicted_cool_ddpm(cfg, chrom_sizes_path)
        if cfg.model in ("mar", "both"):
            mar_cool_path = ensure_predicted_cool_mar(cfg, chrom_sizes_path)

        loops_dir = cfg.results_dir / "loops"
        tol_bins = args.loop_tolerance_bins

        # Mustache
        if args.mustache_template:
            # Experimental Micro-C and Hi-C.
            microc_mustache = run_loop_caller_template(
                template=args.mustache_template,
                label="microc",
                cool_path=microc_exp_path,
                resolution=cfg.resolution,
                chromosomes=cfg.chromosomes,
                fdr_values=(0.05, 0.1),
                out_dir=loops_dir,
                tool_name="mustache",
            )
            hic_mustache = run_loop_caller_template(
                template=args.mustache_template,
                label="hic",
                cool_path=hic_exp_path,
                resolution=cfg.resolution,
                chromosomes=cfg.chromosomes,
                fdr_values=(0.05, 0.1),
                out_dir=loops_dir,
                tool_name="mustache",
            )

            loops_microc = parse_loops_bedpe(microc_mustache, cfg.resolution)
            loops_hic = parse_loops_bedpe(hic_mustache, cfg.resolution)

            if ddpm_cool_path is not None:
                ddpm_mustache = run_loop_caller_template(
                    template=args.mustache_template,
                    label="ddpm",
                    cool_path=ddpm_cool_path,
                    resolution=cfg.resolution,
                    chromosomes=cfg.chromosomes,
                    fdr_values=(0.05, 0.1),
                    out_dir=loops_dir,
                    tool_name="mustache",
                )
                loops_ddpm = parse_loops_bedpe(ddpm_mustache, cfg.resolution)
            else:
                loops_ddpm = []

            if mar_cool_path is not None:
                mar_mustache = run_loop_caller_template(
                    template=args.mustache_template,
                    label="mar",
                    cool_path=mar_cool_path,
                    resolution=cfg.resolution,
                    chromosomes=cfg.chromosomes,
                    fdr_values=(0.05, 0.1),
                    out_dir=loops_dir,
                    tool_name="mustache",
                )
                loops_mar = parse_loops_bedpe(mar_mustache, cfg.resolution)
            else:
                loops_mar = []

            # Compute overlap metrics vs Micro-C loops.
            loops_section["mustache_hic_vs_microc"] = loop_overlap_metrics(
                loops_microc,
                loops_hic,
                tol_bins,
            )
            loops_section["mustache_ddpm_vs_microc"] = loop_overlap_metrics(
                loops_microc,
                loops_ddpm,
                tol_bins,
            )
            loops_section["mustache_mar_vs_microc"] = loop_overlap_metrics(
                loops_microc,
                loops_mar,
                tol_bins,
            )

            # APA using Micro-C loops as reference.
            apa_dir = cfg.results_dir / "apa"
            apa_microc_mat, apa_microc_score = compute_apa(
                microc_exp_path, loops_microc, cfg.resolution, args.apa_window_bins
            )
            apa_hic_mat, apa_hic_score = compute_apa(
                hic_exp_path, loops_microc, cfg.resolution, args.apa_window_bins
            )
            apa_section["microc"] = {"score": apa_microc_score}
            apa_section["hic"] = {"score": apa_hic_score}
            save_apa_heatmap(
                apa_microc_mat,
                apa_dir / "apa_microc_from_microc_loops.png",
                "APA Micro-C (Micro-C loops)",
            )
            save_apa_heatmap(
                apa_hic_mat,
                apa_dir / "apa_hic_from_microc_loops.png",
                "APA Hi-C (Micro-C loops)",
            )

            if ddpm_cool_path is not None:
                apa_ddpm_mat, apa_ddpm_score = compute_apa(
                    ddpm_cool_path, loops_microc, cfg.resolution, args.apa_window_bins
                )
                apa_section["ddpm"] = {"score": apa_ddpm_score}
                save_apa_heatmap(
                    apa_ddpm_mat,
                    apa_dir / "apa_ddpm_from_microc_loops.png",
                    "APA DDPM (Micro-C loops)",
                )
            if mar_cool_path is not None:
                apa_mar_mat, apa_mar_score = compute_apa(
                    mar_cool_path, loops_microc, cfg.resolution, args.apa_window_bins
                )
                apa_section["mar"] = {"score": apa_mar_score}
                save_apa_heatmap(
                    apa_mar_mat,
                    apa_dir / "apa_mar_from_microc_loops.png",
                    "APA MAR (Micro-C loops)",
                )

        # SIP (optional, same structure; metrics stored separately).
        if args.sip_template:
            sip_loops_dir = cfg.results_dir / "loops"
            microc_sip = run_loop_caller_template(
                template=args.sip_template,
                label="microc",
                cool_path=microc_exp_path,
                resolution=cfg.resolution,
                chromosomes=cfg.chromosomes,
                fdr_values=(0.01, 0.05),
                out_dir=sip_loops_dir,
                tool_name="sip",
            )
            hic_sip = run_loop_caller_template(
                template=args.sip_template,
                label="hic",
                cool_path=hic_exp_path,
                resolution=cfg.resolution,
                chromosomes=cfg.chromosomes,
                fdr_values=(0.01, 0.05),
                out_dir=sip_loops_dir,
                tool_name="sip",
            )
            loops_microc_sip = parse_loops_bedpe(microc_sip, cfg.resolution)
            loops_hic_sip = parse_loops_bedpe(hic_sip, cfg.resolution)

            if ddpm_cool_path is not None:
                ddpm_sip = run_loop_caller_template(
                    template=args.sip_template,
                    label="ddpm",
                    cool_path=ddpm_cool_path,
                    resolution=cfg.resolution,
                    chromosomes=cfg.chromosomes,
                    fdr_values=(0.01, 0.05),
                    out_dir=sip_loops_dir,
                    tool_name="sip",
                )
                loops_ddpm_sip = parse_loops_bedpe(ddpm_sip, cfg.resolution)
            else:
                loops_ddpm_sip = []

            if mar_cool_path is not None:
                mar_sip = run_loop_caller_template(
                    template=args.sip_template,
                    label="mar",
                    cool_path=mar_cool_path,
                    resolution=cfg.resolution,
                    chromosomes=cfg.chromosomes,
                    fdr_values=(0.01, 0.05),
                    out_dir=sip_loops_dir,
                    tool_name="sip",
                )
                loops_mar_sip = parse_loops_bedpe(mar_sip, cfg.resolution)
            else:
                loops_mar_sip = []

            loops_section["sip_hic_vs_microc"] = loop_overlap_metrics(
                loops_microc_sip,
                loops_hic_sip,
                tol_bins,
            )
            loops_section["sip_ddpm_vs_microc"] = loop_overlap_metrics(
                loops_microc_sip,
                loops_ddpm_sip,
                tol_bins,
            )
            loops_section["sip_mar_vs_microc"] = loop_overlap_metrics(
                loops_microc_sip,
                loops_mar_sip,
                tol_bins,
            )

    if loops_section:
        summary["loops"] = loops_section
    if apa_section:
        summary["apa"] = apa_section

    out_json = cfg.results_dir / "evaluation_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Wrote summary to {out_json}")


if __name__ == "__main__":  # pragma: no cover
    main()
