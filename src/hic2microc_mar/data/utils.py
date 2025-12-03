from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cooler
import numpy as np
import scipy.sparse as sp

DEFAULT_MAXV_5KB: float = 0.05
DEFAULT_MAXV_1KB: float = 0.08


@dataclass(frozen=True)
class PatchIndex:
    """Index of a 2D window in binned coordinates.

    The indices are half-open, i.e. ``row_start`` is inclusive and
    ``row_end`` is exclusive (same for columns).
    """

    chrom: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.row_end - self.row_start, self.col_end - self.col_start


def normalize_contacts(matrix: np.ndarray, max_value: float) -> np.ndarray:
    """Normalize a contact map into the [-1, 1] range.

    This follows the HiC2MicroC preprocessing:

    1. Clip contacts above ``max_value``.
    2. Linearly map [0, ``max_value``] to [1, 10].
    3. Apply log10, obtaining values in [0, 1].
    4. Linearly map [0, 1] to [-1, 1].
    """
    mat = np.nan_to_num(matrix, copy=True)
    mat[mat > max_value] = max_value

    # [0, max_value] -> [1, 10]
    mat = np.log10((9.0 / max_value) * mat + 1.0)

    # [0, 1] -> [-1, 1]
    mat = 2.0 * mat - 1.0
    return mat


def denormalize_contacts(values: np.ndarray, max_value: float) -> np.ndarray:
    """Invert :func:`normalize_contacts` back to contact counts."""
    vals = (values + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    factor = max_value / 9.0
    vals = (10.0**vals - 1.0) * factor  # [0, 1] -> [0, max_value]
    return vals


def _diag_indices(n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices for the main and upper diagonals up to distance ``k``."""
    rows, cols = np.diag_indices(n)
    rows, cols = list(rows), list(cols)
    rows2, cols2 = rows.copy(), cols.copy()
    for i in range(1, k + 1):
        rows2 += rows[:-i]
        cols2 += cols[i:]
    return np.array(rows2), np.array(cols2)


def _sparse_divide_nonzero(a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
    inv_b = b.copy()
    inv_b.data = 1.0 / inv_b.data
    return a.multiply(inv_b)


def default_patch_params(resolution: int) -> Tuple[int, int]:
    """Return (step, extend_steps) defaults for a given resolution."""
    if resolution == 5000:
        return 50, 5
    if resolution == 1000:
        return 100, 20
    # Conservative generic fallback: small step and moderate extension.
    return max(1, 2000000 // max(resolution, 1) // 4), 8


def compute_patch_indices(
    chrom: str,
    chrom_length_bp: int,
    resolution: int,
    image_size: int = 256,
    step: int | None = None,
    extend_steps: int | None = None,
) -> List[PatchIndex]:
    """Compute 2D window indices along a chromosome.

    This mirrors the ``get_submat_idx`` logic from the original
    HiC2MicroC implementation, but returns structured :class:`PatchIndex`
    objects that are reusable for both training and inference.
    """
    if step is None or extend_steps is None:
        step_default, extend_default = default_patch_params(resolution)
        step = step or step_default
        extend_steps = extend_steps or extend_default

    n_bins = math.ceil(chrom_length_bp / resolution)
    all_inds = np.arange(0, max(n_bins - image_size, 1), step, dtype=int)
    if all_inds.size == 0:
        return []

    last_ind = int(all_inds[-1])
    if last_ind + image_size < n_bins:
        all_inds = np.append(all_inds, n_bins - image_size)

    indices: List[PatchIndex] = []
    for j in all_inds:
        row_start = int(j)
        row_end = row_start + image_size  # exclusive

        for k in range(extend_steps):
            col_start = row_start + k * step
            col_end = col_start + image_size
            if col_end > n_bins:
                continue

            indices.append(
                PatchIndex(
                    chrom=chrom,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                )
            )

    return indices


def merge_patch_predictions(
    predictions: np.ndarray,
    max_value: float,
    chrom_length_bp: int,
    patch_indices: Sequence[PatchIndex],
    resolution: int,
    max_bin: int = 456,
    image_size: int = 256,
) -> sp.coo_matrix:
    """Merge overlapping patch predictions back into a sparse contact map.

    Parameters
    ----------
    predictions:
        Array of shape ``[N, H, W]`` containing normalized predictions
        in the ``[-1, 1]`` range for each patch.
    max_value:
        Maximum contact value used for normalization (``maxV``).
    chrom_length_bp:
        Chromosome length in base pairs.
    patch_indices:
        Sequence of :class:`PatchIndex` instances describing each patch.
    resolution:
        Bin size in base pairs.
    max_bin:
        Maximum genomic distance in bins to keep in the sparse matrix.
    image_size:
        Expected patch size (height/width).
    """
    if predictions.shape[0] != len(patch_indices):
        raise ValueError(
            f"Number of predictions ({predictions.shape[0]}) does not match "
            f"number of patch indices ({len(patch_indices)})."
        )

    bins = math.ceil(chrom_length_bp / resolution)
    rows, cols = _diag_indices(bins, max_bin - 1)

    mp = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(bins, bins))
    mp = mp + mp.T - sp.diags(mp.diagonal())

    mn = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(bins, bins))
    mn = mn + mn.T - sp.diags(mn.diagonal())

    for pred, idx in zip(predictions, patch_indices):
        if idx.shape != (image_size, image_size):
            raise ValueError(
                f"PatchIndex {idx} has shape {idx.shape}, "
                f"expected ({image_size}, {image_size})."
            )
        mp[idx.row_start : idx.row_end, idx.col_start : idx.col_end] += pred
        mn[idx.row_start : idx.row_end, idx.col_start : idx.col_end] += np.ones(
            (image_size, image_size)
        )

    mp.data -= 1.0
    mn.data -= 1.0
    mp.eliminate_zeros()
    mn.eliminate_zeros()

    mpn = _sparse_divide_nonzero(mp, mn)
    mpn_2 = sp.coo_matrix(mpn)

    pred_converted = denormalize_contacts(mpn_2.data, max_value)
    mpn_2.data = pred_converted
    return mpn_2


def get_chr_start_indices(
    chrom_ids: Sequence[str],
    chrom_sizes_bp: Sequence[int],
    resolution: int,
) -> dict[str, int]:
    """Return linearized start indices for each chromosome."""
    chr_start_index: dict[str, int] = {}
    cumulative = 0
    for chrom, length_bp in zip(chrom_ids, chrom_sizes_bp):
        num_bins = math.ceil(length_bp / resolution)
        chr_start_index[chrom] = cumulative
        cumulative += num_bins
    return chr_start_index


def add_weight_column(cool_path: str | Path) -> None:
    """Ensure a ``weight`` column (all ones) exists in the bins table."""
    clr = cooler.Cooler(str(cool_path))
    n_bins = clr.bins().shape[0]

    if "weight" not in clr.bins().columns:
        h5opts = dict(compression="gzip", compression_opts=6)
        with clr.open("r+") as f:
            f["bins"].create_dataset("weight", data=np.ones(n_bins), **h5opts)

