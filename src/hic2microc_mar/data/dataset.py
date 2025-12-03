from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cooler
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    DEFAULT_MAXV_1KB,
    DEFAULT_MAXV_5KB,
    PatchIndex,
    compute_patch_indices,
    normalize_contacts,
)


@dataclass(frozen=True)
class HiC2MicroCPatch:
    """Metadata describing a single Hi-C / Micro-C patch."""

    chrom: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int


class HiC2MicroCDataset(Dataset):
    """Paired Hi-C → Micro-C patch dataset backed by cooler files.

    Parameters
    ----------
    hic_path:
        Path to the Hi-C ``.cool`` file.
    microc_path:
        Path to the Micro-C ``.cool`` file.
    resolution:
        Bin size in base pairs (e.g. 5000 or 1000).
    chromosomes:
        Chromosome IDs to include, e.g. ``[\"chr1\", \"chr2\"]``.
    window_size:
        Patch size (height/width) in bins. Defaults to 256.
    max_hic_value:
        Maximum Hi-C contact value used for normalization. If ``None``,
        defaults to 0.05 for 5 kb and 0.08 for 1 kb.
    max_microc_value:
        Maximum Micro-C contact value used for normalization. If ``None``,
        the same default as Hi-C is used for the given resolution.
    step:
        Optional override for diagonal step size in bins. If ``None``,
        resolution-dependent defaults are used.
    extend_steps:
        Optional override for the number of steps to the right. If ``None``,
        resolution-dependent defaults are used.
    """

    def __init__(
        self,
        hic_path: str | Path,
        microc_path: str | Path,
        resolution: int,
        chromosomes: Sequence[str],
        window_size: int = 256,
        max_hic_value: float | None = None,
        max_microc_value: float | None = None,
        step: int | None = None,
        extend_steps: int | None = None,
    ) -> None:
        super().__init__()

        self.hic_path = Path(hic_path)
        self.microc_path = Path(microc_path)
        self.resolution = int(resolution)
        self.window_size = int(window_size)

        if max_hic_value is None or max_microc_value is None:
            if self.resolution == 5000:
                default_max = DEFAULT_MAXV_5KB
            elif self.resolution == 1000:
                default_max = DEFAULT_MAXV_1KB
            else:
                default_max = DEFAULT_MAXV_5KB
        self.max_hic_value = max_hic_value if max_hic_value is not None else default_max
        self.max_microc_value = (
            max_microc_value if max_microc_value is not None else default_max
        )

        self._hic = cooler.Cooler(str(self.hic_path))
        self._microc = cooler.Cooler(str(self.microc_path))

        if self._hic.binsize != self.resolution:
            raise ValueError(
                f"Hi-C cooler {self.hic_path} has binsize={self._hic.binsize}, "
                f"expected {self.resolution}."
            )
        if self._microc.binsize != self.resolution:
            raise ValueError(
                f"Micro-C cooler {self.microc_path} has binsize={self._microc.binsize}, "
                f"expected {self.resolution}."
            )

        self.chromosomes: List[str] = list(chromosomes)

        # Precompute patch indices for all chromosomes.
        self._patches: List[HiC2MicroCPatch] = []
        for chrom in self.chromosomes:
            if chrom not in self._hic.chromsizes:
                raise KeyError(f"Chromosome {chrom!r} not found in Hi-C cooler.")
            if chrom not in self._microc.chromsizes:
                raise KeyError(f"Chromosome {chrom!r} not found in Micro-C cooler.")

            chrom_len = int(self._hic.chromsizes[chrom])
            patch_indices: List[PatchIndex] = compute_patch_indices(
                chrom=chrom,
                chrom_length_bp=chrom_len,
                resolution=self.resolution,
                image_size=self.window_size,
                step=step,
                extend_steps=extend_steps,
            )
            for idx in patch_indices:
                self._patches.append(
                    HiC2MicroCPatch(
                        chrom=idx.chrom,
                        row_start=idx.row_start,
                        row_end=idx.row_end,
                        col_start=idx.col_start,
                        col_end=idx.col_end,
                    )
                )

    @property
    def max_values(self) -> Tuple[float, float]:
        """Return (max_hic_value, max_microc_value)."""
        return self.max_hic_value, self.max_microc_value

    @property
    def patches(self) -> Sequence[HiC2MicroCPatch]:
        """Return metadata for all patches in the dataset."""
        return self._patches

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._patches)

    def _fetch_patch(
        self,
        clr: cooler.Cooler,
        patch: HiC2MicroCPatch,
    ) -> np.ndarray:
        chr_len = int(clr.chromsizes[patch.chrom])

        # Mirror the original HiC2MicroC indexing logic:
        # start = idx_start * res, end = idx_end * res + res, then clamp to chr_len.
        row_start_bp = patch.row_start * self.resolution
        row_end_bp = (patch.row_end - 1) * self.resolution + self.resolution
        col_start_bp = patch.col_start * self.resolution
        col_end_bp = (patch.col_end - 1) * self.resolution + self.resolution

        row_end_bp = min(row_end_bp, chr_len)
        col_end_bp = min(col_end_bp, chr_len)

        row_region = (patch.chrom, row_start_bp, row_end_bp)
        col_region = (patch.chrom, col_start_bp, col_end_bp)
        mat = clr.matrix(balance=True).fetch(row_region, col_region)
        return np.nan_to_num(mat, copy=True)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        patch = self._patches[index]

        hic_mat = self._fetch_patch(self._hic, patch)
        microc_mat = self._fetch_patch(self._microc, patch)

        hic_norm = normalize_contacts(hic_mat, self.max_hic_value)
        microc_norm = normalize_contacts(microc_mat, self.max_microc_value)

        hic_tensor = torch.from_numpy(hic_norm).unsqueeze(0).float()
        microc_tensor = torch.from_numpy(microc_norm).unsqueeze(0).float()
        return hic_tensor, microc_tensor
