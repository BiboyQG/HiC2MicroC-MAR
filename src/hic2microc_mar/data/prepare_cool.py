from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cooler
from cooler.fileops import cp, list_coolers


def _discover_mcool(
    root: Path,
    cell_type: str,
    assay: str,
) -> Path:
    """Return the expected .mcool path for a given cell type and assay.

    The helper first looks for the canonical ``{cell_type}_{assay}.mcool``
    file name under ``root``. If that is not found, it falls back to a
    simple glob search.
    """
    expected = root / f"{cell_type}_{assay}.mcool"
    if expected.exists():
        return expected

    candidates: List[Path] = list(root.glob(f"*{cell_type}*{assay}*.mcool"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find .mcool for cell_type={cell_type!r}, assay={assay!r} "
            f"under {root}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple .mcool files found for cell_type={cell_type!r}, assay={assay!r}: "
            f"{[str(c) for c in candidates]}"
        )
    return candidates[0]


def _available_resolutions(mcool_path: Path) -> List[int]:
    """List integer resolutions available in a multi-resolution cooler."""
    uris = list(list_coolers(str(mcool_path)))
    resolutions: List[int] = []
    for uri in uris:
        # URIs look like ``/resolutions/5000``.
        parts = uri.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "resolutions":
            try:
                resolutions.append(int(parts[1]))
            except ValueError:
                continue
    return sorted(set(resolutions))


def _resolution_label(resolution_bp: int) -> str:
    """Human-readable resolution label for output file names."""
    if resolution_bp == 5000:
        return "5kb"
    if resolution_bp == 1000:
        return "1kb"
    return f"{resolution_bp}bp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a single resolution from .mcool into .cool files "
        "in a standardized layout."
    )
    parser.add_argument(
        "--mcool-path",
        type=str,
        required=True,
        help="Directory containing {cell_type}_{assay}.mcool files "
        "(e.g. HiC2MicroC/data).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Root directory for output .cool files (e.g. data/cool).",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        required=True,
        help="Cell type identifier, e.g. HFFc6 or H1ESC.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Target resolution in base pairs (e.g. 5000 for 5 kb, 1000 for 1 kb).",
    )
    parser.add_argument(
        "--assays",
        type=str,
        nargs="+",
        default=("HiC", "MicroC"),
        help="Assays to extract (default: HiC MicroC).",
    )
    return parser.parse_args()


def extract_coolers(
    mcool_root: Path,
    out_root: Path,
    cell_type: str,
    resolution: int,
    assays: Iterable[str],
) -> None:
    """Extract the given resolution for each assay into ``out_root``.

    The resulting directory structure follows::

        {out_root}/{cell_type}/{assay}/{resolution_label}.cool

    where ``resolution_label`` is ``5kb`` or ``1kb`` for the common
    resolutions, and ``{resolution}bp`` otherwise.
    """
    mcool_root = mcool_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()

    for assay in assays:
        mcool_path = _discover_mcool(mcool_root, cell_type, assay)
        resolutions = _available_resolutions(mcool_path)
        if resolution not in resolutions:
            raise ValueError(
                f"Resolution {resolution} bp not found in {mcool_path}; "
                f"available resolutions: {resolutions}"
            )

        res_label = _resolution_label(resolution)
        assay_dir = out_root / cell_type / assay
        assay_dir.mkdir(parents=True, exist_ok=True)
        out_path = assay_dir / f"{res_label}.cool"

        src_uri = f"{mcool_path}::/resolutions/{resolution}"
        print(f"[prepare_cool] Copying {src_uri} -> {out_path}")
        cp(src_uri, str(out_path))

        # Basic sanity check.
        clr = cooler.Cooler(str(out_path))
        if clr.binsize != resolution:
            raise RuntimeError(
                f"Copied cooler at {out_path} has binsize={clr.binsize}, "
                f"expected {resolution}"
            )


def main() -> None:
    """Entry point for prepare_cool CLI."""
    args = parse_args()
    extract_coolers(
        mcool_root=Path(args.mcool_path),
        out_root=Path(args.out_root),
        cell_type=args.cell_type,
        resolution=args.resolution,
        assays=args.assays,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
