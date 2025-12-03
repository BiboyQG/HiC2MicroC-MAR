# hic2microc-mar

Unified Hi-C → Micro-C project with a DDPM baseline and a MAR + Diffusion Loss model, built on top of the original HiC2MicroC and MAR codebases.

All new code lives under `src/hic2microc_mar/` and is managed with `uv`. The original `HiC2MicroC/` and `mar/` directories are kept as references and are not modified.

---

## Environment setup (uv)

From the repository root:

```bash
uv sync
```

This creates `.venv/` with the required dependencies, including:

- `torch` (CPU/MPS on macOS, CUDA on Linux if available)
- `numpy`, `scipy`, `pandas`
- `cooler`, `cooltools`
- `tqdm`, `pyyaml`, `matplotlib`, `einops`, `timm`
- `torch-fidelity` (for FID-like image metrics)

All commands below assume you run them from the repository root, with the `uv` environment active.

---

## Data preparation

The project expects `.cool` files under `data/cool/{cell_type}/{assay}/{resolution}.cool`. You can extract single-resolution coolers from the provided `.mcool` files (e.g. in `HiC2MicroC/data/`) using:

```bash
uv run prepare-cool \
  --mcool-path HiC2MicroC/data \
  --out-root data/cool \
  --cell-type HFFc6 \
  --resolution 5000
```

This creates:

- `data/cool/HFFc6/HiC/5kb.cool`
- `data/cool/HFFc6/MicroC/5kb.cool`

You can repeat for other cell types (`H1ESC`) and resolutions (e.g. `1000` for 1 kb; output label becomes `1kb.cool`).

---

## DDPM baseline: training and inference

### Training

The DDPM baseline uses a HiC2MicroC-style U-Net and WaveGrad-style diffusion utilities. Configuration lives in `configs/ddpm_5kb.yaml` (with a smaller `configs/ddpm_5kb_debug.yaml` for quick tests).

Train on HFFc6 5 kb:

```bash
uv run train-ddpm --config configs/ddpm_5kb.yaml
```

Key features:

- Data: `HiC2MicroCDataset` over paired Hi-C / Micro-C 256×256 patches from `.cool`.
- Splits (for HFFc6 5 kb):
  - Train on all chromosomes except `chr1` (reserved for test) and `chr17` (validation).
- Diffusion:
  - 1000 training steps, linear β schedule.
  - 50 inference steps (short chain), as in the HiC2MicroC paper.
- Loss: L1 (or L2, configurable) on predicted noise.

Checkpoints and training curves are stored under `checkpoints/ddpm_*`.

### Inference

Given a trained checkpoint:

```bash
uv run infer-ddpm \
  --config configs/ddpm_5kb.yaml \
  --checkpoint checkpoints/ddpm_hff_5kb/ddpm_best.pt \
  --cell-type HFFc6 \
  --resolution 5000 \
  --chromosomes chr1 \
  --hic-root data/cool \
  --chrom-sizes HiC2MicroC/data/hg38.sizes \
  --out-prefix results/ddpm/HFFc6_5kb_chr1 \
  --device auto
```

This will:

- Read Hi-C from `data/cool/HFFc6/HiC/5kb.cool`,
- Extract 256×256 normalized patches using the same sliding-window scheme as training,
- Run DDPM sampling to generate Micro-C patches,
- Merge overlapping patches back into a sparse genome-wide contact matrix,
- Write `results/ddpm/HFFc6_5kb_chr1.cool` with a `weight` column suitable for downstream tools.

---

## MAR + Diffusion Loss: training and inference

### Training

The MAR model operates on continuous patch tokens from Hi-C and Micro-C and uses a per-token Diffusion Loss head (cosine schedule) to model masked Micro-C tokens.

Train MAR on HFFc6 5 kb:

```bash
uv run train-mar --config configs/mar_5kb.yaml
```

Key components:

- `HiCTokenizer`:
  - Splits 256×256 patches into 16×16 non-overlapping 16×16 patches.
  - Projects each patch to a `token_dim`-dimensional embedding with learnable 2D positional and token-type embeddings.
- `HiC2MicroCMAR`:
  - Encoder-decoder Transformer built with `timm.models.vision_transformer.Block`.
  - Encoder sees CLS tokens + Hi-C tokens + unmasked Micro-C tokens.
  - Decoder reconstructs conditioning vectors `z_i` for Micro-C tokens, with learnable diffusion positional embeddings.
- `DiffusionLoss`:
  - Cosine noise schedule (`create_diffusion(...)` adapted from MAR).
  - AdaLN-conditioned MLP head (`SimpleMLPAdaLN`) operating on per-token vectors.
  - Training loss applied only on masked Micro-C tokens.

Checkpoints (MAR, tokenizer, DiffusionLoss head) and training history are stored under `checkpoints/mar_*`.

### Inference

Given a trained MAR checkpoint:

```bash
uv run infer-mar \
  --config configs/mar_5kb.yaml \
  --checkpoint checkpoints/mar_hff_5kb/mar_best.pt \
  --cell-type HFFc6 \
  --resolution 5000 \
  --chromosomes chr1 \
  --hic-root data/cool \
  --chrom-sizes HiC2MicroC/data/hg38.sizes \
  --out-prefix results/mar/HFFc6_5kb_chr1 \
  --device auto \
  --num-iter 64 \
  --num-diffusion-steps 50 \
  --temperature 1.0
```

For each chromosome, this will:

- Extract and normalize Hi-C patches.
- Initialize Micro-C tokens as fully masked.
- Run a MAR mask-decoding schedule over `num_iter` steps (MaskGIT-style cosine schedule) to gradually reveal tokens:
  - At each step, MAR produces `z_i` for tokens decoded in that step.
  - The DiffusionLoss sampler generates continuous token vectors conditioned on `z_i`.
- Decode Micro-C tokens back to 256×256 patches via `HiCTokenizer.decode_microc`.
- Merge patches into a `.cool` file, analogous to the DDPM pipeline.

---

## Evaluation: FID-like patch metrics and speed

The unified evaluation script provides FID-like metrics on patches and simple speed benchmarks. It can reuse the inference logic to generate predictions on the fly.

Example (DDPM only):

```bash
uv run evaluate-hic2microc \
  --model ddpm \
  --cell-type HFFc6 \
  --resolution 5000 \
  --chromosomes chr1 \
  --results-dir results/hff_5kb_eval \
  --device auto \
  --ddpm-config configs/ddpm_5kb.yaml \
  --ddpm-checkpoint checkpoints/ddpm_hff_5kb/ddpm_best.pt
```

Example (MAR only or both):

```bash
uv run evaluate-hic2microc \
  --model both \
  --cell-type HFFc6 \
  --resolution 5000 \
  --chromosomes chr1 \
  --results-dir results/hff_5kb_eval \
  --device auto \
  --ddpm-config configs/ddpm_5kb.yaml \
  --ddpm-checkpoint checkpoints/ddpm_hff_5kb/ddpm_best.pt \
  --mar-config configs/mar_5kb.yaml \
  --mar-checkpoint checkpoints/mar_hff_5kb/mar_best.pt
```

The script currently computes:

- FID-like scores between predicted and ground-truth Micro-C patches using `torch-fidelity`.
- Basic speed metrics:
  - Average per-window inference time.
  - Total runtime and number of windows processed.

The scaffold for loop calling (Mustache, SIP) and APA analysis is in place and can be extended to run those tools on experimental and predicted `.cool` files. This keeps the core evaluation logic in a single place while allowing you to plug in your preferred loop callers and APA workflows.

---

## Reproducibility and devices

- All training scripts accept a `seed` in their YAML configs; this seed is applied to Python, NumPy, and PyTorch (including CUDA, if available).
- Devices are configurable via `device` in configs or `--device` CLI flags:
  - `cpu`, `cuda`, `mps`, or `auto` (prefers CUDA, then MPS, then CPU).
- DiffusionLoss uses a cosine noise schedule imported from the MAR diffusion utilities, with small runtime patches to support CPU and MPS devices.

---

## Directory overview

- `pyproject.toml` – uv project definition and dependencies.
- `src/hic2microc_mar/`
  - `data/`
    - `prepare_cool.py` – `.mcool` to `.cool` extraction.
    - `dataset.py` – cooler-backed `HiC2MicroCDataset`.
    - `utils.py` – normalization, patch indexing, and COO merging utilities.
  - `models/`
    - `ddpm_unet.py` – HiC2MicroC-style U-Net.
    - `hictokenizer.py` – continuous patch tokenizer.
    - `mar_transformer.py` – MAR encoder-decoder transformer.
  - `diffusion/`
    - `ddpm_wavegrad.py` – WaveGrad-style DDPM utilities.
    - `diffusion_loss.py` – per-token Diffusion Loss (cosine schedule).
  - `train_ddpm.py`, `infer_ddpm.py` – DDPM training and inference CLIs.
  - `train_mar.py`, `infer_mar.py` – MAR training and inference CLIs.
  - `evaluate_hic2microc.py` – unified evaluation CLI.
- `configs/`
  - `ddpm_5kb.yaml`, `ddpm_5kb_debug.yaml`
  - `mar_5kb.yaml`, `mar_5kb_debug.yaml`

---

## Notes

- The original `HiC2MicroC/` and `mar/` directories are treated as references; the new project does not modify them.
- Loop calling (Mustache, SIP) and APA analyses are not fully wired up yet, but the evaluation scaffold and `.cool` outputs from DDPM and MAR are structured for easy integration with those tools.
