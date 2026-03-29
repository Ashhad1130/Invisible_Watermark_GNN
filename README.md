# Invisible Watermarking for AI-Generated Images

Replication and extension of [Tree-Ring Watermarks](https://arxiv.org/abs/2305.20030) (Wen et al., NeurIPS 2023). We reproduce the original results on Stable Diffusion v1.5, then push the robustness further with three progressive modifications — more DDIM inversion steps, a dual-band Fourier mask, and all-channel embedding.

> **Built on top of the original Tree-Ring Watermark implementation by Yuxin Wen et al.**
> Original repository: [https://github.com/YuxinWenRick/tree-ring-watermark](https://github.com/YuxinWenRick/tree-ring-watermark)
> We use their core DDIM inversion and watermark embedding/detection code as the foundation and extend it with our own modifications.

The full write-up is in `Watermark/Instructions/Latex/main.tex` (compiled PDF also included).

---

## What this project does

The core idea of Tree-Ring Watermarking is elegant: embed a secret pattern into the Fourier spectrum of the initial diffusion noise *before* generation starts. Because DDIM denoising is deterministic, the watermark gets baked into every pixel of the output image. At detection time, you invert the image back to noise and check whether the ring pattern is still there.

We replicate this baseline exactly, then test three extensions:

| Config | Description |
|---|---|
| **Baseline (B)** | Paper-accurate replication — 50 DDIM steps, single channel, circular ring mask |
| **Optimized (O)** | 100 DDIM steps — reduces inversion error and improves watermark recovery |
| **Multi-Ring (MR)** | Dual-band Fourier mask (inner disc + outer ring) — more robust to cropping |
| **MR-AllChan (AC)** | MR + watermark in all 4 latent channels — better SNR under rotation |

We evaluate all four across **13 attack conditions** (JPEG, crop, rotation, blur, noise, brightness) on 50 generated images per condition, reporting AUC, balanced accuracy, and TPR@1%FPR.

---

## Requirements

- Python 3.10–3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- GPU recommended (CUDA or Apple Silicon MPS). CPU works but is very slow.
- ~6 GB disk for the Stable Diffusion v1.5 model (downloaded automatically on first run)

---

## Setup

```bash
git clone https://github.com/Ashhad1130/Invisible_Watermark_GNN.git
cd Invisible_Watermark_GNN/Watermark
poetry install
```

That's it. No manual model download needed — the first run fetches `runwayml/stable-diffusion-v1-5` from HuggingFace and caches it locally.

---

## Running experiments

### Quick test (5 images, ~5 min on GPU)

```bash
cd Watermark
./run_all.sh small
```

### Full replication (50 images, all 13 attacks — matches the paper results)

```bash
./run_all.sh large
```

### Run individual configurations

```bash
# Baseline only
poetry run python experiment/run_experiment.py --scale large --mode baseline

# All four configs in one go
poetry run python experiment/run_experiment.py --scale large --mode all

# Just regenerate plots from existing checkpoint data
poetry run python experiment/run_experiment.py --scale large --mode compare
```

### With CLIP perceptual scores (slower, requires extra model download)

```bash
./run_all.sh large --with-clip
```

---

## Folder structure

```
Invisible_Watermark_GNN/
│
├── Watermark/
│   ├── experiment/                  # Our code
│   │   ├── configs.py               # All experiment configurations (scale, attacks, masks)
│   │   ├── run_experiment.py        # Main entry point — run baseline/optimized/all/compare
│   │   ├── run_baseline.py          # Baseline experiment logic
│   │   ├── run_multiring.py         # Multi-ring + all-channel logic
│   │   ├── run_optimized.py         # Optimized (100-step) logic
│   │   ├── compare_results.py       # Result comparison and plot generation
│   │   ├── category_prompts.py      # 8-category prompt set (no internet needed)
│   │   ├── landscape_prompts.py     # Landscape-only prompt set
│   │   ├── device_compat.py         # CUDA / MPS / CPU device detection
│   │   ├── mps_compat.py            # Apple Silicon compatibility patches
│   │   │
│   │   ├── outputs/                 # Generated images (per config, per attack)
│   │   │   └── large_baseline_category/
│   │   │       └── no_attack/
│   │   │           ├── img_0016_wm.png          # watermarked
│   │   │           └── img_0016_no_wm.png       # unwatermarked (reference)
│   │   │
│   │   ├── results/                 # CSVs and plots
│   │   │   ├── large_category_comparison.csv
│   │   │   └── large_category_plots/
│   │   │       ├── auc.png
│   │   │       └── tpr_at_1fpr.png
│   │   │
│   │   └── checkpoints/             # Per-attack score data (JSON) — used for ROC curves
│   │
│   ├── tree-ring-watermark/         # Original code from github.com/YuxinWenRick/tree-ring-watermark
│   │   ├── inverse_stable_diffusion.py   # DDIM inversion (their implementation)
│   │   ├── optim_utils.py                # Watermark embed/detect utilities (their implementation)
│   │   └── src/tree_ring_watermark/      # Core watermark library
│   │
│   ├── Instructions/
│   │   └── Latex/
│   │       ├── main.tex             # Full report source
│   │       ├── main.pdf             # Compiled report
│   │       ├── biblio.bib
│   │       └── figures/             # Diagrams and plots used in the report
│   │
│   ├── pyproject.toml               # Poetry dependency file
│   ├── run_all.sh                   # One-command pipeline script
│   └── setup_and_run.sh             # Full from-scratch setup script (useful for new machines)
```

---

## Reproducing the report

The LaTeX source compiles with a standard `pdflatex` + `bibtex` setup (MiKTeX or TeX Live):

```bash
cd Watermark/Instructions/Latex
pdflatex main.tex
bibtex main
pdflatex main.tex
```

All images referenced in the report are committed to the `report_latex` branch.

---

## Key results

MR-AllChan (dual-band mask + all-channel embedding) achieves:
- AUC ≥ 0.97 on **12/13** attacks (vs 9/13 for baseline)
- Resolves all 4 FAIL conditions of the baseline
- Statistically significant improvement over baseline (Wilcoxon signed-rank, p = 0.012)

The one remaining WARN condition is `rotation_75` — 75° rotation creates a large, correlated phase shift across all latent channels that channel averaging can't fully cancel. Fixing this would require rotation angle estimation before inversion.

---

## Authors

Ashhad Raza Quadri, Maria Alejandra Pabon Galindo, Saliq Neyaz
*Generative Neural Networks for the Sciences, Ruprecht-Karls-Universität Heidelberg, WS 2025/26*
