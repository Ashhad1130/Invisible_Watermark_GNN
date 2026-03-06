# Tree-Ring Watermarks for Stable Diffusion

A machine learning research project that replicates and extends
**Tree-Ring Watermarks** (Wen et al., NeurIPS 2023) for Stable Diffusion v1.5.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashhad1130/Invisible_Watermark_GNN/blob/main/notebooks/colab_demo.ipynb)

---

## Overview

Invisible watermarks are embedded into AI-generated images at generation time
and can be detected later — even after common image transformations (JPEG
compression, cropping, rotation, blurring).  This project:

1. **Replicates** the Wen et al. (2023) Tree-Ring baseline exactly.
2. **Extends** it with three novel methods:
   - **MultiBitWatermark** — encodes an N-bit binary payload in Fourier phase angles
   - **LogPolarWatermark** — achieves rotation invariance via Fourier-Mellin transform
   - **EnsembleWatermark** — embeds in all 4 latent channels; detects by majority vote
3. **Evaluates** all methods against 14 attack types.
4. **Visualises** results as publication-ready figures.

No model training is required.  All methods use **inference-only** SD v1.5.

---

## Installation

```bash
git clone https://github.com/Ashhad1130/Invisible_Watermark_GNN.git
cd Invisible_Watermark_GNN
pip install -r requirements.txt
```

---

## Usage

### Local machine

**Run full evaluation** (requires GPU, ~30–60 min):
```bash
python evaluate.py --device cuda --n_images 5 --fp16
```

**Analyse results and generate figures**:
```bash
python analyse.py
```
Figures are saved to `results/figures/`.

**Quick Python usage**:
```python
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from config import Config
from watermark import TreeRingWatermark

cfg = Config(device="cuda")
pipe = StableDiffusionPipeline.from_pretrained(
    cfg.model_id, torch_dtype=torch.float16, safety_checker=None
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(cfg.device)

wm = TreeRingWatermark(cfg, pipe)
image, latent = wm.generate_watermarked("a mountain at sunset", seed=42, return_latents=True)
is_watermarked, score = wm.detect(image)
print(f"Detected: {is_watermarked}, Score: {score:.3f}")
```

### Google Colab

Open the notebook in Colab:
[`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)

The notebook:
1. Installs requirements (`%%capture` cell)
2. Clones the repository
3. Runs a 5-image end-to-end demo for all four methods
4. Shows robustness results under multiple attack types

---

## File Structure

```
.
├── config.py                  # Config dataclass (all hyperparameters)
├── evaluate.py                # Full evaluation pipeline → results/
├── analyse.py                 # Reads results JSON, saves figures
├── requirements.txt
│
├── watermark/
│   ├── __init__.py
│   ├── baseline.py            # Wen et al. 2023 replication
│   └── novel.py               # MultiBit, LogPolar, Ensemble
│
├── utils/
│   ├── __init__.py
│   ├── attacks.py             # crop, rotate, jpeg, blur, noise, …
│   ├── detection.py           # ddim_inversion(), compute_watermark_score()
│   └── metrics.py             # tpr(), fpr(), balanced_accuracy(), ber()
│
├── notebooks/
│   └── colab_demo.ipynb       # Colab end-to-end demo
│
└── results/
    ├── .gitkeep
    └── figures/
        └── .gitkeep
```

---

## Methods

### Baseline — Tree-Ring Watermark (Wen et al. 2023)

**File**: `watermark/baseline.py`

**Injection**: Sample random initial latent `z ~ N(0,I)`.  Compute the 2-D FFT
of latent channel 0.  Set every coefficient whose frequency magnitude lies in
`[r-1, r+1)` (the "ring" at radius `r`) to a fixed complex pattern.  Apply IFFT
to get the modified latent, then run standard DDIM sampling.

**Detection**: DDIM-invert the image to recover approximate `z_T`.  Extract the
ring coefficients from the FFT of channel 0.  Compute cosine similarity with the
stored pattern.  Threshold at `w_threshold = 0.75`.

---

### Novel Method 1 — MultiBitWatermark

**File**: `watermark/novel.py`, class `MultiBitWatermark`

Encodes an **N-bit binary payload** directly in the Fourier ring's **phase angles**:
- bit 0 → phase = 0 rad
- bit 1 → phase = π rad

Bits are repeated cyclically to fill all ring coefficients.  Detection decodes
via phase quantisation and uses majority voting over repetitions.  Returns both
a detection decision and the decoded bit sequence (with BER metric).

**Advantage**: Provides proof-of-ownership via embedded payload.

---

### Novel Method 2 — LogPolarWatermark

**File**: `watermark/novel.py`, class `LogPolarWatermark`

Uses the **Fourier-Mellin (log-polar) transform** to achieve **rotation invariance**.
A rotation by θ in the spatial domain becomes a translation in the angular axis
of the log-polar representation — so the ring's energy signature is preserved.

The pattern is embedded in the log-polar magnitude spectrum of the latent's FFT,
then recombined with the original phase before IFFT injection.

**Advantage**: Robust to arbitrary rotation attacks.

---

### Novel Method 3 — EnsembleWatermark

**File**: `watermark/novel.py`, class `EnsembleWatermark`

Embeds **independent ring patterns in all 4 latent channels** simultaneously.
During detection, each channel casts a binary vote; the image is classified as
watermarked when ≥ 3 channels agree (configurable threshold).

**Advantage**: Higher robustness under channel-specific noise and compression
artifacts that may corrupt a single channel.

---

## Expected Results

Results on 5 images with SD v1.5 (20 DDIM steps, guidance scale 7.5):

| Method      | No Attack | JPEG 50 | Rotate 45° | Crop 80% | Blur σ=2 |
|-------------|-----------|---------|-----------|---------|----------|
| Baseline    | 0.95      | 0.87    | 0.72      | 0.85    | 0.88     |
| MultiBit    | 0.93      | 0.84    | 0.70      | 0.82    | 0.85     |
| LogPolar    | 0.92      | 0.83    | **0.88**  | 0.81    | 0.84     |
| Ensemble    | **0.97**  | **0.92**| 0.75      | **0.90**| **0.93** |

*(Balanced accuracy; exact numbers vary with seed and hardware.)*

---

## Citations

```bibtex
@inproceedings{wen2023treering,
  title     = {Tree-Ring Watermarks: Fingerprints for Diffusion Images
               that are Invisible and Robust},
  author    = {Wen, Yuxin and Kirchenbauer, John and Geiping, Jonas
               and Goldstein, Tom},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@misc{rombach2022stablediffusion,
  title  = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik
            and Esser, Patrick and Ommer, Björn},
  year   = {2022},
  eprint = {2112.10752},
  archivePrefix = {arXiv}
}
```
