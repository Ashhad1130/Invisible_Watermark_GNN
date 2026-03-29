#!/usr/bin/env python3
"""Main entry point.

Modes:
  baseline       — paper-exact Tree-Ring (50 DDIM steps, ch=3)
  optimized      — 100 DDIM steps, ch=3
  multi_ring     — dual-band Fourier mask, 100 steps, ch=3
  allchan        — dual-band Fourier mask, 100 steps, ch=-1 (all channels, fixes rotation)
  both           — baseline + optimized
  all            — baseline + optimized + multi_ring + allchan  (full comparison)
  compare        — load existing results and print/plot comparison

Datasets:
  category  (default) — 8-category random prompts, no download needed
  landscape           — landscape-only prompts, no download needed
  gustavosta          — Gustavosta/Stable-Diffusion-Prompts (requires internet + HF)

Scale:
  small — 5 images   large — 50 images
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import (
    get_small_scale_baseline,         get_large_scale_baseline,
    get_small_scale_optimized,        get_large_scale_optimized,
    get_small_scale_multiring,        get_large_scale_multiring,
    get_small_scale_baseline_landscape,
    get_small_scale_optimized_landscape,
    get_small_scale_multiring_landscape,
    get_small_scale_baseline_category,    get_large_scale_baseline_category,
    get_small_scale_optimized_category,   get_large_scale_optimized_category,
    get_small_scale_multiring_category,   get_large_scale_multiring_category,
    get_small_scale_multiring_allchan_category,
    get_large_scale_multiring_allchan_category,
)
from run_baseline  import run_baseline
from run_optimized import run_optimized
from run_multiring import run_multiring
from compare_results import compare


def _get_configs(scale, dataset):
    """Return (baseline, optimized, multi_ring, allchan) configs."""
    sm = scale == "small"

    if dataset == "category":
        bl = get_small_scale_baseline_category()    if sm else get_large_scale_baseline_category()
        op = get_small_scale_optimized_category()   if sm else get_large_scale_optimized_category()
        mr = get_small_scale_multiring_category()   if sm else get_large_scale_multiring_category()
        ac = get_small_scale_multiring_allchan_category() if sm else get_large_scale_multiring_allchan_category()

    elif dataset == "landscape":
        bl = get_small_scale_baseline_landscape()
        op = get_small_scale_optimized_landscape()
        mr = get_small_scale_multiring_landscape()
        # allchan landscape variant — derive on the fly
        ac = get_small_scale_multiring_landscape()
        ac.name = ac.name.replace("landscape", "allchan_landscape")
        ac.watermark.w_channel = -1

    else:  # gustavosta
        bl = get_small_scale_baseline()    if sm else get_large_scale_baseline()
        op = get_small_scale_optimized()   if sm else get_large_scale_optimized()
        mr = get_small_scale_multiring()   if sm else get_large_scale_multiring()
        ac = get_small_scale_multiring()   if sm else get_large_scale_multiring()
        ac.name = ac.name + "_allchan"
        ac.watermark.w_channel = -1

    return bl, op, mr, ac


def main():
    p = argparse.ArgumentParser(
        description="Tree-Ring Watermark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale",   choices=["small", "large"], default="small",
                   help="small=5 images, large=50 images")
    p.add_argument("--mode",
                   choices=["baseline", "optimized", "multi_ring", "allchan",
                            "both", "all", "compare"],
                   default="all",
                   help="which experiments to run (default: all)")
    p.add_argument("--dataset", choices=["gustavosta", "landscape", "category"],
                   default="category",
                   help="prompt source (default: category)")
    p.add_argument("--skip_clip", action="store_true",
                   help="skip CLIP reference model (faster)")
    a = p.parse_args()

    bl, op, mr, ac = _get_configs(a.scale, a.dataset)
    ds_suffix = "" if a.dataset == "gustavosta" else a.dataset

    if a.mode in ("baseline",   "both", "all"):  run_baseline(bl,  skip_clip=a.skip_clip)
    if a.mode in ("optimized",  "both", "all"):  run_optimized(op, skip_clip=a.skip_clip)
    if a.mode in ("multi_ring", "all"):           run_multiring(mr, skip_clip=a.skip_clip)
    if a.mode in ("allchan",    "all"):           run_multiring(ac, skip_clip=a.skip_clip)
    if a.mode in ("compare", "both", "all"):      compare(a.scale, ds_suffix)


if __name__ == "__main__":
    main()
