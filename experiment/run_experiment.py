#!/usr/bin/env python3
"""Main entry point. Runs baseline, optimized, or both, then compares."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import (get_small_scale_baseline, get_small_scale_optimized,
                     get_large_scale_baseline, get_large_scale_optimized)
from run_baseline import run_baseline
from run_optimized import run_optimized
from compare_results import compare

def main():
    p = argparse.ArgumentParser(description="Tree-Ring Watermark: Baseline vs Optimized")
    p.add_argument("--scale", choices=["small","large"], default="small")
    p.add_argument("--mode", choices=["baseline","optimized","both","compare"], default="both")
    p.add_argument("--skip_clip", action="store_true")
    p.add_argument("--scale_factor", type=float, default=1.1)
    a = p.parse_args()

    bl = get_small_scale_baseline() if a.scale=="small" else get_large_scale_baseline()
    op = get_small_scale_optimized() if a.scale=="small" else get_large_scale_optimized()

    if a.mode in ("baseline","both"): run_baseline(bl, skip_clip=a.skip_clip)
    if a.mode in ("optimized","both"): run_optimized(op, skip_clip=a.skip_clip, scale_factor=a.scale_factor)
    if a.mode in ("compare","both"): compare(a.scale)

if __name__=="__main__": main()
