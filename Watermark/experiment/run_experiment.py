#!/usr/bin/env python3
"""Main entry point. Runs baseline, optimized, multi_ring, or all, then compares."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import (get_small_scale_baseline, get_small_scale_optimized,
                     get_large_scale_baseline, get_large_scale_optimized,
                     get_small_scale_multiring, get_large_scale_multiring)
from run_baseline import run_baseline
from run_optimized import run_optimized
from run_multiring import run_multiring
from compare_results import compare

def main():
    p = argparse.ArgumentParser(description="Tree-Ring Watermark: Baseline vs Optimized vs Multi-Ring")
    p.add_argument("--scale", choices=["small","large"], default="small")
    p.add_argument("--mode", choices=["baseline","optimized","multi_ring","both","all","compare"],
                   default="both",
                   help="'both'=baseline+optimized, 'all'=baseline+optimized+multi_ring")
    p.add_argument("--skip_clip", action="store_true")
    a = p.parse_args()

    bl = get_small_scale_baseline()  if a.scale=="small" else get_large_scale_baseline()
    op = get_small_scale_optimized() if a.scale=="small" else get_large_scale_optimized()
    mr = get_small_scale_multiring() if a.scale=="small" else get_large_scale_multiring()

    if a.mode in ("baseline","both","all"):   run_baseline(bl,  skip_clip=a.skip_clip)
    if a.mode in ("optimized","both","all"):  run_optimized(op, skip_clip=a.skip_clip)
    if a.mode in ("multi_ring","all"):        run_multiring(mr, skip_clip=a.skip_clip)
    if a.mode in ("compare","both","all"):    compare(a.scale)

if __name__=="__main__": main()
