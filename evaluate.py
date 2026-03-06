"""
Full evaluation pipeline for Tree-Ring Watermark methods.

Loads the Stable Diffusion v1.5 pipeline, runs all four watermarking
methods (Baseline, MultiBit, LogPolar, Ensemble) against all attack
types, and saves a JSON summary to ``results/evaluation_results.json``.

Usage
-----
    python evaluate.py [--device cuda] [--n_images 5] [--output_dir results]

The script is designed to run on a single GPU (Google Colab T4 or
equivalent) with 8–16 GB VRAM.  Enable ``--fp16`` for faster inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from config import Config
from watermark.baseline import TreeRingWatermark
from watermark.novel import MultiBitWatermark, LogPolarWatermark, EnsembleWatermark
from utils import attacks as atk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROMPTS = [
    "a serene mountain landscape at sunset, photorealistic",
    "a futuristic city skyline with neon lights, digital art",
    "a bouquet of wildflowers in a glass vase, oil painting",
    "an astronaut floating in deep space, cinematic",
    "a golden retriever playing in autumn leaves, photography",
]

ATTACK_REGISTRY: dict[str, Any] = {
    "none":          lambda img: img,
    "crop_80":       lambda img: atk.crop(img, 0.80),
    "crop_60":       lambda img: atk.crop(img, 0.60),
    "rotate_15":     lambda img: atk.rotate(img, 15.0),
    "rotate_45":     lambda img: atk.rotate(img, 45.0),
    "rotate_90":     lambda img: atk.rotate(img, 90.0),
    "jpeg_80":       lambda img: atk.jpeg(img, 80),
    "jpeg_50":       lambda img: atk.jpeg(img, 50),
    "blur_2":        lambda img: atk.gaussian_blur(img, 2.0),
    "blur_4":        lambda img: atk.gaussian_blur(img, 4.0),
    "noise_005":     lambda img: atk.gaussian_noise(img, 0.05),
    "noise_01":      lambda img: atk.gaussian_noise(img, 0.10),
    "brightness_15": lambda img: atk.brightness(img, 1.5),
    "contrast_15":   lambda img: atk.contrast(img, 1.5),
}


def _load_pipeline(cfg: Config, fp16: bool = False) -> StableDiffusionPipeline:
    """Load SD v1.5 with a DDIMScheduler."""
    torch_dtype = torch.float16 if fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _eval_method(
    method_name: str,
    watermarker,
    prompts: list[str],
    attack_registry: dict[str, Any],
    cfg: Config,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one watermarking method against all attacks and return results."""
    results: dict[str, Any] = {"method": method_name, "attacks": {}}

    for attack_name, attack_fn in attack_registry.items():
        scores_wm: list[float] = []
        scores_clean: list[float] = []
        detected_wm: list[bool] = []
        detected_clean: list[bool] = []

        first_wm_image = None
        for idx, prompt in enumerate(prompts):
            seed = idx * 100

            # --- Watermarked image ---
            if method_name in ("MultiBit", "Ensemble"):
                img_wm, _ = watermarker.generate_watermarked(prompt, seed=seed, return_latents=True)
                attacked = attack_fn(img_wm)
                is_wm, score, _ = watermarker.detect(attacked)
            else:
                img_wm, _ = watermarker.generate_watermarked(prompt, seed=seed, return_latents=True)
                attacked = attack_fn(img_wm)
                is_wm, score = watermarker.detect(attacked)

            scores_wm.append(score)
            detected_wm.append(is_wm)

            # Keep the first watermarked image for saving
            if idx == 0:
                first_wm_image = img_wm

            # --- Clean (unwatermarked) image ---
            img_clean = watermarker.generate_unwatermarked(prompt, seed=seed + 999) \
                if hasattr(watermarker, "generate_unwatermarked") \
                else _generate_clean(watermarker.pipe, prompt, seed + 999, cfg)
            attacked_clean = attack_fn(img_clean)

            if method_name in ("MultiBit", "Ensemble"):
                is_cl, score_cl, _ = watermarker.detect(attacked_clean)
            else:
                is_cl, score_cl = watermarker.detect(attacked_clean)

            scores_clean.append(score_cl)
            detected_clean.append(is_cl)

        # Save the first watermarked image for the no-attack condition
        if attack_name == "none" and first_wm_image is not None:
            save_path = output_dir / f"{method_name}_sample.png"
            first_wm_image.save(save_path)

        from utils.metrics import tpr as _tpr, fpr as _fpr, balanced_accuracy as _ba

        y_true_combined = [True] * len(detected_wm) + [False] * len(detected_clean)
        y_pred_combined = list(detected_wm) + list(detected_clean)

        results["attacks"][attack_name] = {
            "tpr": _tpr(y_true_combined, y_pred_combined),
            "fpr": _fpr(y_true_combined, y_pred_combined),
            "balanced_accuracy": _ba(y_true_combined, y_pred_combined),
            "mean_score_wm": float(sum(scores_wm) / max(len(scores_wm), 1)),
            "mean_score_clean": float(sum(scores_clean) / max(len(scores_clean), 1)),
        }

    return results


def _generate_clean(pipe, prompt: str, seed: int, cfg: Config):
    """Generate a clean image without any watermark."""
    device = pipe.device
    dtype = pipe.unet.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        output_type="pil",
    )
    return output.images[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tree-Ring Watermark methods")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_images", type=int, default=5, help="Number of images per method")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--fp16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "multibit", "logpolar", "ensemble"],
        default=["baseline", "multibit", "logpolar", "ensemble"],
    )
    args = parser.parse_args()

    cfg = Config(device=args.device, output_dir=args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS[: args.n_images]

    print(f"Loading Stable Diffusion pipeline ({cfg.model_id}) on {cfg.device} …")
    pipe = _load_pipeline(cfg, fp16=args.fp16)
    print("Pipeline loaded.\n")

    all_results: list[dict] = []
    method_map = {
        "baseline": ("Baseline", lambda: TreeRingWatermark(cfg, pipe)),
        "multibit":  ("MultiBit",  lambda: MultiBitWatermark(cfg, pipe)),
        "logpolar":  ("LogPolar",  lambda: LogPolarWatermark(cfg, pipe)),
        "ensemble":  ("Ensemble",  lambda: EnsembleWatermark(cfg, pipe)),
    }

    for key in args.methods:
        method_name, factory = method_map[key]
        print(f"Evaluating: {method_name}")
        watermarker = factory()
        t0 = time.time()
        result = _eval_method(
            method_name, watermarker, prompts, ATTACK_REGISTRY, cfg, output_dir
        )
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 2)
        all_results.append(result)
        print(f"  Done in {elapsed:.1f}s\n")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
