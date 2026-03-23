"""Device compatibility layer: handles FFT dtype issues on MPS, passthrough on CUDA/CPU.
Also patches image_distortion to fix lazy-load JPEG bug in newer Pillow."""
import sys, io
from pathlib import Path
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent / "tree-ring-watermark"
sys.path.insert(0, str(REPO_ROOT))
import optim_utils as _ou

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _needs_fp32_fft(device):
    return device in ("mps", "cpu")

def get_watermarking_pattern(pipe, args, device, shape=None):
    if _needs_fp32_fft(device):
        orig_dtype = pipe.text_encoder.dtype
        pipe.text_encoder = pipe.text_encoder.float()
        result = _ou.get_watermarking_pattern(pipe, args, device, shape)
        pipe.text_encoder = pipe.text_encoder.to(orig_dtype)
        return result
    return _ou.get_watermarking_pattern(pipe, args, device, shape)

def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    if _needs_fp32_fft(init_latents_w.device.type):
        orig_dtype = init_latents_w.dtype
        result = _ou.inject_watermark(init_latents_w.float(), watermarking_mask, gt_patch, args)
        return result.to(orig_dtype)
    return _ou.inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if _needs_fp32_fft(reversed_latents_no_w.device.type):
        return _ou.eval_watermark(
            reversed_latents_no_w.float(), reversed_latents_w.float(),
            watermarking_mask, gt_patch, args)
    return _ou.eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)

# Patch: fix JPEG lazy-load bug in image_distortion.
# Original code saves img1 to file, opens lazily, then overwrites with img2
# before img1 is fully loaded. Use in-memory BytesIO instead.
_orig_image_distortion = _ou.image_distortion

def _patched_image_distortion(img1, img2, seed, args):
    if args.jpeg_ratio is not None:
        buf1 = io.BytesIO()
        img1.save(buf1, format="JPEG", quality=args.jpeg_ratio)
        buf1.seek(0)
        img1 = Image.open(buf1)
        img1.load()

        buf2 = io.BytesIO()
        img2.save(buf2, format="JPEG", quality=args.jpeg_ratio)
        buf2.seek(0)
        img2 = Image.open(buf2)
        img2.load()

        # Temporarily disable jpeg_ratio so original function doesn't redo it
        orig_ratio = args.jpeg_ratio
        args.jpeg_ratio = None
        img1, img2 = _orig_image_distortion(img1, img2, seed, args)
        args.jpeg_ratio = orig_ratio
        return img1, img2
    return _orig_image_distortion(img1, img2, seed, args)

# Monkey-patch so both run_baseline and run_optimized get the fix
_ou.image_distortion = _patched_image_distortion

def image_distortion(img1, img2, seed, args):
    return _ou.image_distortion(img1, img2, seed, args)
