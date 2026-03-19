"""
MPS compatibility: wraps FFT-dependent functions to cast to fp32 on MPS.
MPS does not support fp16 FFT. On CUDA/CPU these are zero-overhead passthroughs.
get_watermarking_pattern returns COMPLEX tensors — we must preserve that dtype.
"""
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent / "tree-ring-watermark"
sys.path.insert(0, str(REPO_ROOT))
import optim_utils as _ou

class _Fp32Context:
    def __init__(self, pipe):
        self.pipe = pipe
    def __enter__(self):
        self.orig_dtype = self.pipe.text_encoder.dtype
        self.pipe.text_encoder = self.pipe.text_encoder.float()
        return self
    def __exit__(self, *args):
        self.pipe.text_encoder = self.pipe.text_encoder.to(self.orig_dtype)

def get_watermarking_pattern(pipe, args, device, shape=None):
    if device == "mps":
        with _Fp32Context(pipe):
            return _ou.get_watermarking_pattern(pipe, args, device, shape)
    return _ou.get_watermarking_pattern(pipe, args, device, shape)

def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    if init_latents_w.device.type == "mps":
        orig_dtype = init_latents_w.dtype
        result = _ou.inject_watermark(init_latents_w.float(), watermarking_mask, gt_patch, args)
        return result.to(orig_dtype)
    return _ou.inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if reversed_latents_no_w.device.type == "mps":
        return _ou.eval_watermark(
            reversed_latents_no_w.float(), reversed_latents_w.float(),
            watermarking_mask, gt_patch, args)
    return _ou.eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)
