"""
Image attack functions for robustness evaluation.

Each function accepts a PIL Image and returns a PIL Image (or a NumPy
array for gaussian_noise / brightness / contrast which also accept and
return PIL images for consistency).

Supported attacks
-----------------
crop            – random centre crop then resize back to original size
rotate          – rotation by a given angle (degrees, counter-clockwise)
jpeg            – JPEG compression at a given quality level
gaussian_blur   – Gaussian blur with a given radius
gaussian_noise  – additive Gaussian noise with a given standard deviation
brightness      – multiply pixel values by a given factor
contrast        – adjust contrast by a given factor
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def crop(image: Image.Image, crop_fraction: float = 0.8) -> Image.Image:
    """Centre-crop *image* to *crop_fraction* of its size, then resize back.

    Parameters
    ----------
    image:
        Input PIL image.
    crop_fraction:
        Fraction of the original dimensions to keep (0 < crop_fraction ≤ 1).
    """
    if not (0 < crop_fraction <= 1.0):
        raise ValueError(f"crop_fraction must be in (0, 1], got {crop_fraction}")
    w, h = image.size
    new_w = int(w * crop_fraction)
    new_h = int(h * crop_fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def rotate(image: Image.Image, angle: float = 45.0) -> Image.Image:
    """Rotate *image* counter-clockwise by *angle* degrees.

    Uses ``expand=False`` to keep the original canvas size (matching how
    an attacker might rotate without revealing the original resolution).

    Parameters
    ----------
    image:
        Input PIL image.
    angle:
        Rotation angle in degrees (counter-clockwise).
    """
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)


def jpeg(image: Image.Image, quality: int = 50) -> Image.Image:
    """Re-encode *image* as JPEG at *quality* and decode back.

    Parameters
    ----------
    image:
        Input PIL image.
    quality:
        JPEG quality factor (1 = worst, 95 = near lossless).
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Apply Gaussian blur with the given *radius*.

    Parameters
    ----------
    image:
        Input PIL image.
    radius:
        Standard deviation of the Gaussian kernel (pixels).
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def gaussian_noise(image: Image.Image, std: float = 0.05) -> Image.Image:
    """Add zero-mean Gaussian noise with standard deviation *std*.

    Pixel values are clipped to [0, 255] after adding noise.

    Parameters
    ----------
    image:
        Input PIL image (RGB).
    std:
        Noise standard deviation in the [0, 1] normalised range, so
        *std* = 0.05 adds noise with σ ≈ 12.75 in [0, 255] space.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def brightness(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Adjust image brightness by *factor*.

    Parameters
    ----------
    image:
        Input PIL image.
    factor:
        1.0 = original brightness; >1.0 = brighter; <1.0 = darker.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Adjust image contrast by *factor*.

    Parameters
    ----------
    image:
        Input PIL image.
    factor:
        1.0 = original contrast; >1.0 = higher contrast; <1.0 = lower.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)
