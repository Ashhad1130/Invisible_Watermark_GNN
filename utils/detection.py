"""
DDIM inversion and watermark scoring helpers.

ddim_inversion
    Runs DDIM inversion on a PIL image to recover the approximate
    initial noise latent.  Used by all detection methods.

compute_watermark_score
    Computes a normalised cosine similarity between the extracted ring
    coefficients and the stored watermark pattern.  Returns a value in
    [0, 1] (higher → more likely watermarked).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from config import Config


# ---------------------------------------------------------------------------
# DDIM Inversion
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_inversion(
    image: Image.Image,
    pipe: "StableDiffusionPipeline",
    cfg: "Config",
) -> torch.Tensor:
    """Invert a PIL *image* back to approximate initial noise via DDIM.

    The procedure follows the deterministic DDIM inversion formula:
    starting from the image's VAE-encoded latent, we iterate *forward*
    through the denoising schedule (i.e. from low-noise to high-noise)
    to approximate the original initial noise latent ``z_T``.

    Parameters
    ----------
    image:
        PIL image to invert.  Should be the same resolution used during
        generation (``cfg.image_size × cfg.image_size``).
    pipe:
        Pre-loaded Stable Diffusion pipeline with a DDIMScheduler.
    cfg:
        Experiment configuration (``num_inference_steps``, ``guidance_scale``).

    Returns
    -------
    latent : torch.Tensor
        Shape ``(1, 4, H//8, W//8)`` approximate initial noise.
    """
    device = pipe.device
    dtype = pipe.unet.dtype

    # Encode image to latent space
    img_tensor = _pil_to_tensor(image).to(device=device, dtype=dtype)
    latent = pipe.vae.encode(img_tensor).latent_dist.mean
    latent = latent * pipe.vae.config.scaling_factor

    # Encode unconditional + empty text embeddings for CFG
    uncond_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(
        uncond_input.input_ids.to(device)
    )[0]

    # Set the scheduler for inversion
    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps.flip(0)  # low → high noise

    for t in timesteps:
        # Predict noise residual (classifier-free guidance)
        latent_model_input = pipe.scheduler.scale_model_input(latent, t)
        noise_pred_uncond = pipe.unet(
            latent_model_input, t, encoder_hidden_states=uncond_embeddings
        ).sample

        # DDIM inversion step (one step forward in noise)
        latent = _ddim_inversion_step(pipe, latent, noise_pred_uncond, t)

    return latent


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a (1, 3, H, W) tensor in [-1, 1]."""
    image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _ddim_inversion_step(
    pipe: "StableDiffusionPipeline",
    latent: torch.Tensor,
    noise_pred: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Single deterministic DDIM forward (inversion) step.

    Given the current latent ``x_t`` and the noise prediction, compute
    ``x_{t+1}`` (moving toward higher noise).
    """
    scheduler = pipe.scheduler
    alpha_prod_t = scheduler.alphas_cumprod[t]
    # Next timestep index (with clamping to stay in range)
    t_next = min(int(t) + scheduler.config.num_train_timesteps // scheduler.num_inference_steps,
                 scheduler.config.num_train_timesteps - 1)
    alpha_prod_t_next = scheduler.alphas_cumprod[t_next]

    beta_prod_t = 1 - alpha_prod_t

    # Predict x_0
    pred_original_sample = (latent - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

    # Direction pointing to x_t
    pred_sample_direction = (1 - alpha_prod_t_next).sqrt() * noise_pred

    # x_{t+1}
    prev_latent = alpha_prod_t_next.sqrt() * pred_original_sample + pred_sample_direction
    return prev_latent


# ---------------------------------------------------------------------------
# Watermark Scoring
# ---------------------------------------------------------------------------

def compute_watermark_score(
    extracted: np.ndarray,
    pattern: np.ndarray,
) -> float:
    """Normalised cosine similarity between extracted ring and stored pattern.

    Both inputs are complex-valued 1-D arrays of the same length.
    The score is computed over the concatenated real and imaginary parts
    so that both magnitude and phase contribute.

    Parameters
    ----------
    extracted:
        Ring coefficients extracted from the inverted latent.
    pattern:
        The watermark pattern stored during generation.

    Returns
    -------
    score : float
        Value in [0, 1].  0.5 corresponds to uncorrelated noise;
        1.0 indicates a perfect match.
    """
    a = np.concatenate([extracted.real, extracted.imag]).astype(np.float64)
    b = np.concatenate([pattern.real, pattern.imag]).astype(np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Map from [-1, 1] to [0, 1]
    return (cosine_sim + 1.0) / 2.0
