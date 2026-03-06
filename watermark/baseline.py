"""
Exact replication of Wen et al. (2023) Tree-Ring Watermarks.

Reference
---------
Wen, Y., Kirchenbauer, J., Geiping, J., & Goldstein, T. (2023).
Tree-Ring Watermarks: Fingerprints for Diffusion Images that are
Invisible and Robust.  NeurIPS 2023.

Algorithm summary
-----------------
Injection (generate_watermarked):
  1. Sample a random latent z ~ N(0,I).
  2. Compute the 2-D FFT of channel `w_channel`.
  3. Set every coefficient whose frequency magnitude falls within
     [r-1, r+1) to a fixed complex pattern (the "ring").
  4. Convert back with IFFT; use the modified latent as the initial
     noise for DDIM sampling.

Detection (detect):
  1. Run DDIM inversion on the image to recover the approximate
     initial noise z'.
  2. Compute FFT of channel `w_channel` of z'.
  3. Extract the ring region and compare to the stored pattern via
     a normalised dot-product (cosine similarity).
  4. Return True if the score exceeds `w_threshold`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from config import Config
from utils.detection import ddim_inversion, compute_watermark_score


class TreeRingWatermark:
    """Baseline Tree-Ring watermark (Wen et al. 2023).

    Parameters
    ----------
    cfg:
        Experiment configuration.
    pipe:
        A pre-loaded :class:`StableDiffusionPipeline`.  Must already be
        on the correct device.
    """

    def __init__(self, cfg: Config, pipe: StableDiffusionPipeline) -> None:
        self.cfg = cfg
        self.pipe = pipe
        self._pattern: Optional[torch.Tensor] = None  # stored ring pattern

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_ring_mask(self, h: int, w: int) -> torch.Tensor:
        """Boolean mask selecting the ring region in frequency space.

        Returns a (h, w) boolean tensor that is True for pixels whose
        distance from the DC component (centre after fftshift) lies
        within [radius-1, radius+1).
        """
        cy, cx = h // 2, w // 2
        y = torch.arange(h, dtype=torch.float32) - cy
        x = torch.arange(w, dtype=torch.float32) - cx
        dist = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
        mask = (dist >= self.cfg.w_radius - 1) & (dist < self.cfg.w_radius + 1)
        return mask

    def _inject_ring(self, latent: torch.Tensor) -> torch.Tensor:
        """Return a copy of *latent* with the ring pattern injected.

        Parameters
        ----------
        latent:
            Shape ``(1, 4, H, W)`` – the initial noise tensor.
        """
        latent = latent.clone()
        ch = self.cfg.w_channel
        h, w = latent.shape[-2], latent.shape[-1]

        # Work in float32 numpy for FFT
        channel_np = latent[0, ch].cpu().float().numpy()

        # 2D FFT → shift DC to centre
        freq = np.fft.fftshift(np.fft.fft2(channel_np))

        mask = self._get_ring_mask(h, w).numpy()

        # Generate (or reuse) a fixed complex pattern for the ring
        if self._pattern is None:
            rng = np.random.default_rng(seed=42)
            real = rng.standard_normal(mask.sum())
            imag = rng.standard_normal(mask.sum())
            self._pattern = (real + 1j * imag).astype(np.complex64)

        np.place(freq, mask, self._pattern)

        # Shift back and IFFT → real part
        injected = np.fft.ifft2(np.fft.ifftshift(freq)).real.astype(np.float32)
        latent[0, ch] = torch.from_numpy(injected).to(latent.device)
        return latent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_watermarked(
        self,
        prompt: str,
        seed: int = 0,
        return_latents: bool = False,
    ):
        """Generate a watermarked image using SD v1-5.

        Parameters
        ----------
        prompt:
            Text prompt for generation.
        seed:
            Random seed for initial noise.
        return_latents:
            If True, also return the modified initial latent.

        Returns
        -------
        image : PIL.Image.Image
        initial_latent : torch.Tensor  (only when *return_latents* is True)
        """
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        generator = torch.Generator(device=device).manual_seed(seed)

        # Sample random initial noise
        latent_shape = (
            1,
            self.pipe.unet.config.in_channels,
            self.cfg.image_size // 8,
            self.cfg.image_size // 8,
        )
        init_latent = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

        # Inject watermark ring
        init_latent = self._inject_ring(init_latent)

        # Run DDIM sampling from the watermarked latent
        output = self.pipe(
            prompt=prompt,
            latents=init_latent,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            output_type="pil",
        )
        image = output.images[0]

        if return_latents:
            return image, init_latent
        return image

    def generate_unwatermarked(self, prompt: str, seed: int = 0):
        """Generate an unwatermarked baseline image (clean DDIM)."""
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        generator = torch.Generator(device=device).manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            output_type="pil",
        )
        return output.images[0]

    def detect(self, image) -> tuple[bool, float]:
        """Detect the watermark in *image*.

        Parameters
        ----------
        image:
            A PIL image (must have been generated at ``cfg.image_size``).

        Returns
        -------
        is_watermarked : bool
        score : float  (0 … 1, higher → more likely watermarked)
        """
        if self._pattern is None:
            raise RuntimeError(
                "No watermark pattern stored.  Call generate_watermarked() first."
            )

        # DDIM inversion to recover approximate initial noise
        inverted = ddim_inversion(image, self.pipe, self.cfg)

        ch = self.cfg.w_channel
        h, w = inverted.shape[-2], inverted.shape[-1]
        channel_np = inverted[0, ch].cpu().float().numpy()

        freq = np.fft.fftshift(np.fft.fft2(channel_np))
        mask = self._get_ring_mask(h, w).numpy()

        score = compute_watermark_score(freq[mask], self._pattern)
        return bool(score >= self.cfg.w_threshold), float(score)
