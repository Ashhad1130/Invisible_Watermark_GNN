"""
Three novel watermarking methods extending Tree-Ring Watermarks.

(1) MultiBitWatermark
    Encodes an N-bit binary payload by mapping each bit to a phase angle
    on the Fourier ring.  Detection decodes the payload via phase
    quantisation and computes the bit-error rate.

(2) LogPolarWatermark
    Achieves rotation invariance by embedding the ring pattern in the
    log-polar (Fourier-Mellin) transform of the latent channel.
    Rotation in the spatial domain becomes a translation in the
    log-polar domain, so the ring energy is preserved.

(3) EnsembleWatermark
    Embeds independent ring patterns in all four latent channels and
    uses majority voting during detection, improving robustness against
    channel-specific noise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from config import Config
from utils.detection import ddim_inversion, compute_watermark_score


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_ring_mask(h: int, w: int, radius: int) -> np.ndarray:
    """Boolean mask (h, w) selecting a 2-pixel-wide ring at *radius*."""
    cy, cx = h // 2, w // 2
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    dist = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    return (dist >= radius - 1) & (dist < radius + 1)


def _inject_pattern_into_channel(
    latent: torch.Tensor,
    channel: int,
    pattern: np.ndarray,
    mask: np.ndarray,
) -> torch.Tensor:
    """Inject *pattern* at ring *mask* positions in *channel* of *latent*."""
    latent = latent.clone()
    channel_np = latent[0, channel].cpu().float().numpy()
    freq = np.fft.fftshift(np.fft.fft2(channel_np))
    np.place(freq, mask, pattern)
    injected = np.fft.ifft2(np.fft.ifftshift(freq)).real.astype(np.float32)
    latent[0, channel] = torch.from_numpy(injected).to(latent.device)
    return latent


# ---------------------------------------------------------------------------
# (1) MultiBitWatermark
# ---------------------------------------------------------------------------

class MultiBitWatermark:
    """Encode an N-bit payload in the Fourier ring phase angles.

    Each ring coefficient is set to unit magnitude with a phase angle
    determined by the corresponding payload bit: 0 → 0 rad, 1 → π rad.
    Bits are distributed evenly around the ring; if the ring has fewer
    coefficients than ``n_bits``, bits are repeated cyclically.

    Parameters
    ----------
    cfg:
        Experiment configuration.  ``cfg.n_bits`` controls payload size.
    pipe:
        Pre-loaded Stable Diffusion pipeline on the target device.
    """

    def __init__(self, cfg: Config, pipe: StableDiffusionPipeline) -> None:
        self.cfg = cfg
        self.pipe = pipe
        self._payload: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def _encode_payload(self, payload: np.ndarray, n_coeffs: int) -> np.ndarray:
        """Map binary *payload* to complex ring coefficients.

        Phase 0 → bit 0, phase π → bit 1.
        """
        bits = np.tile(payload, int(np.ceil(n_coeffs / len(payload))))[:n_coeffs]
        phases = bits.astype(np.float32) * np.pi  # 0 or π
        return (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)

    def _decode_payload(self, ring_coeffs: np.ndarray) -> np.ndarray:
        """Recover bits from ring coefficients via phase quantisation."""
        phases = np.angle(ring_coeffs)  # in (-π, π]
        # Map phase ≈ 0 → 0, phase ≈ ±π → 1
        bits = (np.abs(phases) > np.pi / 2).astype(np.uint8)
        return bits

    # ------------------------------------------------------------------

    def generate_watermarked(
        self,
        prompt: str,
        payload: Optional[np.ndarray] = None,
        seed: int = 0,
        return_latents: bool = False,
    ):
        """Generate an image carrying *payload* bits in the Fourier ring.

        Parameters
        ----------
        prompt:
            Text prompt.
        payload:
            Binary array of length ``cfg.n_bits``.  Random bits if None.
        seed:
            RNG seed for initial noise.
        """
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        if payload is None:
            rng = np.random.default_rng(seed)
            payload = rng.integers(0, 2, size=self.cfg.n_bits).astype(np.uint8)
        self._payload = payload.copy()

        generator = torch.Generator(device=device).manual_seed(seed)
        latent_shape = (
            1,
            self.pipe.unet.config.in_channels,
            self.cfg.image_size // 8,
            self.cfg.image_size // 8,
        )
        init_latent = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

        h, w = init_latent.shape[-2], init_latent.shape[-1]
        self._mask = _get_ring_mask(h, w, self.cfg.w_radius)
        n_coeffs = int(self._mask.sum())
        pattern = self._encode_payload(payload, n_coeffs)

        init_latent = _inject_pattern_into_channel(
            init_latent, self.cfg.w_channel, pattern, self._mask
        )

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

    def detect(self, image) -> tuple[bool, float, np.ndarray]:
        """Detect and decode the watermark payload.

        Returns
        -------
        is_watermarked : bool
        score : float  (1 - BER, higher → more likely watermarked)
        decoded_bits : np.ndarray  (length n_bits)
        """
        if self._payload is None or self._mask is None:
            raise RuntimeError(
                "No payload stored.  Call generate_watermarked() first."
            )

        inverted = ddim_inversion(image, self.pipe, self.cfg)
        ch = self.cfg.w_channel
        channel_np = inverted[0, ch].cpu().float().numpy()
        freq = np.fft.fftshift(np.fft.fft2(channel_np))

        ring_coeffs = freq[self._mask]
        decoded_bits_full = self._decode_payload(ring_coeffs)

        # Majority vote over repeated bit pattern
        n_bits = self.cfg.n_bits
        n_coeffs = len(decoded_bits_full)
        repeats = int(np.ceil(n_coeffs / n_bits))
        padded = np.zeros(repeats * n_bits, dtype=np.float32)
        padded[:n_coeffs] = decoded_bits_full
        votes = padded.reshape(repeats, n_bits).mean(axis=0)
        decoded_bits = (votes >= 0.5).astype(np.uint8)

        ber_val = float(np.mean(decoded_bits != self._payload[:n_bits]))
        score = 1.0 - ber_val
        return bool(score >= self.cfg.w_threshold), score, decoded_bits


# ---------------------------------------------------------------------------
# (2) LogPolarWatermark
# ---------------------------------------------------------------------------

class LogPolarWatermark:
    """Rotation-invariant watermark using the log-polar (Fourier-Mellin) transform.

    Rotation by angle θ in the spatial domain corresponds to a translation
    by θ in the angular axis of the log-polar representation.  By embedding
    the ring pattern *before* the log-polar transform, the ring energy is
    preserved under rotation of the final image.

    Implementation notes
    --------------------
    * We approximate the Fourier-Mellin transform via scipy's
      ``ndimage.geometric_transform``.
    * The pattern is embedded in the log-polar magnitude spectrum; the
      phase is left unchanged to preserve spatial structure.
    """

    def __init__(self, cfg: Config, pipe: StableDiffusionPipeline) -> None:
        self.cfg = cfg
        self.pipe = pipe
        self._pattern: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    @staticmethod
    def _to_log_polar(img: np.ndarray) -> np.ndarray:
        """Convert a 2-D array to log-polar coordinates (magnitude spectrum)."""
        from scipy.ndimage import map_coordinates

        h, w = img.shape
        cy, cx = h / 2.0, w / 2.0
        max_r = np.sqrt(cx ** 2 + cy ** 2)

        log_r = np.linspace(0, np.log(max_r), h)
        theta = np.linspace(0, 2 * np.pi, w, endpoint=False)

        r_grid = np.exp(log_r)[:, None]  # (h, 1)
        theta_grid = theta[None, :]       # (1, w)

        x_coords = cx + r_grid * np.cos(theta_grid)
        y_coords = cy + r_grid * np.sin(theta_grid)

        return map_coordinates(img, [y_coords.ravel(), x_coords.ravel()],
                               order=1).reshape(h, w)

    def _build_pattern(self, h: int, w: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self._mask = _get_ring_mask(h, w, self.cfg.w_radius)
        n_coeffs = int(self._mask.sum())
        real = rng.standard_normal(n_coeffs)
        imag = rng.standard_normal(n_coeffs)
        return (real + 1j * imag).astype(np.complex64)

    # ------------------------------------------------------------------

    def generate_watermarked(
        self,
        prompt: str,
        seed: int = 0,
        return_latents: bool = False,
    ):
        """Generate a rotation-invariantly watermarked image."""
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        generator = torch.Generator(device=device).manual_seed(seed)
        latent_shape = (
            1,
            self.pipe.unet.config.in_channels,
            self.cfg.image_size // 8,
            self.cfg.image_size // 8,
        )
        init_latent = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

        h, w = init_latent.shape[-2], init_latent.shape[-1]
        self._pattern = self._build_pattern(h, w, seed=42)

        # Embed in the log-polar frequency domain
        ch = self.cfg.w_channel
        channel_np = init_latent[0, ch].cpu().float().numpy()

        # FFT → log-polar magnitude injection → back
        freq = np.fft.fftshift(np.fft.fft2(channel_np))
        magnitude = np.abs(freq)
        phase = np.angle(freq)

        lp_magnitude = self._to_log_polar(magnitude)
        lp_freq = np.fft.fftshift(np.fft.fft2(lp_magnitude))

        mask = self._mask
        assert mask is not None
        np.place(lp_freq, mask, self._pattern)

        lp_magnitude_wm = np.fft.ifft2(np.fft.ifftshift(lp_freq)).real
        # Blend back: use the watermarked magnitude with original phase
        freq_wm = lp_magnitude_wm * np.exp(1j * phase)
        injected = np.fft.ifft2(np.fft.ifftshift(freq_wm)).real.astype(np.float32)

        init_latent = init_latent.clone()
        init_latent[0, ch] = torch.from_numpy(injected).to(device)

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

    def detect(self, image) -> tuple[bool, float]:
        """Detect the rotation-invariant watermark."""
        if self._pattern is None or self._mask is None:
            raise RuntimeError(
                "No watermark pattern stored.  Call generate_watermarked() first."
            )

        inverted = ddim_inversion(image, self.pipe, self.cfg)
        ch = self.cfg.w_channel
        channel_np = inverted[0, ch].cpu().float().numpy()

        freq = np.fft.fftshift(np.fft.fft2(channel_np))
        magnitude = np.abs(freq)
        lp_magnitude = self._to_log_polar(magnitude)
        lp_freq = np.fft.fftshift(np.fft.fft2(lp_magnitude))

        score = compute_watermark_score(lp_freq[self._mask], self._pattern)
        return bool(score >= self.cfg.w_threshold), float(score)


# ---------------------------------------------------------------------------
# (3) EnsembleWatermark
# ---------------------------------------------------------------------------

class EnsembleWatermark:
    """Multi-channel ensemble watermark with majority-vote detection.

    Independent ring patterns are embedded in all four latent channels.
    During detection, each channel casts a binary vote; the image is
    classified as watermarked when at least ``vote_threshold`` channels
    agree (default: majority ≥ 3/4).

    Parameters
    ----------
    cfg:
        Experiment configuration.
    pipe:
        Pre-loaded Stable Diffusion pipeline.
    vote_threshold:
        Minimum number of positive-voting channels to declare detection.
    """

    def __init__(
        self,
        cfg: Config,
        pipe: StableDiffusionPipeline,
        vote_threshold: int = 3,
    ) -> None:
        self.cfg = cfg
        self.pipe = pipe
        self.vote_threshold = vote_threshold
        self._patterns: Optional[list[np.ndarray]] = None
        self._mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def _build_patterns(self, h: int, w: int) -> list[np.ndarray]:
        """Generate one independent ring pattern per latent channel."""
        self._mask = _get_ring_mask(h, w, self.cfg.w_radius)
        n_coeffs = int(self._mask.sum())
        patterns = []
        for ch in range(4):
            rng = np.random.default_rng(seed=100 + ch)
            real = rng.standard_normal(n_coeffs)
            imag = rng.standard_normal(n_coeffs)
            patterns.append((real + 1j * imag).astype(np.complex64))
        return patterns

    # ------------------------------------------------------------------

    def generate_watermarked(
        self,
        prompt: str,
        seed: int = 0,
        return_latents: bool = False,
    ):
        """Generate an image watermarked in all four latent channels."""
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        generator = torch.Generator(device=device).manual_seed(seed)
        latent_shape = (
            1,
            self.pipe.unet.config.in_channels,
            self.cfg.image_size // 8,
            self.cfg.image_size // 8,
        )
        init_latent = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

        h, w = init_latent.shape[-2], init_latent.shape[-1]
        self._patterns = self._build_patterns(h, w)

        for ch, pattern in enumerate(self._patterns):
            assert self._mask is not None
            init_latent = _inject_pattern_into_channel(
                init_latent, ch, pattern, self._mask
            )

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

    def detect(self, image) -> tuple[bool, float, list[float]]:
        """Detect watermark via majority vote across all channels.

        Returns
        -------
        is_watermarked : bool
        score : float  (fraction of positive-voting channels, 0 … 1)
        channel_scores : list[float]  per-channel cosine similarity
        """
        if self._patterns is None or self._mask is None:
            raise RuntimeError(
                "No watermark patterns stored.  Call generate_watermarked() first."
            )

        inverted = ddim_inversion(image, self.pipe, self.cfg)
        channel_scores: list[float] = []

        for ch, pattern in enumerate(self._patterns):
            channel_np = inverted[0, ch].cpu().float().numpy()
            freq = np.fft.fftshift(np.fft.fft2(channel_np))
            s = compute_watermark_score(freq[self._mask], pattern)
            channel_scores.append(float(s))

        votes = sum(s >= self.cfg.w_threshold for s in channel_scores)
        score = votes / 4.0
        return bool(votes >= self.vote_threshold), score, channel_scores
