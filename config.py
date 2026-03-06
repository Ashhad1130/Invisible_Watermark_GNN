"""
Configuration dataclass for Tree-Ring Watermark experiments.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Central configuration for all watermark experiments.

    Attributes
    ----------
    model_id:
        HuggingFace model identifier for the Stable Diffusion pipeline.
    device:
        PyTorch device string, e.g. "cuda" or "cpu".
    image_size:
        Spatial resolution (height == width) for generated images.
    num_inference_steps:
        Number of DDIM denoising steps used during generation.
    guidance_scale:
        Classifier-free guidance scale.
    w_channel:
        Latent channel index that receives the watermark ring (baseline).
    w_radius:
        Radius (in pixels) of the Fourier ring pattern.
    w_threshold:
        Detection threshold for watermark score (0-1).
    n_bits:
        Number of payload bits for MultiBitWatermark.
    output_dir:
        Directory where evaluation results and figures are saved.
    """

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    image_size: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    w_channel: int = 0
    w_radius: int = 10
    w_threshold: float = 0.75
    n_bits: int = 16
    output_dir: str = "results"
