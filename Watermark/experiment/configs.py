"""Experiment configurations for baseline vs optimized comparison."""
from dataclasses import dataclass, field
from typing import Optional, List

# Add support for landscape-only dataset (optional alternative to Gustavosta)
DATASET_LANDSCAPE = "landscape"  # Placeholder dataset name

@dataclass
class WatermarkConfig:
    w_seed: int = 999999
    w_channel: int = 3
    w_pattern: str = "ring"
    w_mask_shape: str = "circle"
    w_radius: int = 10
    w_radius_inner: int = 4  # outer radius of inner band for multi_ring mask
    w_measurement: str = "l1_complex"
    w_injection: str = "complex"
    w_pattern_const: float = 0.0

@dataclass
class AttackConfig:
    name: str = "no_attack"
    r_degree: Optional[float] = None
    jpeg_ratio: Optional[int] = None
    crop_scale: Optional[float] = None
    crop_ratio: Optional[float] = None
    gaussian_blur_r: Optional[int] = None
    gaussian_std: Optional[float] = None
    brightness_factor: Optional[float] = None
    rand_aug: int = 0

@dataclass
class ExperimentConfig:
    name: str = "experiment"
    approach: str = "baseline"
    model_id: str = "runwayml/stable-diffusion-v1-5"
    image_length: int = 512
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    test_num_inference_steps: Optional[int] = None
    gen_seed: int = 0
    start: int = 0
    end: int = 10
    reference_model: Optional[str] = "ViT-g-14"
    reference_model_pretrain: Optional[str] = "laion2b_s12b_b42k"
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    attacks: List[AttackConfig] = field(default_factory=list)
    checkpoint_every: int = 5
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    outputs_dir: str = "outputs"

    def __post_init__(self):
        if self.test_num_inference_steps is None:
            self.test_num_inference_steps = self.num_inference_steps

# Paper-exact attacks (Table 2 in Wen et al. 2023)
BASIC_ATTACKS = [
    AttackConfig(name="no_attack"),
    AttackConfig(name="rotation_75",       r_degree=75.0),
    AttackConfig(name="jpeg_25",           jpeg_ratio=25),
    AttackConfig(name="crop_0.75",         crop_scale=0.75, crop_ratio=0.75),
    AttackConfig(name="gaussian_blur_8",   gaussian_blur_r=8),
    AttackConfig(name="gaussian_noise_0.1", gaussian_std=0.1),
    AttackConfig(name="brightness_6",      brightness_factor=6.0),
]

EXTENDED_ATTACKS = BASIC_ATTACKS + [
    AttackConfig(name="rotation_45",       r_degree=45.0),
    AttackConfig(name="jpeg_50",           jpeg_ratio=50),
    AttackConfig(name="crop_0.5",          crop_scale=0.5, crop_ratio=0.5),
    AttackConfig(name="gaussian_blur_4",   gaussian_blur_r=4),
    AttackConfig(name="gaussian_noise_0.05", gaussian_std=0.05),
    AttackConfig(name="brightness_3",      brightness_factor=3.0),
]

def get_small_scale_baseline():
    """Paper-exact baseline: DDIM, r=10, 50 steps, single channel."""
    return ExperimentConfig(
        name="small_baseline", approach="baseline", start=0, end=10,
        num_inference_steps=50, test_num_inference_steps=50,
        watermark=WatermarkConfig(w_channel=3, w_pattern="ring", w_radius=10),
        attacks=BASIC_ATTACKS, checkpoint_every=5)

def get_small_scale_optimized():
    """Optimized: 100 DDIM steps for more accurate inversion; r=10 kept (safe for real images)."""
    return ExperimentConfig(
        name="small_optimized", approach="optimized", start=0, end=10,
        num_inference_steps=100, test_num_inference_steps=100,
        watermark=WatermarkConfig(w_channel=3, w_pattern="ring", w_radius=10),
        attacks=BASIC_ATTACKS, checkpoint_every=5)

def get_large_scale_baseline():
    """Paper-exact baseline, large scale."""
    return ExperimentConfig(
        name="large_baseline", approach="baseline", start=0, end=100,
        num_inference_steps=50, test_num_inference_steps=50,
        watermark=WatermarkConfig(w_channel=3, w_pattern="ring", w_radius=10),
        attacks=EXTENDED_ATTACKS, checkpoint_every=10)

def get_large_scale_optimized():
    """Optimized, large scale."""
    return ExperimentConfig(
        name="large_optimized", approach="optimized", start=0, end=100,
        num_inference_steps=100, test_num_inference_steps=100,
        watermark=WatermarkConfig(w_channel=3, w_pattern="ring", w_radius=10),
        attacks=EXTENDED_ATTACKS, checkpoint_every=10)


def get_small_scale_multiring():
    """Multi-ring: two rotation-invariant Fourier bands + 100 DDIM steps.
    Inner band (r=0..4): survives heavy cropping (near-DC energy).
    Outer band (r=6..10): carries the distinct signal for detection.
    Gap at r=5 keeps the two bands spectrally separate.
    Still rotation-invariant because both bands are annuli.
    """
    return ExperimentConfig(
        name="small_multi_ring", approach="multi_ring", start=0, end=10,
        num_inference_steps=100, test_num_inference_steps=100,
        watermark=WatermarkConfig(
            w_channel=3, w_pattern="ring", w_mask_shape="multi_ring",
            w_radius=10, w_radius_inner=4),
        attacks=BASIC_ATTACKS, checkpoint_every=5)


def get_large_scale_multiring():
    """Multi-ring, large scale."""
    return ExperimentConfig(
        name="large_multi_ring", approach="multi_ring", start=0, end=100,
        num_inference_steps=100, test_num_inference_steps=100,
        watermark=WatermarkConfig(
            w_channel=3, w_pattern="ring", w_mask_shape="multi_ring",
            w_radius=10, w_radius_inner=4),
        attacks=EXTENDED_ATTACKS, checkpoint_every=10)


# ==============================================================================
# LANDSCAPE-ONLY VARIANTS (Optional: use local landscape prompts instead of Gustavosta)
# ==============================================================================
def get_small_scale_baseline_landscape():
    """Baseline with landscape prompts (no dataset download needed)."""
    cfg = get_small_scale_baseline()
    cfg.name = "small_baseline_landscape"
    cfg.reference_model = None  # Skip CLIP for faster iteration
    # The dataset will be set to "landscape" in run_baseline.py --dataset flag
    return cfg


def get_small_scale_optimized_landscape():
    """Optimized with landscape prompts."""
    cfg = get_small_scale_optimized()
    cfg.name = "small_optimized_landscape"
    cfg.reference_model = None
    return cfg


def get_small_scale_multiring_landscape():
    """Multi-ring with landscape prompts."""
    cfg = get_small_scale_multiring()
    cfg.name = "small_multi_ring_landscape"
    cfg.reference_model = None
    return cfg
