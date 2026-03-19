#!/usr/bin/env bash
# =============================================================================
# Tree-Ring Watermark: COMPLETE Setup + Run Script
# Copy this single file to any Mac (M4 Pro) and run it. It handles everything:
#   1. Installs Homebrew Python 3.10 (if needed)
#   2. Installs Poetry (if needed)
#   3. Clones the tree-ring-watermark repo
#   4. Copies experiment files
#   5. Installs all Python dependencies
#   6. Downloads the Stable Diffusion model
#   7. Runs baseline experiment
#   8. Runs optimized experiment
#   9. Compares results
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh                    # small scale, skip CLIP (fastest)
#   ./setup_and_run.sh small              # same as above
#   ./setup_and_run.sh large              # large scale (100 images, 14 attacks)
#   ./setup_and_run.sh small --with-clip  # with CLIP scores (needs extra download)
# =============================================================================
set -euo pipefail

# Allow MPS to use all available memory
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# --- Parse arguments ---
SCALE="${1:-small}"
WITH_CLIP="${2:-}"
SKIP_CLIP_FLAG="--skip_clip"
if [[ "$WITH_CLIP" == "--with-clip" ]]; then
    SKIP_CLIP_FLAG=""
fi

# Project directory (where this script lives, or current dir)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "########################################################"
echo "#  Tree-Ring Watermark: Full Pipeline"
echo "#  Scale: $SCALE"
echo "#  CLIP scores: $([ -z "$SKIP_CLIP_FLAG" ] && echo 'YES' || echo 'SKIPPED')"
echo "#  Project dir: $PROJECT_DIR"
echo "########################################################"
echo ""

# =============================================================================
# STEP 1: Ensure Python 3.10-3.12 is available
# =============================================================================
echo "============================================"
echo "STEP 1/9: Checking Python..."
echo "============================================"

PYTHON_BIN=""
for v in python3.10 python3.11 python3.12; do
    if command -v "$v" &>/dev/null; then
        PYTHON_BIN="$(command -v "$v")"
        break
    fi
    for prefix in /opt/homebrew/opt /usr/local/opt; do
        candidate="${prefix}/python@${v#python}/bin/$v"
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break 2
        fi
    done
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "  Python 3.10-3.12 not found. Installing Python 3.10 via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "  Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
    fi
    brew install python@3.10
    PYTHON_BIN="$(brew --prefix python@3.10)/bin/python3.10"
fi

echo "  Using: $PYTHON_BIN ($($PYTHON_BIN --version))"
echo ""

# =============================================================================
# STEP 2: Ensure Poetry is available
# =============================================================================
echo "============================================"
echo "STEP 2/9: Checking Poetry..."
echo "============================================"

if ! command -v poetry &>/dev/null; then
    echo "  Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | "$PYTHON_BIN" -
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "  $(poetry --version)"
echo ""

# =============================================================================
# STEP 3: Clone tree-ring-watermark repo (if needed)
# =============================================================================
echo "============================================"
echo "STEP 3/9: Setting up tree-ring-watermark repo..."
echo "============================================"

if [[ ! -d "$PROJECT_DIR/tree-ring-watermark" ]]; then
    echo "  Cloning tree-ring-watermark..."
    git clone https://github.com/YuxinWenRick/tree-ring-watermark.git "$PROJECT_DIR/tree-ring-watermark"
else
    echo "  tree-ring-watermark/ already exists. Skipping clone."
fi
echo ""

# =============================================================================
# STEP 4: Create experiment files
# =============================================================================
echo "============================================"
echo "STEP 4/9: Creating experiment files..."
echo "============================================"

mkdir -p "$PROJECT_DIR/experiment/checkpoints" \
         "$PROJECT_DIR/experiment/results" \
         "$PROJECT_DIR/experiment/outputs"

# --- pyproject.toml ---
cat > "$PROJECT_DIR/pyproject.toml" << 'PYPROJECT_EOF'
[tool.poetry]
name = "tree-ring-watermark-experiment"
version = "1.0.0"
description = "Baseline vs Optimized Tree-Ring Watermark comparison"
authors = ["Ashhad Quadri"]

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
# Paper-exact versions (Wen et al. 2023):
# "PyTorch==1.13.0, transformers==4.23.1, diffusers==0.11.1"
# With diffusers==0.11.1, DPMSolverMultistepScheduler.init_noise_sigma==1.0 (correct).
# Higher diffusers versions have init_noise_sigma~14.6, causing double-scaling bug.
torch = ">=1.13.0"
torchvision = ">=0.14.0"
diffusers = "0.11.1"
transformers = ">=4.26.0,<4.36"
accelerate = ">=0.11.0,<0.21"
safetensors = ">=0.2"
huggingface-hub = ">=0.10.0,<0.17"
datasets = ">=1.0"
setuptools = "<70"
scipy = "*"
scikit-learn = "*"
Pillow = "*"
numpy = "<2.0"
tqdm = "*"
open-clip-torch = "*"
matplotlib = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
PYPROJECT_EOF

# --- experiment/__init__.py ---
touch "$PROJECT_DIR/experiment/__init__.py"

# --- experiment/landscape_prompts.py ---
cat > "$PROJECT_DIR/experiment/landscape_prompts.py" << 'PROMPTS_EOF'
"""Curated landscape/scenery prompts for watermark experiments."""

LANDSCAPE_PROMPTS = [
    "A breathtaking panoramic view of snow-capped mountains at golden hour with a crystal clear alpine lake in the foreground",
    "Misty mountain peaks emerging from clouds at dawn, with pine forests covering the lower slopes",
    "A dramatic mountain landscape with a winding river valley, autumn colors on the hillsides",
    "Towering granite cliffs reflected perfectly in a still mountain lake surrounded by wildflowers",
    "A serene alpine meadow with snow-capped peaks in the background under a clear blue sky",
    "An enchanted old-growth forest with sunbeams filtering through ancient moss-covered trees",
    "A peaceful birch forest in autumn with golden leaves carpeting the ground and soft morning light",
    "Dense bamboo forest with shafts of light creating patterns on the forest floor",
    "A redwood forest scene with massive tree trunks and ferns in the misty understory",
    "A winding path through a magical forest with colorful wildflowers and dappled sunlight",
    "A dramatic seascape with waves crashing against rocky cliffs at sunset with vibrant orange and purple sky",
    "A tranquil tropical beach with crystal clear turquoise water and palm trees swaying in the breeze",
    "Rugged coastal cliffs overlooking a deep blue ocean with seabirds soaring overhead",
    "A peaceful cove with emerald green water surrounded by limestone karst formations",
    "Golden hour over a calm ocean with gentle waves lapping on a sandy shore",
    "A vast desert landscape with towering red sandstone formations under a star-filled night sky",
    "Grand Canyon at sunset with layers of colorful rock strata and deep shadows",
    "Rolling sand dunes in the Sahara desert with beautiful patterns created by wind",
    "A desert oasis with palm trees and a small pond surrounded by golden sand dunes",
    "Dramatic desert mesa formations with a thunderstorm approaching in the distance",
    "A mirror-like lake at dawn reflecting autumn trees and mountains in perfect symmetry",
    "A cascading waterfall flowing into a turquoise pool surrounded by lush tropical vegetation",
    "A gentle river winding through a valley of wildflowers with mountains in the distance",
    "A frozen lake surrounded by snow-covered pine trees under northern lights aurora borealis",
    "A peaceful pond with lily pads and lotus flowers surrounded by weeping willows at sunset",
    "Endless lavender fields in Provence stretching to the horizon under a golden sunset sky",
    "A rolling countryside with green hills, stone walls, and scattered wildflowers in spring",
    "A sunflower field at golden hour with warm light illuminating thousands of blooms",
    "Terraced rice paddies reflecting the sunset sky with mountains in the background",
    "A vast prairie grassland with dramatic cumulus clouds and golden afternoon light",
    "A frozen waterfall with blue ice formations and snow-covered evergreen trees",
    "A winter wonderland with fresh snow covering a peaceful village and surrounding mountains",
    "Ice caves with stunning blue ice formations and light filtering through from above",
    "A snow-covered mountain pass with a lone traveler and dramatic cloud formations",
    "Arctic tundra landscape with icebergs floating in calm waters under midnight sun",
    "Lush tropical rainforest with a hidden waterfall and exotic birds and butterflies",
    "A volcanic landscape with lava fields and a smoking crater under dramatic skies",
    "Terraced hillsides with tea plantations stretching into misty mountain valleys",
    "A coral reef visible through crystal clear shallow water with a tropical island behind",
    "Dense jungle canopy seen from above with a river snaking through the green expanse",
    "A dramatic sunset over rolling hills with layers of clouds painted in red and gold",
    "A double rainbow arching over a lush green valley after a summer rainstorm",
    "Northern lights dancing over a snowy mountain landscape with a clear starry sky",
    "Dramatic cumulonimbus clouds towering over a peaceful countryside at golden hour",
    "A foggy morning in a valley with treetops emerging from the mist like islands",
    "A traditional Japanese zen garden with raked gravel, moss, and carefully pruned trees",
    "An English cottage garden overflowing with roses, foxgloves, and delphiniums in summer",
    "A cherry blossom avenue in full bloom with pink petals falling like snow",
    "A formal French garden with geometric hedges, fountains, and symmetrical flower beds",
    "A wild garden meadow with poppies, cornflowers, and butterflies in warm afternoon light",
    "Dramatic fjord landscape with steep cliffs and deep blue water under overcast skies",
    "A vast savanna with acacia trees silhouetted against a spectacular African sunset",
    "Limestone karst mountains rising from emerald rice paddies in morning mist",
    "A hidden valley with waterfalls cascading down mossy cliffs into a turquoise pool",
    "Ancient sequoia grove with massive trees and dappled sunlight on the forest floor",
    "A coastal marsh at low tide with golden grasses and reflective tidal pools",
    "Snow-dusted volcanic peaks rising above a sea of clouds at sunrise",
    "A winding mountain road through autumn foliage with fog in the valley below",
    "Glacial valley with U-shaped walls, a braided river, and distant ice fields",
    "A Mediterranean hillside village surrounded by olive groves and cypress trees at sunset",
    "Turquoise hot springs terraces with steam rising against a backdrop of mountains",
    "A field of tulips in perfect rows stretching to a windmill on the horizon",
    "Dramatic badlands erosion patterns in striped sedimentary rock under a stormy sky",
    "A peaceful monastery perched on a cliff overlooking a misty mountain valley",
    "Bioluminescent bay at night with glowing blue water and a starry sky above",
    "A sprawling vineyard in autumn with golden and red leaves and rolling hills beyond",
    "Crystal clear cenote with light beams penetrating deep turquoise water surrounded by jungle",
    "A vast flower field with mixed wildflowers stretching to distant blue mountains",
    "Majestic redrock canyon with a narrow slot revealing blue sky above",
    "A tranquil bamboo grove with a stone path and soft diffused green light",
    "Spectacular icefall on a glacier with blue crevasses and mountain peaks behind",
    "A highland plateau with grazing yaks and prayer flags against Himalayan peaks",
    "Tropical waterfall surrounded by giant ferns and orchids in a cloud forest",
    "A peaceful countryside with a stone bridge over a stream and rolling green hills",
    "Dramatic sea stacks and arches along a rugged coastline at low tide sunset",
    "An autumn forest reflected in a perfectly still beaver pond at dawn",
    "A desert canyon at night with the Milky Way spanning the sky overhead",
    "Lush terraced hillsides with cascading waterfalls in a tropical paradise",
    "A snow-covered boreal forest stretching to the horizon under pink twilight sky",
    "Dramatic thunder clouds over a golden wheat field with a lone tree",
    "A coral atoll seen from above with rings of turquoise and deep blue water",
    "Misty Scottish highlands with heather-covered hills and a distant loch",
    "A bamboo raft on a emerald river winding through towering limestone peaks",
    "Spring cherry blossoms framing a view of a snow-capped volcano",
    "An ancient stone circle on a windswept moorland under dramatic skies",
    "A tropical sunset over calm waters with silhouetted palm trees and fishing boats",
    "Vast salt flats creating a perfect mirror reflection of clouds and mountains",
    "A hidden grotto with crystal stalactites reflecting in an underground lake",
    "Rolling Tuscan hills with cypress-lined roads and golden wheat fields at sunset",
    "A dramatic waterfall plunging into a deep gorge surrounded by rainforest",
    "Arctic ice shelf edge with turquoise icebergs and a polar bear in the distance",
]

def get_prompt(index):
    return LANDSCAPE_PROMPTS[index % len(LANDSCAPE_PROMPTS)]

def get_prompts(start, end):
    return [get_prompt(i) for i in range(start, end)]
PROMPTS_EOF

# --- experiment/configs.py ---
cat > "$PROJECT_DIR/experiment/configs.py" << 'CONFIGS_EOF'
"""Experiment configurations for baseline vs optimized comparison."""
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class WatermarkConfig:
    w_seed: int = 999999
    w_channel: int = 3
    w_pattern: str = "ring"
    w_mask_shape: str = "circle"
    w_radius: int = 10
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
CONFIGS_EOF

# --- experiment/device_compat.py ---
cat > "$PROJECT_DIR/experiment/device_compat.py" << 'COMPAT_EOF'
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
    return device == "mps"

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
COMPAT_EOF

# --- experiment/run_baseline.py ---
cat > "$PROJECT_DIR/experiment/run_baseline.py" << 'BASELINE_EOF'
"""Baseline runner: paper-exact Tree-Ring watermark (Wen et al. 2023).

Follows the paper exactly:
  - Generate AI images from Gustavosta/Stable-Diffusion-Prompts
  - DPMSolverMultistepScheduler, 50 steps, guidance_scale=7.5
  - Embed: inject ring watermark into random noise z_T, then run SD generation
  - Detect: DDIM inversion (empty prompt, guidance_scale=1) -> L1 vs key
  - Attacks: rotation 75, JPEG q=25, crop 0.75, blur r=8, noise 0.1, brightness 6
"""
import sys, json, time, copy, argparse
from pathlib import Path
from tqdm import tqdm
from statistics import mean
from sklearn import metrics
import torch, numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent / "tree-ring-watermark"
sys.path.insert(0, str(REPO_ROOT))
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import (set_random_seed, transform_img, get_watermarking_mask,
                         measure_similarity, get_dataset)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_compat import get_device, get_watermarking_pattern, inject_watermark, eval_watermark, image_distortion
from configs import ExperimentConfig, AttackConfig


def build_args(config, attack):
    a = argparse.Namespace()
    a.model_id=config.model_id; a.image_length=config.image_length; a.num_images=config.num_images
    a.guidance_scale=config.guidance_scale; a.num_inference_steps=config.num_inference_steps
    a.test_num_inference_steps=config.test_num_inference_steps; a.gen_seed=config.gen_seed
    a.start=config.start; a.end=config.end; a.run_name=f"{config.name}_{attack.name}"
    a.with_tracking=False; a.max_num_log_image=100
    wm=config.watermark
    a.w_seed=wm.w_seed; a.w_channel=wm.w_channel; a.w_pattern=wm.w_pattern
    a.w_mask_shape=wm.w_mask_shape; a.w_radius=wm.w_radius; a.w_measurement=wm.w_measurement
    a.w_injection=wm.w_injection; a.w_pattern_const=wm.w_pattern_const
    a.r_degree=attack.r_degree; a.jpeg_ratio=attack.jpeg_ratio; a.crop_scale=attack.crop_scale
    a.crop_ratio=attack.crop_ratio; a.gaussian_blur_r=attack.gaussian_blur_r
    a.gaussian_std=attack.gaussian_std; a.brightness_factor=attack.brightness_factor; a.rand_aug=attack.rand_aug
    a.dataset="Gustavosta/Stable-Diffusion-Prompts"
    a.reference_model=config.reference_model; a.reference_model_pretrain=config.reference_model_pretrain
    return a


def load_ckpt(p):
    if p.exists():
        with open(p) as f: return json.load(f)
    return None


def save_ckpt(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f: json.dump(data, f, indent=2)


def run_baseline(config, skip_clip=False):
    device = get_device()

    sep = "="*60
    print(f"\n{sep}\nBASELINE: {config.name} | Device: {device} | Images: {config.start}-{config.end}")
    print(f"Watermark: ch={config.watermark.w_channel}, pattern={config.watermark.w_pattern}, r={config.watermark.w_radius}\n{sep}\n")

    base_dir = Path(__file__).resolve().parent
    ckpt_dir = base_dir/config.checkpoint_dir/config.name
    res_dir  = base_dir/config.results_dir/config.name
    out_dir  = base_dir/config.outputs_dir/config.name
    for d in [ckpt_dir, res_dir, out_dir]: d.mkdir(parents=True, exist_ok=True)

    args = build_args(config, config.attacks[0] if config.attacks else AttackConfig())

    print("Loading Stable Diffusion pipeline...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, scheduler=scheduler, torch_dtype=torch.float16,
        safety_checker=None)
    pipe = pipe.to(device)

    ref_model = ref_clip_preprocess = ref_tokenizer = None
    if not skip_clip and config.reference_model:
        print(f"Loading CLIP: {config.reference_model}...")
        import open_clip
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            config.reference_model, pretrained=config.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(config.reference_model)

    # Detection always uses empty prompt (prompt unknown at detection time — paper Section 3)
    text_embeddings = pipe.get_text_embedding("")
    gt_patch = get_watermarking_pattern(pipe, args, device)

    print("Loading Gustavosta/Stable-Diffusion-Prompts dataset...")
    dataset, prompt_key = get_dataset(args)

    all_results = {}
    for attack in config.attacks:
        args_a = build_args(config, attack)
        ckpt_path = ckpt_dir/f"{attack.name}_checkpoint.json"
        ckpt = load_ckpt(ckpt_path)

        if ckpt and ckpt.get("completed"):
            print(f"  {attack.name}: already done, loading.")
            all_results[attack.name] = ckpt["final_results"]; continue

        start_idx = config.start
        results=[]; no_w_metrics=[]; w_metrics=[]; clips=[]; clips_w=[]; saved=[]
        if ckpt and not ckpt.get("completed"):
            start_idx = ckpt["last_idx"] + 1
            results      = ckpt.get("results", [])
            no_w_metrics = ckpt.get("no_w_metrics", [])
            w_metrics    = ckpt.get("w_metrics", [])
            clips        = ckpt.get("clips", [])
            clips_w      = ckpt.get("clips_w", [])
            saved        = ckpt.get("saved", [])
            print(f"  {attack.name}: resuming from {start_idx}")

        t0 = time.time()
        for i in tqdm(range(start_idx, config.end), desc=f"  {attack.name}"):
            seed = i + config.gen_seed
            current_prompt = dataset[i][prompt_key]

            # --- Paper approach: AI image generation from watermarked noise ---
            set_random_seed(seed)
            init_latents_no_w = pipe.get_random_latents()

            # Non-watermarked image
            outputs_no_w = pipe(
                current_prompt, num_images_per_prompt=1,
                guidance_scale=args_a.guidance_scale,
                num_inference_steps=args_a.num_inference_steps,
                height=args_a.image_length, width=args_a.image_length,
                latents=init_latents_no_w)
            img_no_w = outputs_no_w.images[0]

            # Watermarked image: inject ring into same noise, re-generate
            init_latents_w = copy.deepcopy(init_latents_no_w)
            mask = get_watermarking_mask(init_latents_w, args_a, device)
            init_latents_w = inject_watermark(init_latents_w, mask, gt_patch, args_a)
            outputs_w = pipe(
                current_prompt, num_images_per_prompt=1,
                guidance_scale=args_a.guidance_scale,
                num_inference_steps=args_a.num_inference_steps,
                height=args_a.image_length, width=args_a.image_length,
                latents=init_latents_w)
            img_w = outputs_w.images[0]

            # Apply attack
            img_no_w_a, img_w_a = image_distortion(img_no_w, img_w, seed, args_a)

            # Save first 5 pairs
            if len(saved) < 5:
                d = out_dir/attack.name; d.mkdir(parents=True, exist_ok=True)
                img_no_w.save(d/f"img_{i:04d}_no_wm.png")
                img_w.save(d/f"img_{i:04d}_wm.png")
                img_no_w_a.save(d/f"img_{i:04d}_no_wm_attacked.png")
                img_w_a.save(d/f"img_{i:04d}_wm_attacked.png")
                saved.append(i)

            # --- Detection: DDIM inversion with empty prompt, guidance_scale=1 ---
            t_no = transform_img(img_no_w_a).unsqueeze(0).to(text_embeddings.dtype).to(device)
            rev_no = pipe.forward_diffusion(
                latents=pipe.get_image_latents(t_no, sample=False),
                text_embeddings=text_embeddings, guidance_scale=1,
                num_inference_steps=args_a.test_num_inference_steps)

            t_w = transform_img(img_w_a).unsqueeze(0).to(text_embeddings.dtype).to(device)
            rev_w = pipe.forward_diffusion(
                latents=pipe.get_image_latents(t_w, sample=False),
                text_embeddings=text_embeddings, guidance_scale=1,
                num_inference_steps=args_a.test_num_inference_steps)

            nm, wm_ = eval_watermark(rev_no, rev_w, mask, gt_patch, args_a)
            cs_no = cs_w = 0.0
            if ref_model:
                sims = measure_similarity([img_no_w, img_w], current_prompt,
                                          ref_model, ref_clip_preprocess, ref_tokenizer, device)
                cs_no = sims[0].item(); cs_w = sims[1].item()

            results.append({"i": i, "prompt": current_prompt, "no_w": nm, "w": wm_,
                            "clip_no": cs_no, "clip_w": cs_w})
            no_w_metrics.append(-nm); w_metrics.append(-wm_)
            clips.append(cs_no); clips_w.append(cs_w)

            if (i - config.start + 1) % config.checkpoint_every == 0 or i == config.end - 1:
                save_ckpt(ckpt_path, {"attack": attack.name, "last_idx": i, "completed": False,
                    "results": results, "no_w_metrics": no_w_metrics, "w_metrics": w_metrics,
                    "clips": clips, "clips_w": clips_w, "saved": saved})

        elapsed = time.time() - t0
        preds  = no_w_metrics + w_metrics
        labels = [0]*len(no_w_metrics) + [1]*len(w_metrics)
        fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        low = tpr[np.where(fpr < .01)[0][-1]] if len(np.where(fpr < .01)[0]) > 0 else 0.0

        final = {"attack": attack.name, "approach": "baseline", "num_images": len(no_w_metrics),
            "auc": float(auc), "acc": float(acc), "tpr_at_1fpr": float(low),
            "mean_no_w_metric": float(mean([-m for m in no_w_metrics])),
            "mean_w_metric":    float(mean([-m for m in w_metrics])),
            "clip_score_mean":   float(mean(clips))   if clips   and any(c != 0 for c in clips)   else None,
            "clip_score_w_mean": float(mean(clips_w)) if clips_w and any(c != 0 for c in clips_w) else None,
            "elapsed_seconds": elapsed,
            "watermark_params": {"w_channel": config.watermark.w_channel,
                                 "w_pattern": config.watermark.w_pattern,
                                 "w_radius":  config.watermark.w_radius}}
        all_results[attack.name] = final
        save_ckpt(ckpt_path, {"attack": attack.name, "last_idx": config.end - 1, "completed": True,
            "results": results, "no_w_metrics": no_w_metrics, "w_metrics": w_metrics,
            "clips": clips, "clips_w": clips_w, "saved": saved, "final_results": final})
        print(f"  AUC:{auc:.4f} Acc:{acc:.4f} TPR@1%FPR:{low:.4f} Time:{elapsed:.1f}s")

    with open(res_dir/"all_attacks_results.json", "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nBaseline results: {res_dir/'all_attacks_results.json'}")
    return all_results


if __name__ == "__main__":
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument("--scale", default="small")
    p.add_argument("--skip_clip", action="store_true")
    a = p.parse_args()
    from configs import get_small_scale_baseline, get_large_scale_baseline
    run_baseline(get_small_scale_baseline() if a.scale == "small" else get_large_scale_baseline(),
                 skip_clip=a.skip_clip)

BASELINE_EOF

# --- experiment/run_optimized.py ---
cat > "$PROJECT_DIR/experiment/run_optimized.py" << 'OPTIMIZED_EOF'
"""Optimized runner: 100 DDIM steps for more accurate watermark detection.

Key optimization over baseline: 100 inference steps (vs 50).

Why more steps helps:
  DDIM inversion (detection) approximates the noise z_T from a watermarked image.
  Each step introduces approximation error. With 50 steps the error compounds;
  with 100 steps each step is smaller -> z_T is closer to the true watermarked noise
  -> the ring pattern is better recovered -> higher AUC/TPR under attacks.

Both generation and detection use 100 steps for consistency.
"""
import sys, json, time, copy, argparse
from pathlib import Path
from tqdm import tqdm
from statistics import mean
from sklearn import metrics
import torch, numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent / "tree-ring-watermark"
sys.path.insert(0, str(REPO_ROOT))
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import (set_random_seed, transform_img, get_watermarking_mask,
                         measure_similarity, get_dataset)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_compat import get_device, get_watermarking_pattern, inject_watermark, eval_watermark, image_distortion
from configs import ExperimentConfig, AttackConfig


def build_args(config, attack):
    a = argparse.Namespace()
    a.model_id=config.model_id; a.image_length=config.image_length; a.num_images=config.num_images
    a.guidance_scale=config.guidance_scale; a.num_inference_steps=config.num_inference_steps
    a.test_num_inference_steps=config.test_num_inference_steps; a.gen_seed=config.gen_seed
    a.start=config.start; a.end=config.end; a.run_name=f"{config.name}_{attack.name}"
    a.with_tracking=False; a.max_num_log_image=100
    wm=config.watermark
    a.w_seed=wm.w_seed; a.w_channel=wm.w_channel; a.w_pattern=wm.w_pattern
    a.w_mask_shape=wm.w_mask_shape; a.w_radius=wm.w_radius; a.w_measurement=wm.w_measurement
    a.w_injection=wm.w_injection; a.w_pattern_const=wm.w_pattern_const
    a.r_degree=attack.r_degree; a.jpeg_ratio=attack.jpeg_ratio; a.crop_scale=attack.crop_scale
    a.crop_ratio=attack.crop_ratio; a.gaussian_blur_r=attack.gaussian_blur_r
    a.gaussian_std=attack.gaussian_std; a.brightness_factor=attack.brightness_factor; a.rand_aug=attack.rand_aug
    a.dataset="Gustavosta/Stable-Diffusion-Prompts"
    a.reference_model=config.reference_model; a.reference_model_pretrain=config.reference_model_pretrain
    return a


def load_ckpt(p):
    if p.exists():
        with open(p) as f: return json.load(f)
    return None


def save_ckpt(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f: json.dump(data, f, indent=2)


def run_optimized(config, skip_clip=False):
    device = get_device()

    sep = "="*60
    print(f"\n{sep}\nOPTIMIZED: {config.name} | Device: {device} | Images: {config.start}-{config.end}")
    print(f"Watermark: ch={config.watermark.w_channel}, pattern={config.watermark.w_pattern}, r={config.watermark.w_radius}")
    print(f"Optimization: {config.num_inference_steps} steps (vs baseline 50)\n{sep}\n")

    base_dir = Path(__file__).resolve().parent
    ckpt_dir = base_dir/config.checkpoint_dir/config.name
    res_dir  = base_dir/config.results_dir/config.name
    out_dir  = base_dir/config.outputs_dir/config.name
    for d in [ckpt_dir, res_dir, out_dir]: d.mkdir(parents=True, exist_ok=True)

    args = build_args(config, config.attacks[0] if config.attacks else AttackConfig())

    print("Loading Stable Diffusion pipeline...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, scheduler=scheduler, torch_dtype=torch.float16,
        safety_checker=None)
    pipe = pipe.to(device)
    try: pipe.enable_xformers_memory_efficient_attention(); print("  xformers enabled")
    except: pass

    ref_model = ref_clip_preprocess = ref_tokenizer = None
    if not skip_clip and config.reference_model:
        print(f"Loading CLIP: {config.reference_model}...")
        import open_clip
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            config.reference_model, pretrained=config.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(config.reference_model)

    # Detection always uses empty prompt (prompt unknown at detection time — paper Section 3)
    text_embeddings = pipe.get_text_embedding("")
    gt_patch = get_watermarking_pattern(pipe, args, device)

    print("Loading Gustavosta/Stable-Diffusion-Prompts dataset...")
    dataset, prompt_key = get_dataset(args)

    all_results = {}
    for attack in config.attacks:
        args_a = build_args(config, attack)
        ckpt_path = ckpt_dir/f"{attack.name}_checkpoint.json"
        ckpt = load_ckpt(ckpt_path)

        if ckpt and ckpt.get("completed"):
            print(f"  {attack.name}: already done, loading.")
            all_results[attack.name] = ckpt["final_results"]; continue

        start_idx = config.start
        results=[]; no_w_metrics=[]; w_metrics=[]; clips=[]; clips_w=[]; saved=[]
        if ckpt and not ckpt.get("completed"):
            start_idx = ckpt["last_idx"] + 1
            results      = ckpt.get("results", [])
            no_w_metrics = ckpt.get("no_w_metrics", [])
            w_metrics    = ckpt.get("w_metrics", [])
            clips        = ckpt.get("clips", [])
            clips_w      = ckpt.get("clips_w", [])
            saved        = ckpt.get("saved", [])
            print(f"  {attack.name}: resuming from {start_idx}")

        t0 = time.time()
        for i in tqdm(range(start_idx, config.end), desc=f"  {attack.name}"):
            seed = i + config.gen_seed
            current_prompt = dataset[i][prompt_key]

            # --- Paper approach: AI image generation from watermarked noise ---
            set_random_seed(seed)
            init_latents_no_w = pipe.get_random_latents()

            # Non-watermarked image (100 steps)
            outputs_no_w = pipe(
                current_prompt, num_images_per_prompt=1,
                guidance_scale=args_a.guidance_scale,
                num_inference_steps=args_a.num_inference_steps,
                height=args_a.image_length, width=args_a.image_length,
                latents=init_latents_no_w)
            img_no_w = outputs_no_w.images[0]

            # Watermarked image: inject ring into same noise, re-generate (100 steps)
            init_latents_w = copy.deepcopy(init_latents_no_w)
            mask = get_watermarking_mask(init_latents_w, args_a, device)
            init_latents_w = inject_watermark(init_latents_w, mask, gt_patch, args_a)
            outputs_w = pipe(
                current_prompt, num_images_per_prompt=1,
                guidance_scale=args_a.guidance_scale,
                num_inference_steps=args_a.num_inference_steps,
                height=args_a.image_length, width=args_a.image_length,
                latents=init_latents_w)
            img_w = outputs_w.images[0]

            # Apply attack
            img_no_w_a, img_w_a = image_distortion(img_no_w, img_w, seed, args_a)

            # Save first 5 pairs
            if len(saved) < 5:
                d = out_dir/attack.name; d.mkdir(parents=True, exist_ok=True)
                img_no_w.save(d/f"img_{i:04d}_no_wm.png")
                img_w.save(d/f"img_{i:04d}_wm.png")
                img_no_w_a.save(d/f"img_{i:04d}_no_wm_attacked.png")
                img_w_a.save(d/f"img_{i:04d}_wm_attacked.png")
                saved.append(i)

            # --- Detection: DDIM inversion with empty prompt, guidance_scale=1 (100 steps) ---
            t_no = transform_img(img_no_w_a).unsqueeze(0).to(text_embeddings.dtype).to(device)
            rev_no = pipe.forward_diffusion(
                latents=pipe.get_image_latents(t_no, sample=False),
                text_embeddings=text_embeddings, guidance_scale=1,
                num_inference_steps=args_a.test_num_inference_steps)

            t_w = transform_img(img_w_a).unsqueeze(0).to(text_embeddings.dtype).to(device)
            rev_w = pipe.forward_diffusion(
                latents=pipe.get_image_latents(t_w, sample=False),
                text_embeddings=text_embeddings, guidance_scale=1,
                num_inference_steps=args_a.test_num_inference_steps)

            nm, wm_ = eval_watermark(rev_no, rev_w, mask, gt_patch, args_a)
            cs_no = cs_w = 0.0
            if ref_model:
                sims = measure_similarity([img_no_w, img_w], current_prompt,
                                          ref_model, ref_clip_preprocess, ref_tokenizer, device)
                cs_no = sims[0].item(); cs_w = sims[1].item()

            results.append({"i": i, "prompt": current_prompt, "no_w": nm, "w": wm_,
                            "clip_no": cs_no, "clip_w": cs_w})
            no_w_metrics.append(-nm); w_metrics.append(-wm_)
            clips.append(cs_no); clips_w.append(cs_w)

            if (i - config.start + 1) % config.checkpoint_every == 0 or i == config.end - 1:
                save_ckpt(ckpt_path, {"attack": attack.name, "last_idx": i, "completed": False,
                    "results": results, "no_w_metrics": no_w_metrics, "w_metrics": w_metrics,
                    "clips": clips, "clips_w": clips_w, "saved": saved})

        elapsed = time.time() - t0
        preds  = no_w_metrics + w_metrics
        labels = [0]*len(no_w_metrics) + [1]*len(w_metrics)
        fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        low = tpr[np.where(fpr < .01)[0][-1]] if len(np.where(fpr < .01)[0]) > 0 else 0.0

        final = {"attack": attack.name, "approach": "optimized", "num_images": len(no_w_metrics),
            "auc": float(auc), "acc": float(acc), "tpr_at_1fpr": float(low),
            "mean_no_w_metric": float(mean([-m for m in no_w_metrics])),
            "mean_w_metric":    float(mean([-m for m in w_metrics])),
            "clip_score_mean":   float(mean(clips))   if clips   and any(c != 0 for c in clips)   else None,
            "clip_score_w_mean": float(mean(clips_w)) if clips_w and any(c != 0 for c in clips_w) else None,
            "elapsed_seconds": elapsed,
            "watermark_params": {"w_channel": config.watermark.w_channel,
                                 "w_pattern": config.watermark.w_pattern,
                                 "w_radius":  config.watermark.w_radius},
            "optimizations": {"embed_steps": config.num_inference_steps,
                              "detect_steps": config.test_num_inference_steps}}
        all_results[attack.name] = final
        save_ckpt(ckpt_path, {"attack": attack.name, "last_idx": config.end - 1, "completed": True,
            "results": results, "no_w_metrics": no_w_metrics, "w_metrics": w_metrics,
            "clips": clips, "clips_w": clips_w, "saved": saved, "final_results": final})
        print(f"  AUC:{auc:.4f} Acc:{acc:.4f} TPR@1%FPR:{low:.4f} Time:{elapsed:.1f}s")

    with open(res_dir/"all_attacks_results.json", "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nOptimized results: {res_dir/'all_attacks_results.json'}")
    return all_results


if __name__ == "__main__":
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument("--scale", default="small")
    p.add_argument("--skip_clip", action="store_true")
    a = p.parse_args()
    from configs import get_small_scale_optimized, get_large_scale_optimized
    run_optimized(get_small_scale_optimized() if a.scale == "small" else get_large_scale_optimized(),
                  skip_clip=a.skip_clip)

OPTIMIZED_EOF

# --- experiment/compare_results.py ---
cat > "$PROJECT_DIR/experiment/compare_results.py" << 'COMPARE_EOF'
"""Compare baseline vs optimized results. Prints table, saves CSV + plots."""
import json, csv, sys
from pathlib import Path
from typing import Dict

def load_results(d):
    p = d/"all_attacks_results.json"
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)

def fmt(v, f=".4f"):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:{f}}"
    return str(v)

def compare(scale="small"):
    base = Path(__file__).resolve().parent/"results"
    bl = load_results(base/f"{scale}_baseline")
    op = load_results(base/f"{scale}_optimized")
    if not bl and not op: print(f"No results for {scale}. Run experiments first."); return

    print(f"\n{'='*100}\n  COMPARISON: BASELINE vs OPTIMIZED  ({scale.upper()} SCALE)\n{'='*100}")
    print(f"\n{'Attack':<22} | {'Metric':<16} | {'Baseline':>10} | {'Optimized':>10} | {'Delta':>10} | {'Winner':>8}")
    print("-"*100)

    attacks = sorted(set(list(bl.keys())+list(op.keys())))
    ms = [("auc","AUC",True),("acc","Accuracy",True),("tpr_at_1fpr","TPR@1%FPR",True),
          ("mean_w_metric","W Metric",False),("clip_score_w_mean","CLIP (W)",True)]
    summary = {"opt":0,"base":0,"tie":0}

    for atk in attacks:
        b,o = bl.get(atk,{}), op.get(atk,{})
        for mk,mn,hb in ms:
            bv,ov = b.get(mk), o.get(mk)
            if bv is None and ov is None: continue
            bs,os_ = fmt(bv), fmt(ov)
            if bv is not None and ov is not None:
                d=ov-bv; ds=f"{d:+.4f}"
                w="OPT" if (d>0.001 if hb else d<-0.001) else ("BASE" if (d<-0.001 if hb else d>0.001) else "TIE")
                summary["opt" if w=="OPT" else "base" if w=="BASE" else "tie"]+=1
            else: ds="N/A"; w="N/A"
            print(f"{atk:<22} | {mn:<16} | {bs:>10} | {os_:>10} | {ds:>10} | {w:>8}")
        bt,ot = b.get("elapsed_seconds"), o.get("elapsed_seconds")
        if bt and ot: print(f"{atk:<22} | {'Time (s)':<16} | {bt:>10.1f} | {ot:>10.1f} | {bt/ot if ot else 0:>9.2f}x | {'---':>8}")
        print("-"*100)

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  Optimized wins: {summary['opt']}\n  Baseline wins:  {summary['base']}\n  Ties:           {summary['tie']}")
    if bl:
        p=next(iter(bl.values())).get("watermark_params",{})
        print(f"\n  Baseline:  ch={p.get('w_channel')}, pattern={p.get('w_pattern')}, radius={p.get('w_radius')}")
    if op:
        p=next(iter(op.values())).get("watermark_params",{}); o_=next(iter(op.values())).get("optimizations",{})
        print(f"  Optimized: ch={p.get('w_channel')}, pattern={p.get('w_pattern')}, radius={p.get('w_radius')}")
        print(f"  Opts:      embed_steps={o_.get('embed_steps')}, detect_steps={o_.get('detect_steps')}")

    # CSV
    csv_path = base/f"{scale}_comparison.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["Attack","Approach","AUC","Accuracy","TPR@1%FPR","Mean_NoW","Mean_W","CLIP_NoW","CLIP_W","Time_s"])
        for atk in attacks:
            for ap_,res in [("baseline",bl),("optimized",op)]:
                r=res.get(atk,{});
                if not r: continue
                w.writerow([atk,ap_,fmt(r.get("auc")),fmt(r.get("acc")),fmt(r.get("tpr_at_1fpr")),
                    fmt(r.get("mean_no_w_metric")),fmt(r.get("mean_w_metric")),
                    fmt(r.get("clip_score_mean")),fmt(r.get("clip_score_w_mean")),fmt(r.get("elapsed_seconds"),".1f")])
    print(f"\n  CSV: {csv_path}")

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pd = base/f"{scale}_plots"; pd.mkdir(parents=True, exist_ok=True)
        for mk,mn in [("auc","AUC"),("acc","Accuracy"),("tpr_at_1fpr","TPR@1%FPR")]:
            bv=[bl.get(a,{}).get(mk,0) or 0 for a in attacks]
            ov=[op.get(a,{}).get(mk,0) or 0 for a in attacks]
            x=range(len(attacks)); w_=0.35
            fig,ax=plt.subplots(figsize=(max(10,len(attacks)*1.5),6))
            ax.bar([i-w_/2 for i in x],bv,w_,label="Baseline",color="#4C72B0")
            ax.bar([i+w_/2 for i in x],ov,w_,label="Optimized",color="#DD8452")
            ax.set_ylabel(mn); ax.set_title(f"{mn}: Baseline vs Optimized")
            ax.set_xticks(list(x)); ax.set_xticklabels(attacks,rotation=45,ha="right")
            ax.legend(); ax.set_ylim(0,1.05); plt.tight_layout()
            plt.savefig(pd/f"comparison_{mk}.png",dpi=150); plt.close()
        print(f"  Plots: {pd}")
    except ImportError: print("  matplotlib not available, skipping plots.")
    print()

if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("--scale",default="small"); a=p.parse_args()
    if a.scale=="both": compare("small"); compare("large")
    else: compare(a.scale)
COMPARE_EOF

# --- experiment/run_experiment.py ---
cat > "$PROJECT_DIR/experiment/run_experiment.py" << 'ENTRY_EOF'
#!/usr/bin/env python3
"""Main entry point. Runs baseline, optimized, or both, then compares."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import (get_small_scale_baseline, get_small_scale_optimized,
                     get_large_scale_baseline, get_large_scale_optimized)
from run_baseline import run_baseline
from run_optimized import run_optimized
from compare_results import compare

def main():
    p = argparse.ArgumentParser(description="Tree-Ring Watermark: Baseline vs Optimized")
    p.add_argument("--scale", choices=["small","large"], default="small")
    p.add_argument("--mode", choices=["baseline","optimized","both","compare"], default="both")
    p.add_argument("--skip_clip", action="store_true")
    a = p.parse_args()

    bl = get_small_scale_baseline() if a.scale=="small" else get_large_scale_baseline()
    op = get_small_scale_optimized() if a.scale=="small" else get_large_scale_optimized()

    if a.mode in ("baseline","both"): run_baseline(bl, skip_clip=a.skip_clip)
    if a.mode in ("optimized","both"): run_optimized(op, skip_clip=a.skip_clip)
    if a.mode in ("compare","both"): compare(a.scale)

if __name__=="__main__": main()
ENTRY_EOF

echo "  All experiment files created."
echo ""

# =============================================================================
# STEP 5: Setup Poetry env + install deps
# =============================================================================
echo "============================================"
echo "STEP 5/9: Installing Python dependencies..."
echo "============================================"

poetry env use "$PYTHON_BIN" 2>&1 | grep -v "^$" || true
# Remove stale lock file so Poetry resolves fresh from pyproject.toml
rm -f "$PROJECT_DIR/poetry.lock"
poetry install --no-root 2>&1 | tail -5
echo "  Dependencies installed."
echo ""

# Verify
poetry run python -c "
import warnings; warnings.filterwarnings('ignore')
import torch, diffusers, transformers
print(f'  torch:        {torch.__version__}')
print(f'  diffusers:    {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')
d = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'  device:       {d}')
"
echo ""

# =============================================================================
# STEP 6: Download model
# =============================================================================
echo "============================================"
echo "STEP 6/9: Checking Stable Diffusion model..."
echo "============================================"

poetry run python -c "
import warnings; warnings.filterwarnings('ignore')
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
model_id = 'runwayml/stable-diffusion-v1-5'
try:
    DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler', local_files_only=True)
    StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
    print('  Model already cached.')
except:
    print('  Downloading model (~5GB, one-time)...')
    DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
    StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    print('  Model downloaded.')
"
echo ""

# =============================================================================
# STEP 7: Run Baseline
# =============================================================================
echo "============================================"
echo "STEP 7/9: Running BASELINE experiment..."
echo "============================================"

poetry run python experiment/run_experiment.py --scale "$SCALE" --mode baseline $SKIP_CLIP_FLAG
echo ""

# =============================================================================
# STEP 8: Run Optimized
# =============================================================================
echo "============================================"
echo "STEP 8/9: Running OPTIMIZED experiment..."
echo "============================================"

poetry run python experiment/run_experiment.py --scale "$SCALE" --mode optimized $SKIP_CLIP_FLAG
echo ""

# =============================================================================
# STEP 9: Compare
# =============================================================================
echo "============================================"
echo "STEP 9/9: Comparing results..."
echo "============================================"

poetry run python experiment/run_experiment.py --scale "$SCALE" --mode compare

echo ""
echo "########################################################"
echo "#  ALL DONE!"
echo "#"
echo "#  Results:     experiment/results/${SCALE}_comparison.csv"
echo "#  Plots:       experiment/results/${SCALE}_plots/"
echo "#  Images:      experiment/outputs/"
echo "#  Checkpoints: experiment/checkpoints/"
echo "########################################################"
echo ""
