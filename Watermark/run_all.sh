#!/usr/bin/env bash
# =============================================================================
# Tree-Ring Watermark: Full Pipeline Script
# One command to setup, run baseline, run optimized, and compare.
#
# Usage:
#   ./run_all.sh                    # small scale, skip CLIP (fastest)
#   ./run_all.sh small              # small scale, skip CLIP
#   ./run_all.sh small --with-clip  # small scale, with CLIP scores
#   ./run_all.sh large              # large scale, skip CLIP
#   ./run_all.sh large --with-clip  # large scale, with CLIP scores (slowest)
# =============================================================================
set -euo pipefail

# Allow MPS to use all available memory (prevents OOM on large models)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parse arguments ---
SCALE="${1:-small}"
WITH_CLIP="${2:-}"
SKIP_CLIP_FLAG="--skip_clip"
if [[ "$WITH_CLIP" == "--with-clip" ]]; then
    SKIP_CLIP_FLAG=""
fi

echo ""
echo "########################################################"
echo "#  Tree-Ring Watermark: Baseline vs Optimized Pipeline"
echo "#  Scale: $SCALE"
echo "#  CLIP scores: $([ -z "$SKIP_CLIP_FLAG" ] && echo 'YES' || echo 'SKIPPED')"
echo "#  Working dir: $SCRIPT_DIR"
echo "########################################################"
echo ""

# =============================================================================
# STEP 1: Environment Setup
# =============================================================================
echo "============================================"
echo "STEP 1/5: Setting up Poetry environment..."
echo "============================================"

if ! command -v poetry &>/dev/null; then
    echo "ERROR: poetry not found. Install it: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Ensure we use a compatible Python version (3.10-3.12)
# Poetry needs this to create the right virtualenv
PYTHON_BIN=""
for v in python3.10 python3.11 python3.12; do
    if command -v "$v" &>/dev/null; then
        PYTHON_BIN="$v"
        break
    fi
    # Check homebrew paths on macOS
    for prefix in /opt/homebrew/opt /usr/local/opt; do
        candidate="${prefix}/python@${v#python}/bin/$v"
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break 2
        fi
    done
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Python 3.10-3.12 not found. Install with: brew install python@3.10"
    exit 1
fi

echo "  Using Python: $($PYTHON_BIN --version)"
poetry env use "$PYTHON_BIN" 2>&1 | grep -v "^$" || true

# Install dependencies (skips if already installed)
poetry install --no-root 2>&1 | tail -5
echo "Dependencies installed."
echo ""

# Verify key packages
poetry run python -c "
import torch, diffusers, transformers
print(f'  torch:        {torch.__version__}')
print(f'  diffusers:    {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'  device:       {device}')
"
echo ""

# =============================================================================
# STEP 2: Download model (if needed)
# =============================================================================
echo "============================================"
echo "STEP 2/5: Checking model availability..."
echo "============================================"

poetry run python -c "
import warnings; warnings.filterwarnings('ignore')
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch, os

model_id = 'runwayml/stable-diffusion-v1-5'

# Check if model is cached
try:
    DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler', local_files_only=True)
    StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
    print('  Model already cached. Skipping download.')
except Exception:
    print('  Downloading model (~5GB, one-time)...')
    DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
    StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    print('  Model downloaded and cached.')
"
echo ""

# =============================================================================
# STEP 3: Run Baseline
# =============================================================================
echo "============================================"
echo "STEP 3/5: Running BASELINE experiment..."
echo "============================================"
echo "  Config: $SCALE scale"
echo ""

poetry run python experiment/run_experiment.py \
    --scale "$SCALE" \
    --mode baseline \
    $SKIP_CLIP_FLAG

echo ""

# =============================================================================
# STEP 4: Run Optimized
# =============================================================================
echo "============================================"
echo "STEP 4/5: Running OPTIMIZED experiment..."
echo "============================================"
echo "  Config: $SCALE scale"
echo "  Optimizations: multi-channel, larger radius, more inversion steps,"
echo "                 amplitude scaling, ensemble detection"
echo ""

poetry run python experiment/run_experiment.py \
    --scale "$SCALE" \
    --mode optimized \
    $SKIP_CLIP_FLAG

echo ""

# =============================================================================
# STEP 5: Compare Results
# =============================================================================
echo "============================================"
echo "STEP 5/5: Comparing results..."
echo "============================================"
echo ""

poetry run python experiment/run_experiment.py \
    --scale "$SCALE" \
    --mode compare

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
