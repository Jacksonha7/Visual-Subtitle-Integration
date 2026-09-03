#!/bin/bash
# Sets up a conda environment for VSI, including the YOLO-World detector
# adapter used by TStar/interface_yolo.py.
#
# This exists because the "obvious" install path (pip install -r
# requirements.txt + pip install YOLO-World) hits a chain of real version
# conflicts between YOLO-World's own pyproject.toml, mmdet's runtime version
# check, and openmmlab's prebuilt wheel index. See the "Verified Environment /
# Known Issues" section of README.md for the full explanation of every step
# below. This script encodes the fixes so you don't have to rediscover them.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Configuration (override via environment variables before running):
#   CONDA_ENV_NAME   conda environment name (default: vsi)
#   YOLO_WORLD_DIR   where to clone/find YOLO-World (default: ./third_party/YOLO-World)
#   PYTHON_VERSION   python version for the conda env (default: 3.10)

set -euo pipefail
set -x

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vsi}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_WORLD_DIR="${YOLO_WORLD_DIR:-$REPO_ROOT/third_party/YOLO-World}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# --- 1. Conda environment -----------------------------------------------
if ! command -v conda &> /dev/null; then
  for candidate in "/apps/local/anaconda3" "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
      source "$candidate/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if ! command -v conda &> /dev/null; then
  echo "conda was not found on PATH and no known install location worked." >&2
  echo "Activate/initialize conda yourself, then re-run this script." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^${CONDA_ENV_NAME} "; then
  conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
fi
conda activate "$CONDA_ENV_NAME"

export PIP_CONSTRAINT="$REPO_ROOT/constraints.txt"

# --- 2. torch/torchvision (pinned combo — see constraints.txt) ----------
# torch2.1.x/cu121 has no prebuilt mmcv<2.1.0 wheel, and mmdet==3.0.0 requires
# mmcv<2.1.0 at import time. torch2.0.1/cu118 is the newest combo that has one.
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# --- 3. Packages that must be pinned *before* anything else pulls them in ---
pip install "setuptools<81" "numpy==1.26.4" "opencv-python==4.9.0.80" "transformers==4.36.2"

# --- 4. mmcv (prebuilt wheel matching the torch2.0.1/cu118 combo above) -----
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# --- 5. YOLO-World itself ------------------------------------------------
if [ ! -d "$YOLO_WORLD_DIR" ]; then
  git clone --recurse-submodules https://github.com/AILab-CVC/YOLO-World.git "$YOLO_WORLD_DIR"
fi
if [ ! -e "$YOLO_WORLD_DIR/third_party/mmyolo/setup.py" ]; then
  git -C "$YOLO_WORLD_DIR" submodule update --init --recursive
fi

# YOLO-World's own pyproject.toml declares `torchvision>=0.16.2`, which
# conflicts with the mmdet<2.1.0-compatible torch2.0.1/torchvision0.15.2 combo
# we just installed. We already have a self-consistent, working set of
# versions, so skip pip's dependency resolution here and install YOLO-World's
# own code (+ mmyolo's own code) directly; the handful of small deps it needs
# that aren't covered above are installed explicitly right after.
pip install --no-build-isolation --no-deps -e "$YOLO_WORLD_DIR"
pip install --no-build-isolation --no-deps -e "$YOLO_WORLD_DIR/third_party/mmyolo"
pip install mmdet==3.0.0 supervision==0.19.0 prettytable

# Known upstream bug in AILab-CVC/YOLO-World: `reparameterize()` assigns to
# the literal `None`, which is a SyntaxError in Python and breaks importing
# `yolo_world` entirely. Patch it locally (harmless no-op if already patched
# or if upstream has fixed it and the old line is no longer present).
YOLO_DETECTOR_FILE="$YOLO_WORLD_DIR/yolo_world/models/detectors/yolo_world.py"
if grep -q "self.text_feats, None = self.backbone.forward_text(texts)" "$YOLO_DETECTOR_FILE" 2>/dev/null; then
  sed -i "s/self.text_feats, None = self.backbone.forward_text(texts)/self.text_feats, _ = self.backbone.forward_text(texts)/" "$YOLO_DETECTOR_FILE"
  echo "Patched upstream syntax bug in $YOLO_DETECTOR_FILE"
fi

# --- 6. VSI's own dependencies -------------------------------------------
pip install -r "$REPO_ROOT/requirements.txt"
# lvis is only needed because the bundled YOLO-World pretrain configs build a
# real LVIS-format val dataset just to read off class-name metainfo at
# init_detector() time (see step 7).
pip install lvis

# --- 7. LVIS class metadata the bundled YOLO-World configs need at init ----
# init_detector() builds the config's val/test dataset just to read its
# `metainfo` (class names), which for the LVIS-minival pretrain configs means
# it needs the actual annotation file and YOLO-World's bundled class-text
# json on disk — even for pure custom-vocabulary inference. Paths are
# resolved relative to the CWD you run VSI scripts from (i.e. $REPO_ROOT).
mkdir -p "$REPO_ROOT/data/coco/lvis"
if [ ! -f "$REPO_ROOT/data/coco/lvis/lvis_v1_minival_inserted_image_name.json" ]; then
  curl -sL -o "$REPO_ROOT/data/coco/lvis/lvis_v1_minival_inserted_image_name.json" \
    "https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json"
fi
ln -sfn "$YOLO_WORLD_DIR/data/texts" "$REPO_ROOT/data/texts"

echo "===VSI_SETUP_DONE==="
python -c "
import torch, mmcv, mmdet, mmengine, mmyolo, yolo_world
print('torch', torch.__version__, torch.cuda.is_available())
print('mmcv', mmcv.__version__)
print('mmdet', mmdet.__version__)
print('mmyolo', mmyolo.__version__)
import sentence_transformers, decord
print('sentence_transformers OK, decord OK')
"

cat <<EOF

Setup complete. Before running any VSI script, make sure the repo root is on
PYTHONPATH (examples/run_keyframe_search.py and VSI_keyframe_search.py both
import the TStar package by absolute name, which only resolves if the repo
root — not the script's own directory — is on sys.path):

  export PYTHONPATH="$REPO_ROOT:\$PYTHONPATH"

Point --yoloworld_config / --yoloworld_ckpt / --checkpoint_path at a config
under "$YOLO_WORLD_DIR/configs/pretrain/" and a matching checkpoint from
https://huggingface.co/wondervictor/YOLO-World-V2.1 (see README for the
model-name/file-name mismatch to watch out for).
EOF
