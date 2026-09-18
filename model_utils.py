"""
Shared model loaders and Gemini configuration for the Path E gaze-redirection
pipeline (manual_process_path_e.py, dataset_pipeline.py, estimate_gaze_batch.py).

Everything here is lazy-loaded on first use so that importing this module is
cheap even if only one model is actually needed.
"""

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

# ─────────────── Project paths ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

L2CS_DIR = PROJECT_ROOT / "L2CS-Net"
L2CS_MODEL = L2CS_DIR / "models" / "L2CSNet_gaze360.pkl"
sys.path.insert(0, str(L2CS_DIR))

SAM_CHECKPOINT = SCRIPT_DIR / "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# ─────────────── Gemini API ───────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
GEMINI_MODEL = "gemini-3-pro-image-preview"


# ══════════════════════════════════════════════════════════════
#  SAM (person segmentation)
# ══════════════════════════════════════════════════════════════

def ensure_sam_checkpoint() -> Path:
    if SAM_CHECKPOINT.exists():
        return SAM_CHECKPOINT
    print("[SAM] Downloading checkpoint (~2.5 GB) ...")
    urllib.request.urlretrieve(SAM_URL, str(SAM_CHECKPOINT))
    return SAM_CHECKPOINT


def load_sam_predictor():
    """Load a SAM (ViT-H) predictor for interactive point-prompted masking."""
    from segment_anything import sam_model_registry, SamPredictor
    ckpt = ensure_sam_checkpoint()
    print("[SAM] Loading model ...")
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"[SAM] Ready on {device}")
    return predictor


# ══════════════════════════════════════════════════════════════
#  Depth-Anything-V2 (monocular depth, drives the "3D" gaze target)
# ══════════════════════════════════════════════════════════════

_depth_anything_model = None


def get_depth_anything():
    """Lazy-load Depth-Anything-V2 (ViT-S)."""
    global _depth_anything_model
    if _depth_anything_model is not None:
        return _depth_anything_model

    print("[Depth] Loading Depth-Anything-V2 ...")
    da_path = str(PROJECT_ROOT / "Depth-Anything-V2")
    if da_path not in sys.path:
        sys.path.insert(0, da_path)

    from depth_anything_v2.dpt import DepthAnythingV2
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = "vits"
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    }

    ckpt_dir = os.path.join(da_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"depth_anything_v2_{encoder}.pth")

    if not os.path.exists(ckpt_path):
        print(f"[Depth] Downloading {encoder} checkpoint ...")
        repo_id = "depth-anything/Depth-Anything-V2-Small"
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=f"depth_anything_v2_{encoder}.pth")
        import shutil
        shutil.copy(downloaded_path, ckpt_path)

    _depth_anything_model = DepthAnythingV2(**model_configs[encoder])
    _depth_anything_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    _depth_anything_model = _depth_anything_model.to(device).eval()

    print(f"[Depth] Ready on {device}.")
    return _depth_anything_model


# ══════════════════════════════════════════════════════════════
#  L2CS-Net (gaze estimation, used by estimate_gaze_batch.py to
#  score how close the Gemini-edited gaze is to the requested target)
# ══════════════════════════════════════════════════════════════

_l2cs_pipeline = None


def get_l2cs_pipeline():
    """Lazy-load L2CS-Net pipeline."""
    global _l2cs_pipeline
    if _l2cs_pipeline is not None:
        return _l2cs_pipeline

    from l2cs import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[L2CS] Loading model (device={device}) ...")
    _l2cs_pipeline = Pipeline(
        weights=L2CS_MODEL,
        arch="ResNet50",
        device=device,
    )
    print("[L2CS] Ready.")
    return _l2cs_pipeline


# ══════════════════════════════════════════════════════════════
#  RED TARGET-DOT CLEANUP
# ══════════════════════════════════════════════════════════════

def remove_red_marker(image_bgr, pad=6):
    """
    Blacken any leftover red target-dot pixels in a Gemini-edited image, so
    the downstream black-gap repair pass fills them in along with everything
    else in a single call.

    Gemini is told not to move the red dot, but on large pose changes (e.g.
    turning a person all the way around) it may relocate or redraw it, so a
    fixed coordinate can't be trusted — this detects the marker by color
    instead.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red_mask = (
        cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 120, 120), (180, 255, 255))
    )
    if cv2.countNonZero(red_mask) == 0:
        return image_bgr
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    out = image_bgr.copy()
    out[red_mask > 0] = 0
    return out


# ══════════════════════════════════════════════════════════════
#  GEMINI OUTPUT NOISE CLEANUP
# ══════════════════════════════════════════════════════════════
#
# Gemini's isolated-person edit is supposed to return the person on a pure
# black background, but its "black" is often not exactly (0,0,0) - low-level
# compression-like noise (pixel values ~1-4) can cover a large fraction of
# the supposedly-black region. A naive >0 threshold treats all of that as
# foreground, which then gets composited onto the real background as fake
# "person" pixels. There can also be genuine tiny near-black dropout pixels
# inside the person (e.g. on a face), which become tiny holes that the
# background-repair pass also tries to fill - sometimes causing it to
# re-render (and re-noise) the whole image instead of just the actual gap
# left by the pose change.

FOREGROUND_THRESHOLD = 20  # well above observed near-black compression noise (~1-4)


def clean_foreground_mask(gemini_bgr, close_kernel=15, open_kernel=5):
    """
    Build a clean binary mask (0/255) of Gemini's actual (non-near-black)
    foreground: closes small internal holes, opens away small external
    speckles, and keeps only the single largest connected blob (the person).
    """
    gray = cv2.cvtColor(gemini_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, FOREGROUND_THRESHOLD, 255, cv2.THRESH_BINARY)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


def inpaint_small_holes(gemini_bgr, foreground_mask):
    """
    Locally inpaint any black holes inside `foreground_mask` (e.g. speckle
    dropouts on a face) so they don't need to go through the (much more
    expensive, and sometimes noisy) background-repair image-edit pass.
    """
    gray = cv2.cvtColor(gemini_bgr, cv2.COLOR_BGR2GRAY)
    _, non_black = cv2.threshold(gray, FOREGROUND_THRESHOLD, 255, cv2.THRESH_BINARY)
    holes = cv2.bitwise_and(foreground_mask, cv2.bitwise_not(non_black))
    if cv2.countNonZero(holes) == 0:
        return gemini_bgr
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    holes = cv2.dilate(holes, kernel, iterations=1)
    return cv2.inpaint(gemini_bgr, holes, 5, cv2.INPAINT_TELEA)
