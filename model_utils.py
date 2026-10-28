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

def red_marker_mask(image_bgr, pad=5):
    """
    Mask (0/255) of the red target dot in a Gemini-edited image.

    Gemini is told not to move the red dot, but on large pose changes (e.g.
    turning a person all the way around) it may relocate or redraw it, so a
    fixed coordinate can't be trusted — the marker is found by color instead.
    Only small, compact, near-pure-red blobs count: an earlier HSV hue-range
    test (H 0-10, S/V >= 120) also matched warm skin, sunset-lit hair, lips
    and red clothing, and blacked out large parts of faces.
    """
    b, g, r = (image_bgr[..., i].astype(np.int16) for i in range(3))
    cand = ((r >= 140) & (r - np.maximum(g, b) >= 100)).astype(np.uint8) * 255
    out = np.zeros(cand.shape, np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not 120 <= area <= 4000:               # redrawn dots seen: 138-2984 px; lips ~60
            continue
        if not 0.5 <= bw / bh <= 2.0 or area / (bw * bh) < 0.55:   # a disc fills ~0.785
            continue
        out[labels == i] = 255
    if pad and cv2.countNonZero(out):
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad)))
    return out


def remove_red_marker(image_bgr, pad=5):
    """Blacken the red target dot (see red_marker_mask)."""
    m = red_marker_mask(image_bgr, pad)
    if cv2.countNonZero(m) == 0:
        return image_bgr
    out = image_bgr.copy()
    out[m > 0] = 0
    return out


# ══════════════════════════════════════════════════════════════
#  GEMINI OUTPUT NOISE CLEANUP
# ══════════════════════════════════════════════════════════════
#
# Gemini's isolated-person edit is supposed to return the person on a pure
# black background, but its "black" is not exactly (0,0,0): over 215 eval
# outputs the noise is <= 3 at the 99.9th percentile in most images, with
# sparse specks up to ~19 in a few. Dark hair/clothing/shadow on the person
# overlaps that range, so brightness alone can't separate them; the mask
# below combines a low threshold with blob-size filtering and hole filling.
#
# Earlier version (threshold 20, keep only the largest blob, inpaint every
# dark pixel inside it) dropped whole heads whenever the person was split
# in two by a black region (e.g. another person's arm cut out of the mask),
# and turned dark hair/clothing into background-coloured holes.

FOREGROUND_THRESHOLD = 8   # on the max channel; above typical noise, below most dark hair


def foreground_mask(gemini_bgr, person_mask=None):
    """
    Binary mask (0/255) of the person Gemini actually drew.

    Keeps every sizeable blob, not only the largest: blobs touching the
    original SAM mask are kept regardless, other blobs only if they're at
    least 2% the size of the largest (speckle noise is far smaller). Small
    enclosed holes (pupils, dark folds, the removed marker) are filled;
    large ones are left out, since they're usually background the new pose
    exposes.
    """
    h, w = gemini_bgr.shape[:2]
    m = (gemini_bgr.max(axis=2) > FOREGROUND_THRESHOLD).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    min_area = max(64, int(2e-4 * h * w))
    near = None
    if person_mask is not None:
        near = cv2.dilate((person_mask > 0).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    keep = np.zeros(n, bool)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        if a >= 0.02 * largest:
            keep[i] = True
        elif near is not None:
            x, y, bw, bh = stats[i, :4]
            comp = labels[y:y + bh, x:x + bw] == i
            keep[i] = near[y:y + bh, x:x + bw][comp].mean() >= 0.3
    m = np.where(keep[labels], 255, 0).astype(np.uint8)

    # Fill enclosed holes (not connected to the image border) up to 0.1% of the frame.
    inv = cv2.bitwise_not(m)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    x, y, bw, bh, a = (st2[:, i] for i in range(5))
    fill = (a <= int(1e-3 * h * w)) & (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h)
    fill[0] = False
    m[fill[lab2]] = 255
    return m


def composite_onto_background(original_bgr, gemini_bgr, person_mask, return_fg=False):
    """
    Paste Gemini's redirected person over the original photo.

    Returns (composite, gap_mask): composite has the edited person on top,
    the original background elsewhere, and black (0) in gap_mask — pixels
    the old person covered that the new pose no longer does. Repair passes
    must use gap_mask rather than "pixel == 0", because genuinely black hair
    or clothing is also 0. With return_fg=True also returns the mask (0/255)
    of the pasted person, i.e. the edited person's silhouette.
    """
    h, w = original_bgr.shape[:2]
    if gemini_bgr.shape[:2] != (h, w):
        gemini_bgr = cv2.resize(gemini_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    marker = red_marker_mask(gemini_bgr)
    gem = gemini_bgr.copy()
    gem[marker > 0] = 0
    fg = foreground_mask(gem, person_mask)

    # Marker drawn over the person: patch it from the surrounding person pixels.
    on_person = cv2.bitwise_and(marker, fg)
    if cv2.countNonZero(on_person):
        gem = cv2.inpaint(gem, on_person, 5, cv2.INPAINT_TELEA)

    # Gemini's silhouette edge is blended toward its black background; pasting
    # it as-is leaves a thin dark outline. Pull the edge in by 2 px (the
    # original background, or the gap repair, covers the difference).
    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    composite = original_bgr.copy()
    composite[fg > 0] = gem[fg > 0]
    gap = ((person_mask > 2) & (fg == 0)).astype(np.uint8) * 255
    composite[gap > 0] = 0
    return (composite, gap, fg) if return_fg else (composite, gap)
