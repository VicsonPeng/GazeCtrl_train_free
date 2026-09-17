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
