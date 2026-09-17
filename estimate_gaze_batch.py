"""
Phase 2 — Batch Gaze Estimation on Pipeline Results
=====================================================
Reads metadata.json files from dataset_pipeline.py output,
runs a gaze model on each 07_final.png, computes angular error
between predicted gaze and target_gaze, and updates metadata.json.

Supported models
----------------
  l2cs        L2CS-Net     — default, no extra venv needed
  6drepnet    6DRepNet     — run from the 6DRepNet venv
  3dgazenet   3DGazeNet    — run from the WSL conda env (conda activate 3DGazeNet)

Usage
-----
  # Default (L2CS-Net, any env):
  python estimate_gaze_batch.py --results-dir e_results

  # 6DRepNet (activate 6DRepNet venv first):
  python estimate_gaze_batch.py --results-dir e_results --model 6drepnet

  # 3DGazeNet (run inside WSL conda 3DGazeNet env):
  python estimate_gaze_batch.py --results-dir e_results --model 3dgazenet

  # Recompute angular error only (no model inference):
  python estimate_gaze_batch.py --results-dir e_results --model none
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))


# ══════════════════════════════════════════════════════════════════════════════
#  ANGULAR ERROR  (between two unit vectors)
# ══════════════════════════════════════════════════════════════════════════════

def angular_error_deg(v1: dict, v2: dict) -> float:
    """
    Compute angular error in degrees between two gaze dicts with keys dx, dy, dz.
    Handles the case where either vector has yaw_deg/pitch_deg but no dx/dy/dz.
    """
    def to_vec(g):
        if all(k in g for k in ("dx", "dy", "dz")):
            return np.array([g["dx"], g["dy"], g["dz"]], dtype=float)
        yaw   = math.radians(g.get("yaw_deg", 0.0))
        pitch = math.radians(g.get("pitch_deg", 0.0))
        dx = math.cos(pitch) * math.sin(yaw)
        dy = -math.sin(pitch)
        dz = math.cos(pitch) * math.cos(yaw)
        return np.array([dx, dy, dz], dtype=float)

    a, b = to_vec(v1), to_vec(v2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    cos_a = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


# ══════════════════════════════════════════════════════════════════════════════
#  L2CS-NET  (default — same env as dataset_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

_l2cs_pipeline = None

def predict_gaze_l2cs(image_bgr):
    """Run L2CS-Net on an image; returns gaze dict or None."""
    global _l2cs_pipeline
    if _l2cs_pipeline is None:
        from model_utils import get_l2cs_pipeline
        _l2cs_pipeline = get_l2cs_pipeline()

    results = _l2cs_pipeline.step(image_bgr)
    if results.pitch.shape[0] == 0:
        return None

    # L2CS variable names are swapped vs. geometric convention
    true_pitch_rad = float(results.yaw[0])
    true_yaw_rad   = -float(results.pitch[0])

    dx = math.cos(true_pitch_rad) * math.sin(true_yaw_rad)
    dy = -math.sin(true_pitch_rad)
    dz = math.cos(true_pitch_rad) * math.cos(true_yaw_rad)

    return {
        "model":     "l2cs",
        "dx":        round(dx, 6),
        "dy":        round(dy, 6),
        "dz":        round(dz, 6),
        "yaw_deg":   round(math.degrees(true_yaw_rad),   3),
        "pitch_deg": round(math.degrees(true_pitch_rad), 3),
        "yaw_rad":   round(true_yaw_rad, 6),
        "pitch_rad": round(true_pitch_rad, 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  6DREPNET  (must be run from 6DRepNet venv)
# ══════════════════════════════════════════════════════════════════════════════

_sixdrep_model = None
_sixdrep_head_yolo = None
_sixdrep_transform = None


def _load_6drepnet():
    global _sixdrep_model, _sixdrep_head_yolo, _sixdrep_transform
    import torch
    from torchvision import transforms
    from ultralytics import YOLO

    sixdrep_dir = str(PROJECT_ROOT / "6DRepNet" / "sixdrepnet")
    if sixdrep_dir not in sys.path:
        sys.path.insert(0, sixdrep_dir)

    from torchvision.models.resnet import Bottleneck
    from model import SixDRepNet2
    import utils as sixdrep_utils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = PROJECT_ROOT / "6DRepNet" / "sixdrepnet" / \
              "6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
    saved = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved)
    model = SixDRepNet2(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    _sixdrep_model = (model, device, sixdrep_utils)

    yolo_head = str(PROJECT_ROOT / "yolo_seg" / "yolov8_head.pt")
    _sixdrep_head_yolo = YOLO(yolo_head)

    _sixdrep_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print("[6DRepNet] Loaded.")


def predict_gaze_6drepnet(image_bgr):
    """Run 6DRepNet on image; returns gaze dict or None."""
    import torch
    from PIL import Image

    if _sixdrep_model is None:
        _load_6drepnet()

    model, device, utils = _sixdrep_model

    # Head detection
    head_res = _sixdrep_head_yolo(image_bgr, verbose=False)
    boxes = head_res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = int(torch.argmax(boxes.conf))
    bx = boxes.xyxy[best].cpu().numpy()
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    bw, bh = (bx[2] - bx[0]) * 1.4, (bx[3] - bx[1]) * 1.4
    x1 = max(0, int(cx - bw / 2)); y1 = max(0, int(cy - bh / 2))
    x2 = min(image_bgr.shape[1], int(cx + bw / 2))
    y2 = min(image_bgr.shape[0], int(cy + bh / 2))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return None

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = _sixdrep_transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        R = model(tensor)
    euler = utils.compute_euler_angles_from_rotation_matrices(R) * 180 / np.pi
    pitch_deg = float(euler[:, 0].cpu().numpy()[0])
    yaw_deg   = float(euler[:, 1].cpu().numpy()[0])
    roll_deg  = float(euler[:, 2].cpu().numpy()[0])

    pitch_rad = math.radians(pitch_deg)
    yaw_rad   = math.radians(yaw_deg)
    dx = -(abs(math.cos(pitch_rad)) * math.sin(yaw_rad))
    dy = -math.sin(pitch_rad)
    dz =  math.cos(pitch_rad) * math.cos(yaw_rad)
    norm = math.sqrt(dx**2 + dy**2 + dz**2)
    if norm > 1e-8:
        dx /= norm; dy /= norm; dz /= norm

    return {
        "model":     "6drepnet",
        "dx":        round(dx, 6),
        "dy":        round(dy, 6),
        "dz":        round(dz, 6),
        "pitch_deg": round(pitch_deg, 3),
        "yaw_deg":   round(yaw_deg, 3),
        "roll_deg":  round(roll_deg, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3DGAZENET  (run inside WSL conda env)
# ══════════════════════════════════════════════════════════════════════════════

def predict_gaze_3dgazenet(image_bgr, gazenet_inference):
    """Run 3DGazeNet on image using pre-loaded GazeNetInference. Returns gaze dict or None."""
    import torch

    # Face detection
    faces = gazenet_inference.face_detector.model.get(image_bgr)
    if len(faces) == 0:
        return None
    face = faces[0]
    if np.any(np.isnan(face.kps)):
        return None

    kps = face.kps.astype(int)
    with torch.no_grad():
        result = gazenet_inference.gaze_predictor(image_bgr, kps, undo_roll=True)
    if result is None:
        return None

    vec = result.get("gaze_combined") or result.get("gaze_out")
    if vec is None or np.any(np.isnan(vec)):
        return None

    # Convention correction (raw X and Y are inverted vs. image-space)
    dx =  float(vec[0])
    dy = -float(vec[1])
    dz =  float(vec[2])

    # Compute pitch/yaw from unit vector
    norm = math.sqrt(dx**2 + dy**2 + dz**2)
    if norm > 1e-8:
        dx /= norm; dy /= norm; dz /= norm
    pitch_rad = math.asin(max(-1.0, min(1.0, -dy)))
    yaw_rad   = math.atan2(dx, dz)

    return {
        "model":     "3dgazenet",
        "dx":        round(dx, 6),
        "dy":        round(dy, 6),
        "dz":        round(dz, 6),
        "yaw_deg":   round(math.degrees(yaw_rad),   3),
        "pitch_deg": round(math.degrees(pitch_rad), 3),
        "det_score": round(float(face.det_score), 4),
    }


def load_3dgazenet():
    """Load GazeNetInference (WSL / conda 3DGazeNet env required)."""
    import os
    demo_dir = str(PROJECT_ROOT / "3DGazeNet" / "demo")
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)
    os.chdir(demo_dir)
    from inference import GazeNetInference
    net = GazeNetInference()
    print("[3DGazeNet] Loaded.")
    return net


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Estimate gaze on pipeline results and compute angular error")
    parser.add_argument(
        "--results-dir", type=str, default="e_results",
        help="Directory produced by dataset_pipeline.py (default: e_results)")
    parser.add_argument(
        "--model", choices=["l2cs", "6drepnet", "3dgazenet", "none"],
        default="l2cs",
        help="Gaze estimation model (default: l2cs)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run even if predicted_gaze is already set in metadata.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[Error] Results directory not found: {results_dir}")
        sys.exit(1)

    # Collect all samples that have a final image + metadata
    samples = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path  = d / "metadata.json"
        final_path = d / "07_final.png"
        if meta_path.exists() and final_path.exists():
            samples.append((d, meta_path, final_path))

    if not samples:
        print(f"[Error] No completed samples found in {results_dir}")
        sys.exit(1)

    print(f"[Phase 2] {len(samples)} samples  |  model: {args.model}")

    # Load gaze model once
    gazenet = None
    if args.model == "3dgazenet":
        gazenet = load_3dgazenet()
    elif args.model == "6drepnet":
        _load_6drepnet()
    elif args.model == "l2cs":
        from model_utils import get_l2cs_pipeline
        get_l2cs_pipeline()  # warm up

    done = errors = skipped = 0

    for sample_dir, meta_path, final_path in samples:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if not args.overwrite and meta.get("predicted_gaze") is not None:
            skipped += 1
            continue

        image_bgr = cv2.imread(str(final_path))
        if image_bgr is None:
            print(f"  [Error] Cannot read {final_path}")
            errors += 1
            continue

        # Run gaze prediction
        pred = None
        try:
            if args.model == "l2cs":
                pred = predict_gaze_l2cs(image_bgr)
            elif args.model == "6drepnet":
                pred = predict_gaze_6drepnet(image_bgr)
            elif args.model == "3dgazenet":
                pred = predict_gaze_3dgazenet(image_bgr, gazenet)
            elif args.model == "none":
                pred = meta.get("predicted_gaze")  # keep existing
        except Exception as e:
            print(f"  [Error] {sample_dir.name}: {e}")
            errors += 1
            continue

        if pred is None:
            print(f"  [Warning] {sample_dir.name}: no gaze detected")
            meta["predicted_gaze"] = None
            meta["angular_error_deg"] = None
            meta["quality_flag"] = "no_face"
        else:
            target = meta.get("target_gaze", {})
            err_deg = angular_error_deg(target, pred)
            meta["predicted_gaze"]    = pred
            meta["angular_error_deg"] = round(err_deg, 3) if not math.isnan(err_deg) else None
            meta["quality_flag"]      = None  # set by build_training_set.py

        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        done += 1

        err_str = (f"{meta['angular_error_deg']:.1f}°"
                   if meta.get("angular_error_deg") is not None else "N/A")
        print(f"  {sample_dir.name}  error={err_str}")

    # Regenerate master manifest
    all_meta = []
    for d in sorted(results_dir.iterdir()):
        mf = d / "metadata.json"
        if d.is_dir() and mf.exists():
            try:
                all_meta.append(json.loads(mf.read_text(encoding="utf-8")))
            except Exception:
                pass
    if all_meta:
        (results_dir / "manifest.json").write_text(
            json.dumps({"samples": all_meta, "total": len(all_meta)},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\n[Phase 2] Done: {done}  Skipped: {skipped}  Errors: {errors}")
    if all_meta:
        errors_deg = [m["angular_error_deg"] for m in all_meta
                      if m.get("angular_error_deg") is not None]
        if errors_deg:
            print(f"  Angular error — "
                  f"mean: {sum(errors_deg)/len(errors_deg):.1f}°  "
                  f"median: {sorted(errors_deg)[len(errors_deg)//2]:.1f}°  "
                  f"max: {max(errors_deg):.1f}°")


if __name__ == "__main__":
    main()
