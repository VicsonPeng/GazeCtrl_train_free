"""
GazeCtrl Path B — Batch Evaluation Pipeline

Automatically runs the gaze redirection pipeline on a folder of images,
estimates gaze with L2CS-Net, and produces error analysis.

Usage:
    python batch_evaluate.py --input-dir ./eval_images [--warp-mode {none,shift,tps}] [--max-images 100]
"""

import os
import sys
import json
import glob
import math
import argparse
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk, ImageFont
import tkinter as tk
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# ─────────────── Project paths ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
L2CS_DIR = PROJECT_ROOT / "L2CS-Net"
L2CS_MODEL = L2CS_DIR / "models" / "L2CSNet_gaze360.pkl"

SAM_CHECKPOINT = SCRIPT_DIR / "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Add L2CS-Net to path
sys.path.insert(0, str(L2CS_DIR))

# Gemini API
GEMINI_API_KEY = "AIzaSyB5a_-zFj7QZ2naroAMxDkzzAT4I-x3hWI"
GEMINI_MODEL = "gemini-3-pro-image-preview"  # configurable via --model

# ─────────────── Virtual pinhole camera ───────────────
# We approximate gaze angles using a pinhole camera model:
#   yaw   = arctan(du / f)
#   pitch = arctan(dv / f)
# where f is the virtual focal length (estimated from face size)
VIRTUAL_FOCAL_MULTIPLIER = 2.0  # f = face_width * multiplier


# ══════════════════════════════════════════════════════════════
#  SAM MODEL
# ══════════════════════════════════════════════════════════════

def ensure_sam_checkpoint() -> Path:
    if SAM_CHECKPOINT.exists():
        return SAM_CHECKPOINT
    print(f"[SAM] Downloading checkpoint (~2.5 GB) …")
    urllib.request.urlretrieve(SAM_URL, str(SAM_CHECKPOINT))
    return SAM_CHECKPOINT


def load_sam_predictor():
    """Load SAM predictor."""
    from segment_anything import sam_model_registry, SamPredictor
    ckpt = ensure_sam_checkpoint()
    print("[SAM] Loading model …")
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"[SAM] Ready on {device}")
    return predictor


# ══════════════════════════════════════════════════════════════
#  L2CS-NET
# ══════════════════════════════════════════════════════════════

_l2cs_pipeline = None


def get_l2cs_pipeline():
    """Lazy-load L2CS-Net pipeline."""
    global _l2cs_pipeline
    if _l2cs_pipeline is not None:
        return _l2cs_pipeline

    from l2cs import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[L2CS] Loading model (device={device}) …")
    _l2cs_pipeline = Pipeline(
        weights=L2CS_MODEL,
        arch="ResNet50",
        device=device,
    )
    print("[L2CS] Ready.")
    return _l2cs_pipeline


def estimate_gaze_l2cs(image_bgr: np.ndarray) -> dict:
    """
    Run L2CS-Net on an image and return gaze estimation.

    Returns:
        tuple (dict, raw_results) or (None, None)
    """
    pipeline = get_l2cs_pipeline()
    results = pipeline.step(image_bgr)

    if results.pitch.shape[0] == 0:
        return None, None

    # WARNING: L2CS-Net authors swapped the variable names in their dataset loaders!
    # L2CS `results.yaw` actually controls vertical movement (True Pitch).
    # L2CS `results.pitch` actually controls horizontal movement (True Yaw).
    # L2CS `results.yaw` > 0 means looking UP.
    # L2CS `results.pitch` > 0 means looking LEFT.
    
    # We map them to standard True Geometric Angles:
    # True Pitch > 0 for UP. True Yaw > 0 for RIGHT.
    true_pitch_rad = float(results.yaw[0])
    true_yaw_rad = -float(results.pitch[0])
    
    dic = {
        "yaw_rad": true_yaw_rad,
        "pitch_rad": true_pitch_rad,
        "yaw_deg": math.degrees(true_yaw_rad),
        "pitch_deg": math.degrees(true_pitch_rad),
    }
    return dic, results

# ══════════════════════════════════════════════════════════════
#  DEPTH-ANYTHING-V2
# ══════════════════════════════════════════════════════════════

_depth_anything_model = None

def get_depth_anything():
    """Lazy-load DepthAnythingV2 model."""
    global _depth_anything_model
    if _depth_anything_model is not None:
        return _depth_anything_model
        
    print("[Depth] Loading Depth-Anything-V2...")
    da_path = str(PROJECT_ROOT / "Depth-Anything-V2")
    if da_path not in sys.path:
        sys.path.insert(0, da_path)
        
    from depth_anything_v2.dpt import DepthAnythingV2
    from huggingface_hub import hf_hub_download
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = 'vits'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    ckpt_dir = os.path.join(da_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'depth_anything_v2_{encoder}.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"[Depth] Downloading {encoder} checkpoint...")
        repo_id = "depth-anything/Depth-Anything-V2-Small"
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=f"depth_anything_v2_{encoder}.pth")
        import shutil
        shutil.copy(downloaded_path, ckpt_path)
        
    _depth_anything_model = DepthAnythingV2(**model_configs[encoder])
    _depth_anything_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    _depth_anything_model = _depth_anything_model.to(device).eval()
    
    print(f"[Depth] Ready on {device}.")
    return _depth_anything_model


# ══════════════════════════════════════════════════════════════
#  FACE DETECTION (RetinaFace — same detector L2CS-Net uses)
# ══════════════════════════════════════════════════════════════

_retinaface_detector = None


def _get_retinaface():
    """Lazy-load RetinaFace detector (shared with L2CS-Net)."""
    global _retinaface_detector
    if _retinaface_detector is not None:
        return _retinaface_detector
    from face_detection import RetinaNetMobileNetV1 as RetinaFace
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _retinaface_detector = RetinaFace(
        confidence_threshold=0.3,
        nms_iou_threshold=0.4,
        device=device,
        max_resolution=None,
        fp16_inference=False,
        clip_boxes=False,
    )
    print("[RetinaFace] Detector ready")
    return _retinaface_detector


def detect_face_center(image_bgr: np.ndarray, mask: np.ndarray = None, click_point: tuple = None) -> dict:
    """
    Detect face using RetinaFace (same detector L2CS-Net uses).

    If click_point is provided, finds the face bounding box that contains this point.
    Otherwise, picks the highest confidence face.
    """
    detector = _get_retinaface()

    boxes, landmarks = detector.batched_detect_with_landmarks(
        np.expand_dims(image_bgr, 0)
    )

    if len(boxes) == 0 or boxes[0].shape[0] == 0:
        return None

    face_boxes = boxes[0]
    face_lms = landmarks[0]

    best_idx = -1
    
    # Strategy 1: If SAM mask is provided, find the face that overlaps most with the mask
    if mask is not None:
        best_overlap = 0
        for i, box in enumerate(face_boxes):
            x1, y1, x2, y2, conf = [int(v) for v in box]
            # Ensure bounds
            h, w = mask.shape[:2]
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            if y2 > y1 and x2 > x1:
                # Count non-zero pixels of the mask inside this face's bounding box
                overlap_pixels = cv2.countNonZero(mask[y1:y2, x1:x2])
                box_area = (y2 - y1) * (x2 - x1)
                overlap_ratio = overlap_pixels / float(box_area) if box_area > 0 else 0
                
                # If more than 30% of the face box is covered by the person mask, it's a strong candidate
                if overlap_ratio > 0.3 and overlap_pixels > best_overlap:
                    best_overlap = overlap_pixels
                    best_idx = i

    # Strategy 2: If no mask overlap found, fallback to the click point
    if best_idx == -1 and click_point is not None:
        cx, cy = click_point
        # Find which bounding box contains the click point
        for i, box in enumerate(face_boxes):
            x1, y1, x2, y2, conf = box
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                best_idx = i
                break
                
    # Strategy 3: Fallback to highest confidence
    if best_idx == -1:
        best_idx = int(np.argmax(face_boxes[:, 4]))

    box = face_boxes[best_idx]
    lm = face_lms[best_idx]  # shape (5, 2): left_eye, right_eye, nose, mouth_l, mouth_r

    # Extract eye positions (RetinaFace landmark order)
    left_eye = (int(lm[0, 0]), int(lm[0, 1]))
    right_eye = (int(lm[1, 0]), int(lm[1, 1]))
    nose = (int(lm[2, 0]), int(lm[2, 1]))

    eye_center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2,
    )

    return {
        "face_center": nose,
        "eye_center": eye_center,
        "left_iris": left_eye,
        "right_iris": right_eye,
        "bbox": box[:4].tolist(),
        "confidence": float(box[4]),
    }


def select_person_point(image_bgr: np.ndarray, img_name: str) -> tuple:
    """
    Show image in Tkinter window for user to click on the person.
    Returns (x, y) in original image coordinates.
    """
    h, w = image_bgr.shape[:2]

    # Scale for display
    max_display = 800
    scale = min(max_display / w, max_display / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    disp = cv2.resize(image_bgr, (dw, dh))
    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(disp_rgb)

    clicked_point = [None]

    root = tk.Tk()
    root.title(f"Click on person — {img_name}")

    tk_img = ImageTk.PhotoImage(pil_img)
    canvas = tk.Canvas(root, width=dw, height=dh)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    # Instruction text
    canvas.create_text(dw // 2, 20, text="Click on the person's face",
                       fill="yellow", font=("Arial", 14, "bold"))

    def on_click(event):
        # Convert display coords back to original image coords
        ox = int(event.x / scale)
        oy = int(event.y / scale)
        clicked_point[0] = (ox, oy)
        root.destroy()

    canvas.bind("<Button-1>", on_click)
    root.mainloop()

    if clicked_point[0] is None:
        # Fallback: image center
        return (w // 2, h // 2)
    return clicked_point[0]


def generate_random_gaze_target(image_bgr: np.ndarray, face_center: tuple) -> tuple:
    """Generate a random gaze target visible from the person's face."""
    h, w = image_bgr.shape[:2]
    cx, cy = face_center

    # Random angle and distance
    angle = random.uniform(0, 2 * math.pi)
    min_dist = min(w, h) * 0.15
    max_dist = min(w, h) * 0.45
    dist = random.uniform(min_dist, max_dist)

    gx = int(cx + dist * math.cos(angle))
    gy = int(cy + dist * math.sin(angle))

    # Clamp to image bounds
    gx = max(10, min(w - 10, gx))
    gy = max(10, min(h - 10, gy))

    return (gx, gy)


# ══════════════════════════════════════════════════════════════
#  DESIRED GAZE COMPUTATION (virtual pinhole camera)
# ══════════════════════════════════════════════════════════════

def compute_desired_gaze_3d(face_center: tuple, gaze_target: tuple,
                            face_width_px: float, depth_map: np.ndarray,
                            image_shape: tuple,
                            target_mode: str = "virtual",
                            virtual_depth_ratio: float = 1.3) -> dict:
    """
    Compute desired gaze direction using a 3D vector derived from Depth-Anything-V2.
    """
    u1, v1 = int(face_center[0]), int(face_center[1])
    u2, v2 = int(gaze_target[0]), int(gaze_target[1])
    
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    
    f = face_width_px * VIRTUAL_FOCAL_MULTIPLIER
    if f < 1:
        f = 100.0
        
    # Depth map normalization to make it reasonable
    # DPT gives relative inverse depth maps. Higher value usually means closer.
    D = depth_map.astype(np.float32)
    d_min, d_max = np.min(D), np.max(D)
    if d_max - d_min > 1e-5:
        D_norm = (D - d_min) / (d_max - d_min)
    else:
        D_norm = D
        
    # Add epsilon to prevent div by zero
    D_norm = D_norm + 0.05 
    
    # Get disparity at eye and target. We use a small median window for robustness.
    ey1, ey2 = max(0, v1-2), min(h, v1+3)
    ex1, ex2 = max(0, u1-2), min(w, u1+3)
    d1 = np.median(D_norm[ey1:ey2, ex1:ex2])
    
    ty1, ty2 = max(0, v2-2), min(h, v2+3)
    tx1, tx2 = max(0, u2-2), min(w, u2+3)
    d2 = np.median(D_norm[ty1:ty2, tx1:tx2])
    
    # Assume Z is proportional to 1/D. Anchor Z1 to focal length f.
    Z1 = f
    if target_mode == "physical":
        Z2 = f * (d1 / d2)
    else:  # virtual
        Z2 = Z1 * virtual_depth_ratio
    
    # Calculate 3D points based on camera coordinates
    X1 = (u1 - cx) * Z1 / f
    Y1 = (v1 - cy) * Z1 / f
    
    X2 = (u2 - cx) * Z2 / f
    Y2 = (v2 - cy) * Z2 / f
    
    # Gaze vector V = P2 - P1
    Vx = X2 - X1
    Vy = Y2 - Y1
    Vz = Z2 - Z1
    
    print(f"  [Depth Vector] 3D Vector components (World space): dx={Vx:.2f}, dy={Vy:.2f}, dz={Vz:.2f}")
    
    # Normalize vector
    mag = math.sqrt(Vx**2 + Vy**2 + Vz**2)
    if mag < 1e-5:
        yaw_rad, pitch_rad = 0.0, 0.0
    else:
        # Gemini edits keep the face forward-facing. If depth puts the dot "behind" the user (Vz > 0), 
        # it results in an angle > 90. We mirror Z to the front hemisphere (-Z is towards the camera)
        Vz_front = -abs(Vz) 
        
        # Normalize vector
        Vx /= mag
        Vy /= mag
        Vz_front /= mag
        
        # Standard True Geometric Angles:
        # pitch > 0 means UP (Vy < 0)
        pitch_rad = math.asin(np.clip(-Vy, -1.0, 1.0))
        
        # yaw > 0 means RIGHT (Vx > 0). Vz_front is > 0 (pointing into screen away from eye)
        yaw_rad = math.atan2(Vx, -Vz_front)
    
    return {
        "yaw_rad": float(yaw_rad),
        "pitch_rad": float(pitch_rad),
        "yaw_deg": float(math.degrees(yaw_rad)),
        "pitch_deg": float(math.degrees(pitch_rad)),
        "focal_px": float(f),
        "Z_eye": float(Z1),
        "Z_target": float(Z2)
    }


def compute_angular_error(desired: dict, predicted: dict) -> float:
    """
    Compute angular error between desired and predicted gaze vectors.

    Converts yaw/pitch to 3D unit vectors, then computes the angle between them.
    """
    # Convert standard yaw/pitch to 3D unit vectors
    def to_vec(yaw, pitch):
        # yaw > 0 is Right (X > 0), pitch > 0 is Up (Y < 0). Z > 0 is Forward into screen.
        x = math.cos(pitch) * math.sin(yaw)
        y = -math.sin(pitch)
        z = math.cos(pitch) * math.cos(yaw)
        return np.array([x, y, z])

    v_desired = to_vec(desired["yaw_rad"], desired["pitch_rad"])
    v_predicted = to_vec(predicted["yaw_rad"], predicted["pitch_rad"])

    # Angle between vectors
    cos_angle = np.clip(np.dot(v_desired, v_predicted), -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


# ══════════════════════════════════════════════════════════════
#  PIPELINE PHASES (reused from gazectrl_pathb.py)
# ══════════════════════════════════════════════════════════════

def phase1_masking(image_bgr, person_pt, sam_predictor):
    """SAM masking (same as gazectrl_pathb.py)."""
    print("[Phase 1] SAM masking …")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(rgb)

    input_point = np.array([[person_pt[0], person_pt[1]]])
    input_label = np.array([1])

    masks, scores, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    best = np.argmax(scores)
    raw_mask = (masks[best] * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(raw_mask, kernel, iterations=2)

    print(f"  Mask coverage: {cv2.countNonZero(dilated)/(raw_mask.shape[0]*raw_mask.shape[1])*100:.1f}%")
    return raw_mask, dilated


def phase2_iris_shift_direct(
    image_bgr: np.ndarray,
    gaze_target: tuple,
    left_iris: tuple,
    right_iris: tuple,
    max_shift_px: int = 8,
    eye_radius: int = 18,
) -> np.ndarray:
    """Iris shift... (existing logic)"""
    # ... existing implementation omitted for brevity in chunk but I'll keep it correct in the actual edit ...
    print("[Phase 2] Iris pixel shift …")
    h, w = image_bgr.shape[:2]
    gx, gy = gaze_target
    result = image_bgr.copy()

    for label, (ix, iy) in [("L", left_iris), ("R", right_iris)]:
        dx, dy = gx - ix, gy - iy
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1e-3: continue
        shift_mag = min(mag * 0.1, max_shift_px)
        shift_x, shift_y = int(round(dx/mag*shift_mag)), int(round(dy/mag*shift_mag))
        
        ey1, ey2 = max(0, iy-eye_radius), min(h, iy+eye_radius)
        ex1, ex2 = max(0, ix-eye_radius), min(w, ix+eye_radius)
        yy, xx = np.mgrid[0:ey2-ey1, 0:ex2-ex1]
        dist = np.sqrt((xx-(ix-ex1))**2 + (yy-(iy-ey1))**2)
        pm = (np.exp(-0.5*(dist/(eye_radius*0.6))**2) * (dist<=eye_radius)).astype(np.float32)

        sy1, sy2 = max(0, ey1-shift_y), min(h, ey2-shift_y)
        sx1, sx2 = max(0, ex1-shift_x), min(w, ex2-shift_x)
        dy1, dy2 = max(0, sy1+shift_y), min(h, sy2+shift_y)
        dx1, dx2 = max(0, sx1+shift_x), min(w, sx2+shift_x)
        rh, rw = min(sy2-sy1, dy2-dy1), min(sx2-sx1, dx2-dx1)
        if rh<=0 or rw<=0: continue

        pm_patch = pm[max(0, dy1-ey1):max(0, dy1-ey1)+rh, max(0, dx1-ex1):max(0, dx1-ex1)+rw, np.newaxis]
        result[dy1:dy1+rh, dx1:dx1+rw] = np.clip(
            image_bgr[sy1:sy1+rh, sx1:sx1+rw]*pm_patch + result[dy1:dy1+rh, dx1:dx1+rw]*(1-pm_patch), 
            0, 255).astype(np.uint8)
    return result

def get_head_mask(image_bgr, face_info):
    """Create a mask covering only the head region using RetinaFace bbox."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    bbox = face_info["bbox"]  # [x1, y1, x2, y2]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x1 = max(0, int(bbox[0] - bw * 0.15))
    y1 = max(0, int(bbox[1] - bh * 0.3))  # extra room for hair
    x2 = min(w, int(bbox[2] + bw * 0.15))
    y2 = min(h, int(bbox[3] + bh * 0.1))
    cv2.ellipse(mask,
        center=((x1 + x2) // 2, (y1 + y2) // 2),
        axes=((x2 - x1) // 2, (y2 - y1) // 2),
        angle=0, startAngle=0, endAngle=360,
        color=255, thickness=-1)
    return cv2.GaussianBlur(mask, (21, 21), 0)  # soft edge

def get_head_fallback_mask(sam_mask: np.ndarray, click_point: tuple) -> np.ndarray:
    """Fallback if no face detected: estimate head region from SAM mask."""
    y, x = np.where(sam_mask > 0)
    if len(y) == 0: return sam_mask
    
    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)
    
    # Estimate head size as 25% of the bounding box's max dimension
    head_radius = int(max(y_max - y_min, x_max - x_min) * 0.25)
    head_radius = max(30, min(200, head_radius))  # clamped bounds
    
    cx, cy = click_point
    # If user clicked in the upper half of the body, assume they clicked the head
    if cy < y_min + (y_max - y_min) * 0.5:
        center_x, center_y = cx, cy
    else:
        # Otherwise, place the head at the top center of the mask
        top_mask = (y < y_min + head_radius * 2)
        if np.any(top_mask):
            center_x = int(np.median(x[top_mask]))
        else:
            center_x = (x_min + x_max) // 2
        center_y = y_min + head_radius

    fallback = np.zeros_like(sam_mask)
    cv2.ellipse(fallback,
        center=(center_x, center_y),
        axes=(int(head_radius * 0.8), head_radius),
        angle=0, startAngle=0, endAngle=360,
        color=255, thickness=-1)
    
    # Intersect with the SAM mask to keep exact silhouettes where possible
    result = cv2.bitwise_and(fallback, sam_mask)
    return cv2.GaussianBlur(result, (21, 21), 0)


def apply_vector_prompting(image_bgr, eye_center, gaze_target, length=60):
    """Draw a directional gradient shadow towards the target as a 'Vector Cue'."""
    print("[Phase 2] Applying Vector Prompting (Gradient Shadow) …")
    out = image_bgr.copy()
    ex, ey = eye_center
    gx, gy = gaze_target
    dx, dy = gx - ex, gy - ey
    mag = np.sqrt(dx**2 + dy**2)
    if mag < 1e-3: return out

    # Draw a subtle directional line (gradient-like)
    unit_x, unit_y = dx/mag, dy/mag
    for i in range(length):
        px, py = int(ex + i*unit_x), int(ey + i*unit_y)
        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            alpha = 1.0 - (i/length)
            color = out[py, px].astype(np.float32)
            # Mix with a 'directional shadow' (darker gray)
            out[py, px] = (color * (1 - 0.4*alpha)).astype(np.uint8)
    return out


def draw_red_dot(image_bgr, point, radius=8):
    """Draw red dot on image."""
    out = image_bgr.copy()
    cv2.circle(out, point, radius, (0, 0, 255), -1)
    cv2.circle(out, point, radius + 2, (0, 0, 180), 2)
    return out


def _gemini_edit(image_bgr, prompt_text, max_retries=3):
    """Core Gemini image edit call with auto-retry on 503. Returns edited BGR image or None."""
    from google import genai
    from google.genai import types
    import io, time

    client = genai.Client(api_key=GEMINI_API_KEY)
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    orig_h, orig_w = image_bgr.shape[:2]

    # Save PIL image to bytes for the new API
    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    for attempt in range(1, max_retries + 1):
        try:
            if "gemini-3" in GEMINI_MODEL:
                # New gemini-3 streaming API
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt_text),
                            types.Part(inline_data=types.Blob(
                                mime_type="image/png", data=img_bytes)),
                        ],
                    ),
                ]
                config = types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        image_size="1K",
                    ),
                    response_modalities=["IMAGE", "TEXT"],
                )
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                ):
                    if chunk.parts is None:
                        continue
                    for part in chunk.parts:
                        if part.inline_data and part.inline_data.data:
                            result_pil = Image.open(io.BytesIO(part.inline_data.data))
                            if result_pil.size != (orig_w, orig_h):
                                result_pil = result_pil.resize((orig_w, orig_h), Image.LANCZOS)
                            return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                print("  [Warning] No image in Gemini response")
                return None
            else:
                # Legacy gemini-2.x API
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[prompt_text, pil_img],
                    config=types.GenerateContentConfig(
                        response_modalities=["Image"],
                    ),
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        result_pil = Image.open(io.BytesIO(part.inline_data.data))
                        if result_pil.size != (orig_w, orig_h):
                            result_pil = result_pil.resize((orig_w, orig_h), Image.LANCZOS)
                        return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                print("  [Warning] No image in Gemini response")
                return None
        except Exception as e:
            err_str = str(e)
            if "503" in err_str and attempt < max_retries:
                wait = 30 * attempt  # 30s, 60s, 90s
                print(f"  [Retry {attempt}/{max_retries}] 503 overloaded — waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  [Error] Gemini failed: {e}")
                return None
    return None


# ── Pass 1 Prompt Strategies (gaze redirect, keep red dot) ──
PASS1_PROMPTS = {
    "standard": (
        "First, remove the green contour and remember the person inside it — that is the person to edit. "
        "Then make that person look at the red dot. "
        "Keep the red dot visible and do NOT move it."
    ),
    "contrastive": (
        "First, remove the green contour and remember the person inside it — that is the person to edit. "
        "Ignore their current gaze direction completely. "
        "Redirect their eyes, pupils, and head orientation to look precisely at the red dot. "
        "Preserve their original face identity — same face shape, skin tone, and features. "
        "Keep the red dot visible and do NOT move, remove, or repaint it."
    ),
    "vector_focus": (
        "First, remove the green contour and remember the person inside it — that is the person to edit. "
        "Follow the directional gradient shadow on the face as a guide for where to redirect the gaze. "
        "Make the person look at the red dot by aligning their eyes and head with the gradient direction. "
        "Keep the red dot visible and do NOT move it."
    ),
    "p1_gaze_only": (
        "First, remove the green contour and remember the person inside it — that is the person to edit. "
        "Then make that person look directly at the red dot by changing their eye direction and head orientation. "
        "Keep the person's face exactly the same — same identity, same skin, same features. "
        "Do NOT move, resize, or change the red dot in any way. Keep the red dot visible."
    ),
    "p1_contrastive_gaze": (
        "First, remove the green contour and remember the person inside it — that is the person to edit. "
        "Ignore their current gaze direction. Redirect their eyes and head to look precisely at the red dot. "
        "You MUST keep the person's original face identity — same face shape, skin tone, hair, and expression. "
        "Do NOT move, remove, or repaint the red dot. The red dot must stay exactly where it is."
    ),
}


def phase3_gemini_refine(warped_bgr, dilated_mask, prompt_style="standard", head_mask=None,
                         target_mode="virtual", target_desc=""):
    """Single-pass Gemini refinement (legacy/backward compatible)."""
    print("[Phase 3] Gemini refinement (single-pass) …")
    marked = warped_bgr.copy()
    contour_mask = head_mask if head_mask is not None else dilated_mask
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, contours, -1, (0, 255, 0), 2)

    prompt = PASS1_PROMPTS.get(prompt_style, PASS1_PROMPTS["standard"])
    if target_mode == "virtual":
        prompt += " Ignore everything except the person and the red dot."
    elif target_mode == "physical":
        desc = target_desc if target_desc else "a physical object or background element in the scene"
        prompt = f"The red dot marks a specific physical object: {desc}. Make the person look specifically at that object. {prompt}"

    result = _gemini_edit(marked, prompt)
    if result is not None:
        print("  -> Gemini refinement complete")
        return result, marked
    return warped_bgr.copy(), marked

def phase4_remove_anchor(image_bgr, gaze_target, dot_radius=18):
    """Remove red dot."""
    print("[Phase 4] Removing red dot …")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    gx, gy = gaze_target
    h, w = image_bgr.shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(region_mask, (gx, gy), dot_radius + 10, 255, -1)
    inpaint_mask = cv2.bitwise_and(red_mask, region_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    if cv2.countNonZero(inpaint_mask) == 0:
        cv2.circle(inpaint_mask, (gx, gy), dot_radius, 255, -1)

    return cv2.inpaint(image_bgr, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ══════════════════════════════════════════════════════════════
#  SINGLE IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════

def process_single_image(
    image_path: str,
    sam_predictor,
    warp_mode: str = "none",
    use_vector: bool = False,
    prompt_style: str = "standard",
    output_dir: Path = None,
    target_mode: str = "virtual",
    target_desc: str = "",
    virtual_depth_ratio: float = 1.3,
) -> dict:
    """
    Process a single image through the full pipeline + evaluation.
    No interactive UI — auto-detects everything.
    """
    img_name = Path(image_path).stem
    print(f"\n{'═' * 60}")
    print(f"  Processing: {img_name}")
    print(f"{'═' * 60}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [Error] Cannot read: {image_path}")
        return None

    # Resize if too large
    max_dim = 1024
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    h, w = image_bgr.shape[:2]

    # ── Detect face automatically (No manual selection) ──
    face_info = detect_face_center(image_bgr)
    if face_info is None:
        print(f"  [Warning] No face detected in {img_name} — skipping.")
        return None

    eye_center = face_info["eye_center"]
    left_iris = face_info.get("left_iris")
    right_iris = face_info.get("right_iris")
    person_pt = face_info["face_center"]  # use nose as person point
    print(f"  Auto-detected face center (RetinaFace): {eye_center}")

    # ── Phase 1: SAM masking ──
    raw_mask, dilated_mask = phase1_masking(image_bgr, person_pt, sam_predictor)

    # ── Phase 0.5: Estimate Depth Map ──
    print("[Eval] Estimating depth map ...")
    depth_model = get_depth_anything()
    depth_map = depth_model.infer_image(image_bgr)

    # ── Generate random gaze target ──
    gaze_target = generate_random_gaze_target(image_bgr, eye_center)
    dx = gaze_target[0] - person_pt[0]
    dy = gaze_target[1] - person_pt[1]
    dist = math.sqrt(dx**2 + dy**2)
    angle_deg = math.degrees(math.atan2(-dy, dx))  # screen coords: y-down
    print(f"  Face center: {person_pt}")
    print(f"  Gaze target: {gaze_target}")
    print(f"  Vector (2D): dx={dx:+d} dy={dy:+d}  dist={dist:.1f}px  angle={angle_deg:.1f}°")

    # ── Draw red dot ──
    image_with_dot = draw_red_dot(image_bgr, gaze_target)

    # ── Phase 2: Warp / Neutralize ──
    if warp_mode == "none":
        warped = image_with_dot.copy()
        print("[Phase 2] SKIPPED")
    elif warp_mode == "shift":
        warped = phase2_iris_shift_direct(
            image_with_dot, gaze_target, left_iris, right_iris
        )
    else:
        # TPS warp
        sys.path.insert(0, str(SCRIPT_DIR))
        from gazectrl_pathb import phase2_tps_warp
        warped = phase2_tps_warp(
            image_with_dot, gaze_target, dilated_mask,
            left_iris=left_iris, right_iris=right_iris
        )

    if use_vector:
        warped = apply_vector_prompting(warped, eye_center, gaze_target)

    # ── Phase 3: Gemini (redirect gaze, keep red dot) ──
    head_mask = get_head_mask(image_bgr, face_info)
    gemini_result, marked_img = phase3_gemini_refine(
        warped, dilated_mask, prompt_style=prompt_style, head_mask=head_mask,
        target_mode=target_mode, target_desc=target_desc)

    # ── Phase 4: Remove red dot (OpenCV inpaint) ──
    final = phase4_remove_anchor(gemini_result, gaze_target)

    # ── L2CS-Net gaze estimation on final output ──
    print("[Eval] L2CS-Net gaze estimation …")
    predicted_gaze = estimate_gaze_l2cs(final)

    # ── Compute desired gaze (virtual pinhole) ──
    face_width = math.sqrt(
        (right_iris[0] - left_iris[0]) ** 2 +
        (right_iris[1] - left_iris[1]) ** 2
    ) * 3  # approximate face width from inter-iris distance

    # Compute using face_center (person_pt) instead of eye_center
    desired_gaze = compute_desired_gaze_3d(person_pt, gaze_target, face_width, depth_map, image_bgr.shape, target_mode, virtual_depth_ratio)

    # ── L2CS-Net gaze estimation on final output ──
    print("[Eval] L2CS-Net gaze estimation …")
    predicted_gaze, raw_l2cs_results = estimate_gaze_l2cs(final)

    # ── Angular error ──
    if predicted_gaze is not None:
        angular_error = compute_angular_error(desired_gaze, predicted_gaze)
        print(f"  Desired:  yaw={desired_gaze['yaw_deg']:.1f}°  pitch={desired_gaze['pitch_deg']:.1f}°")
        print(f"  L2CS:     yaw={predicted_gaze['yaw_deg']:.1f}°  pitch={predicted_gaze['pitch_deg']:.1f}°")
        print(f"  Error:    {angular_error:.2f}°")
    else:
        angular_error = None
        print("  [Warning] L2CS-Net detected no face in output")

    # ── Save outputs ──
    if output_dir:
        img_dir = output_dir / img_name
        img_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_dir / "01_input.png"), image_bgr)
        cv2.imwrite(str(img_dir / "02_mask.png"),
                     cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(img_dir / "03_with_dot.png"), image_with_dot)
        cv2.imwrite(str(img_dir / "04_warped.png"), warped)
        cv2.imwrite(str(img_dir / "04b_contour.png"), marked_img)
        cv2.imwrite(str(img_dir / "05_gemini.png"), gemini_result)
        cv2.imwrite(str(img_dir / "06_final.png"), final)
        
        if raw_l2cs_results is not None:
            from l2cs import render
            l2cs_rendered = render(final.copy(), raw_l2cs_results)
            cv2.imwrite(str(img_dir / "07_l2cs_gaze.png"), l2cs_rendered)

        eval_data = {
            "image": img_name,
            "eye_center": list(eye_center),
            "gaze_target": list(gaze_target),
            "desired_gaze": desired_gaze,
            "predicted_gaze": predicted_gaze,
            "angular_error_deg": angular_error,
            "warp_mode": warp_mode,
            "use_vector": use_vector,
            "prompt_style": prompt_style,
            "target_mode": target_mode,
        }
        with open(img_dir / "evaluation.json", "w") as f:
            json.dump(eval_data, f, indent=2)

    return {
        "image": img_name,
        "angular_error_deg": angular_error,
        "desired_gaze": desired_gaze,
        "predicted_gaze": predicted_gaze,
    }


# ══════════════════════════════════════════════════════════════
#  ANALYSIS & VISUALIZATION
# ══════════════════════════════════════════════════════════════

def generate_analysis(all_results: list, output_dir: Path):
    """Generate error distribution plots and summary statistics."""
    print(f"\n{'═' * 60}")
    print("  Generating Analysis")
    print(f"{'═' * 60}")

    # Filter valid results
    valid = [r for r in all_results if r and r["angular_error_deg"] is not None]
    errors = [r["angular_error_deg"] for r in valid]

    if not errors:
        print("  [Error] No valid results to analyze")
        return

    errors = np.array(errors)

    # ── Summary statistics ──
    stats = {
        "total_images": len(all_results),
        "valid_results": len(valid),
        "failed": len(all_results) - len(valid),
        "mean_error_deg": float(np.mean(errors)),
        "median_error_deg": float(np.median(errors)),
        "std_error_deg": float(np.std(errors)),
        "min_error_deg": float(np.min(errors)),
        "max_error_deg": float(np.max(errors)),
        "p25_error_deg": float(np.percentile(errors, 25)),
        "p75_error_deg": float(np.percentile(errors, 75)),
        "p90_error_deg": float(np.percentile(errors, 90)),
        "errors": errors.tolist(),
        "per_image": valid,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # ── Text report ──
    report = f"""GazeCtrl Path B — Batch Evaluation Report
{'=' * 50}
Total images processed: {stats['total_images']}
Valid results:          {stats['valid_results']}
Failed:                 {stats['failed']}

Angular Error (degrees):
  Mean:     {stats['mean_error_deg']:.2f}°
  Median:   {stats['median_error_deg']:.2f}°
  Std:      {stats['std_error_deg']:.2f}°
  Min:      {stats['min_error_deg']:.2f}°
  Max:      {stats['max_error_deg']:.2f}°
  P25:      {stats['p25_error_deg']:.2f}°
  P75:      {stats['p75_error_deg']:.2f}°
  P90:      {stats['p90_error_deg']:.2f}°
"""
    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report)
    print(report)

    # ── Error Distribution Histogram ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=20, color='#4CAF50', edgecolor='white', alpha=0.85)
    ax1.axvline(np.mean(errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(errors):.1f}°')
    ax1.axvline(np.median(errors), color='blue', linestyle='--',
                label=f'Median: {np.median(errors):.1f}°')
    ax1.set_xlabel('Angular Error (degrees)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Gaze Angular Error Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(errors, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='#81C784', alpha=0.7))
    ax2.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax2.set_title('Error Box Plot', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "error_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: error_distribution.png")

    # ── Per-image error bar chart (if not too many) ──
    if len(valid) <= 50:
        fig, ax = plt.subplots(figsize=(max(10, len(valid) * 0.3), 5))
        names = [r["image"][:15] for r in valid]
        ax.bar(range(len(errors)), errors, color='#4CAF50', alpha=0.8)
        ax.set_xticks(range(len(errors)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Angular Error (degrees)')
        ax.set_title('Per-Image Gaze Angular Error')
        ax.axhline(np.mean(errors), color='red', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(str(output_dir / "per_image_errors.png"), dpi=150)
        plt.close()
        print(f"  Saved: per_image_errors.png")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GazeCtrl Path B — Batch Evaluation"
    )
    parser.add_argument("--input-dir", type=str, default="./eval_images",
                        help="Directory containing test images")
    parser.add_argument("--output-dir", type=str, default="./batch_outputs",
                        help="Directory for outputs")
    parser.add_argument("--model", type=str, default="gemini-3-pro-image-preview",
                        help="Gemini model name (e.g. gemini-3-pro-image-preview)")
    parser.add_argument("--warp-mode", choices=["none", "shift", "tps"],
                        default="none", help="Phase 2 warp method")
    parser.add_argument("--use-vector", action="store_true",
                        help="Enable directional vector prompting")
    parser.add_argument("--prompt-style",
                        choices=["standard", "contrastive", "vector_focus",
                                 "p1_gaze_only", "p1_contrastive_gaze"],
                        default="standard", help="Gemini prompt strategy")
    parser.add_argument("--target-mode", choices=["virtual", "physical"],
                        default="virtual", help="Target interpretation mode")
    parser.add_argument("--target-desc", type=str, default="",
                        help="Description of physical target (only used in physical mode)")
    parser.add_argument("--virtual-depth-ratio", type=float, default=1.3,
                        help="Ratio of target depth to eye depth in virtual mode")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Maximum images to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for gaze target generation")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set model globally
    global GEMINI_MODEL
    GEMINI_MODEL = args.model

    print("=" * 60)
    print("  GazeCtrl Path B — Batch Evaluation")
    print(f"  Input:     {args.input_dir}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Model:     {args.model}")
    print(f"  Warp mode: {args.warp_mode}")
    print(f"  Vector:    {args.use_vector}")
    print(f"  Prompt:    {args.prompt_style}")
    print(f"  Target Mode: {args.target_mode}")
    if args.target_mode == "virtual":
        print(f"  Virtual Depth Ratio: {args.virtual_depth_ratio}")
    else:
        print(f"  Target Desc: {args.target_desc}")
    print(f"  Max imgs:  {args.max_images}")
    print("=" * 60)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(str(input_dir / ext)))
    image_files.sort()
    image_files = image_files[:args.max_images]

    if not image_files:
        print(f"\n[Error] No images in {input_dir}")
        print("  Place test images in the eval_images/ folder.")
        sys.exit(1)

    print(f"\nFound {len(image_files)} image(s)")

    # Load models
    sam_predictor = load_sam_predictor()
    _ = get_l2cs_pipeline()  # pre-load
    _ = get_depth_anything()  # pre-load

    # Process all images
    all_results = []
    for idx, img_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}]")
        try:
            result = process_single_image(
                img_path, sam_predictor,
                warp_mode=args.warp_mode,
                use_vector=args.use_vector,
                prompt_style=args.prompt_style,
                output_dir=output_dir,
                target_mode=args.target_mode,
                target_desc=args.target_desc,
                virtual_depth_ratio=args.virtual_depth_ratio,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  [Error] Failed: {e}")
            all_results.append(None)

    # Generate analysis
    generate_analysis(all_results, output_dir)

    print("\n" + "=" * 60)
    print("  Batch evaluation complete!")
    print(f"  Results in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
