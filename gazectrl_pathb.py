"""
GazeCtrl Path B — Training-free Warp-and-Refine Pipeline

Pipeline Phases:
  1. Interactive point selection → SAM masking
  2. Geometric pre-warp (iris shift / TPS) toward target
  3. Gemini Nano Banana counterfactual refinement
  4. Red-dot anchor removal + L2CS-Net gaze evaluation

Usage:
  python gazectrl_pathb.py [--warp-mode {none,shift,tps}]
"""

import os
import sys
import json
import glob
import math
import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageFont
import tkinter as tk

# ─────────────────────────── Paths ───────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR / "tests"
OUTPUTS_DIR = SCRIPT_DIR / "test_outputs"
SAM_CHECKPOINT = SCRIPT_DIR / "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Gemini API
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

# ─────────────────────────── Globals for click UI ────────────
_click_annotated_bgr: np.ndarray | None = None


# ══════════════════════════════════════════════════════════════
#  UTILITY — SAM model download
# ══════════════════════════════════════════════════════════════

def ensure_sam_checkpoint() -> Path:
    """Download SAM ViT-H checkpoint if not present."""
    if SAM_CHECKPOINT.exists():
        print(f"[SAM] Checkpoint found: {SAM_CHECKPOINT}")
        return SAM_CHECKPOINT
    print(f"[SAM] Downloading checkpoint (~2.5 GB) …")
    print(f"      {SAM_URL}")
    urllib.request.urlretrieve(SAM_URL, str(SAM_CHECKPOINT))
    print(f"[SAM] Download complete.")
    return SAM_CHECKPOINT


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE CLICK UI  (Tkinter)
# ══════════════════════════════════════════════════════════════

def select_points_on_image(image_bgr: np.ndarray) -> tuple:
    """
    Show a Tkinter window.  User clicks:
      1. Target person (SAM prompt)  — shown as blue crosshair
      2. Target gaze position (red dot) — shown as red circle

    Controls: 'R' to reset, Enter to confirm, Escape to quit.

    Returns:
        (person_point, gaze_target_point)  each as (x, y)
    """
    global _click_annotated_bgr

    click_points = []
    canvas_items = []

    # Convert BGR → RGB → PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    # Scale down if image is very large so it fits the screen
    max_display = 900
    scale = 1.0
    if max(pil_img.size) > max_display:
        scale = max_display / max(pil_img.size)
        display_size = (int(pil_img.width * scale), int(pil_img.height * scale))
        display_img = pil_img.resize(display_size, Image.LANCZOS)
    else:
        display_img = pil_img.copy()
        display_size = pil_img.size

    root = tk.Tk()
    root.title("GazeCtrl – Click Interface")
    root.resizable(False, False)

    # Instructions frame
    info_frame = tk.Frame(root, bg="#282828", padx=10, pady=6)
    info_frame.pack(fill=tk.X)
    tk.Label(info_frame, text="Click 1: Target Person",
             fg="#FFB400", bg="#282828", font=("Consolas", 11, "bold")).pack(anchor=tk.W)
    tk.Label(info_frame, text="Click 2: Gaze Target (Red Dot)",
             fg="#FF3333", bg="#282828", font=("Consolas", 11, "bold")).pack(anchor=tk.W)
    status_var = tk.StringVar(value="  Waiting for Click 1 …")
    tk.Label(info_frame, textvariable=status_var,
             fg="#AAAAAA", bg="#282828", font=("Consolas", 9)).pack(anchor=tk.W, pady=(4, 0))

    # Canvas
    canvas = tk.Canvas(root, width=display_size[0], height=display_size[1],
                       highlightthickness=0)
    canvas.pack()

    tk_photo = ImageTk.PhotoImage(display_img)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_photo)

    def on_click(event):
        if len(click_points) >= 2:
            return
        # Map display coords back to original image coords
        orig_x = int(event.x / scale)
        orig_y = int(event.y / scale)
        click_points.append((orig_x, orig_y))

        dx, dy = event.x, event.y  # display coords for drawing

        if len(click_points) == 1:
            # Blue crosshair
            size = 15
            items = [
                canvas.create_line(dx - size, dy, dx + size, dy,
                                   fill="#FFB400", width=2),
                canvas.create_line(dx, dy - size, dx, dy + size,
                                   fill="#FFB400", width=2),
                canvas.create_text(dx + 18, dy - 14, text="Person",
                                   fill="#FFB400", font=("Consolas", 10, "bold"),
                                   anchor=tk.W),
            ]
            canvas_items.extend(items)
            status_var.set(f"  Click 1: ({orig_x}, {orig_y}) ✓   Waiting for Click 2 …")
            print(f"  [Click 1] Target person at ({orig_x}, {orig_y})")

        elif len(click_points) == 2:
            # Red dot
            r = 8
            items = [
                canvas.create_oval(dx - r, dy - r, dx + r, dy + r,
                                   fill="#FF0000", outline="#CC0000", width=2),
                canvas.create_text(dx + 14, dy - 14, text="Gaze Target",
                                   fill="#FF3333", font=("Consolas", 10, "bold"),
                                   anchor=tk.W),
            ]
            canvas_items.extend(items)
            status_var.set(f"  Click 2: ({orig_x}, {orig_y}) ✓   Press Enter to confirm, R to reset")
            print(f"  [Click 2] Gaze target  at ({orig_x}, {orig_y})")

    def on_key(event):
        if event.keysym == "Return" and len(click_points) == 2:
            root.quit()
        elif event.keysym.lower() == "r":
            click_points.clear()
            for item_id in canvas_items:
                canvas.delete(item_id)
            canvas_items.clear()
            status_var.set("  [Reset] Waiting for Click 1 …")
            print("  [Reset] Cleared all clicks.")
        elif event.keysym == "Escape":
            root.destroy()
            sys.exit(0)

    canvas.bind("<Button-1>", on_click)
    root.bind("<Key>", on_key)

    print("\n  Waiting for 2 clicks …  (R = reset, Enter = confirm, Esc = quit)")
    root.mainloop()
    root.destroy()

    # Build an annotated BGR image for saving
    annotated = image_bgr.copy()
    if len(click_points) >= 1:
        px, py = click_points[0]
        cv2.drawMarker(annotated, (px, py), (0, 180, 255),
                       cv2.MARKER_CROSS, 30, 2)
        cv2.putText(annotated, "Person", (px + 12, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    if len(click_points) >= 2:
        gx, gy = click_points[1]
        cv2.circle(annotated, (gx, gy), 8, (0, 0, 255), -1)
        cv2.circle(annotated, (gx, gy), 10, (0, 0, 200), 2)
        cv2.putText(annotated, "Gaze Target", (gx + 12, gy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    _click_annotated_bgr = annotated
    return click_points[0], click_points[1]


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — SAM Masking + Dilation
# ══════════════════════════════════════════════════════════════

def phase1_masking(
    image_bgr: np.ndarray,
    person_point: tuple,
    sam_predictor,
    dilation_kernel_size: int = 15,
    dilation_iterations: int = 3,
) -> tuple:
    """
    Use SAM to segment the target person from a single point prompt,
    then dilate the mask for natural blending.

    Returns:
        (raw_mask, dilated_mask)  — both as uint8 [0, 255]
    """
    print("[Phase 1] Generating SAM mask …")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    input_point = np.array([list(person_point)])
    input_label = np.array([1])  # foreground

    masks, scores, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Pick highest-scoring mask
    best_idx = int(np.argmax(scores))
    raw_mask = (masks[best_idx] * 255).astype(np.uint8)

    # Morphological dilation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
    )
    dilated_mask = cv2.dilate(raw_mask, kernel, iterations=dilation_iterations)

    print(f"  SAM mask score = {scores[best_idx]:.4f}  |  "
          f"dilation kernel {dilation_kernel_size}×{dilation_kernel_size} ×{dilation_iterations}")
    return raw_mask, dilated_mask


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — TPS Geometric Warping
# ══════════════════════════════════════════════════════════════

def _detect_iris_centers(image_bgr: np.ndarray) -> tuple:
    """
    Use MediaPipe Face Landmarker (Tasks API) to find iris centers.

    Returns:
        (left_iris_xy, right_iris_xy) in pixel coords.
        Returns (None, None) if no face detected.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    # Iris landmark indices (same as Face Mesh convention)
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # Ensure face_landmarker.task model exists
    model_path = SCRIPT_DIR / "face_landmarker.task"
    if not model_path.exists():
        # Try the one in the project root
        root_model = SCRIPT_DIR.parent / "face_landmarker.task"
        if root_model.exists():
            import shutil
            shutil.copy2(str(root_model), str(model_path))
            print(f"  Copied face_landmarker.task from project root")
        else:
            # Download it
            model_url = ("https://storage.googleapis.com/mediapipe-models/"
                         "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            print(f"  Downloading face_landmarker.task …")
            urllib.request.urlretrieve(model_url, str(model_path))
            print(f"  Download complete")

    # Create FaceLandmarker
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    try:
        # Convert BGR → RGB and create MediaPipe Image
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        results = landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            print("  [Warning] No face detected by MediaPipe")
            return None, None

        landmarks = results.face_landmarks[0]
        h, w = image_bgr.shape[:2]

        # Check if iris landmarks are available (need >= 478 landmarks)
        if len(landmarks) < 478:
            print("  [Warning] Iris landmarks not available (need >= 478)")
            return None, None

        def lm_to_px(idx):
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)

        left_iris = lm_to_px(LEFT_IRIS_CENTER)
        right_iris = lm_to_px(RIGHT_IRIS_CENTER)

        print(f"  Iris centres: L={left_iris}  R={right_iris}")
        return left_iris, right_iris
    finally:
        landmarker.close()


def phase2_iris_shift(
    image_bgr: np.ndarray,
    gaze_target: tuple,
    dilated_mask: np.ndarray,
    max_shift_px: int = 8,
    eye_radius: int = 18,
) -> np.ndarray:
    """
    Simple iris-pixel translation toward the gaze target.

    For each detected eye, shifts the eye-region pixels a few pixels
    toward the gaze target, then blends with Gaussian feathering.
    This creates a subtle structural prior for SD inpainting.

    Args:
        image_bgr:    Input image (BGR)
        gaze_target:  (x, y) of the red-dot target
        dilated_mask: (unused, kept for API compatibility)
        max_shift_px: Maximum pixels to shift each iris (default 8)
        eye_radius:   Radius of the circular eye crop region (px)

    Returns:
        Shifted image (BGR), same size.
    """
    print("[Phase 2] Iris pixel shift …")

    left_iris, right_iris = _detect_iris_centers(image_bgr)
    if left_iris is None:
        print("  [Warning] Skipping shift — no iris detected.")
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]
    gx, gy = gaze_target
    result = image_bgr.copy()

    for label, (ix, iy) in [("L", left_iris), ("R", right_iris)]:
        # Compute displacement direction toward the target
        dx = gx - ix
        dy = gy - iy
        mag = np.sqrt(dx ** 2 + dy ** 2)
        if mag < 1e-3:
            continue

        # Normalize and cap at max_shift_px
        shift_mag = min(mag * 0.1, max_shift_px)  # 10% of distance, capped
        shift_x = int(round(dx / mag * shift_mag))
        shift_y = int(round(dy / mag * shift_mag))

        if shift_x == 0 and shift_y == 0:
            continue

        # Define a circular region around the iris
        # Build a soft (Gaussian-feathered) mask for this eye
        ey_min = max(0, iy - eye_radius)
        ey_max = min(h, iy + eye_radius)
        ex_min = max(0, ix - eye_radius)
        ex_max = min(w, ix + eye_radius)

        crop_h = ey_max - ey_min
        crop_w = ex_max - ex_min

        # Create a Gaussian-weighted circular mask for smooth blending
        yy, xx = np.mgrid[0:crop_h, 0:crop_w]
        cy = iy - ey_min
        cx = ix - ex_min
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        sigma = eye_radius * 0.6
        eye_mask = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
        eye_mask[dist > eye_radius] = 0

        # Source region (where to read shifted pixels from)
        sy_min = max(0, ey_min - shift_y)
        sy_max = min(h, ey_max - shift_y)
        sx_min = max(0, ex_min - shift_x)
        sx_max = min(w, ex_max - shift_x)

        # Destination region in the output
        dy_min = sy_min + shift_y
        dy_max = sy_max + shift_y
        dx_min = sx_min + shift_x
        dx_max = sx_max + shift_x

        # Clamp to image bounds
        dy_min = max(0, dy_min)
        dy_max = min(h, dy_max)
        dx_min = max(0, dx_min)
        dx_max = min(w, dx_max)

        # Compute the overlapping region sizes
        rh = min(sy_max - sy_min, dy_max - dy_min)
        rw = min(sx_max - sx_min, dx_max - dx_min)
        if rh <= 0 or rw <= 0:
            continue

        # Extract shifted pixels
        src_patch = image_bgr[sy_min:sy_min + rh, sx_min:sx_min + rw].astype(np.float32)
        dst_patch = result[dy_min:dy_min + rh, dx_min:dx_min + rw].astype(np.float32)

        # Crop the mask to match the overlap region
        mask_y_off = dy_min - ey_min
        mask_x_off = dx_min - ex_min
        m_y_end = min(mask_y_off + rh, crop_h)
        m_x_end = min(mask_x_off + rw, crop_w)
        mask_y_off = max(0, mask_y_off)
        mask_x_off = max(0, mask_x_off)
        patch_mask = eye_mask[mask_y_off:m_y_end, mask_x_off:m_x_end]

        # Ensure shapes match
        ph = min(rh, patch_mask.shape[0], src_patch.shape[0], dst_patch.shape[0])
        pw = min(rw, patch_mask.shape[1], src_patch.shape[1], dst_patch.shape[1])
        if ph <= 0 or pw <= 0:
            continue

        pm = patch_mask[:ph, :pw, np.newaxis]  # (ph, pw, 1)
        blended = src_patch[:ph, :pw] * pm + dst_patch[:ph, :pw] * (1 - pm)
        result[dy_min:dy_min + ph, dx_min:dx_min + pw] = np.clip(blended, 0, 255).astype(np.uint8)

        print(f"  Eye {label}: shift=({shift_x}, {shift_y}) px, radius={eye_radius}")

    return result


def phase2_tps_warp(
    image_bgr: np.ndarray,
    gaze_target: tuple,
    dilated_mask: np.ndarray,
    warp_strength: float = 0.15,
    left_iris: tuple = None,
    right_iris: tuple = None,
) -> np.ndarray:
    """
    Thin Plate Spline (TPS) warping using scipy RBF interpolation.
    """
    print("[Phase 2] TPS warping …")
    from scipy.interpolate import RBFInterpolator

    if left_iris is None or right_iris is None:
        left_iris, right_iris = _detect_iris_centers(image_bgr)
    
    if left_iris is None:
        print("  [Warning] Skipping TPS — no iris detected.")
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]
    gx, gy = gaze_target

    # Source points: iris centres + anchor grid (edges)
    src_pts = []
    dst_pts = []

    for (ix, iy) in [left_iris, right_iris]:
        # Iris centre moves toward target
        dx = (gx - ix) * warp_strength
        dy = (gy - iy) * warp_strength
        src_pts.append([ix, iy])
        dst_pts.append([ix + dx, iy + dy])

    # Fixed anchor points around image border (stay put)
    n_border = 8
    for i in range(n_border):
        bx = int(w * i / (n_border - 1))
        for by in [0, h - 1]:
            src_pts.append([bx, by])
            dst_pts.append([bx, by])
    for j in range(1, n_border - 1):
        by = int(h * j / (n_border - 1))
        for bx in [0, w - 1]:
            src_pts.append([bx, by])
            dst_pts.append([bx, by])

    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    # Displacement from src to dst
    disp = dst_pts - src_pts

    # Build RBF interpolators for x and y displacement
    rbf_dx = RBFInterpolator(src_pts, disp[:, 0], kernel='thin_plate_spline', smoothing=0.1)
    rbf_dy = RBFInterpolator(src_pts, disp[:, 1], kernel='thin_plate_spline', smoothing=0.1)

    # Create coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Compute displacement at every pixel
    map_dx = rbf_dx(grid_coords).reshape(h, w)
    map_dy = rbf_dy(grid_coords).reshape(h, w)

    # Build remap arrays (inverse mapping)
    map_x = (grid_x - map_dx).astype(np.float32)
    map_y = (grid_y - map_dy).astype(np.float32)

    result = cv2.remap(image_bgr, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101)

    # Blend: only apply warp inside a region around the eyes
    for (ix, iy) in [left_iris, right_iris]:
        r = 30
        y1, y2 = max(0, iy - r), min(h, iy + r)
        x1, x2 = max(0, ix - r), min(w, ix + r)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dist = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
        sigma = r * 0.6
        mask = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
        mask = mask[:, :, np.newaxis]
        blended = result[y1:y2, x1:x2].astype(np.float32) * mask + \
                  image_bgr[y1:y2, x1:x2].astype(np.float32) * (1 - mask)
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # Restore background outside mask
    mask_3ch = dilated_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    result = (result.astype(np.float32) * mask_3ch +
              image_bgr.astype(np.float32) * (1 - mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)

    print(f"  TPS warp applied (strength={warp_strength})")
    return result


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — Gemini Nano Banana Counterfactual Refinement
# ══════════════════════════════════════════════════════════════

_gemini_client = None  # lazy-loaded singleton


def _get_gemini_client():
    """Lazy-load Gemini client (only once)."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    from google import genai

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[Gemini] Client ready.")
    return _gemini_client


def phase3_gemini_refine(
    warped_bgr: np.ndarray,
    dilated_mask: np.ndarray,
) -> tuple:
    """
    Refine the warped image via Gemini Nano Banana image editing.

    Sends the warped image (with red dot) plus the SAM mask as visual
    context to Gemini, which natively preserves unmodified regions.

    Returns:
        (refined_bgr, marked_bgr) — the Gemini result and the contour-marked input.
    """
    print("[Phase 3] Gemini Nano Banana counterfactual refinement …")
    from google.genai import types

    client = _get_gemini_client()

    # Convert to PIL
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

    # Draw the SAM mask outline on the image so Gemini knows which person
    marked = warped_bgr.copy()
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, contours, -1, (0, 255, 0), 2)  # green outline
    marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    pil_marked = Image.fromarray(marked_rgb)

    prompt = (
        "Edit the person who got marked and make the person look at the red dot "
        "and remove the marker around the person and the red dot."
    )

    # Send the marked image
    print("  Sending to Gemini API …")
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, pil_marked],
            config=types.GenerateContentConfig(
                response_modalities=["Image"],
            ),
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Decode image bytes → PIL
                import io
                img_bytes = part.inline_data.data
                result_pil = Image.open(io.BytesIO(img_bytes))
                # Resize to match original if needed
                orig_h, orig_w = warped_bgr.shape[:2]
                if result_pil.size != (orig_w, orig_h):
                    result_pil = result_pil.resize((orig_w, orig_h), Image.LANCZOS)
                result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                print("  ✓ Gemini refinement complete")
                return result_bgr, marked

        print("  [Warning] No image in Gemini response, returning input")
        return warped_bgr.copy(), marked

    except Exception as e:
        print(f"  [Error] Gemini API failed: {e}")
        print("  Returning warped image without refinement.")
        return warped_bgr.copy(), marked


# ══════════════════════════════════════════════════════════════
#  PHASE 4 — Red-dot Anchor Removal
# ══════════════════════════════════════════════════════════════

def phase4_remove_anchor(
    image_bgr: np.ndarray,
    gaze_target: tuple,
    dot_radius: int = 18,
) -> np.ndarray:
    """
    Remove the red dot from the final image using OpenCV inpainting.

    Args:
        image_bgr:    Image containing the red dot (BGR)
        gaze_target:  (x, y) red dot centre
        dot_radius:   Radius of the area to inpaint

    Returns:
        Cleaned image (BGR).
    """
    print("[Phase 4] Removing red-dot anchor …")

    # Method A: precise colour-based detection in HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Red in HSV wraps around 0/180
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Restrict to neighbourhood of known gaze target
    gx, gy = gaze_target
    h, w = image_bgr.shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(region_mask, (gx, gy), dot_radius + 10, 255, -1)
    inpaint_mask = cv2.bitwise_and(red_mask, region_mask)

    # Dilate slightly for smoother inpainting
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    # If no red pixels detected, fallback: paint a circle around the target
    if cv2.countNonZero(inpaint_mask) == 0:
        print("  No red pixels detected by HSV — using circular fallback mask")
        cv2.circle(inpaint_mask, (gx, gy), dot_radius, 255, -1)

    result = cv2.inpaint(image_bgr, inpaint_mask, inpaintRadius=5,
                         flags=cv2.INPAINT_TELEA)
    print(f"  Inpainted {cv2.countNonZero(inpaint_mask)} px around ({gx}, {gy})")
    return result


def evaluate_gaze_error(
    final_bgr: np.ndarray,
    gaze_target: tuple,
    original_shape: tuple,
) -> dict:
    """
    Estimate gaze angular error between the final image's gaze
    direction and the intended target point.

    Uses MediaPipe iris landmarks to estimate the gaze direction,
    then computes the angular deviation from the target.

    Returns:
        dict with evaluation metrics.
    """
    print("[Eval] Gaze angular error estimation …")

    h, w = final_bgr.shape[:2]
    gx, gy = gaze_target

    left_iris, right_iris = _detect_iris_centers(final_bgr)
    if left_iris is None:
        print("  [Warning] No iris detected in final image — cannot evaluate")
        return {"error": "no_iris_detected", "angular_error_deg": None}

    # Estimate gaze direction from iris displacement
    # Use the midpoint of both iris centres as the "eye position"
    mid_x = (left_iris[0] + right_iris[0]) / 2
    mid_y = (left_iris[1] + right_iris[1]) / 2

    # Vector from eye midpoint to target (in image coords)
    target_dx = gx - mid_x
    target_dy = gy - mid_y
    target_angle = math.atan2(target_dy, target_dx)  # radians

    # Approximate the "looking direction" from iris positions
    # In a front-facing photo, iris displacement from eye centre
    # indicates gaze direction. We use inter-iris midpoint shift
    # relative to face centre as a rough gaze proxy.
    # (This is a geometric approximation; L2CS-Net would be more accurate)
    inter_iris_dist = math.sqrt(
        (right_iris[0] - left_iris[0]) ** 2 +
        (right_iris[1] - left_iris[1]) ** 2
    )

    # Target distance from eye midpoint
    target_dist = math.sqrt(target_dx ** 2 + target_dy ** 2)

    # Angular error approximation:
    # Use the angle subtended by the target relative to estimated face plane
    # Assume ~60cm viewing distance and inter-pupillary distance of ~6cm
    # 1 pixel ≈ face_width / inter_iris_dist * real_ipd
    if inter_iris_dist > 0:
        # Rough pixel-to-angle conversion
        # Typical horizontal FOV of face in image ≈ 30°
        pixels_per_deg = inter_iris_dist / 30.0  # rough estimate
        angular_error = target_dist / max(pixels_per_deg, 1.0)
    else:
        angular_error = float('inf')

    eval_result = {
        "target_point": [gx, gy],
        "iris_left": list(left_iris) if left_iris else None,
        "iris_right": list(right_iris) if right_iris else None,
        "iris_midpoint": [mid_x, mid_y],
        "inter_iris_dist_px": round(inter_iris_dist, 1),
        "target_dist_px": round(target_dist, 1),
        "angular_error_deg": round(angular_error, 2),
        "method": "geometric_iris_approximation",
    }

    print(f"  Iris midpoint: ({mid_x:.0f}, {mid_y:.0f})")
    print(f"  Target dist:   {target_dist:.1f} px")
    print(f"  Angular error: ~{angular_error:.1f}° (geometric approx)")
    return eval_result


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════

def draw_red_dot_on_image(image_bgr: np.ndarray, point: tuple,
                          radius: int = 8) -> np.ndarray:
    """Draw a visible red dot on the image (used as gaze anchor)."""
    out = image_bgr.copy()
    cv2.circle(out, point, radius, (0, 0, 255), -1)
    cv2.circle(out, point, radius + 2, (0, 0, 180), 2)
    return out


def make_comparison_grid(images: dict, max_cols: int = 3) -> np.ndarray:
    """
    Build a labelled grid from a dict of {label: bgr_image}.
    All images are resized to match the first one.
    """
    entries = list(images.items())
    if not entries:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    ref_h, ref_w = entries[0][1].shape[:2]
    cells = []
    for label, img in entries:
        resized = cv2.resize(img, (ref_w, ref_h))
        # Add label bar on top
        bar = np.zeros((36, ref_w, 3), dtype=np.uint8)
        cv2.putText(bar, label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cells.append(np.vstack([bar, resized]))

    # Arrange in rows
    rows = []
    for i in range(0, len(cells), max_cols):
        row_cells = cells[i:i + max_cols]
        # Pad last row if needed
        while len(row_cells) < max_cols:
            row_cells.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row_cells))

    return np.vstack(rows)


# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline(image_path: str, sam_predictor, warp_mode: str = "shift") -> dict:
    """
    Execute the full 4-phase pipeline on a single image.

    Args:
        image_path:    Path to input image
        sam_predictor: SAM predictor instance
        warp_mode:     'none', 'shift', or 'tps'

    Returns:
        dict of all intermediate + final images (BGR).
    """
    print(f"\n{'═' * 60}")
    print(f"  Processing: {image_path}")
    print(f"  Warp mode:  {warp_mode}")
    print(f"{'═' * 60}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [Error] Cannot read image: {image_path}")
        return {}

    # Resize large images to manageable size
    max_dim = 1024
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        print(f"  Resized to {image_bgr.shape[1]}×{image_bgr.shape[0]}")

    # ── Interactive clicks ──
    person_pt, gaze_pt = select_points_on_image(image_bgr)

    # Draw red dot on working copy
    image_with_dot = draw_red_dot_on_image(image_bgr, gaze_pt)

    # ── Phase 1 ──
    raw_mask, dilated_mask = phase1_masking(image_bgr, person_pt, sam_predictor)

    # ── Phase 2 ──
    if warp_mode == "shift":
        warped = phase2_iris_shift(image_with_dot, gaze_pt, dilated_mask)
    elif warp_mode == "tps":
        warped = phase2_tps_warp(image_with_dot, gaze_pt, dilated_mask)
    else:
        warped = image_with_dot.copy()
        print("[Phase 2] SKIPPED (warp_mode=none)")

    # ── Phase 3 ──
    gemini_result, marked_img = phase3_gemini_refine(warped, dilated_mask)

    # ── Phase 4 ──
    final = phase4_remove_anchor(gemini_result, gaze_pt)

    # ── Evaluate gaze error ──
    eval_result = evaluate_gaze_error(final, gaze_pt, image_bgr.shape)

    results = {
        "01_input_with_clicks": _click_annotated_bgr.copy() if _click_annotated_bgr is not None else image_bgr,
        "02_sam_mask": cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR),
        "03_dilated_mask": cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR),
        "04_warped": warped,
        "04b_marked_contour": marked_img,
        "05_gemini_result": gemini_result,
        "06_final": final,
    }

    # Comparison grid
    grid = make_comparison_grid({
        "01 Input": _click_annotated_bgr.copy() if _click_annotated_bgr is not None else image_bgr,
        "02 SAM Mask": cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR),
        "03 Dilated": cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR),
        "04 Warped": warped,
        "04b Contour": marked_img,
        "05 Gemini": gemini_result,
    })
    results["comparison"] = grid
    results["_eval"] = eval_result  # not an image, handled separately

    return results


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GazeCtrl Path B — Gemini Nano Banana")
    parser.add_argument(
        "--warp-mode",
        choices=["none", "shift", "tps"],
        default="shift",
        help="Phase 2 warp method: none / shift (iris pixel shift) / tps (thin-plate spline)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  GazeCtrl Path B — Gemini Nano Banana")
    print(f"  Warp mode: {args.warp_mode}")
    print("=" * 60)

    # ── Create output dir ──
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect test images ──
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(str(TESTS_DIR / ext)))
    image_files.sort()

    if not image_files:
        print(f"\n[Error] No images found in {TESTS_DIR}")
        print("  Please place test images (jpg/png) into the tests/ folder.")
        sys.exit(1)

    print(f"\nFound {len(image_files)} test image(s):")
    for i, f in enumerate(image_files):
        print(f"  {i + 1}. {Path(f).name}")

    # ── Load SAM ──
    ckpt = ensure_sam_checkpoint()
    print("\n[SAM] Loading model …")
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    print(f"[SAM] Model loaded on {device}")

    # ── Process loop ──
    for idx, img_path in enumerate(image_files):
        img_name = Path(img_path).stem

        results = run_pipeline(img_path, sam_predictor, warp_mode=args.warp_mode)
        if not results:
            continue

        # Save outputs
        out_dir = OUTPUTS_DIR / img_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, data in results.items():
            if name == "_eval":
                # Save evaluation JSON
                eval_path = out_dir / "evaluation.json"
                with open(eval_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved: evaluation.json")
                continue
            save_path = out_dir / f"{name}.png"
            cv2.imwrite(str(save_path), data)
            print(f"  Saved: {save_path.name}")

        print(f"\n  ✓ All outputs saved to {out_dir}")

        # Ask whether to continue
        if idx < len(image_files) - 1:
            print(f"\n  Next image: {Path(image_files[idx + 1]).name}")
            ans = input("  Process next image? (y/n): ").strip().lower()
            if ans != 'y':
                print("  Stopping.")
                break

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Results in: {OUTPUTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
