"""
Dataset Gaze Control Pipeline
Automated batch gaze redirection over a folder of images.

Flow per image:
  1. Depth-Anything-V2 → depth map
  2. Interactive UI (2-step):
       Step 1 — click on person → SAM generates mask
       Step 2 — click gaze target + adjust depth slider
  3. Gemini edit: isolated person + depth context → redirected person
  4. Composite with original background
  5. Gap repair: Gemini 2nd-pass or OpenCV inpainting

Usage:
    python dataset_pipeline.py --dataset ../condition_dataset_v2/source
    python dataset_pipeline.py --dataset ../test_imgs --repair inpaint
    python dataset_pipeline.py --dataset ../condition_dataset_v2/source --resume
    python dataset_pipeline.py --dataset ../condition_dataset_v2/source \\
        --prompt-extra "Keep clothing and background unchanged."
"""

import io
import sys
import time
import math
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_utils import (
    load_sam_predictor,
    get_depth_anything,
    remove_red_marker,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════════════════
#  DEPTH MAP
# ══════════════════════════════════════════════════════════════════════════════

def generate_depth_map(image_bgr):
    model = get_depth_anything()
    raw = model.infer_image(image_bgr)
    d_min, d_max = raw.min(), raw.max()
    norm = ((raw - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return raw, norm, color


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PERSON SELECTION + GAZE TARGET UI
# ══════════════════════════════════════════════════════════════════════════════

def select_person_and_target(image_bgr, depth_color, depth_norm, sam_predictor, img_name):
    """
    3-panel interactive UI.

    Step 1 — click on person in the Original panel → SAM generates mask (shown in panel 3)
    Step 2 — click gaze target in the Original panel + adjust depth slider

    Returns (target_pt, chosen_depth, sam_mask)  or  (None, None, None) to skip.
    """
    h, w = image_bgr.shape[:2]
    scale = min(380 / w, 380 / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    target_pt    = [None]
    chosen_depth = [128]
    sam_mask     = [None]
    ui_step      = [0]   # 0 = pick person, 1 = pick gaze target

    root = tk.Tk()
    root.title(f"Gaze Control — {img_name}")

    # ── 3 labelled panel columns ──────────────────────────────────────────────
    panel = tk.Frame(root)
    panel.pack(padx=4, pady=4)

    canvases = []
    for label in ("Original  (click here)", "Depth Map", "SAM Mask"):
        col = tk.Frame(panel)
        col.pack(side=tk.LEFT, padx=4)
        tk.Label(col, text=label, font=("Arial", 9, "bold")).pack()
        c = tk.Canvas(col, width=dw, height=dh, bg="black")
        c.pack()
        canvases.append(c)
    c_orig, c_depth, c_mask = canvases

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl = tk.Frame(root)
    ctrl.pack(pady=6)

    step_lbl = tk.Label(
        ctrl, text="STEP 1 — Click on the person",
        font=("Arial", 11, "bold"), fg="red"
    )
    step_lbl.grid(row=0, column=0, columnspan=3, pady=(0, 4))

    tk.Label(ctrl, text="Target Depth  (0 = far · 255 = near):").grid(
        row=1, column=0, columnspan=3)
    depth_slider = tk.Scale(ctrl, from_=0, to=255,
                            orient=tk.HORIZONTAL, length=460)
    depth_slider.set(128)
    depth_slider.grid(row=2, column=0, columnspan=3)

    confirm_btn = tk.Button(
        ctrl, text="Confirm & Send to Gemini",
        state=tk.DISABLED, font=("Arial", 12)
    )
    confirm_btn.grid(row=3, column=0, pady=6, padx=6)

    def _reset():
        ui_step[0] = 0
        sam_mask[0] = None
        target_pt[0] = None
        step_lbl.config(text="STEP 1 — Click on the person", fg="red")
        confirm_btn.config(state=tk.DISABLED)
        _refresh()

    tk.Button(ctrl, text="Reset",      command=_reset).grid(row=3, column=1, padx=4)
    tk.Button(ctrl, text="Skip Image", command=root.destroy).grid(row=3, column=2, padx=4)

    # ── Render helpers ────────────────────────────────────────────────────────
    def _show(bgr, canvas):
        disp   = cv2.resize(bgr, (dw, dh))
        tk_img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._tk_img = tk_img          # prevent GC
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    def _refresh():
        img_draw = image_bgr.copy()
        dep_draw = depth_color.copy()
        msk_draw = np.zeros_like(image_bgr)

        if sam_mask[0] is not None:
            msk_draw[sam_mask[0] > 0] = (0, 200, 200)
            cv2.putText(msk_draw, "LOCKED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if target_pt[0] is not None:
            tx, ty = target_pt[0]
            dv = depth_slider.get()
            cv2.circle(img_draw, (tx, ty), 10, (0, 0, 255), -1)
            patch    = np.full((1, 1), dv, dtype=np.uint8)
            dot_bgr  = cv2.applyColorMap(patch, cv2.COLORMAP_INFERNO)[0, 0].tolist()
            cv2.circle(dep_draw, (tx, ty), 12, dot_bgr,        -1)
            cv2.circle(dep_draw, (tx, ty), 12, (255, 255, 255), 3)
            cv2.circle(dep_draw, (tx, ty), 12, (0, 0, 255),     2)
            cv2.circle(dep_draw, (tx, ty),  1, (0, 0, 255),    -1)
            cv2.circle(msk_draw, (tx, ty), 10, (0, 0, 255),    -1)

        _show(img_draw, c_orig)
        _show(dep_draw, c_depth)
        _show(msk_draw, c_mask)

    # ── Click handler ─────────────────────────────────────────────────────────
    def on_click(event):
        ox = max(0, min(w - 1, int(event.x / scale)))
        oy = max(0, min(h - 1, int(event.y / scale)))

        if ui_step[0] == 0:
            # Step 1: SAM
            print(f"  [SAM] Prompting at ({ox}, {oy}) ...")
            sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            masks, scores, _ = sam_predictor.predict(
                point_coords=np.array([[ox, oy]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            best = np.argmax(scores)
            sam_mask[0] = (masks[best].astype(np.uint8) * 255)
            pct = cv2.countNonZero(sam_mask[0]) / (h * w) * 100
            print(f"  [SAM] Mask coverage: {pct:.1f}%")
            ui_step[0] = 1
            step_lbl.config(
                text="Person LOCKED  ✓  —  STEP 2: Click the gaze target",
                fg="blue")

        elif ui_step[0] == 1:
            # Step 2: gaze target
            target_pt[0] = (ox, oy)
            depth_slider.set(int(depth_norm[oy, ox]))
            confirm_btn.config(state=tk.NORMAL)
            step_lbl.config(
                text=f"Target ({ox}, {oy}) set.  Adjust depth then confirm.",
                fg="darkgreen")

        _refresh()

    c_orig.bind("<Button-1>", on_click)
    depth_slider.config(command=lambda _: _refresh())
    confirm_btn.config(command=lambda: [
        chosen_depth.__setitem__(0, depth_slider.get()),
        root.destroy(),
    ])

    _refresh()
    root.mainloop()

    if sam_mask[0] is None or target_pt[0] is None:
        return None, None, None
    return target_pt[0], chosen_depth[0], sam_mask[0]


# ══════════════════════════════════════════════════════════════════════════════
#  ISOLATE PERSON + DEPTH  (inputs to Gemini)
# ══════════════════════════════════════════════════════════════════════════════

def isolate_person_and_depth(image_bgr, depth_color, depth_norm,
                             person_mask, gaze_target, chosen_depth):
    """
    Image 1 — person on black background, solid red dot.
    Image 2 — full depth map, hollow red ring showing target depth colour.
    """
    tx, ty = gaze_target

    isolated = np.zeros_like(image_bgr)
    isolated[person_mask > 0] = image_bgr[person_mask > 0]
    cv2.circle(isolated, (tx, ty), 8, (0, 0, 255), -1)

    depth_out = depth_color.copy()
    patch    = np.full((1, 1), int(chosen_depth), dtype=np.uint8)
    dot_bgr  = cv2.applyColorMap(patch, cv2.COLORMAP_INFERNO)[0, 0].tolist()
    cv2.circle(depth_out, (tx, ty), 12, dot_bgr,        -1)
    cv2.circle(depth_out, (tx, ty), 12, (255, 255, 255), 3)
    cv2.circle(depth_out, (tx, ty), 12, (0, 0, 255),     2)
    cv2.circle(depth_out, (tx, ty),  1, (0, 0, 255),    -1)

    return isolated, depth_out


# ══════════════════════════════════════════════════════════════════════════════
#  DEPTH-AWARE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def build_depth_aware_prompt(chosen_depth, person_depth, extra=""):
    diff = float(chosen_depth) - float(person_depth)
    if abs(diff) < 30:
        rel = "at the same depth as"
    elif diff > 30:
        rel = "IN FRONT OF (closer to camera than)"
    else:
        rel = "BEHIND"

    prompt = (
        f"Image 2 is a STATIC DEPTH REFERENCE. The hollow circle marks the red dot's "
        f"3D position, which is {rel} the person.\n"
        "OUTPUT: Return Image 1 with the person's new gaze. "
        "DO NOT move the red dot. Keep background black.\n"
        "TASK: On Image 1, IGNORE the person's current gaze and REDIRECT the person's "
        "eyes and HEAD to face the exact 3D position of the solid red dot.\n"
    )
    if rel == "BEHIND":
        prompt += (
            "Because the target is so much farther away than the person, the only way "
            "to actually look at it is to turn all the way around: the person must end "
            "up with their BACK almost fully facing the camera, having rotated their "
            "whole body away from its current orientation, with their head turned to "
            "look toward the red dot, which now sits further ahead of them in this new "
            "orientation, deeper into the scene. Do not just glance back over the "
            "shoulder while staying mostly front-on to the camera — the body itself "
            "must rotate away.\n"
        )
    if extra:
        prompt += extra.strip() + "\n"
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI EDIT  (dual-image)
# ══════════════════════════════════════════════════════════════════════════════

def gemini_edit_dual_image(image_bgr, depth_bgr, prompt_text, max_retries=3):
    from google import genai
    from google.genai import types

    client   = genai.Client(api_key=GEMINI_API_KEY)
    orig_h, orig_w = image_bgr.shape[:2]

    def _png(bgr):
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        return buf.getvalue()

    img_bytes   = _png(image_bgr)
    depth_bytes = _png(depth_bgr)

    for attempt in range(1, max_retries + 1):
        try:
            contents = [types.Content(role="user", parts=[
                types.Part.from_text(text=prompt_text),
                types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
                types.Part(inline_data=types.Blob(mime_type="image/png", data=depth_bytes)),
            ])]
            cfg = types.GenerateContentConfig(
                image_config=types.ImageConfig(image_size="1K"),
                response_modalities=["IMAGE", "TEXT"],
            )
            for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL, contents=contents, config=cfg):
                if chunk.parts is None:
                    continue
                for part in chunk.parts:
                    if part.inline_data and part.inline_data.data:
                        pil = Image.open(io.BytesIO(part.inline_data.data))
                        if pil.size != (orig_w, orig_h):
                            pil = pil.resize((orig_w, orig_h), Image.LANCZOS)
                        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            print("  [Warning] No image in Gemini response.")
            return None
        except Exception as e:
            if "503" in str(e) and attempt < max_retries:
                wait = 30 * attempt
                print(f"  [Retry {attempt}/{max_retries}] 503 — waiting {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  [Gemini Error] {e}")
                return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  GAP REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def repair_gemini(composite_bgr, max_retries=3):
    """Gemini 2nd pass: fill black gaps with plausible background."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    orig_h, orig_w = composite_bgr.shape[:2]

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
    img_bytes = buf.getvalue()

    repair_prompt = (
        "This composite image has black areas (RGB 0,0,0) where the background was removed. "
        "Fill ONLY these black areas with a seamless, realistic background that matches "
        "the surrounding scene. Do NOT alter the person, their gaze, or any non-black region."
    )

    for attempt in range(1, max_retries + 1):
        try:
            contents = [types.Content(role="user", parts=[
                types.Part.from_text(text=repair_prompt),
                types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
            ])]
            cfg = types.GenerateContentConfig(
                image_config=types.ImageConfig(image_size="1K"),
                response_modalities=["IMAGE", "TEXT"],
            )
            for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL, contents=contents, config=cfg):
                if chunk.parts is None:
                    continue
                for part in chunk.parts:
                    if part.inline_data and part.inline_data.data:
                        pil = Image.open(io.BytesIO(part.inline_data.data))
                        return cv2.cvtColor(
                            np.array(pil.resize((orig_w, orig_h))), cv2.COLOR_RGB2BGR)
            return None
        except Exception as e:
            if attempt < max_retries:
                time.sleep(20)
            else:
                print(f"  [Gemini Repair Error] {e}")
                return None
    return None


def repair_inpaint(composite_bgr):
    """OpenCV TELEA inpainting: fill near-black gaps."""
    gray = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(composite_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND COMPOSITE + REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def composite_and_repair(original_bgr, gemini_result, person_mask, repair_mode):
    h, w = original_bgr.shape[:2]
    if gemini_result.shape[:2] != (h, w):
        gemini_result = cv2.resize(gemini_result, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Clear any leftover red target-dot pixels (Gemini can relocate/redraw the
    # marker on large pose changes, so a fixed coordinate can't be trusted).
    gemini_result = remove_red_marker(gemini_result)

    background = original_bgr.copy()
    background[person_mask > 2] = 0

    gray = cv2.cvtColor(gemini_result, cv2.COLOR_BGR2GRAY)
    _, new_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    composite = background.copy()
    composite[new_mask > 0] = 0
    composite = cv2.add(composite, gemini_result)

    if repair_mode == "inpaint":
        final = repair_inpaint(composite)
    else:
        final = repair_gemini(composite)
        if final is None:
            print("  [Warning] Gemini repair failed — falling back to inpaint.")
            final = repair_inpaint(composite)

    return composite, final


# ══════════════════════════════════════════════════════════════════════════════
#  GAZE VECTOR COMPUTATION  (target gaze from red dot + depth)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_face_from_mask(sam_mask):
    """
    Estimate face center and approximate face width from the SAM body mask.
    Head is assumed to occupy the top ~20% of the mask bounding box.
    Returns ((cx, cy), face_width_est_px).
    """
    ys, xs = np.where(sam_mask > 0)
    if len(ys) == 0:
        h, w = sam_mask.shape
        return (w // 2, h // 4), 80

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    body_height = max(y_max - y_min, 1)

    # Take pixels in the top 20% of the body height as the head region
    head_y_max = y_min + body_height * 0.20
    in_head = ys < head_y_max

    if np.any(in_head):
        cy = int(np.median(ys[in_head]))
        cx = int(np.median(xs[in_head]))
        head_w = int(xs[in_head].max() - xs[in_head].min())
        face_width_est = max(30, head_w)
    else:
        cy = y_min
        cx = (x_min + x_max) // 2
        face_width_est = max(30, (x_max - x_min) // 3)

    return (cx, cy), face_width_est


def compute_target_gaze(face_center, gaze_target, chosen_depth, person_depth_median,
                        image_shape):
    """
    Compute target gaze direction using a simplified pinhole camera model.

    Inputs:
      face_center        — (cx, cy) in image pixels
      gaze_target        — (tx, ty) in image pixels
      chosen_depth       — user-set depth value (0-255), higher = closer
      person_depth_median — median depth_norm value inside SAM mask (0-255)
      image_shape        — (H, W[, C]) of the image

    Returns dict with keys: dx, dy, dz (unit vector), yaw_deg, pitch_deg,
                            yaw_rad, pitch_rad.

    Convention (matches 3DGazeNet / 6DRepNet):
      dx > 0 = right   dy > 0 = down   dz > 0 = into screen (forward)
    """
    h, w = image_shape[:2]
    face_cx, face_cy = float(face_center[0]), float(face_center[1])
    tx, ty = float(gaze_target[0]), float(gaze_target[1])
    cx_img, cy_img = w / 2.0, h / 2.0

    # Virtual focal length heuristic
    f = w * 0.8

    # depth_norm ~ disparity (1/Z):  Z ∝ 1/depth_norm
    # Anchor face at Z=1; compute target Z relative to face.
    p_d = max(float(person_depth_median), 1.0)
    c_d = max(float(chosen_depth), 1.0)
    Z_face   = 1.0
    Z_target = Z_face * (p_d / c_d)

    # Back-project to 3D (y-down camera coordinates)
    X_face   = (face_cx - cx_img) / f * Z_face
    Y_face   = (face_cy - cy_img) / f * Z_face
    X_target = (tx - cx_img) / f * Z_target
    Y_target = (ty - cy_img) / f * Z_target

    Vx = X_target - X_face
    Vy = Y_target - Y_face
    Vz = Z_target - Z_face          # negative when target is closer

    # Force forward hemisphere (gaze is always roughly into screen)
    Vz_fwd = -abs(Vz)

    mag = math.sqrt(Vx ** 2 + Vy ** 2 + Vz_fwd ** 2)
    if mag < 1e-8:
        return {"dx": 0.0, "dy": 0.0, "dz": 1.0,
                "yaw_deg": 0.0, "pitch_deg": 0.0,
                "yaw_rad": 0.0, "pitch_rad": 0.0}

    dx = Vx / mag
    dy = Vy / mag
    dz = -Vz_fwd / mag              # positive = into screen

    # pitch > 0 = up (negate dy because dy>0 is down)
    pitch_rad = math.asin(max(-1.0, min(1.0, -dy)))
    yaw_rad   = math.atan2(dx, dz)  # yaw > 0 = right

    return {
        "dx":        round(dx, 6),
        "dy":        round(dy, 6),
        "dz":        round(dz, 6),
        "yaw_deg":   round(math.degrees(yaw_rad),   3),
        "pitch_deg": round(math.degrees(pitch_rad), 3),
        "yaw_rad":   round(yaw_rad,   6),
        "pitch_rad": round(pitch_rad, 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_image(image_path, sam_predictor, args):
    img_name = Path(image_path).stem
    print(f"\n{'═' * 60}")
    print(f"  Image: {img_name}")

    out_dir = Path(args.output) / img_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and (out_dir / "07_final.png").exists():
        print("  [Resume] Already done — skipping.")
        return True

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print("  [Error] Cannot read image.")
        return False

    h, w = image_bgr.shape[:2]
    if max(h, w) > 1500:
        s = 1500 / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * s), int(h * s)))

    # ── Step 1: Depth map ─────────────────────────────────────────────────────
    print("  [1/5] Depth-Anything-V2 ...")
    _, depth_norm, depth_color = generate_depth_map(image_bgr)
    cv2.imwrite(str(out_dir / "01_depth.png"), depth_color)

    # ── Step 2: Interactive UI (person selection + gaze target) ───────────────
    gaze_target, chosen_depth, sam_mask = select_person_and_target(
        image_bgr, depth_color, depth_norm, sam_predictor, img_name
    )
    if gaze_target is None:
        print("  [Skip] User skipped.")
        return False

    cv2.imwrite(str(out_dir / "02_sam_mask.png"), sam_mask)
    background_img = image_bgr.copy()
    background_img[sam_mask > 0] = 0
    cv2.imwrite(str(out_dir / "03_background.png"), background_img)

    # ── Step 3: Build Gemini inputs ───────────────────────────────────────────
    isolated_person, isolated_depth = isolate_person_and_depth(
        image_bgr, depth_color, depth_norm, sam_mask, gaze_target, chosen_depth
    )
    cv2.imwrite(str(out_dir / "04a_isolated_person.png"), isolated_person)
    cv2.imwrite(str(out_dir / "04b_isolated_depth.png"),  isolated_depth)

    person_depth = (float(np.median(depth_norm[sam_mask > 0]))
                    if np.any(sam_mask > 0) else 128.0)
    prompt = build_depth_aware_prompt(chosen_depth, person_depth, args.prompt_extra)
    print(f"\n--- Prompt ---\n{prompt}---\n")
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    # ── Step 4: Gemini gaze edit ──────────────────────────────────────────────
    send = input("Send to Gemini? (y / n=skip): ").strip().lower()
    if send not in ("y", "yes", ""):
        print("  Skipped.")
        return False

    print("  [4/5] Gemini gaze edit ...")
    gemini_result = gemini_edit_dual_image(isolated_person, isolated_depth, prompt)
    if gemini_result is None:
        print("  [Error] Gemini returned no image.")
        return False
    cv2.imwrite(str(out_dir / "05_gemini_raw.png"), gemini_result)

    # ── Step 5: Composite + repair ────────────────────────────────────────────
    print(f"  [5/5] Composite + repair ({args.repair}) ...")
    composite, final = composite_and_repair(
        image_bgr, gemini_result, sam_mask, args.repair
    )
    cv2.imwrite(str(out_dir / "06_composite_unfilled.png"), composite)
    cv2.imwrite(str(out_dir / "07_final.png"), final)

    # ── Metadata ──────────────────────────────────────────────────────────────
    face_center_px, face_width_est = estimate_face_from_mask(sam_mask)
    target_gaze = compute_target_gaze(
        face_center_px, gaze_target, chosen_depth, person_depth, image_bgr.shape
    )

    metadata = {
        "sample_id":    img_name,
        "source_image": str(Path(image_path).resolve()),
        "files": {
            "depth":              "01_depth.png",
            "sam_mask":           "02_sam_mask.png",
            "background":         "03_background.png",
            "isolated_person":    "04a_isolated_person.png",
            "isolated_depth":     "04b_isolated_depth.png",
            "gemini_raw":         "05_gemini_raw.png",
            "composite_unfilled": "06_composite_unfilled.png",
            "final":              "07_final.png",
        },
        "image_size":          list(image_bgr.shape[:2]),   # [H, W]
        "face_center_px":      list(face_center_px),
        "face_width_est_px":   face_width_est,
        "gaze_target_px":      list(gaze_target),
        "chosen_depth":        int(chosen_depth),
        "person_depth_median": round(person_depth, 2),
        "sam_mask_coverage_pct": round(
            float(cv2.countNonZero(sam_mask)) /
            (image_bgr.shape[0] * image_bgr.shape[1]) * 100, 2),
        "repair_mode":  args.repair,
        "prompt":       prompt,
        "target_gaze":  target_gaze,
        # Filled by estimate_gaze_batch.py (Phase 2)
        "predicted_gaze":        None,
        "angular_error_deg":     None,
        "quality_flag":          None,
    }

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"  Done → {out_dir}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Gaze Control Pipeline")
    parser.add_argument(
        "--dataset", type=str,
        default=str(PROJECT_ROOT / "condition_dataset_v2" / "source"),
        help="Folder of input images (default: condition_dataset_v2/source)")
    parser.add_argument(
        "--output", type=str, default="e_results",
        help="Output directory (default: e_results)")
    parser.add_argument(
        "--repair", choices=["gemini", "inpaint"], default="gemini",
        help="Gap-repair method (default: gemini)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip images that already have 07_final.png")
    parser.add_argument(
        "--prompt-extra", type=str, default="", metavar="TEXT",
        help="Extra text appended to every Gemini prompt")
    parser.add_argument(
        "--files", nargs="+", metavar="FILENAME",
        help="Process only these specific filenames from --dataset (e.g. foo.jpg bar.jpg)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[Error] Dataset path not found: {dataset_path}")
        sys.exit(1)

    image_files = sorted(
        p for p in dataset_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.files:
        wanted = set(args.files)
        image_files = [p for p in image_files if p.name in wanted]
        missing = wanted - {p.name for p in image_files}
        if missing:
            print(f"[Warning] Not found in dataset: {', '.join(sorted(missing))}")
    if not image_files:
        print(f"[Error] No images to process.")
        sys.exit(1)

    print(f"[Dataset] {len(image_files)} image(s)  →  {dataset_path}")
    print(f"[Output]  {Path(args.output).resolve()}")
    print(f"[Repair]  {args.repair}")

    print("\n[Init] Loading models ...")
    sam_predictor = load_sam_predictor()
    get_depth_anything()
    print("[Init] Ready.\n")

    done = skipped = failed = 0
    for i, img_path in enumerate(image_files):
        print(f"\n[{i + 1}/{len(image_files)}] {img_path.name}")
        try:
            ok = process_image(img_path, sam_predictor, args)
            if ok:
                done += 1
            else:
                skipped += 1
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            break
        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ── Collect all metadata into a master manifest ───────────────────────────
    out_base = Path(args.output)
    all_meta = []
    for sample_dir in sorted(out_base.iterdir()):
        meta_file = sample_dir / "metadata.json"
        if meta_file.exists():
            try:
                all_meta.append(json.loads(meta_file.read_text(encoding="utf-8")))
            except Exception:
                pass

    if all_meta:
        manifest_path = out_base / "manifest.json"
        manifest_path.write_text(
            json.dumps({"samples": all_meta, "total": len(all_meta)},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Manifest: {manifest_path}  ({len(all_meta)} samples)")

    print(f"\n{'=' * 60}")
    print(f"  Done: {done}   Skipped: {skipped}   Failed: {failed}")
    print(f"  Results: {out_base.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
