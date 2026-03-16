"""
Manual Gaze Redirection Process - Path E
(Depth-Guided Dual-Image Pipeline)

1. Load image
2. Generate Depth Map using Depth-Anything-V2
3. User clicks to place the red dot (gaze target)
4. User adjusts the red dot's depth on the depth map via a slider
5. Send original image (with red dot) + depth map (with depth-adjusted red dot) to Gemini
6. Save result
"""

import os
import sys
import io
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk

from batch_evaluate import (
    phase4_remove_anchor,
    get_depth_anything,
    detect_face_center,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


# ── Depth Map Generation ──
def generate_depth_map(image_bgr):
    """Generate depth map. Returns raw float depth + normalized uint8."""
    depth_model = get_depth_anything()
    raw_depth = depth_model.infer_image(image_bgr)
    d_min, d_max = raw_depth.min(), raw_depth.max()
    depth_norm = ((raw_depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    return raw_depth, depth_norm, depth_color


# ── Person Depth & Face Orientation Estimation ──
def estimate_person_depth_and_orientation(image_bgr, depth_norm):
    """
    Use RetinaFace to:
    - Detect the face bounding box → compute median depth in that region
    - Determine if the person is facing away (face not detected = facing away)

    Returns:
        person_depth (float): 0-255 depth value of the person's face region
        facing_away (bool): True if no face detected (person likely facing away)
    """
    face_info = detect_face_center(image_bgr)
    if face_info is not None:
        x1, y1, x2, y2 = [int(v) for v in face_info["bbox"]]
        h, w = depth_norm.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            person_depth = float(np.median(depth_norm[y1:y2, x1:x2]))
        else:
            person_depth = float(np.median(depth_norm))
        facing_away = False
        print(f"  [Depth] Face detected (conf={face_info['confidence']:.2f}). "
              f"Person face depth: {person_depth:.1f}/255")
    else:
        # No face detected — assume person is facing away from camera
        person_depth = float(np.median(depth_norm))
        facing_away = True
        print(f"  [Depth] No face detected — assuming person is facing away. "
              f"Fallback person depth (image median): {person_depth:.1f}/255")

    return person_depth, facing_away


# ── Dynamic Prompt Builder ──
def build_depth_aware_prompt(chosen_depth, person_depth, facing_away,
                             depth_threshold=30):
    """
    Build prompt text based on:
      - chosen_depth: 0-255, brightness of the red dot on the depth map (higher = closer)
      - person_depth:  0-255, median brightness of the person's face region (higher = closer)
      - facing_away:   True if person is facing away from camera
      - depth_threshold: tolerance band (out of 255) for "roughly same depth"

    Returns prompt string.
    """
    depth_diff = float(chosen_depth) - float(person_depth)

    # ── Depth relation ──
    if abs(depth_diff) < depth_threshold:
        depth_relation = "same"
    elif depth_diff < -depth_threshold:
        depth_relation = "behind"   # dot darker (farther) than person → behind person
    else:
        depth_relation = "closer"   # dot brighter (closer) than person → in front of person

    print(f"  [Prompt] chosen_depth={chosen_depth}, person_depth={person_depth:.1f}, "
          f"diff={depth_diff:+.1f} → depth_relation='{depth_relation}', "
          f"facing_away={facing_away}")

    # ── Base header (same for all) ──
    header = (
        "I am providing two images.\n"
        "IMAGE 1 (first image): The original photo with a red dot marking the gaze target.\n"
        "IMAGE 2 (second image): A DEPTH MAP of the exact same scene. "
        "In this depth map, BRIGHTER areas represent objects CLOSER to the camera, "
        "and DARKER areas represent objects FARTHER from the camera.\n"
        "The red dot is also marked on the depth map.\n\n"
    )

    # ── Depth-specific instruction ──
    if depth_relation == "same":
        depth_instruction = (
            f"Based on the depth map, the red dot (depth={chosen_depth}/255) is at "
            f"approximately the same depth as the person (face depth≈{person_depth:.0f}/255). "
            "They are roughly at the same distance from the camera.\n"
            "TASK: Make the person look at the red dot. Since the red dot is at roughly the same "
            "depth, adjust mainly the gaze direction (eyes, pupils) and head yaw/pitch "
            "to point toward the red dot's 2D position in the image.\n"
        )
    elif depth_relation == "behind":
        depth_instruction = (
            f"Based on the depth map, the red dot (depth={chosen_depth}/255) is DARKER than "
            f"the person (face depth≈{person_depth:.0f}/255), meaning the red dot is "
            "BEHIND the person — farther from the camera.\n"
            "TASK: The red dot is behind the person. "
            "Make the person turn their body and/or head around to look at the red dot "
            "that is located behind them. Adjust gaze direction, head pose, and body pose "
            "accordingly so they are looking back at the red dot's 3D position.\n"
        )
    else:  # closer
        depth_instruction = (
            f"Based on the depth map, the red dot (depth={chosen_depth}/255) is BRIGHTER than "
            f"the person (face depth≈{person_depth:.0f}/255), meaning the red dot is "
            "MUCH CLOSER to the camera than the person — it is in front of the person.\n"
            "TASK: The red dot is between the person and the camera (closer to the viewer). "
            "Make the person look toward the camera-side at the red dot's position, "
            "adjusting their gaze direction, head pose, and body pose accordingly.\n"
        )

    # ── Face orientation modifier ──
    if facing_away:
        if depth_relation == "behind":
            # Person facing away + dot even further behind → don't need to turn
            facing_modifier = (
                "NOTE: The person is currently facing AWAY from the camera (back toward the scene). "
                "The red dot is also behind the person (same general direction they are already facing). "
                "Therefore, they do NOT need to turn around — just redirect their gaze toward "
                "the red dot's exact position while keeping them generally facing away from the camera.\n"
            )
        else:
            # Person facing away + dot at same depth or closer → must turn back
            facing_modifier = (
                "NOTE: The person is currently facing AWAY from the camera (back toward the scene). "
                "The red dot is in front of or at the same depth as them. "
                "Make the person turn their head (and body if needed) BACK toward the camera "
                "to look at the red dot.\n"
            )
    else:
        facing_modifier = ""

    # ── Footer ──
    footer = (
        "Preserve the person's identity, appearance, clothing, and the overall scene. "
        "Only change what is necessary for the gaze redirection."
    )

    full_prompt = header + (facing_modifier if facing_modifier else "") + depth_instruction + footer
    return full_prompt


# ── Interactive UI: Click red dot + adjust depth slider ──
def select_target_and_depth(image_bgr, depth_color, depth_norm):
    """
    Opens a Tkinter window showing the original image.
    User clicks to place the red dot, then adjusts a depth slider.
    Returns: (target_x, target_y), chosen_depth_value (0-255)
    """
    h, w = image_bgr.shape[:2]
    
    # Scale for display — keep each panel small enough for side-by-side + controls
    display_max = 450
    scale = min(display_max / w, display_max / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    target_pt = [None]
    chosen_depth = [128]  # default mid-depth
    auto_depth = [128]

    root = tk.Tk()
    root.title("Path E: Click red dot target, then adjust depth")

    # Main frame
    main_frame = tk.Frame(root)
    main_frame.pack()

    # Canvas for image
    canvas = tk.Canvas(main_frame, width=dw, height=dh)
    canvas.pack(side=tk.LEFT, padx=5, pady=5)

    # Canvas for depth map preview  
    depth_canvas = tk.Canvas(main_frame, width=dw, height=dh)
    depth_canvas.pack(side=tk.LEFT, padx=5, pady=5)

    # Controls frame
    ctrl_frame = tk.Frame(root)
    ctrl_frame.pack(pady=10)

    depth_label = tk.Label(ctrl_frame, text="Red dot depth: 128 (mid)", font=("Arial", 12))
    depth_label.pack()

    depth_slider = tk.Scale(ctrl_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                            length=400, label="Depth (bright=close, dark=far)")
    depth_slider.set(128)
    depth_slider.pack()

    auto_btn_text = tk.StringVar(value="Use auto depth")
    
    info_label = tk.Label(ctrl_frame, text="Click on the image to place the red dot.", 
                          font=("Arial", 10), fg="gray")
    info_label.pack(pady=5)

    confirm_btn = tk.Button(ctrl_frame, text="Confirm", font=("Arial", 12),
                            state=tk.DISABLED)
    confirm_btn.pack(pady=5)

    def _display_image(cv_img, canvas_widget):
        disp = cv2.resize(cv_img, (dw, dh))
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(disp_rgb)
        tk_img = _pil_to_tk(pil_img)
        canvas_widget._tk_img = tk_img
        canvas_widget.create_image(0, 0, anchor=tk.NW, image=tk_img)

    def _pil_to_tk(pil_img):
        import tkinter as tk
        from PIL import ImageTk
        return ImageTk.PhotoImage(pil_img)

    # Initial display
    _display_image(image_bgr, canvas)
    _display_image(depth_color, depth_canvas)

    def _refresh_displays():
        """Redraw both images with current red dot and depth."""
        if target_pt[0] is None:
            return
        tx, ty = target_pt[0]
        dv = depth_slider.get()
        chosen_depth[0] = dv

        # Redraw original with red dot
        img_copy = image_bgr.copy()
        cv2.circle(img_copy, (tx, ty), 8, (0, 0, 255), -1)
        _display_image(img_copy, canvas)

        # Redraw depth map with depth-adjusted red dot
        dep_copy = depth_color.copy()
        # Draw the red dot with a circle whose inner fill brightness = chosen depth
        cv2.circle(dep_copy, (tx, ty), 12, (0, 0, 255), 2)  # red outline
        # Fill inner circle with the chosen depth color (grayscale mapped to INFERNO)
        depth_patch = np.full((1, 1), dv, dtype=np.uint8)
        dot_color_bgr = cv2.applyColorMap(depth_patch, cv2.COLORMAP_INFERNO)[0, 0]
        cv2.circle(dep_copy, (tx, ty), 10, dot_color_bgr.tolist(), -1)  # fill
        cv2.circle(dep_copy, (tx, ty), 12, (0, 0, 255), 2)  # red outline on top
        
        # Label
        label_txt = f"depth={dv}"
        cv2.putText(dep_copy, label_txt, (tx + 15, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        _display_image(dep_copy, depth_canvas)

        # Update label
        if dv > 170:
            desc = "very close"
        elif dv > 128:
            desc = "close"
        elif dv > 85:
            desc = "mid-range"
        elif dv > 42:
            desc = "far"
        else:
            desc = "very far"
        depth_label.config(text=f"Red dot depth: {dv} ({desc})")

    def _on_click(event):
        # Convert display coords to original image coords
        ox = int(event.x / scale)
        oy = int(event.y / scale)
        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))
        target_pt[0] = (ox, oy)

        # Auto-set slider to the actual depth at clicked location
        actual_depth = int(depth_norm[oy, ox])
        auto_depth[0] = actual_depth
        depth_slider.set(actual_depth)
        auto_btn_text.set(f"Reset to auto depth ({actual_depth})")

        confirm_btn.config(state=tk.NORMAL)
        info_label.config(text=f"Red dot at ({ox}, {oy}). Adjust depth slider, then click Confirm.")
        _refresh_displays()

    def _on_slider_change(val):
        _refresh_displays()

    def _on_auto_depth():
        depth_slider.set(auto_depth[0])
        _refresh_displays()

    def _on_confirm():
        chosen_depth[0] = depth_slider.get()
        root.quit()
        root.destroy()

    canvas.bind("<Button-1>", _on_click)
    depth_slider.config(command=_on_slider_change)

    auto_btn = tk.Button(ctrl_frame, textvariable=auto_btn_text, command=_on_auto_depth)
    auto_btn.pack(pady=3)

    confirm_btn.config(command=_on_confirm)

    root.mainloop()

    if target_pt[0] is None:
        return None, None
    return target_pt[0], chosen_depth[0]


# ── Gemini Dual-Image API Call ──
def gemini_edit_dual_image(image_bgr, depth_bgr, prompt_text, max_retries=3):
    """Send TWO images (original + depth map) to Gemini. Returns edited BGR or None."""
    from google import genai
    from google.genai import types
    import time

    client = genai.Client(api_key=GEMINI_API_KEY)
    orig_h, orig_w = image_bgr.shape[:2]

    def to_png_bytes(bgr_img):
        pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    img_bytes = to_png_bytes(image_bgr)
    depth_bytes = to_png_bytes(depth_bgr)

    for attempt in range(1, max_retries + 1):
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt_text),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/png", data=img_bytes)),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/png", data=depth_bytes)),
                    ],
                ),
            ]
            config = types.GenerateContentConfig(
                image_config=types.ImageConfig(image_size="1K"),
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
        except Exception as e:
            err_str = str(e)
            if "503" in err_str and attempt < max_retries:
                wait = 30 * attempt
                print(f"  [Retry {attempt}/{max_retries}] 503 overloaded — waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  [Error] Gemini failed: {e}")
                return None
    return None


# ── Main ──
def main():
    parser = argparse.ArgumentParser(
        description="Manual Gaze Redirection - Path E (Depth-Guided)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="manual_out_path_e.png",
                        help="Path to output image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Could not find image at {image_path}")
        sys.exit(1)

    print(f"Loading image: {image_path}")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print("Error: Could not read image.")
        sys.exit(1)

    # Scale down if too large
    max_dim = 1500
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    # 1. Generate Depth Map
    print("[Path E] Generating depth map...")
    raw_depth, depth_norm, depth_color = generate_depth_map(image_bgr)

    # 2. Interactive: click red dot + adjust depth
    print("[Path E] Opening interactive UI...")
    gaze_target, chosen_depth = select_target_and_depth(image_bgr, depth_color, depth_norm)

    if gaze_target is None:
        print("No target selected. Exiting.")
        sys.exit(0)

    tx, ty = gaze_target
    print(f"  Target: ({tx}, {ty}), Chosen depth: {chosen_depth}/255")

    # 3. Prepare images for Gemini
    # Original image with red dot
    image_with_dot = image_bgr.copy()
    cv2.circle(image_with_dot, (tx, ty), 8, (0, 0, 255), -1)

    # Depth map with depth-adjusted red dot
    depth_with_dot = depth_color.copy()
    depth_patch = np.full((1, 1), chosen_depth, dtype=np.uint8)
    dot_color_bgr = cv2.applyColorMap(depth_patch, cv2.COLORMAP_INFERNO)[0, 0]
    cv2.circle(depth_with_dot, (tx, ty), 10, dot_color_bgr.tolist(), -1)
    cv2.circle(depth_with_dot, (tx, ty), 12, (0, 0, 255), 2)

    # 4. Estimate person depth and face orientation
    print("[Path E] Estimating person depth and face orientation...")
    person_depth, facing_away = estimate_person_depth_and_orientation(image_bgr, depth_norm)

    # 5. Build dynamic prompt
    prompt = build_depth_aware_prompt(
        chosen_depth=chosen_depth,
        person_depth=person_depth,
        facing_away=facing_away,
    )
    print("\n[Path E] Generated prompt:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)

    # 6. Send to Gemini
    #send or not
    send_or_not = input("Send to Gemini? (y/n): ")
    if send_or_not == "y":
        print("[Path E] Sending original image + depth map to Gemini...")
        gemini_result = gemini_edit_dual_image(image_with_dot, depth_with_dot, prompt)

        if gemini_result is None:
            print("Error: Gemini returned None.")
        sys.exit(1)

    # 7. Remove red dot
    final_output = phase4_remove_anchor(gemini_result, gaze_target)

    # 8. Save outputs
    cv2.imwrite(args.output, final_output)

    out_dir = Path(args.output).parent
    base_name = Path(args.output).stem
    cv2.imwrite(str(out_dir / f"{base_name}_depth_map.png"), depth_color)
    cv2.imwrite(str(out_dir / f"{base_name}_depth_with_dot.png"), depth_with_dot)
    if send_or_not == "y":
        cv2.imwrite(str(out_dir / f"{base_name}_original_with_dot.png"), image_with_dot)

    print(f"\nDone! Edited image saved to: {args.output}")
    print(f"Debug outputs: {base_name}_depth_map.png, "
          f"{base_name}_depth_with_dot.png, {base_name}_original_with_dot.png")


if __name__ == "__main__":
    main()
