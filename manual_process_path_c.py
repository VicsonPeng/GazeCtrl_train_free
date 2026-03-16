"""
Manual Gaze Redirection Process - Path C 
(Two-Stage Generation + Face Swap)

1. Load image and select person 
2. Masking the person's head and keep the rest of the image
3. Select the red dot position for the anchor
4. Send the image without the person to Gemini with "Generate *any* realistic head facing exactly at the red dot"
5. Swap original head onto Gemini's generated head using InsightFace
6. Clean up anchor
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Try to import InsightFace for Face Swapping
try:
    import insightface
except ImportError:
    print("Error: insightface is not installed. Please install it using:")
    print("pip install insightface onnxruntime")
    sys.exit(1)

from gazectrl_pathb import (
    phase1_masking,
    draw_red_dot_on_image
)
from batch_evaluate import (
    load_sam_predictor, 
    phase4_remove_anchor,
    detect_face_center,
    get_head_mask,
    get_head_fallback_mask,
    _gemini_edit
)


def select_single_point(image_bgr, window_title="Click on the person, then press Enter"):
    """Let user click ONE point on the image. Returns (x, y) or None."""
    clicked = [None]
    display = image_bgr.copy()

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[0] = (x, y)
            vis = image_bgr.copy()
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, f"({x},{y})", (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow(window_title, vis)

    cv2.imshow(window_title, display)
    cv2.setMouseCallback(window_title, _on_mouse)
    print(f"  {window_title}")
    
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # Enter
            break
        elif key == 27:  # Esc
            clicked[0] = None
            break
        elif key == ord('r'):
            clicked[0] = None
            cv2.imshow(window_title, image_bgr.copy())

    cv2.destroyWindow(window_title)
    return clicked[0]


def main():
    parser = argparse.ArgumentParser(description="Manual Gaze Redirection - Path C (Face Swap)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="manual_out_path_c.png", help="Path to output image")
    args = parser.parse_args()

    # Load InsightFace Model
    print("[InsightFace] Loading face swap model...")
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
    if not os.path.exists(swapper_path):
        print(f"Error: inswapper_128.onnx not found at {swapper_path}")
        sys.exit(1)
    try:
        swapper = insightface.model_zoo.get_model(swapper_path)
    except Exception as e:
        print(f"Error loading inswapper_128.onnx: {e}")
        sys.exit(1)

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

    # ── Step 1: Select person ──
    print("\n[Step 1] Click on the person whose gaze you want to redirect.")
    person_pt = select_single_point(image_bgr, "Step 1: Click on the person, then Enter")
    if person_pt is None:
        print("No person selected. Exiting.")
        sys.exit(0)
    print(f"  Selected person at: {person_pt}")

    # Get source face identity for later swapping (before we remove the head)
    source_faces = app.get(image_bgr)
    source_face = None
    if source_faces:
        # Pick the face closest to the click point
        cx, cy = person_pt
        best_face = source_faces[0]
        best_dist = float('inf')
        for face in source_faces:
            fx = (face.bbox[0] + face.bbox[2]) / 2
            fy = (face.bbox[1] + face.bbox[3]) / 2
            dist = (fx - cx)**2 + (fy - cy)**2
            if dist < best_dist:
                best_dist = dist
                best_face = face
        source_face = best_face
        print(f"  Source face captured for identity swap.")
    else:
        print("  Warning: InsightFace could not detect any face in the source image.")

    # ── Step 2: Mask the person's head and inpaint it out ──
    print("\n[Step 2] Masking the person's head...")
    sam_predictor = load_sam_predictor()
    raw_mask, dilated_mask = phase1_masking(image_bgr, person_pt, sam_predictor)

    face_info = detect_face_center(image_bgr, mask=dilated_mask, click_point=person_pt)
    if face_info is None:
        print("  Face detection failed. Using fallback head mask.")
        head_mask = get_head_fallback_mask(dilated_mask, click_point=person_pt)
    else:
        head_mask = get_head_mask(image_bgr, face_info)

    # Inpaint the head area to create a "headless" image
    print("  Inpainting head region out of image...")
    # Dilate head mask slightly for cleaner inpainting
    kernel = np.ones((7, 7), np.uint8)
    head_mask_dilated = cv2.dilate(head_mask, kernel, iterations=2)
    headless_image = cv2.inpaint(image_bgr, head_mask_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    print("  ✓ Headless image ready.")

    # ── Step 3: Select red dot position on the headless image ──
    print("\n[Step 3] Click where you want the gaze target (red dot).")
    gaze_target = select_single_point(headless_image, "Step 3: Click gaze target (red dot), then Enter")
    if gaze_target is None:
        print("No gaze target selected. Exiting.")
        sys.exit(0)
    print(f"  Gaze target at: {gaze_target}")

    # Draw red dot on headless image
    headless_with_dot = draw_red_dot_on_image(headless_image, gaze_target)

    # ── Step 4: Gemini — Generate any head looking at the red dot ──
    print("\n[Step 4] Sending headless image to Gemini...")
    prompt = (
        "The person in this image is missing their head. "
        "Generate a realistic head for this person that is looking exactly at the red dot. "
        "The head should naturally fit the person's body position, neck, and clothing. "
        "Do not worry about matching any specific identity — just make the head realistic "
        "and ensure the eyes, pupils, and face orientation are precisely directed at the red dot. "
        "Keep the red dot visible."
    )

    gemini_result = _gemini_edit(headless_with_dot, prompt)

    if gemini_result is None:
        print("Error: Gemini returned None.")
        sys.exit(1)
    print("  ✓ Gemini generated a new head.")

    # ── Step 5: Face Swap — restore original identity ──
    if source_face is not None:
        print("\n[Step 5] Swapping original face identity onto Gemini's result...")
        target_faces = app.get(gemini_result)
        if target_faces:
            # Pick the face in Gemini's result that's closest to original head position
            best_target = target_faces[0]
            if face_info is not None:
                orig_cx = face_info['eye_center'][0]
                orig_cy = face_info['eye_center'][1]
                best_dist = float('inf')
                for tf in target_faces:
                    fx = (tf.bbox[0] + tf.bbox[2]) / 2
                    fy = (tf.bbox[1] + tf.bbox[3]) / 2
                    dist = (fx - orig_cx)**2 + (fy - orig_cy)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_target = tf
            
            swapped_result = swapper.get(gemini_result, best_target, source_face, paste_back=True)
            gemini_result = swapped_result
            print("  ✓ Face swap complete.")
        else:
            print("  Warning: InsightFace could not detect a face in Gemini's result.")
    else:
        print("\n[Step 5] Skipping face swap (no source face detected).")

    # ── Step 6: Clean up anchor ──
    print("\n[Step 6] Removing red dot...")
    final_output = phase4_remove_anchor(gemini_result, gaze_target)

    # Save outputs
    cv2.imwrite(args.output, final_output)
    
    out_dir = Path(args.output).parent
    base_name = Path(args.output).stem
    cv2.imwrite(str(out_dir / f"{base_name}_head_mask.png"), head_mask)
    cv2.imwrite(str(out_dir / f"{base_name}_headless.png"), headless_image)
    cv2.imwrite(str(out_dir / f"{base_name}_headless_with_dot.png"), headless_with_dot)
        
    print(f"\nDone! Edited image saved to: {args.output}")
    print(f"Debug outputs: {base_name}_head_mask.png, "
          f"{base_name}_headless.png, {base_name}_headless_with_dot.png")

if __name__ == "__main__":
    main()
