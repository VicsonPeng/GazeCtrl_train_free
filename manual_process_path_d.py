"""
Manual Gaze Redirection Process - Path D 
(Isolate-Generate-Paste-Blend)

1. Load image and select person / target
2. SAM Masking
3. Isolate the person against a black background
4. Send the isolated image with red dot to Gemini 
   (so background doesn't distract the generation)
5. Extract the newly generated head
6. Paste and blend the new head back onto the original image
7. Clean up anchor
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

from gazectrl_pathb import (
    select_points_on_image,
    phase1_masking,
    draw_red_dot_on_image
)
from batch_evaluate import (
    load_sam_predictor, 
    phase3_gemini_refine, 
    phase4_remove_anchor,
    detect_face_center,
    get_head_mask,
    get_head_fallback_mask,
    _gemini_edit
)

def main():
    parser = argparse.ArgumentParser(description="Manual Gaze Redirection - Path D (Isolate & Paste)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="manual_out_path_d.png", help="Path to output image")
    parser.add_argument("--target-mode", choices=["virtual", "physical"], default="virtual", help="Virtual or physical target mode")
    parser.add_argument("--target-desc", type=str, default="", help="Description of physical target (if physical mode)")
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

    # 1. Interactive Selection
    try:
        person_pt, gaze_target = select_points_on_image(image_bgr)
    except Exception as e:
        print(f"Selection failed or was cancelled: {e}")
        sys.exit(1)

    if not person_pt or not gaze_target:
        print("Selection was not completed. Exiting.")
        sys.exit(0)

    # 2. SAM Masking
    print("[Path D] Masking person...")
    sam_predictor = load_sam_predictor()
    raw_mask, dilated_mask = phase1_masking(image_bgr, person_pt, sam_predictor)

    # 3. Isolation
    print("[Path D] Isolating person from background...")
    isolated_bgr = np.zeros_like(image_bgr)
    # create a 3-channel mask 
    mask_3d = np.repeat(dilated_mask[:, :, np.newaxis], 3, axis=2)
    isolated_bgr = np.where(mask_3d > 0, image_bgr, isolated_bgr)

    # 4. Draw Red Dot on Isolated Image
    isolated_with_dot = draw_red_dot_on_image(isolated_bgr, gaze_target)

    # 5. Gemini Generation (Isolate mode)
    print("Sending isolated image to Gemini for gaze redirection...")
    prompt = (
        "Make the person look at the red dot and remove the red dot. "
        "Keep the black background completely black."
    )
    if args.target_mode == "physical" and args.target_desc:
        prompt = f"The red dot marks a specific physical object: {args.target_desc}. " + prompt

    gemini_result = _gemini_edit(isolated_with_dot, prompt)

    if gemini_result is None:
        print("Error: Gemini returned None.")
        sys.exit(1)

    # 6. Extract New Body and Inpaint Edges
    print("[Path D] Pasting the full body back and inpainting edges...")
    
    # Simple copy using the SAM dilated mask
    mask_3d = np.repeat(dilated_mask[:, :, np.newaxis], 3, axis=2) / 255.0
    pasted = (gemini_result * mask_3d + image_bgr * (1 - mask_3d)).astype(np.uint8)

    # Find the boundary of the mask for inpainting
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)
    dilated_mask_edges = cv2.dilate(dilated_mask, kernel, iterations=2)
    edge_mask = cv2.subtract(dilated_mask_edges, eroded_mask)

    print("  ✓ Pasting complete. Inpainting seams...")
    # Inpaint the seams using the edge mask
    blended = cv2.inpaint(pasted, edge_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    final_output = blended

    # 7. Final Cleanup (Inpainting any remaining dot if needed - though Gemini was asked to remove it)
    final_output = phase4_remove_anchor(final_output, gaze_target)

    # Save Output
    cv2.imwrite(args.output, final_output)
    
    # Save the masks for debugging
    out_dir = Path(args.output).parent
    base_name = Path(args.output).stem
    cv2.imwrite(str(out_dir / f"{base_name}_sam_mask.png"), dilated_mask)
        
    print(f"\nDone! Edited image saved to: {args.output}")
    print(f"Masks saved as: {base_name}_sam_mask.png")

if __name__ == "__main__":
    main()
