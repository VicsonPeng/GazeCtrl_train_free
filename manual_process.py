"""
Manual Gaze Redirection Process

This script allows the user to:
1. Load an image
2. Click to select the target person (SAM mask region)
3. Click to place the gaze target (red dot)
4. Specify target mode (virtual/physical)
5. Generate the manually redirected image using Gemini
"""

import os
import sys
import argparse
from pathlib import Path
import cv2

# Import from existing pipelines
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
    get_head_fallback_mask
)

def main():
    parser = argparse.ArgumentParser(description="Manual Gaze Redirection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="manual_output.png", help="Path to output image")
    parser.add_argument("--target-mode", choices=["virtual", "physical"], default="virtual", help="Virtual or physical target mode")
    parser.add_argument("--target-desc", type=str, default="", help="Description of physical target (if physical mode)")
    parser.add_argument("--prompt-style", type=str, default="standard", help="Base prompt style")
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
    sam_predictor = load_sam_predictor()
    raw_mask, dilated_mask = phase1_masking(image_bgr, person_pt, sam_predictor)

    # 3. Detect Face and Get Head Mask
    # Pass dilated_mask and person_pt to ensure we get the face corresponding to the clicked person
    face_info = detect_face_center(image_bgr, mask=dilated_mask, click_point=person_pt)
    if face_info is None:
        print("Warning: Face detection failed for the selected person. Using fallback head mask.")
        head_mask = get_head_fallback_mask(dilated_mask, click_point=person_pt)
    else:
        head_mask = get_head_mask(image_bgr, face_info)

    # 4. Draw Red Dot and Green Contour
    image_with_dot = draw_red_dot_on_image(image_bgr, gaze_target)
    
    # Draw contour based on head mask
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_with_dot, contours, -1, (0, 255, 0), 2)

    # 5. Gemini Refinement
    # print("Sending to Gemini for gaze redirection...")
    # gemini_result, _ = phase3_gemini_refine(
    #     image_with_dot, 
    #     dilated_mask, 
    #     prompt_style=args.prompt_style, 
    #     target_mode=args.target_mode, 
    #     target_desc=args.target_desc
    # )

    # if gemini_result is None:
    #     print("Error: Gemini returned None.")
    #     sys.exit(1)

    # # 5. Remove Red Dot
    # final_output = phase4_remove_anchor(gemini_result, gaze_target)

    # Save Output
    cv2.imwrite(args.output, image_with_dot)
    
    # Save the masks for debugging
    out_dir = Path(args.output).parent
    base_name = Path(args.output).stem
    cv2.imwrite(str(out_dir / f"{base_name}_sam_mask.png"), dilated_mask)
    if 'head_mask' in locals():
        cv2.imwrite(str(out_dir / f"{base_name}_head_mask.png"), head_mask)
        
    print(f"\nDone! Edited image saved to: {args.output}")
    print(f"Masks saved as: {base_name}_sam_mask.png and {base_name}_head_mask.png")

if __name__ == "__main__":
    main()
