"""
Path E — Interactive, single-image gaze redirection.

Click on the person (SAM segments them), click on the gaze target and adjust
its depth, then confirm. The person is isolated onto a black background and
paired with a depth map annotated at the target's depth; both go to Gemini
as a dual-image edit request. The result is composited back over the
original background and any gaps are repaired (2nd Gemini pass by default).

Usage:
    python manual_process_path_e.py --image path/to/photo.jpg
"""

import sys
import io
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tkinter as tk

from model_utils import (
    get_depth_anything,
    load_sam_predictor,
    remove_red_marker,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

# ── Color Palette for Person Masking ──
PERSON_COLORS = [
    # (BGR tuple, human-readable name)
    ((255, 255, 0),   "Cyan"),
    ((255, 0, 255),   "Magenta"),
    ((0, 255, 255),   "Yellow"),
    ((0, 255, 0),     "Green"),
    ((0, 165, 255),   "Orange"),
    ((203, 192, 255), "Pink"),
    ((255, 0, 0),     "Blue"),
    ((0, 215, 255),   "Gold"),
    ((208, 224, 64),  "Turquoise"),
    ((147, 20, 255),  "DeepPink"),
]

def gemini_fill_background_gaps(composite_bgr, prompt_text, max_retries=3):
    from google import genai
    from google.genai import types
    import time

    client = genai.Client(api_key=GEMINI_API_KEY)
    orig_h, orig_w = composite_bgr.shape[:2]

    def to_png_bytes(bgr_img):
        pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    img_bytes = to_png_bytes(composite_bgr)
    
    repair_prompt = (
        "Attached is a composite image with a person in a new pose. "
        "There are black gaps (pixel value 0,0,0) where the original background was removed or where the person moved. "
        "TASK: Please fill in these black areas seamlessly with a realistic background. "
        "IMPORTANT: Do not change the person's identity, gaze, or the existing background. Only fill the BLACK areas."
    )

    for attempt in range(1, max_retries + 1):
        try:
            contents = [
                types.Content(role="user", parts=[
                    types.Part.from_text(text=repair_prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
                ])
            ]
            config = types.GenerateContentConfig(
                image_config=types.ImageConfig(image_size="1K"),
                response_modalities=["IMAGE", "TEXT"],
            )
            for chunk in client.models.generate_content_stream(model=GEMINI_MODEL, contents=contents, config=config):
                if chunk.parts:
                    for part in chunk.parts:
                        if part.inline_data:
                            res = Image.open(io.BytesIO(part.inline_data.data))
                            return cv2.cvtColor(np.array(res.resize((orig_w, orig_h))), cv2.COLOR_RGB2BGR)
            return None
        except Exception as e:
            if attempt < max_retries: time.sleep(20)
            else: return None
    return None

# ── Depth Map Generation ──
def generate_depth_map(image_bgr):
    """Generate depth map. Returns raw float depth + normalized uint8."""
    depth_model = get_depth_anything()
    raw_depth = depth_model.infer_image(image_bgr)
    d_min, d_max = raw_depth.min(), raw_depth.max()
    depth_norm = ((raw_depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    return raw_depth, depth_norm, depth_color


# ── SAM Person Color Masking (Removed) ──
# We now use interactive point-prompted SAM directly in the UI.


# ── Person & Depth Isolation ──
def isolate_person_and_depth(image_bgr, depth_color, depth_norm, person_mask, gaze_target, chosen_depth):
    """
    資料流更新：
    1. 主角圖 (Image 1): 隔離主角 + 實心紅點 (BGR: 0, 0, 255)
    2. 深度圖 (Image 2): 完整深度圖 + 空心紅點 (讓 Gemini 看見內部的深度顏色)
    """
    tx, ty = gaze_target
    
    # 1. Image 1: 隔離主角原圖 + 實心紅點
    isolated_image = np.zeros_like(image_bgr)
    isolated_image[person_mask > 0] = image_bgr[person_mask > 0]
    # 在主角圖畫實心紅點 (半徑 8)
    cv2.circle(isolated_image, (tx, ty), 8, (0, 0, 255), -1)
    
    # 2. Image 2: 完整深度圖 + 包含指定深度顏色的點
    full_depth_with_dot = depth_color.copy()
    
    # 取得 chosen_depth 的 INFERNO 顏色
    depth_patch = np.full((1, 1), int(chosen_depth), dtype=np.uint8)
    dot_color_bgr = cv2.applyColorMap(depth_patch, cv2.COLORMAP_INFERNO)[0, 0].tolist()
    
    # 先在圓內部填滿您設定的 depth color
    cv2.circle(full_depth_with_dot, (tx, ty), 12, dot_color_bgr, -1)
    
    # 畫出紅色空心外框標示位置 (外加白底增加對比)
    cv2.circle(full_depth_with_dot, (tx, ty), 12, (255, 255, 255), 3) # 白色襯底線條
    cv2.circle(full_depth_with_dot, (tx, ty), 12, (0, 0, 255), 2)     # 紅色外框
    
    # 中心點一個極小的點，讓 Gemini 更好定位精確中心
    cv2.circle(full_depth_with_dot, (tx, ty), 1, (0, 0, 255), -1)

    print(f"  [Isolate] Target at ({tx}, {ty}). Image 1: Solid Dot, Image 2: Filled Depth Dot with Red Border.")
    return isolated_image, full_depth_with_dot

def paste_back_and_repair(original_bgr, gemini_result, person_mask, out_dir, base_name, original_prompt):
    """
    1. 將原圖舊主角位置挖空。
    2. 讓 Gemini 新生成的圖層完全覆蓋重疊區域（不使用 max，避免數值疊加）。
    3. 送回 Gemini 修補因位移產生的黑色縫隙。
    """
    h, w = original_bgr.shape[:2]
    
    # 1. 尺寸校準
    if gemini_result.shape[:2] != (h, w):
        print(f"  [Fix] Resizing Gemini result to {(h, w)}")
        gemini_result = cv2.resize(gemini_result, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 1b. 清除 Gemini 可能留下的紅點（大幅度姿態變化時，紅點座標不一定跟原本一樣）
    gemini_result = remove_red_marker(gemini_result)

    # 2. 準備「帶洞背景」：先挖掉原本 SAM 抓出的舊位置
    background_with_hole = original_bgr.copy()
    background_with_hole[person_mask > 2] = 0
    
    # 3. 建立「新主角遮罩」：從 Gemini 回傳圖中找出非 0 的區域
    # 這一步確保 Gemini 給的新數值會「完全蓋掉」該位置的背景
    gemini_gray = cv2.cvtColor(gemini_result, cv2.COLOR_BGR2GRAY)
    _, new_person_mask = cv2.threshold(gemini_gray, 1, 255, cv2.THRESH_BINARY)
    
    # 4. 執行覆蓋：
    # 先把 background 中新主角會佔據的地方也挖空 (避免背景殘留)
    composite_image = background_with_hole.copy()
    composite_image[new_person_mask > 0] = 0
    
    # 直接加上 gemini_result (因為 composite 對應位置現在是 0，加法等同覆蓋)
    composite_image = cv2.add(composite_image, gemini_result)
    
    # 儲存帶縫隙的中間產物供 Debug
    composite_path = str(out_dir / f"{base_name}_composite_unfilled.png")
    cv2.imwrite(composite_path, composite_image)
    print(f"  [Path E] Composite created (Priority: Gemini pixels). Sending to Gemini for final repair...")

    # 5. 第二階段：填補黑色區塊 (pixel value 0)
    final_output = gemini_fill_background_gaps(composite_image, original_prompt)
    
    if final_output is None:
        print("  [Warning] Gemini repair failed. Returning unfilled composite.")
        return composite_image
        
    return final_output


# ── Person Depth Estimation ──
def estimate_person_depth_and_orientation(person_mask, depth_norm):
    """
    Compute median depth in the mask region.
    """
    if np.any(person_mask > 0):
        person_depth = float(np.median(depth_norm[person_mask > 0]))
    else:
        person_depth = float(np.median(depth_norm))
    facing_away = False
    print(f"  [Depth] Person median depth: {person_depth:.1f}/255")
    return person_depth, facing_away


# ── Dynamic Prompt Builder (Isolation Mode) ──
def build_depth_aware_prompt(chosen_depth, person_depth, facing_away):
    # 判斷深度關係
    depth_diff = float(chosen_depth) - float(person_depth)
    # 這裡的邏輯：
    # Brighter is closer (larger value). 
    # If chosen_depth > person_depth + 30: target is IN FRONT
    # If chosen_depth < person_depth - 30: target is BEHIND
    if abs(depth_diff) < 30:
        rel = "at the same depth as"
    elif depth_diff > 30:
        rel = "IN FRONT OF (closer to camera than)"
    else:
        rel = "BEHIND"

    prompt = (
        f"Image 2 is a STATIC DEPTH REFERENCE. The hollow circle marks the red dot's 3D position, which is {rel} the person.\n"
        "OUTPUT: Return Image 1 with the person's new gaze. DO NOT move the red dot. Keep background black.\n"
        "TASK: On Image 1, IGNORE the person's current gaze and REDIRECT the person's eyes and HEAD to face the exact 3D position of the solid red dot.\n"
    )
    if rel == "BEHIND":
        prompt += (
            "The target is far away and behind the person's current line of sight, so "
            "they must physically turn their head and upper body around to look back "
            "over their shoulder toward it, the way someone turns around when they hear "
            "their name called from behind — a large, obvious head-and-shoulder "
            "rotation, not just a small glance or eye movement.\n"
        )
    return prompt


def select_target_and_depth(image_bgr, depth_color, depth_norm, sam_predictor):
    h, w = image_bgr.shape[:2]
    num_panels = 3
    display_max = 350
    scale = min(display_max / w, display_max / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    target_pt = [None]
    target_mask = [None]
    chosen_depth = [128]
    ui_step = [0]  # 0: 選人, 1: 選紅點

    root = tk.Tk()
    root.title("Path E: Step-by-Step Selection")

    main_frame = tk.Frame(root); main_frame.pack()
    canvas = tk.Canvas(main_frame, width=dw, height=dh); canvas.pack(side=tk.LEFT, padx=5, pady=5)
    depth_canvas = tk.Canvas(main_frame, width=dw, height=dh); depth_canvas.pack(side=tk.LEFT, padx=5, pady=5)
    mask_canvas = tk.Canvas(main_frame, width=dw, height=dh); mask_canvas.pack(side=tk.LEFT, padx=5, pady=5)
    ctrl_frame = tk.Frame(root); ctrl_frame.pack(pady=10)
    step_label = tk.Label(ctrl_frame, text="STEP 1: Please click on the target person", font=("Arial", 12, "bold"), fg="red")
    step_label.pack()
    person_color_label = tk.Label(ctrl_frame, text="Target: None", font=("Arial", 11)); person_color_label.pack()
    depth_slider = tk.Scale(ctrl_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=400); depth_slider.set(128); depth_slider.pack()
    confirm_btn = tk.Button(ctrl_frame, text="Confirm", state=tk.DISABLED, font=("Arial", 12)); confirm_btn.pack(pady=5)

    def _display_image(cv_img, canvas_widget):
        disp = cv2.resize(cv_img, (dw, dh))
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
        canvas_widget._tk_img = tk_img
        canvas_widget.create_image(0, 0, anchor=tk.NW, image=tk_img)

    def _refresh():
        img_draw = image_bgr.copy()
        dep_draw = depth_color.copy()
        mask_draw = np.zeros_like(image_bgr)

        if target_mask[0] is not None:
            mask_draw[target_mask[0] > 0] = (0, 255, 255)
            cv2.putText(mask_draw, "LOCKED", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if target_pt[0] is not None:
            tx, ty = target_pt[0]
            dv = depth_slider.get()
            cv2.circle(img_draw, (tx, ty), 12, (0, 0, 255), -1)
            depth_patch = np.full((1, 1), dv, dtype=np.uint8)
            dot_color_bgr = cv2.applyColorMap(depth_patch, cv2.COLORMAP_INFERNO)[0, 0]
            cv2.circle(dep_draw, (tx, ty), 12, dot_color_bgr.tolist(), -1)
            cv2.circle(mask_draw, (tx, ty), 12, (0, 0, 255), -1)

        _display_image(img_draw, canvas)
        _display_image(dep_draw, depth_canvas)
        _display_image(mask_draw, mask_canvas)

    def _on_click(event):
        ox, oy = int(event.x / scale), int(event.y / scale)
        ox, oy = max(0, min(w-1, ox)), max(0, min(h-1, oy))

        if ui_step[0] == 0:
            print(f"  [SAM] Prompting at ({ox}, {oy})...")
            input_point = np.array([[ox, oy]])
            input_label = np.array([1])
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            best_idx = np.argmax(scores)
            target_mask[0] = (masks[best_idx].astype(np.uint8) * 255)
            step_label.config(text="Target LOCKED. Now click for Gaze Target.", fg="blue")
            person_color_label.config(text="Locked Target: Custom Click", fg="green")
            ui_step[0] = 1
        elif ui_step[0] == 1:
            target_pt[0] = (ox, oy)
            depth_slider.set(int(depth_norm[oy, ox]))
            confirm_btn.config(state=tk.NORMAL)
        _refresh()

    def _on_reset():
        ui_step[0] = 0
        target_mask[0] = None
        target_pt[0] = None
        step_label.config(text="STEP 1: Please click on the target person", fg="red")
        person_color_label.config(text="Target: None", fg="black")
        confirm_btn.config(state=tk.DISABLED)
        _refresh()

    canvas.bind("<Button-1>", _on_click)
    depth_slider.config(command=lambda x: _refresh())
    reset_btn = tk.Button(ctrl_frame, text="Reset Selection", command=_on_reset); reset_btn.pack(side=tk.LEFT, padx=5)
    confirm_btn.config(command=lambda: [chosen_depth.__setitem__(0, depth_slider.get()), root.destroy()])

    _refresh()
    root.mainloop()
    return target_pt[0], chosen_depth[0], target_mask[0]

# ── Gemini Dual-Image API Call (Isolated Images) ──
def gemini_edit_dual_image(image_bgr, depth_bgr, prompt_text, max_retries=3):
    """Send TWO isolated images (person + depth) to Gemini. Returns edited BGR or None."""
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


def get_next_result_dir(base_dir="e_results"):
    """
    Find the next available numbered directory in base_dir.
    Returns the Path object to the new directory.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    existing_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not existing_dirs:
        next_id = 1
    else:
        ids = [int(d.name) for d in existing_dirs]
        next_id = max(ids) + 1
    
    new_dir = base_path / f"{next_id:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir

# ── Main ──
def main():
    parser = argparse.ArgumentParser(
        description="Manual Gaze Redirection - Path E (Isolation-Based Depth-Guided)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional final filename. If not provided, uses base name of input.")
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

    # --- Setup Automated Output Directory ---
    out_dir = get_next_result_dir("e_results")
    if args.output:
        base_name = Path(args.output).stem
        final_filename = Path(args.output).name
    else:
        base_name = image_path.stem
        final_filename = f"{base_name}_final.png"
    
    print(f"  [Output] Results will be saved to: {out_dir}")

    # 1. 深度與遮罩生成
    raw_depth, depth_norm, depth_color = generate_depth_map(image_bgr)
    sam_predictor = load_sam_predictor()
    print("[SAM] Setting image for predictor...")
    sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # 2. 互動式選擇 (點兩次：選人、選紅點)
    gaze_target, chosen_depth, target_mask = select_target_and_depth(
        image_bgr, depth_color, depth_norm, sam_predictor
    )

    if gaze_target is None or target_mask is None:
        print("Selection incomplete. Exiting."); sys.exit(0)

    # 3. 隔離與寫出
    isolated_image, isolated_depth = isolate_person_and_depth(
        image_bgr, depth_color, depth_norm,
        person_mask=target_mask,
        gaze_target=gaze_target,
        chosen_depth=chosen_depth,
    )
    
    cv2.imwrite(str(out_dir / f"{base_name}_isolated_person.png"), isolated_image)
    cv2.imwrite(str(out_dir / f"{base_name}_isolated_depth.png"), isolated_depth)

    # 額外輸出 原圖*(1-mask) 即背景圖
    background_image = image_bgr.copy()
    background_image[target_mask > 0] = 0
    cv2.imwrite(str(out_dir / f"{base_name}_background.png"), background_image)

    person_depth, facing_away = estimate_person_depth_and_orientation(target_mask, depth_norm)
    prompt = build_depth_aware_prompt(chosen_depth, person_depth, facing_away)

    # Print and save prompt
    print("\n--- Gemini Prompt ---")
    print(prompt)
    print("----------------------\n")
    with open(out_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # 4. Gemini 第一階段：生成轉向後的主角
    send_or_not = input("Send to Gemini? (y/n): ")
    if send_or_not.lower() == 'n': sys.exit(0)

    gemini_result = gemini_edit_dual_image(isolated_image, isolated_depth, prompt)
    if gemini_result is None: sys.exit(1)
    cv2.imwrite(str(out_dir / f"{base_name}_gemini_raw.png"), gemini_result)

    # 5. Gemini 第二階段：貼回並填補背景空缺
    print("[Path E] Pasting result and initiating Gemini-based background fill...")
    final_output = paste_back_and_repair(
        original_bgr=image_bgr,
        gemini_result=gemini_result,
        person_mask=target_mask,
        out_dir=out_dir,
        base_name=base_name,
        original_prompt=prompt
    )

    final_path = out_dir / final_filename
    cv2.imwrite(str(final_path), final_output)
    print(f"Done! Final result saved to: {final_path}")

if __name__ == "__main__":
    main()