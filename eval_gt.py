"""
Evaluation-set GT generator for the training-based GazeCtrl model.

Runs the training-free (Path E) pipeline over a small, hand-picked photo set
to produce (source, edited target, gaze label) triples, with gaze targets
chosen so that the accepted set ends up evenly spread over yaw x pitch.

Workflow (each stage is a subcommand; all state lives under --data):

  1. prep      Interactive, once per image: SAM mask (left click = include,
               right click = exclude) + click the face (eye midpoint).
               Also caches the depth map. No Gemini calls.
  2. plan      For each prepped image, pick N gaze directions from the
               least-filled yaw/pitch bins, and solve for the red-dot
               position + depth that requests exactly that direction.
               Writes Gemini inputs only. No Gemini calls.
  3. run       Sends every planned (or previously failed) candidate to
               Gemini; gap repair uses OpenCV inpaint for screening.
     recomposite  Rebuilds the composites from saved Gemini outputs (no API
               calls) — use after a compositing fix.
  4. review    Keyboard UI: y = OK, n = bad, s = skip, b = back, q = quit.
  5. stats     Accepted/pending counts per yaw/pitch bin and per category.
     gallery   Static HTML view of the dataset (<data>/gallery/index.html):
               label distribution charts + every candidate's source / result /
               result with label direction, filterable; defaults to accepted.
  -> repeat plan/run/review with --round 2, 3, ... until bins are filled.
     label-source  Label the gaze of the person in each source photo.
     repair-select  Inspect each accepted view's inpaint gap fill (Gemini
               output | inpaint result | gap close-up), mark the bad ones.
  6. repair    Gemini gap repair for accepted candidates only (replaces the
               screening inpaint; --selected = only the marked ones). The thin
               rim of the gap is filled locally, only real holes go to Gemini,
               and its full response is kept: `repair-view` shows it,
               `remerge --merge feather|full|gap` recombines without API calls,
               `merge-select` picks full or feather view by view,
               `unrepair` reverts individual samples.
  6b. issue-tag / fix / fix-view   targeted 3rd pass for views that are still
               bad: tag the problem (implausible / artifact / seam), paint
               where it is, one Gemini call per view with a prompt built from
               the tags, blended back inside the painted area only.
     finalize  Exports final/ as scenes: every labelled view of each person
               (original + accepted edits) with its own mask; any two views
               of a scene form a pair (no API calls).

Gaze label convention (matches the training-based pipeline, Part II):
  dx > 0 image-right, dy > 0 UP, dz > 0 TOWARD the camera,
  yaw = atan2(dx, dz)   (0 = facing camera, +/-180 = facing away)
  pitch = asin(dy)
The label is the *requested* direction, computed from the face point, the
red-dot position and its depth with the same pinhole/disparity model that
dataset_pipeline.compute_target_gaze uses (Z proportional to 1/depth,
f = 0.8 * width), but without that function's forward-hemisphere clamp
and with Part II's axis signs.

Usage:
    python eval_gt.py prep
    python eval_gt.py plan --round 1 --per-image 10
    python eval_gt.py run --workers 3
    python eval_gt.py review --round 1
    python eval_gt.py stats
    python eval_gt.py label-source
    python eval_gt.py repair --min-gap-pct 0.2
    python eval_gt.py finalize
"""

import os
import sys
import json
import math
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_utils import composite_onto_background, remove_red_marker
from dataset_pipeline import (
    generate_depth_map,
    isolate_person_and_depth,
    gemini_edit_dual_image,
    composite_and_repair,
    repair_gemini,
    merge_gap_repair,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 1500                      # same resize as dataset_pipeline
FOCAL_SCALE = 0.8                    # f = 0.8 * width, as in compute_target_gaze

CATEGORIES = {
    "coup": "couple", "eat": "eating", "fami": "family", "meet": "meeting",
    "natu": "nature", "norm": "normal", "play": "playing", "sport": "sport",
    "vaca": "vacation", "work": "work",
}

# ── Gaze bins: 12 yaw bins centred on -180, -150, ..., 150 (as in the Part II
#    full-circle sweep) x 3 pitch bins (down / level / up).
YAW_BIN_DEG = 30
N_YAW_BINS = 360 // YAW_BIN_DEG
PITCH_EDGES = [-45.0, -15.0, 15.0, 45.0]
PITCH_NAMES = ["down", "level", "up"]


def yaw_bin(yaw):
    return int(((yaw + 180 + YAW_BIN_DEG / 2) % 360) // YAW_BIN_DEG)


def yaw_bin_center(b):
    return -180 + b * YAW_BIN_DEG


def pitch_bin(pitch):
    for i in range(len(PITCH_EDGES) - 1):
        if PITCH_EDGES[i] <= pitch < PITCH_EDGES[i + 1]:
            return i
    if pitch == PITCH_EDGES[-1]:
        return len(PITCH_EDGES) - 2
    return None


def bin_key(yaw, pitch):
    pb = pitch_bin(pitch)
    return None if pb is None else (yaw_bin(yaw), pb)


def category_of(name):
    for prefix, cat in CATEGORIES.items():
        if name.startswith(prefix):
            return cat
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  STORAGE
# ══════════════════════════════════════════════════════════════════════════════

class Store:
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.raw = self.root / "raw"
        self.prep = self.root / "prep"
        self.cand = self.root / "candidates"
        self.final = self.root / "final"
        self.reviews_path = self.root / "reviews.json"
        self._lock = threading.Lock()

    def raw_images(self):
        return sorted(p for p in self.raw.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    def prep_info(self, name):
        p = self.prep / name / "prep.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def prepped_names(self):
        if not self.prep.exists():
            return []
        return sorted(d.name for d in self.prep.iterdir() if (d / "prep.json").exists())

    def load_prep_arrays(self, name):
        d = self.prep / name
        return (
            cv2.imread(str(d / "image.png")),
            cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(d / "depth_norm.png"), cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(d / "depth_color.png")),
        )

    def candidates(self):
        """All candidate metas, as list of (dir, meta)."""
        out = []
        if not self.cand.exists():
            return out
        for img_dir in sorted(self.cand.iterdir()):
            for c in sorted(img_dir.iterdir()):
                mp = c / "meta.json"
                if mp.exists():
                    out.append((c, json.loads(mp.read_text(encoding="utf-8"))))
        return out

    def write_meta(self, cdir, meta):
        with self._lock:
            tmp = cdir / "meta.json.tmp"
            tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cdir / "meta.json")

    def reviews(self):
        if self.reviews_path.exists():
            return json.loads(self.reviews_path.read_text(encoding="utf-8"))
        return {}

    def save_reviews(self, reviews):
        tmp = self.reviews_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.reviews_path)


def cand_key(meta):
    return f"{meta['image']}/{meta['cand_id']}"


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY  (Part II convention)
# ══════════════════════════════════════════════════════════════════════════════

def gaze_from_target(face_px, target_px, face_depth, target_depth, image_shape):
    """Requested gaze (Part II convention) for a 3D target given as 2D px + depth."""
    h, w = image_shape[:2]
    f = w * FOCAL_SCALE
    cx, cy = w / 2.0, h / 2.0
    z_f = 1.0
    z_t = max(float(face_depth), 1.0) / max(float(target_depth), 1.0)
    pf = np.array([(face_px[0] - cx) / f * z_f, (face_px[1] - cy) / f * z_f, z_f])
    pt = np.array([(target_px[0] - cx) / f * z_t, (target_px[1] - cy) / f * z_t, z_t])
    v = pt - pf                                  # camera coords: x right, y down, z into scene
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    dx, dy, dz = v[0] / n, -v[1] / n, -v[2] / n  # -> y up, z toward camera
    return {
        "dx": round(float(dx), 6), "dy": round(float(dy), 6), "dz": round(float(dz), 6),
        "yaw_deg": round(math.degrees(math.atan2(dx, dz)), 3),
        "pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, dy)))), 3),
    }


def direction_from_angles(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


def angular_diff_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def solve_target(info, mask, yaw_deg, pitch_deg, rng):
    """
    Find an integer red-dot position + depth (0-255) that requests
    (yaw_deg, pitch_deg) from the face point. Returns (px, depth, gaze) or None
    if this direction can't be expressed as an on-screen target for this photo.
    """
    h, w = info["size"]
    f = w * FOCAL_SCALE
    cx, cy = w / 2.0, h / 2.0
    fx, fy = info["face_px"]
    p = float(info["face_depth"])
    want = direction_from_angles(yaw_deg, pitch_deg)
    v = np.array([want[0], -want[1], -want[2]])  # back to camera coords
    pf = np.array([(fx - cx) / f, (fy - cy) / f, 1.0])

    ts = np.linspace(0.03, 4.0, 600)
    pts = pf[None, :] + ts[:, None] * v[None, :]
    z = pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f * pts[:, 0] / z + cx
        vv = f * pts[:, 1] / z + cy
        c = p / z
    margin = max(20, int(0.03 * min(h, w)))
    excl = max(1.3 * info["head_r"], 25)
    ok = (
        (z > 1e-3) & (u >= margin) & (u < w - margin) & (vv >= margin) & (vv < h - margin)
        & (c >= 5) & (c <= 250) & (np.hypot(u - fx, vv - fy) >= excl)
    )
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None

    # Prefer a dot that doesn't sit on the person's own body.
    mask_dil = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    off_body = [i for i in idx if mask_dil[int(vv[i]), int(u[i])] == 0]
    pool = off_body if off_body else list(idx)
    rng.shuffle(pool)
    for i in pool[:40]:
        px = (int(round(u[i])), int(round(vv[i])))
        dep = int(round(c[i]))
        gaze = gaze_from_target((fx, fy), px, p, dep, (h, w))
        if gaze is None:
            continue
        got = np.array([gaze["dx"], gaze["dy"], gaze["dz"]])
        if angular_diff_deg(got, want) < 4.0 and bin_key(gaze["yaw_deg"], gaze["pitch_deg"]) \
                == bin_key(yaw_deg, pitch_deg):
            return px, dep, gaze
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT
# ══════════════════════════════════════════════════════════════════════════════

FULL_TURN_CLAUSE = (
    "The target is so much farther away than the person that they must rotate "
    "their ENTIRE head and body all the way around to face fully away from the "
    "camera, looking straight ahead into the scene toward the target. The "
    "camera must see ONLY the back/crown of their head and hair, and their full "
    "back — absolutely NO part of their face, cheek, ear, or facial profile "
    "should be visible. Do not turn the head sideways to glance back over the "
    "shoulder; the whole body faces directly away from the camera, like someone "
    "who has simply turned around to walk toward something in the distance and "
    "is now seen entirely from behind.\n"
)


def build_eval_prompt(target_depth, face_depth, yaw_deg):
    """
    dataset_pipeline.build_depth_aware_prompt, plus an orientation clause sized
    to the requested yaw. The pipeline's BEHIND clause always asks for a full
    turn away, which would contradict a requested yaw of e.g. 110 deg; here the
    full-turn clause is reserved for |yaw| >= 150.

    The in-front/behind relation uses the same Z ~ 1/depth ratio as the label
    rather than the pipeline's absolute +/-30 depth difference: for a distant
    person (face depth ~40) no target can be 30 units farther, so the absolute
    rule called a target three times farther away "at the same depth".
    """
    z_ratio = max(float(face_depth), 1.0) / max(float(target_depth), 1.0)  # Z_target / Z_face
    if z_ratio < 0.9:
        rel = "IN FRONT OF (closer to camera than)"
    elif z_ratio > 1.1:
        rel = "BEHIND"
    else:
        rel = "at the same depth as"

    prompt = (
        f"Image 2 is a STATIC DEPTH REFERENCE. The hollow circle marks the red dot's "
        f"3D position, which is {rel} the person.\n"
        "OUTPUT: Return Image 1 with the person's new gaze. "
        "DO NOT move the red dot. Keep background black.\n"
        "TASK: On Image 1, IGNORE the person's current gaze and REDIRECT the person's "
        "eyes and HEAD to face the exact 3D position of the solid red dot.\n"
    )
    side = "right" if yaw_deg > 0 else "left"
    a = abs(yaw_deg)
    if a >= 150:
        prompt += FULL_TURN_CLAUSE
    elif a >= 105:
        prompt += (
            f"The target is behind the person and off toward the {side} side of the image. "
            f"Turn the head and upper body away from the camera toward it, so the camera sees "
            f"mostly the back and the {side}-hand side of the head; at most a thin sliver of "
            "the cheek or facial profile may remain visible. Do not keep the face toward the "
            "camera.\n"
        )
    elif a >= 75:
        prompt += (
            f"The target is far to the {side} side of the image, at roughly the person's own "
            f"depth. Turn the head to a clean side profile facing the {side} edge of the "
            "image, eyes looking at the red dot.\n"
        )
    prompt += "Keep the person's identity, hairstyle, hair colour, clothing and body size unchanged.\n"
    return prompt, rel


# ══════════════════════════════════════════════════════════════════════════════
#  1. PREP  (interactive mask + face point)
# ══════════════════════════════════════════════════════════════════════════════

def head_radius(mask, face_px):
    h, w = mask.shape
    x, y = face_px
    widths = []
    for yy in range(max(0, y - 6), min(h, y + 7)):
        row = mask[yy] > 0
        if not row[x]:
            continue
        l = x
        while l > 0 and row[l - 1]:
            l -= 1
        r = x
        while r < w - 1 and row[r + 1]:
            r += 1
        widths.append(r - l + 1)
    if not widths:
        return 40.0
    return float(np.clip(np.median(widths) / 2.0, 12, 0.25 * min(h, w)))


def face_depth_at(depth_norm, mask, face_px, r):
    x, y = face_px
    h, w = mask.shape
    rr = max(4, int(r / 2))
    y0, y1, x0, x1 = max(0, y - rr), min(h, y + rr + 1), max(0, x - rr), min(w, x + rr + 1)
    win_d, win_m = depth_norm[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    vals = win_d[win_m > 0]
    if vals.size == 0:
        vals = depth_norm[mask > 0]
    return float(np.median(vals)) if vals.size else 128.0


def prep_ui(image_bgr, sam_predictor, title):
    """
    Returns dict(mask, face_px, points, labels) or None (skip) or "quit".
    Keys: Enter = done with mask -> click face,  s = save,  z = undo point,
          m = cycle SAM proposals (single point only),  r = reset,
          k = skip image,  Esc = quit.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    h, w = image_bgr.shape[:2]
    root = tk.Tk()
    root.title(title)
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    scale = min((scr_w - 80) / w, (scr_h - 220) / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    st = {"points": [], "labels": [], "proposals": None, "pidx": 0,
          "mask": None, "face": None, "mode": "mask", "result": None}

    canvas = tk.Canvas(root, width=dw, height=dh, bg="black")
    canvas.pack(padx=4, pady=4)
    msg = tk.Label(root, font=("Arial", 11, "bold"))
    msg.pack()
    tk.Label(root, font=("Arial", 9), text=(
        "Mask: left click = include, right click = exclude, z = undo, m = next SAM proposal, "
        "r = reset  |  Enter = mask done -> click the face (eye midpoint)  |  "
        "s = save & next, k = skip image, Esc = quit")).pack(pady=(0, 6))

    def run_sam():
        if not st["points"]:
            st["mask"], st["proposals"] = None, None
            return
        single = len(st["points"]) == 1
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array(st["points"]), point_labels=np.array(st["labels"]),
            multimask_output=single)
        if single:
            order = list(np.argsort(-scores))
            st["proposals"] = [masks[i] for i in order]
            st["pidx"] = 0
        else:
            st["proposals"] = [masks[int(np.argmax(scores))]]
            st["pidx"] = 0
        st["mask"] = (st["proposals"][0].astype(np.uint8) * 255)

    def render():
        disp = image_bgr.copy()
        if st["mask"] is not None:
            tint = disp.copy()
            tint[st["mask"] > 0] = (0, 200, 0)
            disp = cv2.addWeighted(disp, 0.55, tint, 0.45, 0)
        for (px, py), lab in zip(st["points"], st["labels"]):
            cv2.circle(disp, (px, py), 7, (0, 255, 0) if lab == 1 else (0, 0, 255), -1)
            cv2.circle(disp, (px, py), 7, (255, 255, 255), 2)
        if st["face"] is not None:
            r = head_radius(st["mask"], st["face"])
            cv2.circle(disp, st["face"], int(r), (255, 128, 0), 2)
            cv2.drawMarker(disp, st["face"], (255, 128, 0), cv2.MARKER_CROSS, 18, 2)
        img = Image.fromarray(cv2.cvtColor(cv2.resize(disp, (dw, dh)), cv2.COLOR_BGR2RGB))
        canvas._img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=canvas._img)
        if st["mode"] == "mask":
            msg.config(fg="red", text="STEP 1 — click the person to build the mask, then press Enter")
        elif st["face"] is None:
            msg.config(fg="blue", text="STEP 2 — click the face (midpoint between the eyes)")
        else:
            msg.config(fg="darkgreen", text="Press s to save (or click the face again to move it)")

    def to_img(e):
        return (max(0, min(w - 1, int(e.x / scale))), max(0, min(h - 1, int(e.y / scale))))

    def on_click(e, label):
        pt = to_img(e)
        if st["mode"] == "mask":
            st["points"].append(pt)
            st["labels"].append(label)
            run_sam()
        elif label == 1:
            if st["mask"][pt[1], pt[0]] == 0:
                msg.config(fg="red", text="The face point must be inside the mask.")
                return
            st["face"] = pt
        render()

    def on_key(e):
        k = e.keysym.lower()
        if k == "escape":
            st["result"] = "quit"
            root.destroy()
        elif k == "k":
            st["result"] = None
            root.destroy()
        elif k == "r":
            st.update(points=[], labels=[], mask=None, proposals=None, face=None, mode="mask")
        elif k == "z" and st["mode"] == "mask" and st["points"]:
            st["points"].pop()
            st["labels"].pop()
            run_sam()
        elif k == "m" and st["mode"] == "mask" and st["proposals"] and len(st["proposals"]) > 1:
            st["pidx"] = (st["pidx"] + 1) % len(st["proposals"])
            st["mask"] = st["proposals"][st["pidx"]].astype(np.uint8) * 255
        elif k == "return" and st["mask"] is not None:
            st["mode"] = "face"
        elif k == "s" and st["mask"] is not None and st["face"] is not None:
            st["result"] = {"mask": st["mask"], "face_px": st["face"],
                            "points": st["points"], "labels": st["labels"]}
            root.destroy()
            return
        render()

    canvas.bind("<Button-1>", lambda e: on_click(e, 1))
    canvas.bind("<Button-3>", lambda e: on_click(e, 0))
    root.bind("<Key>", on_key)
    render()
    root.mainloop()
    return st["result"]


def cmd_prep(store, args):
    from model_utils import load_sam_predictor, get_depth_anything

    images = store.raw_images()
    if args.images:
        images = [p for p in images if p.stem in set(args.images)]
    todo = [p for p in images if args.redo or store.prep_info(p.stem) is None]
    print(f"[Prep] {len(todo)} image(s) to prepare ({len(images) - len(todo)} already done).")
    if not todo:
        return
    sam = load_sam_predictor()
    get_depth_anything()

    for i, path in enumerate(todo):
        name = path.stem
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        res = prep_ui(image, sam, f"[{i + 1}/{len(todo)}] {name}  ({category_of(name)})")
        if res == "quit":
            print("[Prep] Quit.")
            break
        if res is None:
            print(f"  {name}: skipped")
            continue

        _, depth_norm, depth_color = generate_depth_map(image)
        mask = res["mask"]
        r = head_radius(mask, res["face_px"])
        info = {
            "image": name,
            "category": category_of(name),
            "source_file": path.name,
            "size": list(image.shape[:2]),
            "face_px": list(res["face_px"]),
            "head_r": round(r, 1),
            "face_depth": round(face_depth_at(depth_norm, mask, res["face_px"], r), 2),
            "person_depth_median": round(float(np.median(depth_norm[mask > 0])), 2),
            "sam_points": [list(p) for p in res["points"]],
            "sam_labels": res["labels"],
        }
        d = store.prep / name
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "image.png"), image)
        cv2.imwrite(str(d / "mask.png"), mask)
        cv2.imwrite(str(d / "depth_norm.png"), depth_norm)
        cv2.imwrite(str(d / "depth_color.png"), depth_color)
        (d / "prep.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"  {name}: saved (face depth {info['face_depth']}, head r {info['head_r']})")


# ══════════════════════════════════════════════════════════════════════════════
#  2. PLAN
# ══════════════════════════════════════════════════════════════════════════════

def bin_counts(store, reviews):
    """Per-bin accepted and pending (generated/planned, not yet reviewed) counts,
    plus the same per image."""
    acc, pend = {}, {}
    per_img = {}
    for _, meta in store.candidates():
        b = label_bin(meta)
        v = reviews.get(cand_key(meta), {}).get("verdict")
        if v == "bad":
            continue
        tgt = acc if v == "ok" else pend
        tgt[b] = tgt.get(b, 0) + 1
        per_img.setdefault(meta["image"], {})
        per_img[meta["image"]][b] = per_img[meta["image"]].get(b, 0) + 1
    return acc, pend, per_img


def cmd_plan(store, args):
    rng = np.random.default_rng(args.seed if args.seed is not None else 1000 + args.round)
    names = store.prepped_names()
    if args.images:
        names = [n for n in names if n in set(args.images)]
    if not names:
        print("[Plan] No prepped images. Run `prep` first.")
        return

    reviews = store.reviews()
    acc, pend, per_img = bin_counts(store, reviews)
    all_bins = [(y, p) for y in range(N_YAW_BINS) for p in range(len(PITCH_NAMES))]
    load = {b: acc.get(b, 0) + pend.get(b, 0) for b in all_bins}

    order = list(names)
    rng.shuffle(order)
    total, infeasible = 0, {}
    for name in order:
        info = store.prep_info(name)
        image, mask, depth_norm, depth_color = store.load_prep_arrays(name)
        mine = per_img.get(name, {})
        existing_ids = {m["cand_id"] for _, m in store.candidates() if m["image"] == name}
        if any(cid.startswith(f"r{args.round}_") for cid in existing_ids) and not args.force:
            print(f"  {name}: round {args.round} already planned — skipping (use --force)")
            continue

        # Least-filled bins first; bins this image already covers count extra
        # so each identity also gets spread across directions.
        scored = sorted(all_bins, key=lambda b: (load[b] + 2 * mine.get(b, 0), rng.random()))
        picked = 0
        for b in scored:
            if picked >= args.per_image:
                break
            sol = None
            for _ in range(8):
                yaw = yaw_bin_center(b[0]) + rng.uniform(-YAW_BIN_DEG / 2, YAW_BIN_DEG / 2)
                yaw = (yaw + 180) % 360 - 180
                pitch = rng.uniform(PITCH_EDGES[b[1]], PITCH_EDGES[b[1] + 1])
                sol = solve_target(info, mask, yaw, pitch, rng)
                if sol:
                    break
            if sol is None:
                infeasible[b] = infeasible.get(b, 0) + 1
                continue
            px, dep, gaze = sol
            cid = f"r{args.round}_{picked:02d}"
            cdir = store.cand / name / cid
            cdir.mkdir(parents=True, exist_ok=True)
            iso, iso_depth = isolate_person_and_depth(image, depth_color, depth_norm, mask, px, dep)
            prompt, rel = build_eval_prompt(dep, info["face_depth"], gaze["yaw_deg"])
            cv2.imwrite(str(cdir / "04a_isolated_person.png"), iso)
            cv2.imwrite(str(cdir / "04b_isolated_depth.png"), iso_depth)
            (cdir / "prompt.txt").write_text(prompt, encoding="utf-8")
            meta = {
                "image": name, "cand_id": cid, "round": args.round,
                "category": info["category"], "status": "planned",
                "bin": list(b),
                "requested_yaw_deg": round(yaw, 3), "requested_pitch_deg": round(pitch, 3),
                "target_px": list(px), "target_depth": dep,
                "face_px": info["face_px"], "face_depth": info["face_depth"],
                "relation": rel, "gaze": gaze, "prompt": prompt,
            }
            store.write_meta(cdir, meta)
            load[b] += 1
            mine[b] = mine.get(b, 0) + 1
            picked += 1
            total += 1
        print(f"  {name}: {picked} candidates planned")

    print(f"\n[Plan] {total} candidates planned for round {args.round} "
          f"(= {total} Gemini calls on `run`).")
    if infeasible:
        print("[Plan] Bins that could not be expressed as an on-screen target for some images:")
        for b, n in sorted(infeasible.items(), key=lambda x: -x[1]):
            print(f"    yaw {yaw_bin_center(b[0]):+4d}  pitch {PITCH_NAMES[b[1]]:5s}: {n} image(s)")


# ══════════════════════════════════════════════════════════════════════════════
#  3. RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_one(store, cdir, meta):
    image, mask, _, _ = store.load_prep_arrays(meta["image"])
    iso = cv2.imread(str(cdir / "04a_isolated_person.png"))
    iso_depth = cv2.imread(str(cdir / "04b_isolated_depth.png"))
    t0 = time.time()
    raw = gemini_edit_dual_image(iso, iso_depth, meta["prompt"])
    if raw is None:
        meta.update(status="failed", error="Gemini returned no image")
        store.write_meta(cdir, meta)
        return False
    cv2.imwrite(str(cdir / "05_gemini_raw.png"), raw)
    write_composite(cdir, image, raw, mask)
    meta.update(status="done", error=None, gemini_seconds=round(time.time() - t0, 1))
    store.write_meta(cdir, meta)
    return True


def write_composite(cdir, image, raw, mask):
    composite, final, gap = composite_and_repair(image, raw, mask, "inpaint")
    cv2.imwrite(str(cdir / "06_composite_unfilled.png"), composite)
    cv2.imwrite(str(cdir / "06b_gap_mask.png"), gap)
    cv2.imwrite(str(cdir / "07_final_inpaint.png"), final)
    for stale in ("08_final_gemini.png", "09_final_fix.png"):   # built from an older composite
        (cdir / stale).unlink(missing_ok=True)


def cmd_recomposite(store, args):
    """Rebuild 06/07 from the saved Gemini output (no API calls), e.g. after a
    compositing fix."""
    todo = [(d, m) for d, m in store.candidates()
            if m["status"] == "done" and (d / "05_gemini_raw.png").exists()
            and (args.round is None or m["round"] == args.round)]
    print(f"[Recomposite] {len(todo)} candidate(s).")

    def one(d, m):
        image, mask, _, _ = store.load_prep_arrays(m["image"])
        write_composite(d, image, cv2.imread(str(d / "05_gemini_raw.png")), mask)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, d, m) for d, m in todo]), 1):
            fut.result()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


def cmd_run(store, args):
    todo = [(d, m) for d, m in store.candidates()
            if m["status"] in ("planned", "failed")
            and (args.round is None or m["round"] == args.round)]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("[Run] Nothing to run.")
        return
    print(f"[Run] {len(todo)} candidate(s) -> {len(todo)} Gemini call(s), {args.workers} worker(s).")
    if not args.yes and input("Proceed? (y/N): ").strip().lower() not in ("y", "yes"):
        return

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, store, d, m): m for d, m in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            m = futs[fut]
            try:
                good = fut.result()
            except Exception as e:
                good = False
                m.update(status="failed", error=repr(e))
                store.write_meta(store.cand / m["image"] / m["cand_id"], m)
            ok += good
            fail += not good
            print(f"  [{i}/{len(todo)}] {cand_key(m)}: {'done' if good else 'FAILED'}")
    print(f"\n[Run] done {ok}, failed {fail}. Re-run `run` to retry failures.")


# ══════════════════════════════════════════════════════════════════════════════
#  4. REVIEW
# ══════════════════════════════════════════════════════════════════════════════

def describe_gaze(yaw, pitch):
    a = abs(yaw)
    side = "畫面右" if yaw > 0 else "畫面左"
    if a < 20:
        facing = "朝向鏡頭"
    elif a < 75:
        facing = f"偏{side}"
    elif a < 105:
        facing = f"側臉看{side}"
    elif a < 150:
        facing = f"背對、轉向{side}"
    else:
        facing = "完全背對"
    vert = "往上" if pitch > 15 else "往下" if pitch < -15 else "水平"
    return f"{facing}，{vert}"


def label_of(meta):
    """The gaze label a candidate carries: the reviewer's manual label if one
    was saved in review, otherwise the requested target."""
    if meta.get("label"):
        return meta["label"]
    return {"face_px": meta["face_px"], "target_px": meta["target_px"],
            "face_depth": meta["face_depth"], "target_depth": meta["target_depth"],
            "gaze": meta["gaze"], "source": "requested"}


def label_bin(meta):
    """Bin of the effective label; pitch outside the bin range is clamped to
    the nearest pitch bin so manually labelled extremes still count."""
    g = label_of(meta)["gaze"]
    pitch = min(max(g["pitch_deg"], PITCH_EDGES[0]), PITCH_EDGES[-1])
    return bin_key(g["yaw_deg"], pitch)


def cmd_review(store, args):
    """
    Review + relabel. The result pane shows the candidate's gaze label (face
    point -> target dot); if the generated gaze is good but doesn't match the
    requested target, move the label to where the person actually looks
    before accepting:
      left click/drag on the result   = target position
      right click/drag on the result  = face/head position (in the result;
                                        shown only while the button is held)
      mouse wheel or slider           = target distance relative to the face
      y = OK (saves the label)  n = bad  r = reset label to requested
      h = hide/show all marks  s = skip  b = back  q = quit
    No marker is drawn on the face and the arrow starts outside the head, so
    the eyes stay unobstructed.
    The label uses the same geometry as the requested one (gaze_from_target),
    with the face depth from prep; only the 2D points and the target's
    distance ratio are edited.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    reviews = store.reviews()
    items = [(d, m) for d, m in store.candidates()
             if m["status"] == "done"
             and (args.round is None or m["round"] == args.round)
             and (not args.images or m["image"] in set(args.images))
             and (args.all or cand_key(m) not in reviews)]
    if not items:
        print("[Review] Nothing to review.")
        return
    print(f"[Review] {len(items)} candidate(s).")

    root = tk.Tk()
    root.geometry("+0+0")
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    pane_w, pane_h = (scr_w - 300) // 2, scr_h - 280
    top = tk.Frame(root)
    top.pack()
    c_src = tk.Canvas(top, width=pane_w, height=pane_h, bg="black")
    c_out = tk.Canvas(top, width=pane_w, height=pane_h, bg="black", cursor="crosshair")
    side_panel = tk.Frame(top)
    c_src.pack(side=tk.LEFT, padx=2)
    c_out.pack(side=tk.LEFT, padx=2)
    side_panel.pack(side=tk.LEFT, padx=6, anchor=tk.N)
    tk.Label(side_panel, text="Top view (camera at bottom)\nred = label, grey = requested").pack()
    dial = tk.Canvas(side_panel, width=220, height=220, bg="white")
    dial.pack()
    tk.Label(side_panel, text="Pitch").pack(pady=(10, 0))
    pbar = tk.Canvas(side_panel, width=220, height=120, bg="white")
    pbar.pack()
    tk.Label(side_panel, text="目標距離 ÷ 臉的距離 (log2)").pack(pady=(10, 0))
    ratio_scale = tk.Scale(side_panel, from_=-2.5, to=2.5, resolution=0.05,
                           orient=tk.HORIZONTAL, length=220, takefocus=0)
    ratio_scale.pack()
    info = tk.Label(side_panel, justify=tk.LEFT, font=("Arial", 11), wraplength=230)
    info.pack(pady=8)
    tk.Label(root, font=("Arial", 10), text=(
        "Result pane: left click = target, right click (hold) = face/head, wheel = distance   |   "
        "y = OK (save label)   n = bad   r = reset label   h = hide marks   "
        "s = skip   b = back   q = quit")).pack(pady=4)

    pos = [0]
    st = {}          # current item's editable label + display geometry
    view = {"hide": False}

    def fit(bgr):
        h, w = bgr.shape[:2]
        sc = min(pane_w / w, pane_h / h)
        return cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA), sc

    def put(canvas, bgr):
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        canvas._img = img
        canvas.delete("all")
        canvas.create_image(pane_w // 2, pane_h // 2, image=img)

    def ratio_to_depth(log2_ratio):
        return float(np.clip(st["face_depth"] / (2.0 ** log2_ratio), 1.0, 255.0))

    def current_gaze():
        return gaze_from_target(st["face"], st["target"], st["face_depth"], st["depth"],
                                st["shape"])

    def edited():
        req = st["requested"]
        return (list(st["face"]) != list(req["face_px"])
                or list(st["target"]) != list(req["target_px"])
                or abs(st["depth"] - float(req["target_depth"])) > 1e-6)

    def on_slider(*_):
        # The slider is quantised (0.05 in log2), so it only overrides the
        # stored depth once the user actually moves it.
        if st and not st.get("loading"):
            st["depth"] = ratio_to_depth(ratio_scale.get())
        redraw_label()

    def draw_dir(g, color, width):
        yaw, pitch = g["yaw_deg"], g["pitch_deg"]
        dial.create_line(110, 110, 110 + 85 * math.sin(math.radians(yaw)),
                         110 + 85 * math.cos(math.radians(yaw)),
                         arrow=tk.LAST, width=width, fill=color)
        pbar.create_line(20, 60, 20 + 180 * math.cos(math.radians(pitch)),
                         60 - 50 * math.sin(math.radians(pitch)),
                         arrow=tk.LAST, width=width, fill=color)

    def redraw_label(*_):
        if not st:
            return
        sc, ox, oy = st["sc"], st["ox"], st["oy"]
        c_out.delete("label")
        if not view["hide"]:
            tx, ty = st["target"][0] * sc + ox, st["target"][1] * sc + oy
            a0 = arrow_start(st["face"], st["target"], st["gap"])
            if a0:
                c_out.create_line(a0[0] * sc + ox, a0[1] * sc + oy, tx, ty, fill="#ffd400",
                                  width=3, arrow=tk.LAST, tags="label")
            c_out.create_oval(tx - 7, ty - 7, tx + 7, ty + 7, fill="#ff2020", outline="white",
                              tags="label")
            if st.get("show_face"):        # only while the right button is held
                fx, fy = st["face"][0] * sc + ox, st["face"][1] * sc + oy
                c_out.create_line(fx - 12, fy, fx + 12, fy, fill="#00e5ff", width=2, tags="label")
                c_out.create_line(fx, fy - 12, fx, fy + 12, fill="#00e5ff", width=2, tags="label")

        g = current_gaze() or st["requested"]["gaze"]
        dial.delete("all")
        dial.create_oval(20, 20, 200, 200, outline="#999")
        dial.create_rectangle(98, 204, 122, 216, fill="#333")          # camera
        dial.create_text(110, 12, text="away from camera", font=("Arial", 8))
        pbar.delete("all")
        pbar.create_line(20, 60, 200, 60, fill="#999")
        draw_dir(st["requested"]["gaze"], "#bbb", 3)
        draw_dir(g, "#d33", 4)

        lr = math.log2(st["face_depth"] / st["depth"])
        r = 2.0 ** lr
        dist = "與臉同距離" if abs(lr) < 0.08 else \
            (f"比臉遠 {r:.2f} 倍" if r > 1 else f"比臉近（距離 ×{r:.2f}）")
        m = items[pos[0]][1]
        verdict = reviews.get(cand_key(m), {}).get("verdict", "—")
        rq = st["requested"]["gaze"]
        info.config(text=(
            f"[{pos[0] + 1}/{len(items)}]\n{m['image']} / {m['cand_id']}\n"
            f"category: {m['category']}\n\n"
            f"label      yaw {g['yaw_deg']:+.0f}°  pitch {g['pitch_deg']:+.0f}°\n"
            f"{describe_gaze(g['yaw_deg'], g['pitch_deg'])}\n"
            f"requested  yaw {rq['yaw_deg']:+.0f}°  pitch {rq['pitch_deg']:+.0f}°\n\n"
            f"目標{dist}\n(depth {st['depth']:.1f}, face {st['face_depth']:.0f})\n\n"
            f"label: {'已修改' if edited() else '= requested'}\n"
            f"verdict: {verdict}"))

    def load_label(lab):
        st["face"] = list(lab["face_px"])
        st["target"] = list(lab["target_px"])
        st["depth"] = float(lab["target_depth"])
        st["loading"] = True
        ratio_scale.set(math.log2(st["face_depth"] / max(st["depth"], 1.0)))
        root.update_idletasks()
        st["loading"] = False

    def show():
        d, m = items[pos[0]]
        src = cv2.imread(str(store.prep / m["image"] / "image.png"))
        out = cv2.imread(str(d / "07_final_inpaint.png"))
        info_ = store.prep_info(m["image"]) or {}
        gap = max(25.0, 1.6 * float(info_.get("head_r", 40)))
        if not view["hide"]:
            lw = max(2, src.shape[1] // 300)
            a0 = arrow_start(m["face_px"], m["target_px"], gap)
            if a0:
                cv2.arrowedLine(src, a0, tuple(m["target_px"]), (0, 230, 255), lw,
                                tipLength=0.05)
            cv2.circle(src, tuple(m["target_px"]), 4 * lw, (0, 0, 255), -1)
        put(c_src, fit(src)[0])
        disp, sc = fit(out)
        put(c_out, disp)
        st.clear()
        st.update(sc=sc, ox=(pane_w - disp.shape[1]) // 2, oy=(pane_h - disp.shape[0]) // 2,
                  shape=out.shape[:2], face_depth=float(m["face_depth"]), gap=gap,
                  requested=label_of({k: v for k, v in m.items() if k != "label"}))
        load_label(label_of(m))
        redraw_label()
        root.title(f"Review — {cand_key(m)}")

    def to_img(e):
        h, w = st["shape"]
        return [int(np.clip((e.x - st["ox"]) / st["sc"], 0, w - 1)),
                int(np.clip((e.y - st["oy"]) / st["sc"], 0, h - 1))]

    def set_target(e):
        st["target"] = to_img(e)
        redraw_label()

    def set_face(e):
        st["face"] = to_img(e)
        st["show_face"] = True
        redraw_label()

    def release_face(_):
        st["show_face"] = False
        redraw_label()

    def wheel(e):
        up = getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4
        ratio_scale.set(min(2.5, max(-2.5, ratio_scale.get() + (0.05 if up else -0.05))))
        st["depth"] = ratio_to_depth(ratio_scale.get())
        redraw_label()

    def step(delta):
        pos[0] += delta
        if pos[0] >= len(items):
            print("[Review] Reached the end.")
            root.destroy()
            return
        pos[0] = max(0, pos[0])
        show()

    def on_key(e):
        k = e.keysym.lower()
        d, m = items[pos[0]]
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if k == "y":
            g = current_gaze()
            if g is None:
                return
            m["label"] = {
                "face_px": list(st["face"]), "target_px": list(st["target"]),
                "face_depth": st["face_depth"],
                "target_depth": st["depth"],
                "gaze": g, "source": "manual" if edited() else "requested", "time": now,
            }
            store.write_meta(d, m)
            reviews[cand_key(m)] = {"verdict": "ok", "time": now}
            store.save_reviews(reviews)
            step(1)
        elif k == "n":
            reviews[cand_key(m)] = {"verdict": "bad", "time": now}
            store.save_reviews(reviews)
            step(1)
        elif k == "r":
            load_label(st["requested"])
            redraw_label()
        elif k == "h":
            view["hide"] = not view["hide"]
            keep = {"face_px": list(st["face"]), "target_px": list(st["target"]),
                    "target_depth": st["depth"]}
            show()
            load_label(keep)
            redraw_label()
        elif k == "s":
            step(1)
        elif k == "b":
            step(-1)
        elif k == "q":
            root.destroy()

    c_out.bind("<Button-1>", set_target)
    c_out.bind("<B1-Motion>", set_target)
    c_out.bind("<Button-3>", set_face)
    c_out.bind("<B3-Motion>", set_face)
    c_out.bind("<ButtonRelease-3>", release_face)
    c_out.bind("<MouseWheel>", wheel)
    c_out.bind("<Button-4>", wheel)
    c_out.bind("<Button-5>", wheel)
    ratio_scale.config(command=on_slider)
    root.bind("<Key>", on_key)
    show()
    root.mainloop()
    cmd_stats(store, args)


def cmd_label_source(store, args):
    """
    Label the gaze of the person in each SOURCE photo (one per prepped image),
    with the same geometry and controls as `review`:
      left click/drag = where they look, right click (hold) = face/head,
      wheel / slider = target distance relative to the face,
      y = save & next, r = revert to saved, h = hide marks, s = skip, b = back, q = quit.
    Saved to prep/<image>/prep.json as "source_label".
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    names = store.prepped_names()
    if args.images:
        names = [n for n in names if n in set(args.images)]
    if not args.all:
        names = [n for n in names if not store.prep_info(n).get("source_label")]
    if not names:
        print("[Label source] Nothing to label (use --all to revisit).")
        return
    print(f"[Label source] {len(names)} image(s).")

    root = tk.Tk()
    root.geometry("+0+0")
    root.title("Source gaze label")
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    pane_w, pane_h = scr_w - 300, scr_h - 280
    top = tk.Frame(root)
    top.pack()
    canvas = tk.Canvas(top, width=pane_w, height=pane_h, bg="black", cursor="crosshair")
    canvas.pack(side=tk.LEFT, padx=2)
    side_panel = tk.Frame(top)
    side_panel.pack(side=tk.LEFT, padx=6, anchor=tk.N)
    tk.Label(side_panel, text="Top view (camera at bottom)").pack()
    dial = tk.Canvas(side_panel, width=220, height=220, bg="white")
    dial.pack()
    tk.Label(side_panel, text="Pitch").pack(pady=(10, 0))
    pbar = tk.Canvas(side_panel, width=220, height=120, bg="white")
    pbar.pack()
    tk.Label(side_panel, text="目標距離 ÷ 臉的距離 (log2)").pack(pady=(10, 0))
    ratio_scale = tk.Scale(side_panel, from_=-2.5, to=2.5, resolution=0.05,
                           orient=tk.HORIZONTAL, length=220, takefocus=0)
    ratio_scale.pack()
    info = tk.Label(side_panel, justify=tk.LEFT, font=("Arial", 11), wraplength=230)
    info.pack(pady=8)
    tk.Label(root, font=("Arial", 10), text=(
        "left click = where they look, right click (hold) = face/head, wheel = distance   |   "
        "y = save & next   r = revert   h = hide marks   s = skip   b = back   q = quit")).pack(pady=4)

    pos = [0]
    st = {}
    view = {"hide": False}

    def ratio_to_depth(lr):
        return float(np.clip(st["face_depth"] / (2.0 ** lr), 1.0, 255.0))

    def current_gaze():
        if st.get("target") is None:
            return None
        return gaze_from_target(st["face"], st["target"], st["face_depth"], st["depth"],
                                st["shape"])

    def on_slider(*_):
        if st and not st.get("loading"):
            st["depth"] = ratio_to_depth(ratio_scale.get())
        redraw()

    def redraw(*_):
        if not st:
            return
        sc, ox, oy = st["sc"], st["ox"], st["oy"]
        canvas.delete("label")
        if not view["hide"] and st.get("target") is not None:
            tx, ty = st["target"][0] * sc + ox, st["target"][1] * sc + oy
            a0 = arrow_start(st["face"], st["target"], st["gap"])
            if a0:
                canvas.create_line(a0[0] * sc + ox, a0[1] * sc + oy, tx, ty, fill="#ffd400",
                                   width=3, arrow=tk.LAST, tags="label")
            canvas.create_oval(tx - 7, ty - 7, tx + 7, ty + 7, fill="#ff2020", outline="white",
                               tags="label")
        if st.get("show_face"):
            fx, fy = st["face"][0] * sc + ox, st["face"][1] * sc + oy
            canvas.create_line(fx - 12, fy, fx + 12, fy, fill="#00e5ff", width=2, tags="label")
            canvas.create_line(fx, fy - 12, fx, fy + 12, fill="#00e5ff", width=2, tags="label")

        g = current_gaze()
        dial.delete("all")
        dial.create_oval(20, 20, 200, 200, outline="#999")
        dial.create_rectangle(98, 204, 122, 216, fill="#333")
        dial.create_text(110, 12, text="away from camera", font=("Arial", 8))
        pbar.delete("all")
        pbar.create_line(20, 60, 200, 60, fill="#999")
        name = names[pos[0]]
        saved = (store.prep_info(name) or {}).get("source_label")
        if g:
            yaw, pitch = g["yaw_deg"], g["pitch_deg"]
            dial.create_line(110, 110, 110 + 85 * math.sin(math.radians(yaw)),
                             110 + 85 * math.cos(math.radians(yaw)),
                             arrow=tk.LAST, width=4, fill="#d33")
            pbar.create_line(20, 60, 20 + 180 * math.cos(math.radians(pitch)),
                             60 - 50 * math.sin(math.radians(pitch)),
                             arrow=tk.LAST, width=4, fill="#d33")
            lr = math.log2(st["face_depth"] / st["depth"])
            r = 2.0 ** lr
            dist = "與臉同距離" if abs(lr) < 0.08 else \
                (f"比臉遠 {r:.2f} 倍" if r > 1 else f"比臉近（距離 ×{r:.2f}）")
            body = (f"yaw {yaw:+.0f}°  pitch {pitch:+.0f}°\n{describe_gaze(yaw, pitch)}\n\n"
                    f"目標{dist}\n(depth {st['depth']:.1f}, face {st['face_depth']:.0f})")
        else:
            body = "左鍵點出他正在看的位置"
        info.config(text=(f"[{pos[0] + 1}/{len(names)}]\n{name}  ({category_of(name)})\n\n"
                          f"{body}\n\nsaved: {'yes' if saved else 'no'}"))

    def load(lab):
        st["face"] = list(lab["face_px"])
        st["target"] = list(lab["target_px"]) if lab.get("target_px") else None
        st["depth"] = float(lab.get("target_depth", st["face_depth"]))
        st["loading"] = True
        ratio_scale.set(math.log2(st["face_depth"] / max(st["depth"], 1.0)))
        root.update_idletasks()
        st["loading"] = False

    def show():
        name = names[pos[0]]
        info_ = store.prep_info(name)
        img = cv2.imread(str(store.prep / name / "image.png"))
        h, w = img.shape[:2]
        sc = min(pane_w / w, pane_h / h)
        disp = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        tkimg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._img = tkimg
        canvas.delete("all")
        canvas.create_image(pane_w // 2, pane_h // 2, image=tkimg)
        st.clear()
        st.update(sc=sc, ox=(pane_w - disp.shape[1]) // 2, oy=(pane_h - disp.shape[0]) // 2,
                  shape=(h, w), face_depth=float(info_["face_depth"]),
                  gap=max(25.0, 1.6 * float(info_.get("head_r", 40))))
        load(info_.get("source_label") or {"face_px": info_["face_px"]})
        redraw()
        root.title(f"Source gaze label — {name}")

    def to_img(e):
        h, w = st["shape"]
        return [int(np.clip((e.x - st["ox"]) / st["sc"], 0, w - 1)),
                int(np.clip((e.y - st["oy"]) / st["sc"], 0, h - 1))]

    def set_target(e):
        st["target"] = to_img(e)
        redraw()

    def set_face(e):
        st["face"] = to_img(e)
        st["show_face"] = True
        redraw()

    def release_face(_):
        st["show_face"] = False
        redraw()

    def wheel(e):
        up = getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4
        ratio_scale.set(min(2.5, max(-2.5, ratio_scale.get() + (0.05 if up else -0.05))))
        st["depth"] = ratio_to_depth(ratio_scale.get())
        redraw()

    def step(delta):
        pos[0] += delta
        if pos[0] >= len(names):
            print("[Label source] Reached the end.")
            root.destroy()
            return
        pos[0] = max(0, pos[0])
        show()

    def on_key(e):
        k = e.keysym.lower()
        name = names[pos[0]]
        if k == "y":
            g = current_gaze()
            if g is None:
                return
            info_ = store.prep_info(name)
            info_["source_label"] = {
                "face_px": list(st["face"]), "target_px": list(st["target"]),
                "face_depth": st["face_depth"], "target_depth": st["depth"],
                "gaze": g, "source": "manual", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (store.prep / name / "prep.json").write_text(json.dumps(info_, indent=2),
                                                         encoding="utf-8")
            step(1)
        elif k == "r":
            info_ = store.prep_info(name)
            load(info_.get("source_label") or {"face_px": info_["face_px"]})
            redraw()
        elif k == "h":
            view["hide"] = not view["hide"]
            redraw()
        elif k == "s":
            step(1)
        elif k == "b":
            step(-1)
        elif k == "q":
            root.destroy()

    canvas.bind("<Button-1>", set_target)
    canvas.bind("<B1-Motion>", set_target)
    canvas.bind("<Button-3>", set_face)
    canvas.bind("<B3-Motion>", set_face)
    canvas.bind("<ButtonRelease-3>", release_face)
    canvas.bind("<MouseWheel>", wheel)
    canvas.bind("<Button-4>", wheel)
    canvas.bind("<Button-5>", wheel)
    ratio_scale.config(command=on_slider)
    root.bind("<Key>", on_key)
    show()
    root.mainloop()
    done = sum(1 for n in store.prepped_names() if store.prep_info(n).get("source_label"))
    print(f"[Label source] {done}/{len(store.prepped_names())} source images labelled.")


def accepted_candidates(store):
    reviews = store.reviews()
    return [(d, m) for d, m in store.candidates()
            if reviews.get(cand_key(m), {}).get("verdict") == "ok"]


def gt_image(cdir):
    """
    The best available target image: the targeted fix (`fix`) if kept, else
    the Gemini gap repair if done, else the screening inpaint.
    """
    f = cdir / "09_final_fix.png"
    if f.exists():
        return f, "gemini"
    g = cdir / "08_final_gemini.png"
    return (g, "gemini") if g.exists() else (cdir / "07_final_inpaint.png", "inpaint")


def load_repair_selection(store):
    p = store.root / "repair_selection.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_repair_selection(store, sel):
    p = store.root / "repair_selection.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def cmd_repair_select(store, args):
    """
    Inspect every accepted view's gap fill and mark the ones that need a
    Gemini repair. Panes: Gemini output | composite after inpaint (gap
    outlined) | close-up of the gap area.
      x = needs Gemini fix   space / Enter = inpaint is fine   b = back
      o = toggle gap outline   z = close-up of the next gap region   q = quit
    Saved to <data>/repair_selection.json; then `repair --selected`.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    sel = load_repair_selection(store)
    items = accepted_candidates(store)
    if args.images:
        items = [(d, m) for d, m in items if m["image"] in set(args.images)]
    if not args.all:
        items = [(d, m) for d, m in items if cand_key(m) not in sel]
    if not items:
        print("[Repair select] Nothing to inspect (use --all to revisit).")
        return
    print(f"[Repair select] {len(items)} view(s).")

    root = tk.Tk()
    root.geometry("+0+0")
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    pane_w, pane_h = (scr_w - 30) // 3, scr_h - 200
    top = tk.Frame(root)
    top.pack()
    panes = []
    for title in ("Gemini output", "after inpaint (gap outlined)", "gap close-up"):
        col = tk.Frame(top)
        col.pack(side=tk.LEFT, padx=2)
        tk.Label(col, text=title, font=("Arial", 10, "bold")).pack()
        c = tk.Canvas(col, width=pane_w, height=pane_h, bg="black")
        c.pack()
        panes.append(c)
    status = tk.Label(root, font=("Arial", 12, "bold"))
    status.pack(pady=(4, 0))
    tk.Label(root, font=("Arial", 10), text=(
        "x = needs Gemini fix    space / Enter = inpaint OK    b = back    "
        "o = toggle outline    z = next gap close-up    q = quit")).pack(pady=(0, 4))

    pos = [0]
    view = {"outline": True, "zoom": 0}

    def put(canvas, bgr):
        h, w = bgr.shape[:2]
        sc = min(pane_w / w, pane_h / h)
        disp = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                          interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._img = img
        canvas.delete("all")
        canvas.create_image(pane_w // 2, pane_h // 2, image=img)

    def show():
        d, m = items[pos[0]]
        raw = cv2.imread(str(d / "05_gemini_raw.png"))
        fin = cv2.imread(str(d / "07_final_inpaint.png"))
        gap = cv2.imread(str(d / "06b_gap_mask.png"), cv2.IMREAD_GRAYSCALE)
        h, w = fin.shape[:2]
        shown = fin.copy()
        if view["outline"] and gap is not None and cv2.countNonZero(gap):
            cnts, _ = cv2.findContours(gap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(shown, cnts, -1, (255, 0, 255), max(1, w // 600))
        put(panes[0], raw)
        put(panes[1], shown)
        zinfo = ""
        if gap is not None and cv2.countNonZero(gap):
            # Close-up of one gap region at a time, largest first (z cycles):
            # the gap is usually a thin rim around the whole person plus one
            # or two real holes the pose change opened, which is where
            # inpainting fails.
            n, _, stats, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
            order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])
            i = int(order[view["zoom"] % len(order)])
            x, y, bw, bh = stats[i, :4]
            pad = max(40, int(0.6 * max(bw, bh)), int(0.04 * max(h, w)))
            x0, x1 = max(0, x - pad), min(w, x + bw + pad)
            y0, y1 = max(0, y - pad), min(h, y + bh + pad)
            put(panes[2], shown[y0:y1, x0:x1])
            pct = cv2.countNonZero(gap) / gap.size * 100
            zinfo = f"   close-up {view['zoom'] % len(order) + 1}/{len(order)} (z = next)"
        else:
            panes[2].delete("all")
            pct = 0.0
        marked = sum(v == "fix" for v in sel.values())
        cur = sel.get(cand_key(m), "—")
        status.config(text=(f"[{pos[0] + 1}/{len(items)}]  {cand_key(m)}   gap {pct:.2f}%   "
                            f"current: {cur}   marked for fix: {marked}{zinfo}"),
                      fg="red" if cur == "fix" else "black")
        root.title(f"Repair select — {cand_key(m)}")

    def step(delta):
        pos[0] += delta
        view["zoom"] = 0
        if pos[0] >= len(items):
            print("[Repair select] Reached the end.")
            root.destroy()
            return
        pos[0] = max(0, pos[0])
        show()

    def on_key(e):
        k = e.keysym.lower()
        m = items[pos[0]][1]
        if k == "x":
            sel[cand_key(m)] = "fix"
            save_repair_selection(store, sel)
            step(1)
        elif k in ("space", "return"):
            sel[cand_key(m)] = "ok"
            save_repair_selection(store, sel)
            step(1)
        elif k == "b":
            step(-1)
        elif k == "o":
            view["outline"] = not view["outline"]
            show()
        elif k == "z":
            view["zoom"] += 1
            show()
        elif k == "q":
            root.destroy()

    root.bind("<Key>", on_key)
    show()
    root.lift()
    root.attributes("-topmost", True)                 # open in front of the editor
    root.after(400, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()
    n_fix = sum(v == "fix" for v in sel.values())
    print(f"[Repair select] {len(sel)} inspected, {n_fix} marked for Gemini fix "
          f"-> `python eval_gt.py repair --selected` ({n_fix} Gemini call(s)).")


def split_gap(composite, gap):
    """
    Split the gap into a thin rim (<= ~3 px wide, mostly the silhouette edge)
    and real holes. The rim is filled locally with TELEA, which handles thin
    lines well, and only the holes are left black for Gemini: at its 1K output
    resolution Gemini tends to keep a 1-2 px black line as an outline, which
    came back as a dark seam. Returns (prefilled_composite, hole_mask).
    """
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    holes = cv2.morphologyEx(gap, cv2.MORPH_OPEN, ell(7))
    holes = cv2.bitwise_and(cv2.dilate(holes, ell(5)), gap)   # keep holes' own rim pixels
    rim = cv2.bitwise_and(gap, cv2.bitwise_not(holes))
    pre = composite.copy()
    if cv2.countNonZero(rim):
        pre = cv2.inpaint(pre, cv2.dilate(rim, ell(3)), 3, cv2.INPAINT_TELEA)
        pre[holes > 0] = 0
    return pre, holes


def blend_repair(base, response, holes, person, old_mask, grow=7, band=8):
    """
    Seamless merge of Gemini's repair into the pre-filled composite.

    Taking only the hole pixels left dark seams. The cause was mostly outside
    the holes: the old person's anti-aliased fringe (hair wisps the SAM mask
    missed) stays in the original background around the hole, while Gemini
    repaints it clean. So:
      1. take Gemini's pixels in the holes *and* the old person's fringe
         around them (grown by `grow` px, never onto the new person);
      2. correct Gemini's slight tone shift (~1-2 levels) with a smooth
         per-pixel offset estimated from a ring of original pixels around it;
      3. add back grain where Gemini is smoother than the photo;
      4. fade into the original over `band` px, and give the pasted person an
         anti-aliased edge (blended on the inside of its silhouette only, so
         the black holes next to it never leak in).
    Pixels away from the holes stay the original's, unlike taking Gemini's
    whole image, which also changes the surrounding scene.
    """
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    near_old = cv2.dilate(old_mask, ell(2 * grow + 9))
    grown = (cv2.dilate(holes, ell(2 * grow + 1)) & near_old
             & cv2.bitwise_not(cv2.dilate(person, ell(5))))
    region = cv2.bitwise_or(holes, grown)

    ring = ((cv2.dilate(region, ell(41)) > 0) & (region == 0)
            & (cv2.dilate(person, ell(5)) == 0))
    w = ring.astype(np.float32)
    num = cv2.GaussianBlur((base.astype(np.float32) - response.astype(np.float32)) * w[..., None],
                           (0, 0), 25)
    den = cv2.GaussianBlur(w, (0, 0), 25)[..., None]
    resp = response.astype(np.float32) + np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)

    if ring.any():
        hf = lambda im: (lambda g: (g - cv2.GaussianBlur(g, (0, 0), 1.2))[ring].std())(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
        add = float(np.sqrt(max(0.0, hf(base) ** 2 - hf(response) ** 2)))
        if add > 0:
            noise = np.random.default_rng(0).normal(0, add, base.shape[:2]).astype(np.float32)
            resp += noise[..., None]

    dist = cv2.distanceTransform((region == 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1 - dist / band, 0, 1)
    alpha[holes > 0] = 1
    keep = np.clip(cv2.GaussianBlur((person > 0).astype(np.float32), (0, 0), 1.0) * 2 - 1, 0, 1)
    alpha = (alpha * (1 - keep))[..., None]
    out = alpha * resp + (1 - alpha) * base.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def merge_repair(base, response, holes, mode, person=None, old_mask=None):
    """
    Combine Gemini's repair response with the pre-filled composite.
      feather: blend_repair — holes + old-person fringe, tone/grain matched,
               feathered edges; the rest of the scene stays original (default)
      full:    Gemini's whole image (no seams, but the scene around can change,
               the frame becomes its 1K output upscaled, and it may touch the person)
      gap:     Gemini pixels in the holes only, hard edge (first version)
    """
    if mode == "full":
        return response.copy()
    if mode == "gap":
        out = base.copy()
        m = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out[m > 0] = response[m > 0]
        return out
    return blend_repair(base, response, holes, person, old_mask)


def repair_context(d):
    """(person mask of this view, original person mask) for blend_repair."""
    name = d.parent.name
    prep = d.parent.parent.parent / "prep" / name
    old_mask = cv2.imread(str(prep / "mask.png"), cv2.IMREAD_GRAYSCALE)
    cached = d / "08d_person.png"
    if cached.exists():
        return cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE), old_mask
    image = cv2.imread(str(prep / "image.png"))
    raw = cv2.imread(str(d / "05_gemini_raw.png"))
    _, _, person = composite_onto_background(image, raw, old_mask, return_fg=True)
    cv2.imwrite(str(cached), person)
    return person, old_mask


REPAIR_FILES = ("08_final_gemini.png", "08a_sent.png", "08b_gemini_response.png",
                "08c_holes.png", "08d_person.png", "08_repair.json")


def write_repair_result(d, mode, chosen=False):
    """
    (Re)build 08_final_gemini.png from the saved send/response files.
    chosen=True marks a merge picked by hand in `merge-select`; a plain
    `remerge` leaves those alone.
    """
    base = cv2.imread(str(d / "08a_sent.png"))
    holes = cv2.imread(str(d / "08c_holes.png"), cv2.IMREAD_GRAYSCALE)
    rp = d / "08b_gemini_response.png"
    resp = cv2.imread(str(rp)) if rp.exists() else None
    if resp is None:                                   # rim only: no Gemini call was needed
        out = base
    else:
        person, old_mask = repair_context(d) if mode == "feather" else (None, None)
        out = merge_repair(base, resp, holes, mode, person, old_mask)
    cv2.imwrite(str(d / "08_final_gemini.png"), out)
    (d / "08_repair.json").write_text(json.dumps(
        {"merge": mode if resp is not None else "rim-only (TELEA)",
         "gemini_called": resp is not None,
         "chosen": bool(chosen and resp is not None),
         "hole_pct": round(cv2.countNonZero(holes) / holes.size * 100, 3)}, indent=2),
        encoding="utf-8")


def repair_info(d):
    p = d / "08_repair.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def cmd_repair(store, args):
    """
    Gemini gap repair for accepted views. The thin rim of the gap is filled
    locally first (split_gap); only real holes go to Gemini, so views whose gap
    is only a rim cost no call. Gemini's full response is saved
    (08b_gemini_response.png) and merged per --merge; `remerge` changes the
    merge later without calling Gemini again.
    """
    accepted = accepted_candidates(store)
    if args.selected:
        marked = {k for k, v in load_repair_selection(store).items() if v == "fix"}
        accepted = [(d, m) for d, m in accepted if cand_key(m) in marked]
    if args.ids:
        want = set(args.ids)
        accepted = [(d, m) for d, m in accepted if cand_key(m) in want
                    or f"{m['image']}_{m['cand_id']}" in want]
    todo, rim_only, small = [], [], 0
    for d, m in accepted:
        done = (d / "08b_gemini_response.png").exists() or (
            (d / "08_repair.json").exists() and not (d / "08b_gemini_response.png").exists()
            and (d / "08_final_gemini.png").exists())
        if done and not (args.ids or args.redo):
            continue
        gap = cv2.imread(str(d / "06b_gap_mask.png"), cv2.IMREAD_GRAYSCALE)
        if gap is None:
            print(f"  {cand_key(m)}: no 06b_gap_mask.png — run `recomposite` first")
            continue
        if cv2.countNonZero(gap) / gap.size * 100 < args.min_gap_pct:
            small += 1
            continue
        composite = cv2.imread(str(d / "06_composite_unfilled.png"))
        pre, holes = split_gap(composite, gap)
        (rim_only if not cv2.countNonZero(holes) else todo).append((d, m, pre, holes))
    if args.limit:
        todo = todo[:args.limit]

    # Rim-only views: no API call needed.
    for d, m, pre, holes in rim_only:
        for f in REPAIR_FILES:
            (d / f).unlink(missing_ok=True)
        cv2.imwrite(str(d / "08a_sent.png"), pre)
        cv2.imwrite(str(d / "08c_holes.png"), holes)
        write_repair_result(d, args.merge)
    print(f"[Repair] {len(rim_only)} view(s) had only a thin rim -> filled locally, no Gemini call.")
    print(f"[Repair] {len(todo)} view(s) with real holes -> {len(todo)} Gemini call(s)"
          + (f"; {small} below --min-gap-pct {args.min_gap_pct} keep inpaint" if small else ""))
    if not todo:
        return
    if not args.yes and input("Proceed? (y/N): ").strip().lower() not in ("y", "yes"):
        return

    def one(d, pre, holes):
        rep = repair_gemini(pre)
        if rep is None:
            return False
        for f in REPAIR_FILES:
            (d / f).unlink(missing_ok=True)
        cv2.imwrite(str(d / "08a_sent.png"), pre)
        cv2.imwrite(str(d / "08b_gemini_response.png"), rep)
        cv2.imwrite(str(d / "08c_holes.png"), holes)
        write_repair_result(d, args.merge)
        return True

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, d, pre, holes): m for d, m, pre, holes in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            m = futs[fut]
            try:
                good = fut.result()
            except Exception as e:
                print(f"  {cand_key(m)}: {e!r}")
                good = False
            ok += good
            fail += not good
            print(f"  [{i}/{len(todo)}] {cand_key(m)}: {'repaired' if good else 'FAILED'}")
    print(f"[Repair] repaired {ok}, failed {fail}. Re-run to retry failures; "
          "`repair-view` shows what Gemini returned.")


def cmd_remerge(store, args):
    """
    Re-merge saved Gemini responses with another --merge mode (no API calls).
    Views picked by hand in `merge-select` are skipped unless named in --ids.
    """
    n = kept = 0
    for d, m in accepted_candidates(store):
        if args.ids and cand_key(m) not in args.ids and f"{m['image']}_{m['cand_id']}" not in args.ids:
            continue
        if not (d / "08a_sent.png").exists():
            continue
        if not args.ids and repair_info(d).get("chosen"):
            kept += 1
            continue
        write_repair_result(d, args.merge, chosen=bool(args.ids))
        n += 1
    print(f"[Remerge] {n} view(s) merged with mode '{args.merge}'"
          + (f"; {kept} hand-picked in merge-select kept" if kept else "") + ".")


def cmd_unrepair(store, args):
    """Drop the Gemini repair for the given samples (they fall back to inpaint)."""
    want = set(args.ids)
    n = 0
    for d, m in store.candidates():
        if cand_key(m) in want or f"{m['image']}_{m['cand_id']}" in want:
            if (d / "08_final_gemini.png").exists():
                n += 1
            for f in REPAIR_FILES:
                (d / f).unlink(missing_ok=True)
    print(f"[Unrepair] removed {n} Gemini repair(s).")


def cmd_repair_view(store, args):
    """
    Look at the repairs: what was sent (holes black) | Gemini's full response |
    the merged result used in the dataset.
      space / Enter / n = next   b = back   z = close-up of the next hole
      (whole image first)   q = quit
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    items = [(d, m) for d, m in accepted_candidates(store) if (d / "08a_sent.png").exists()]
    if args.ids:
        items = [(d, m) for d, m in items if cand_key(m) in args.ids
                 or f"{m['image']}_{m['cand_id']}" in args.ids]
    if not items:
        print("[Repair view] No repaired views yet (run `repair`).")
        return
    root = tk.Tk()
    root.geometry("+0+0")
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    pane_w, pane_h = (scr_w - 30) // 3, scr_h - 200
    top = tk.Frame(root)
    top.pack()
    panes = []
    for title in ("sent to Gemini (holes black)", "Gemini response (full image)",
                  "result used in the dataset"):
        col = tk.Frame(top)
        col.pack(side=tk.LEFT, padx=2)
        tk.Label(col, text=title, font=("Arial", 10, "bold")).pack()
        c = tk.Canvas(col, width=pane_w, height=pane_h, bg="black")
        c.pack()
        panes.append(c)
    status = tk.Label(root, font=("Arial", 12, "bold"))
    status.pack(pady=(4, 0))
    tk.Label(root, font=("Arial", 10), text=(
        "space / Enter = next    b = back    z = close-up of next hole (cycles back to "
        "whole image)    q = quit")).pack(pady=(0, 4))
    pos, zoom = [0], [0]

    def put(canvas, bgr):
        h, w = bgr.shape[:2]
        sc = min(pane_w / w, pane_h / h)
        disp = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                          interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._img = img
        canvas.delete("all")
        canvas.create_image(pane_w // 2, pane_h // 2, image=img)

    def show():
        d, m = items[pos[0]]
        sent = cv2.imread(str(d / "08a_sent.png"))
        resp = cv2.imread(str(d / "08b_gemini_response.png"))
        fin = cv2.imread(str(d / "08_final_gemini.png"))
        holes = cv2.imread(str(d / "08c_holes.png"), cv2.IMREAD_GRAYSCALE)
        if resp is None:
            resp = np.zeros_like(sent)
            cv2.putText(resp, "no Gemini call (rim only)", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        h, w = sent.shape[:2]
        crop = (slice(0, h), slice(0, w))
        n, _, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        label = "whole image"
        if zoom[0] % n and n > 1:
            order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])
            i = int(order[(zoom[0] % n) - 1])
            x, y, bw, bh = stats[i, :4]
            pad = max(30, int(0.5 * max(bw, bh)))
            crop = (slice(max(0, y - pad), min(h, y + bh + pad)),
                    slice(max(0, x - pad), min(w, x + bw + pad)))
            label = f"hole {zoom[0] % n}/{n - 1}"
        for c, im in zip(panes, (sent, resp, fin)):
            put(c, im[crop])
        info = json.loads((d / "08_repair.json").read_text(encoding="utf-8"))
        status.config(text=f"[{pos[0] + 1}/{len(items)}]  {cand_key(m)}   merge: {info['merge']}   "
                           f"holes {info['hole_pct']}%   view: {label}")
        root.title(f"Repair view — {cand_key(m)}")

    def on_key(e):
        k = e.keysym.lower()
        if k in ("space", "return", "n"):
            if pos[0] + 1 < len(items):
                pos[0] += 1
                zoom[0] = 0
                show()
        elif k == "b":
            pos[0] = max(0, pos[0] - 1)
            zoom[0] = 0
            show()
        elif k == "z":
            zoom[0] += 1
            show()
        elif k == "q":
            root.destroy()

    root.bind("<Key>", on_key)
    show()
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()


def cmd_merge_select(store, args):
    """
    Pick, per Gemini-repaired view, which result goes into the dataset:
    Gemini's whole image (full) or the feather merge. The choice is written at
    once (08_final_gemini.png + "chosen" in 08_repair.json); no API calls.
      1 / g = Gemini full    2 / f = feather    (both save and go to the next)
      space / n = next without changing    b = back
      z = close-up of the next hole    t = flip both panes to the sent image
      q = quit
    Starts at the first view not picked yet (--all: from the start).
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    items = [(d, m) for d, m in accepted_candidates(store)
             if (d / "08b_gemini_response.png").exists()]
    if args.ids:
        items = [(d, m) for d, m in items if cand_key(m) in args.ids
                 or f"{m['image']}_{m['cand_id']}" in args.ids]
    if not items:
        print("[Merge select] No views with a saved Gemini response (run `repair`).")
        return
    root = tk.Tk()
    root.geometry("+0+0")
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    pane_w, pane_h = (scr_w - 40) // 2, scr_h - 210
    top = tk.Frame(root)
    top.pack()
    panes, frames, titles = [], [], []
    for title in ("[1 / g]  Gemini full image", "[2 / f]  feather merge"):
        col = tk.Frame(top, bd=0, highlightthickness=5, highlightbackground="#444")
        col.pack(side=tk.LEFT, padx=4)
        lab = tk.Label(col, text=title, font=("Arial", 12, "bold"))
        lab.pack()
        c = tk.Canvas(col, width=pane_w, height=pane_h, bg="black", highlightthickness=0)
        c.pack()
        panes.append(c)
        frames.append(col)
        titles.append(lab)
    status = tk.Label(root, font=("Arial", 12, "bold"))
    status.pack(pady=(4, 0))
    tk.Label(root, font=("Arial", 10), text=(
        "1 / g = use Gemini full    2 / f = use feather    space / n = next    b = back    "
        "z = close-up of next hole    t = show sent image    q = quit")).pack(pady=(0, 4))

    def picked(d):
        info = repair_info(d)
        return info.get("merge") if info.get("chosen") else None

    start = 0
    if not args.all:
        start = next((i for i, (d, _) in enumerate(items) if not picked(d)), len(items) - 1)
    pos, zoom, peek = [start], [0], [False]
    cache = {}

    def images(d):
        if d not in cache:
            sent = cv2.imread(str(d / "08a_sent.png"))
            resp = cv2.imread(str(d / "08b_gemini_response.png"))
            holes = cv2.imread(str(d / "08c_holes.png"), cv2.IMREAD_GRAYSCALE)
            person, old_mask = repair_context(d)
            cache.clear()
            cache[d] = (sent, merge_repair(sent, resp, holes, "full"),
                        merge_repair(sent, resp, holes, "feather", person, old_mask), holes)
        return cache[d]

    def put(canvas, bgr):
        h, w = bgr.shape[:2]
        sc = min(pane_w / w, pane_h / h)
        disp = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                          interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._img = img
        canvas.delete("all")
        canvas.create_image(pane_w // 2, pane_h // 2, image=img)

    def show():
        d, m = items[pos[0]]
        sent, full, feather, holes = images(d)
        h, w = sent.shape[:2]
        crop = (slice(0, h), slice(0, w))
        n, _, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        label = "whole image"
        if zoom[0] % n and n > 1:
            order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])
            i = int(order[(zoom[0] % n) - 1])
            x, y, bw, bh = stats[i, :4]
            pad = max(40, int(0.6 * max(bw, bh)))
            crop = (slice(max(0, y - pad), min(h, y + bh + pad)),
                    slice(max(0, x - pad), min(w, x + bw + pad)))
            label = f"hole {zoom[0] % n}/{n - 1}"
        for c, im in zip(panes, (full, feather)):
            put(c, (sent if peek[0] else im)[crop])
        cur = picked(d)
        for f, t, mode in zip(frames, titles, ("full", "feather")):
            on = cur == mode
            f.config(highlightbackground="#2e9e4f" if on else "#444")
            t.config(fg="#2e9e4f" if on else "black")
        n_done = sum(bool(picked(dd)) for dd, _ in items)
        status.config(text=(
            f"[{pos[0] + 1}/{len(items)}]  {cand_key(m)}   picked: {cur or '— (now ' + repair_info(d).get('merge', '?') + ')'}"
            f"   holes {repair_info(d).get('hole_pct', '?')}%   view: {label}"
            f"{'   [showing SENT image]' if peek[0] else ''}   ({n_done}/{len(items)} picked)"))
        root.title(f"Merge select — {cand_key(m)}")

    def go(step):
        pos[0] = min(max(0, pos[0] + step), len(items) - 1)
        zoom[0] = 0
        peek[0] = False
        show()

    def choose(mode):
        d, m = items[pos[0]]
        write_repair_result(d, mode, chosen=True)
        if pos[0] + 1 < len(items):
            go(1)
        else:
            show()
            print(f"[Merge select] last view done.")

    def on_key(e):
        k = e.keysym.lower()
        if k in ("1", "g", "kp_1"):
            choose("full")
        elif k in ("2", "f", "kp_2"):
            choose("feather")
        elif k in ("space", "return", "n"):
            go(1)
        elif k == "b":
            go(-1)
        elif k == "z":
            zoom[0] += 1
            show()
        elif k == "t":
            peek[0] = not peek[0]
            show()
        elif k == "q":
            root.destroy()

    root.bind("<Key>", on_key)
    show()
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()
    picks = [picked(d) for d, _ in items]
    print(f"[Merge select] picked {sum(p == 'full' for p in picks)} full, "
          f"{sum(p == 'feather' for p in picks)} feather, "
          f"{sum(p is None for p in picks)} not picked (keep their current merge). "
          "Run `finalize` to update the dataset.")


# ── Targeted fix (3rd pass) ────────────────────────────────────────────────
#  issue-tag: mark what is wrong with a view (tags + painted region + note)
#  fix:       one Gemini call per tagged view, prompt built from the tags,
#             response blended back inside the painted region only
#  fix-view:  before | Gemini response | result, pick region / full / revert

ISSUE_TAGS = [            # (key, tag, label in the UI)
    ("1", "implausible", "不合理"),
    ("2", "artifact", "artifact"),
    ("3", "seam", "接縫沒補好"),
]
ISSUE_PROMPTS = {
    "implausible": (
        "Something in the marked area is physically implausible: for example distorted or "
        "malformed anatomy (hands, fingers, arms, legs, ears, neck), extra or missing body "
        "parts, objects that merge into each other or into the person, or broken perspective. "
        "Correct it so it looks like a real, coherent photograph."),
    "artifact": (
        "The marked area contains image-generation artifacts: smudges, blur, noise, colour "
        "blotches, melted or repeated textures, or garbled details. Replace them with clean, "
        "realistic detail that matches the rest of the photo."),
    "seam": (
        "The marked area shows seams where an edited person was pasted onto the background: "
        "visible outlines, halos or dark/light fringes along the silhouette, leftover traces "
        "of the previous pose, or background patches whose colour, brightness or texture do "
        "not match their surroundings. Blend them so the transition is invisible and the "
        "background is continuous."),
}
FIX_FILES = ("09_final_fix.png", "09a_input.png", "09b_marked.png", "09c_region.png",
             "09d_response.png", "09_fix.json")


def load_issues(store):
    p = store.root / "issues.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_issues(store, issues):
    p = store.root / "issues.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(issues, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def current_image(d):
    """The image of this view that is in the dataset before a fix."""
    g = d / "08_final_gemini.png"
    return g if g.exists() else d / "07_final_inpaint.png"


def build_fix_prompt(tags, note, has_region, ref_original=False):
    """
    has_region: a 2nd image with the area outlined in red is sent as a guide.
    ref_original: the last image is the original photo, as a reference for
    objects lost in the edit (e.g. what the person was holding).
    """
    where = "the area outlined in red in the second image" if has_region else "the image"
    parts = [
        "The first image is a photograph that needs a local fix. "
        + ("The second image is the same photograph with the area to fix outlined in red; "
           "it is only a guide, and the red outline must NOT appear in your output. "
           if has_region else "")
        + ("The last image is the original photo before the person was edited. Use it ONLY as "
           "a reference for objects and details that went missing (for example what the person "
           "holds); do NOT copy the person's head pose or gaze from it. "
           if ref_original else ""),
        f"Edit {where}:",
    ]
    parts += [f"- {ISSUE_PROMPTS[t]}" for t in tags if t in ISSUE_PROMPTS]
    if note:
        parts.append(f"- Specifically: {note}")
    parts.append(
        "Keep the person's identity, face, hairstyle, clothing, body pose, head orientation and "
        "gaze direction (where the eyes look) exactly as they are. "
        + ("Change nothing outside the marked area. " if has_region else "")
        + "Keep the framing, composition, lighting and colours of the photograph identical, "
          "and return the whole photograph at the same aspect ratio.")
    return "\n".join(parts)


def guide_leaked(base, resp, min_area=300):
    """True if Gemini copied the red guide outline into its output: a new blob
    of pure red (natural red objects are orange-ish or already in the input)."""
    def pure(im):
        b, g, r = [im[..., i].astype(int) for i in range(3)]
        return (r >= 160) & (g <= 80) & (b <= 80) & (r - np.maximum(g, b) >= 120)
    old = cv2.dilate(pure(base).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    new = (pure(resp) & ~old).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(new, connectivity=8)
    return n > 1 and int(st[1:, cv2.CC_STAT_AREA].max()) >= min_area


def mark_region(image, region):
    """Copy of the image with the region's outline drawn in red (guide for Gemini)."""
    out = image.copy()
    cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t = max(3, int(round(max(image.shape[:2]) / 400)))
    cv2.drawContours(out, cs, -1, (0, 0, 255), t)
    return out


def gemini_fix(images_bgr, prompt, max_retries=3):
    """Gemini edit with one or more input images; response resized to the first image."""
    from google import genai
    from google.genai import types
    import io
    from PIL import Image
    import dataset_pipeline as dp

    client = genai.Client(api_key=dp.GEMINI_API_KEY)
    h, w = images_bgr[0].shape[:2]
    parts = [types.Part.from_text(text=prompt)]
    for im in images_bgr:
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=buf.getvalue())))
    cfg = types.GenerateContentConfig(image_config=types.ImageConfig(image_size="1K"),
                                      response_modalities=["IMAGE", "TEXT"])
    for attempt in range(1, max_retries + 1):
        try:
            for chunk in client.models.generate_content_stream(
                    model=dp.GEMINI_MODEL, contents=[types.Content(role="user", parts=parts)],
                    config=cfg):
                for part in chunk.parts or []:
                    if part.inline_data and part.inline_data.data:
                        pil = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                        return cv2.cvtColor(np.array(pil.resize((w, h), Image.LANCZOS)),
                                            cv2.COLOR_RGB2BGR)
            return None
        except Exception as e:
            if attempt < max_retries:
                time.sleep(20)
            else:
                print(f"  [Gemini Fix Error] {e}")
    return None


def align_to(base, resp, max_shift=25):
    """Undo a small global shift of Gemini's output (phase correlation)."""
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (sx, sy), resp_q = cv2.phaseCorrelate(g(base), g(resp))
    if resp_q < 0.05 or abs(sx) > max_shift or abs(sy) > max_shift or abs(sx) + abs(sy) < 0.5:
        return resp
    M = np.float32([[1, 0, -sx], [0, 1, -sy]])
    return cv2.warpAffine(resp, M, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT)


def blend_region(base, resp, region, band=12):
    """
    Gemini's fix inside the painted region only: aligned, tone-matched from a
    ring of original pixels around it, grain added back, feathered over `band` px.
    """
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    resp = align_to(base, resp)
    region = cv2.dilate(region, ell(5))
    ring = (cv2.dilate(region, ell(2 * band + 31)) > 0) & (region == 0)
    w = ring.astype(np.float32)
    num = cv2.GaussianBlur((base.astype(np.float32) - resp.astype(np.float32)) * w[..., None],
                           (0, 0), 25)
    den = cv2.GaussianBlur(w, (0, 0), 25)[..., None]
    out_r = resp.astype(np.float32) + np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    if ring.any():
        hf = lambda im: (lambda gg: (gg - cv2.GaussianBlur(gg, (0, 0), 1.2))[ring].std())(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
        add = float(np.sqrt(max(0.0, hf(base) ** 2 - hf(resp) ** 2)))
        if add > 0:
            out_r += np.random.default_rng(0).normal(0, add, base.shape[:2]).astype(np.float32)[..., None]
    dist = cv2.distanceTransform((region == 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1 - dist / band, 0, 1)[..., None]
    out = alpha * out_r + (1 - alpha) * base.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def write_fix_result(d, mode):
    """(Re)build 09_final_fix.png: mode region | full | revert."""
    info_p = d / "09_fix.json"
    info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else {}
    if mode == "revert":
        (d / "09_final_fix.png").unlink(missing_ok=True)
    else:
        base = cv2.imread(str(d / "09a_input.png"))
        resp = cv2.imread(str(d / "09d_response.png"))
        region = cv2.imread(str(d / "09c_region.png"), cv2.IMREAD_GRAYSCALE)
        if mode == "region" and region is not None and cv2.countNonZero(region):
            out = blend_region(base, resp, region)
        else:
            mode = "full"
            out = resp
        cv2.imwrite(str(d / "09_final_fix.png"), out)
    info["merge"] = mode
    info_p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _select_ids(items, ids):
    if not ids:
        return items
    want = set(ids)
    return [(d, m) for d, m in items if cand_key(m) in want or f"{m['image']}_{m['cand_id']}" in want]


def cmd_issue_tag(store, args):
    """
    Mark what is wrong with each view, for `fix`.
      1 = 不合理 (implausible)   2 = artifact   3 = 接縫 (seam)   (toggle)
      left drag = paint the problem area    right drag = erase
      [ / ] = brush size    wheel = zoom at cursor    middle drag = pan
      t = show: current image / Gemini response / sent to Gemini
      c = clear painting    x = no issue (remove the entry)
      click the note box to type a note (Enter returns)
      space = save + next    b = save + back    q = save + quit
    Views: --ids, --unpicked (Gemini-repaired views not picked in
    merge-select), plus every view already in issues.json.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    issues = load_issues(store)
    acc = accepted_candidates(store)
    keys = set(issues)
    if args.ids:
        keys |= {cand_key(m) for d, m in _select_ids(acc, args.ids)}
    if args.unpicked:
        keys |= {cand_key(m) for d, m in acc if (d / "08b_gemini_response.png").exists()
                 and not repair_info(d).get("chosen")}
    items = [(d, m) for d, m in acc if cand_key(m) in keys]
    if not items:
        print("[Issue tag] No views (use --ids or --unpicked).")
        return
    print(f"[Issue tag] {len(items)} view(s).")

    root = tk.Tk()
    root.geometry("+0+0")
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    cw, ch = scr_w - 30, scr_h - 230
    canvas = tk.Canvas(root, width=cw, height=ch, bg="black", highlightthickness=0, cursor="tcross")
    canvas.pack()
    bar = tk.Frame(root)
    bar.pack(pady=(4, 0))
    tag_vars = {}
    for k, tag, lab in ISSUE_TAGS:
        v = tk.BooleanVar()
        tk.Checkbutton(bar, text=f"[{k}] {lab}", variable=v, font=("Arial", 12, "bold"),
                       takefocus=0).pack(side=tk.LEFT, padx=8)
        tag_vars[tag] = v
    tk.Label(bar, text="note:", font=("Arial", 11)).pack(side=tk.LEFT, padx=(16, 2))
    note = tk.Entry(bar, width=70, font=("Arial", 11))
    note.pack(side=tk.LEFT)
    status = tk.Label(root, font=("Arial", 12, "bold"))
    status.pack(pady=(2, 0))
    tk.Label(root, font=("Arial", 10), text=(
        "1/2/3 = tag    left drag = paint problem area    right drag = erase    [ ] = brush    "
        "wheel = zoom    middle drag = pan    t = current/Gemini/sent    c = clear    "
        "x = no issue    space = save+next    b = back    q = quit")).pack(pady=(0, 4))

    S = {"pos": 0, "img": None, "alts": [], "alt": 0, "mask": None, "scale": 1.0,
         "ox": 0.0, "oy": 0.0, "brush": 18, "pending": False, "pan": None}

    def fit():
        h, w = S["img"].shape[:2]
        S["scale"] = min(cw / w, ch / h)
        S["ox"] = (w - cw / S["scale"]) / 2
        S["oy"] = (h - ch / S["scale"]) / 2

    def redraw():
        S["pending"] = False
        im = S["alts"][S["alt"]]
        h, w = im.shape[:2]
        sc = S["scale"]
        # canvas pixel -> image coordinate: x = ox + u / sc
        M = np.float32([[sc, 0, -S["ox"] * sc], [0, sc, -S["oy"] * sc]])
        interp = cv2.INTER_LINEAR if sc < 1 else cv2.INTER_NEAREST
        disp = cv2.warpAffine(im, M, (cw, ch), flags=interp, borderValue=(0, 0, 0))
        mk = cv2.warpAffine(S["mask"], M, (cw, ch), flags=cv2.INTER_NEAREST)
        sel = mk > 0
        disp[sel] = (0.55 * disp[sel] + 0.45 * np.array([60, 60, 255])).astype(np.uint8)
        edge = cv2.morphologyEx(mk, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        disp[edge] = (0, 0, 255)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._img = img
        canvas.delete("all")
        canvas.create_image(0, 0, image=img, anchor="nw")
        update_status()

    def later():
        if not S["pending"]:
            S["pending"] = True
            root.after(25, redraw)

    def update_status():
        d, m = items[S["pos"]]
        names = ["current image", "Gemini response", "sent to Gemini"]
        area = cv2.countNonZero(S["mask"]) / S["mask"].size * 100
        done = sum(1 for _, mm in items if issues.get(cand_key(mm), {}).get("tags"))
        status.config(text=(
            f"[{S['pos'] + 1}/{len(items)}]  {cand_key(m)}   showing: {names[S['alt']]}   "
            f"painted {area:.2f}%   brush {S['brush']}px   zoom {S['scale']:.2f}   "
            f"({done}/{len(items)} tagged)"))
        root.title(f"Issue tag — {cand_key(m)}")

    def load():
        d, m = items[S["pos"]]
        cur = cv2.imread(str(current_image(d)))
        alts = [cur]
        for f in ("08b_gemini_response.png", "08a_sent.png"):
            a = cv2.imread(str(d / f)) if (d / f).exists() else None
            if a is not None:
                alts.append(cv2.resize(a, (cur.shape[1], cur.shape[0])))
        S.update(img=cur, alts=alts, alt=0)
        rp = d / "09r_region.png"
        mk = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE) if rp.exists() else None
        S["mask"] = mk if mk is not None and mk.shape == cur.shape[:2] else np.zeros(cur.shape[:2], np.uint8)
        e = issues.get(cand_key(m), {})
        for tag, v in tag_vars.items():
            v.set(tag in e.get("tags", []))
        note.delete(0, tk.END)
        note.insert(0, e.get("note", ""))
        fit()
        redraw()

    def save():
        d, m = items[S["pos"]]
        tags = [t for _, t, _ in ISSUE_TAGS if tag_vars[t].get()]
        k = cand_key(m)
        if tags or cv2.countNonZero(S["mask"]) or note.get().strip():
            issues[k] = {"tags": tags, "note": note.get().strip(),
                         "region": bool(cv2.countNonZero(S["mask"]))}
        else:
            issues.pop(k, None)
        if cv2.countNonZero(S["mask"]):
            cv2.imwrite(str(d / "09r_region.png"), S["mask"])
        else:
            (d / "09r_region.png").unlink(missing_ok=True)
        save_issues(store, issues)

    def to_img(e):
        return S["ox"] + e.x / S["scale"], S["oy"] + e.y / S["scale"]

    def paint(e, val):
        x, y = to_img(e)
        r = max(1, int(round(S["brush"] / S["scale"])))
        last = S.get("last")
        p = (int(round(x)), int(round(y)))
        if last:
            cv2.line(S["mask"], last, p, val, 2 * r)
        cv2.circle(S["mask"], p, r, val, -1)
        S["last"] = p
        later()

    def zoom(e):
        f = 1.25 if e.delta > 0 else 0.8
        x, y = to_img(e)
        S["scale"] = min(max(S["scale"] * f, 0.1), 12)
        S["ox"], S["oy"] = x - e.x / S["scale"], y - e.y / S["scale"]
        later()

    def pan_start(e):
        S["pan"] = (e.x, e.y, S["ox"], S["oy"])

    def pan_move(e):
        x0, y0, ox, oy = S["pan"]
        S["ox"], S["oy"] = ox - (e.x - x0) / S["scale"], oy - (e.y - y0) / S["scale"]
        later()

    def go(step):
        save()
        S["pos"] = min(max(0, S["pos"] + step), len(items) - 1)
        load()

    canvas.bind("<Button-1>", lambda e: (canvas.focus_set(), S.update(last=None), paint(e, 255)))
    canvas.bind("<B1-Motion>", lambda e: paint(e, 255))
    canvas.bind("<Button-3>", lambda e: (S.update(last=None), paint(e, 0)))
    canvas.bind("<B3-Motion>", lambda e: paint(e, 0))
    canvas.bind("<ButtonRelease-1>", lambda e: S.update(last=None))
    canvas.bind("<ButtonRelease-3>", lambda e: S.update(last=None))
    canvas.bind("<MouseWheel>", zoom)
    canvas.bind("<Button-2>", pan_start)
    canvas.bind("<B2-Motion>", pan_move)
    note.bind("<Return>", lambda e: (canvas.focus_set(), "break")[1])

    def on_key(e):
        if root.focus_get() is note:
            return
        k = e.keysym.lower()
        for key, tag, _ in ISSUE_TAGS:
            if k in (key, f"kp_{key}"):
                tag_vars[tag].set(not tag_vars[tag].get())
                return
        if k == "space":
            if S["pos"] + 1 < len(items):
                go(1)
            else:
                save()
                print("[Issue tag] last view saved.")
                update_status()
        elif k == "b":
            go(-1)
        elif k == "bracketleft":
            S["brush"] = max(2, int(S["brush"] / 1.3))
            update_status()
        elif k == "bracketright":
            S["brush"] = min(300, int(S["brush"] * 1.3) + 1)
            update_status()
        elif k == "t":
            S["alt"] = (S["alt"] + 1) % len(S["alts"])
            later()
        elif k == "c":
            S["mask"][:] = 0
            later()
        elif k == "x":
            for v in tag_vars.values():
                v.set(False)
            note.delete(0, tk.END)
            S["mask"][:] = 0
            later()
        elif k == "f":
            fit()
            later()
        elif k == "q":
            save()
            root.destroy()

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", lambda: (save(), root.destroy()))
    load()
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    canvas.focus_set()
    root.mainloop()
    n = sum(1 for _, m in items if issues.get(cand_key(m), {}).get("tags"))
    print(f"[Issue tag] {n} view(s) tagged -> {store.root / 'issues.json'}. Next: `fix`.")


def cmd_fix(store, args):
    """
    One Gemini call per view tagged in issue-tag: the current dataset image
    (+ a copy with the painted area outlined in red), prompt built from the
    tags and note. The response is blended back inside the painted area only
    (whole response if nothing was painted). Skips views already fixed with
    the same tags/note/region unless --redo.
    """
    issues = load_issues(store)
    todo, untagged = [], []
    for d, m in _select_ids(accepted_candidates(store), args.ids):
        e = issues.get(cand_key(m))
        if e and not e.get("tags") and (e.get("region") or e.get("note")):
            untagged.append(cand_key(m))
        if not e or not e.get("tags"):
            continue
        region_p = d / "09r_region.png"
        region = cv2.imread(str(region_p), cv2.IMREAD_GRAYSCALE) if region_p.exists() else None
        has_region = region is not None and cv2.countNonZero(region) > 0
        sig = {"tags": e["tags"], "note": e.get("note", ""),
               "region_mtime": region_p.stat().st_mtime if has_region else None}
        if args.ref_original:
            sig["ref_original"] = True
        old = json.loads((d / "09_fix.json").read_text(encoding="utf-8")) \
            if (d / "09_fix.json").exists() else {}
        old_sig = dict(old.get("sig") or {})
        if not args.ref_original:          # a plain `fix` doesn't redo --ref-original fixes
            old_sig.pop("ref_original", None)
        if not args.redo and (d / "09d_response.png").exists() and old_sig == sig:
            continue
        todo.append((d, m, region if has_region else None, e, sig))
    if args.limit:
        todo = todo[:args.limit]
    if untagged:
        print(f"[Fix] WARNING: {len(untagged)} view(s) painted or noted but with no tag, "
              f"skipped (tag them in issue-tag): {' '.join(untagged)}")
    print(f"[Fix] {len(todo)} view(s) -> {len(todo)} Gemini call(s).")
    if not todo:
        return
    if args.show_prompt:
        _, _, r0, e0, _ = todo[0]
        print("--- prompt of the first view ---\n"
              + build_fix_prompt(e0["tags"], e0.get("note", ""), r0 is not None, args.ref_original)
              + "\n---")
    if not args.yes and input("Proceed? (y/N): ").strip().lower() not in ("y", "yes"):
        return

    def one(d, m, region, e, sig):
        # The input is the image before any fix, so a redo never stacks fixes.
        base_p = d / "09a_input.png"
        base = cv2.imread(str(base_p)) if base_p.exists() else cv2.imread(str(current_image(d)))
        ref = []
        if args.ref_original:
            orig = cv2.imread(str(store.prep / m["image"] / "image.png"))
            ref = [cv2.resize(orig, (base.shape[1], base.shape[0]))]
        guide = region is not None
        prompt = build_fix_prompt(e["tags"], e.get("note", ""), guide, args.ref_original)
        imgs = [base] + ([mark_region(base, region)] if guide else []) + ref
        resp = gemini_fix(imgs, prompt)
        if resp is not None and guide and guide_leaked(base, resp):
            # Gemini drew the red outline into the picture: ask again without the
            # guide image; the result is still blended inside the painted area only.
            print(f"  {cand_key(m)}: red guide leaked into the output -> retry without guide")
            guide = False
            prompt = build_fix_prompt(e["tags"], e.get("note", ""), False, args.ref_original)
            resp = gemini_fix([base] + ref, prompt)
        if resp is None:
            return False
        for f in FIX_FILES:
            if f != "09a_input.png":
                (d / f).unlink(missing_ok=True)
        cv2.imwrite(str(base_p), base)
        if region is not None:
            if guide:
                cv2.imwrite(str(d / "09b_marked.png"), imgs[1])
            cv2.imwrite(str(d / "09c_region.png"), region)
        cv2.imwrite(str(d / "09d_response.png"), resp)
        (d / "09_fix.json").write_text(json.dumps(
            {"sig": sig, "prompt": prompt, "guide_sent": guide,
             "red_leak": bool(guide_leaked(base, resp))}, indent=2,
            ensure_ascii=False), encoding="utf-8")
        write_fix_result(d, "region" if region is not None else "full")
        return True

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, d, m, r, e, s): m for d, m, r, e, s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            m = futs[fut]
            try:
                good = fut.result()
            except Exception as e:
                print(f"  {cand_key(m)}: {e!r}")
                good = False
            ok += good
            fail += not good
            print(f"  [{i}/{len(todo)}] {cand_key(m)}: {'fixed' if good else 'FAILED'}")
    print(f"[Fix] fixed {ok}, failed {fail}. Re-run to retry failures; check with `fix-view`.")


def cmd_unfix(store, args):
    """Remove the fix of the given views (back to the image before it)."""
    n = 0
    for d, m in _select_ids(store.candidates(), args.ids):
        if (d / "09_final_fix.png").exists() or (d / "09d_response.png").exists():
            n += 1
        for f in FIX_FILES:
            (d / f).unlink(missing_ok=True)
    print(f"[Unfix] removed {n} fix(es).")


def cmd_fix_view(store, args):
    """
    Check the fixes: before | Gemini response | result in the dataset.
      1 = use region blend   2 = use Gemini full image   3 = revert (keep before)
      (each saves and goes to the next)
      space / n = next   b = back   z = zoom to the painted area / whole image
      q = quit
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    items = [(d, m) for d, m in _select_ids(accepted_candidates(store), args.ids)
             if (d / "09d_response.png").exists()]
    if not items:
        print("[Fix view] No fixed views yet (run `fix`).")
        return
    issues = load_issues(store)
    root = tk.Tk()
    root.geometry("+0+0")
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    pane_w, pane_h = (scr_w - 40) // 3, scr_h - 230
    top = tk.Frame(root)
    top.pack()
    panes, frames = [], []
    for title in ("before", "[2] Gemini response (full)", "[1] region blend"):
        col = tk.Frame(top, highlightthickness=5, highlightbackground="#444")
        col.pack(side=tk.LEFT, padx=3)
        tk.Label(col, text=title, font=("Arial", 11, "bold")).pack()
        c = tk.Canvas(col, width=pane_w, height=pane_h, bg="black", highlightthickness=0)
        c.pack()
        panes.append(c)
        frames.append(col)
    status = tk.Label(root, font=("Arial", 12, "bold"))
    status.pack(pady=(4, 0))
    tag_line = tk.Label(root, font=("Arial", 10))
    tag_line.pack()
    tk.Label(root, font=("Arial", 10), text=(
        "1 = region blend    2 = Gemini full    3 = revert (keep before)    space = next    "
        "b = back    z = zoom to painted area    q = quit")).pack(pady=(0, 4))
    pos, zoom = [0], [False]
    cache = {}

    def info(d):
        p = d / "09_fix.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def imgs(d):
        if d not in cache:
            cache.clear()
            base = cv2.imread(str(d / "09a_input.png"))
            resp = cv2.imread(str(d / "09d_response.png"))
            rp = d / "09c_region.png"
            region = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE) if rp.exists() else None
            blend = blend_region(base, resp, region) if region is not None else resp
            cache[d] = (base, resp, blend, region)
        return cache[d]

    def put(canvas, bgr):
        h, w = bgr.shape[:2]
        sc = min(pane_w / w, pane_h / h)
        disp = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                          interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._img = img
        canvas.delete("all")
        canvas.create_image(pane_w // 2, pane_h // 2, image=img)

    def show():
        d, m = items[pos[0]]
        base, resp, blend, region = imgs(d)
        h, w = base.shape[:2]
        crop = (slice(0, h), slice(0, w))
        if zoom[0] and region is not None and cv2.countNonZero(region):
            x, y, bw, bh = cv2.boundingRect(region)
            pad = max(40, int(0.3 * max(bw, bh)))
            crop = (slice(max(0, y - pad), min(h, y + bh + pad)),
                    slice(max(0, x - pad), min(w, x + bw + pad)))
        shown = [base, resp, blend]
        if region is not None:
            shown[0] = mark_region(base, region)
        for c, im in zip(panes, shown):
            put(c, im[crop])
        mode = info(d).get("merge")
        for f, md in zip(frames, (None, "full", "region")):
            f.config(highlightbackground="#2e9e4f" if mode == md and md else "#444")
        e = issues.get(cand_key(m), {})
        status.config(text=f"[{pos[0] + 1}/{len(items)}]  {cand_key(m)}   in dataset: "
                           f"{ {'region': 'region blend', 'full': 'Gemini full', 'revert': 'before (reverted)'}.get(mode, mode)}"
                           f"   view: {'painted area' if zoom[0] else 'whole image'}")
        tag_line.config(text=f"tags: {', '.join(e.get('tags', []))}    note: {e.get('note', '')}")
        root.title(f"Fix view — {cand_key(m)}")

    def go(step):
        pos[0] = min(max(0, pos[0] + step), len(items) - 1)
        zoom[0] = False
        show()

    def choose(mode):
        d, _ = items[pos[0]]
        if mode == "region" and not (d / "09c_region.png").exists():
            mode = "full"
        write_fix_result(d, mode)
        if pos[0] + 1 < len(items):
            go(1)
        else:
            show()

    def on_key(e):
        k = e.keysym.lower()
        if k in ("1", "kp_1"):
            choose("region")
        elif k in ("2", "kp_2"):
            choose("full")
        elif k in ("3", "kp_3"):
            choose("revert")
        elif k in ("space", "return", "n"):
            go(1)
        elif k == "b":
            go(-1)
        elif k == "z":
            zoom[0] = not zoom[0]
            show()
        elif k == "q":
            root.destroy()

    root.bind("<Key>", on_key)
    show()
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()
    modes = [info(d).get("merge") for d, _ in items]
    print(f"[Fix view] region {modes.count('region')}, full {modes.count('full')}, "
          f"reverted {modes.count('revert')}. Run `finalize` to update the dataset.")


# ── Face point check (final inspection) ──────────────────────────────────────
#  After the edit the head often moved, while the label's face point stayed
#  where the face was in the source. face-check detects faces (insightface)
#  and flags views whose face point is off; face-fix lets you move it.

def _all_views(store):
    """(key, image path, label, kind) for every view in the dataset."""
    out = []
    for name in sorted({m["image"] for _, m in accepted_candidates(store)} | set(store.prepped_names())):
        sl = (store.prep_info(name) or {}).get("source_label")
        if sl:
            out.append((f"{name}/orig", store.prep / name / "image.png", sl, "original"))
    for d, m in accepted_candidates(store):
        out.append((cand_key(m), gt_image(d)[0], label_of(m), "generated"))
    return out


def cmd_face_check(store, args):
    """
    Detect faces on every view and flag face points that are off: farther
    than --rel x face size from the detected eyes, or (no face found, e.g. a
    back view) not on the person mask of the exported view. Writes
    face_check.json (suggested eye midpoint per view) for face-fix.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.35)
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    res = {}
    views = _all_views(store)
    for i, (key, path, lab, kind) in enumerate(views, 1):
        im = cv2.imread(str(path))
        name, vid = key.split("/")
        mp = store.final / name / "masks" / f"{vid}.png"
        mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
        if mk is None or mk.shape != im.shape[:2]:
            mk = cv2.imread(str(store.prep / name / "mask.png"), cv2.IMREAD_GRAYSCALE)
        fx, fy = lab["face_px"]
        on_person = cv2.dilate(mk, ell(25))
        best = None
        for f in app.get(im):
            x0, y0, x1, y1 = [float(v) for v in f.bbox]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if not (0 <= int(cy) < im.shape[0] and 0 <= int(cx) < im.shape[1]) or not on_person[int(cy), int(cx)]:
                continue
            eyes = f.kps[:2].mean(axis=0) if f.kps is not None else np.array([cx, cy])
            dist = float(np.hypot(eyes[0] - fx, eyes[1] - fy))
            if best is None or dist < best["dist"]:
                best = {"eyes": [float(eyes[0]), float(eyes[1])], "bbox": [x0, y0, x1, y1],
                        "size": max(x1 - x0, y1 - y0), "score": float(f.det_score), "dist": dist}
        h, w = mk.shape
        on = bool(on_person[min(h - 1, max(0, int(fy))), min(w - 1, max(0, int(fx)))])
        if best:
            flag = best["dist"] > args.rel * best["size"]
            why = f"{best['dist']:.0f}px from detected eyes ({best['dist'] / best['size']:.2f} face size)"
        else:
            flag = not on
            why = "no face detected; face point off the person" if flag else "no face detected"
        res[key] = {"kind": kind, "face_px": [float(fx), float(fy)], "det": best,
                    "flag": bool(flag), "why": why}
        if i % 50 == 0:
            print(f"  {i}/{len(views)}")
    p = store.root / "face_check.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    flagged = [k for k, r in res.items() if r["flag"]]
    print(f"[Face check] {len(res)} views, {sum(r['det'] is not None for r in res.values())} with a "
          f"detected face; {len(flagged)} flagged -> {p}\n  " + " ".join(flagged))


def _set_face(store, key, new_face):
    """Move a label's face point; the target moves by the same 2D offset so the
    drawn gaze arrow (what was judged in review) stays the same."""
    name, vid = key.split("/")
    if vid == "orig":
        info = store.prep_info(name)
        lab = info["source_label"]
    else:
        d = store.cand / name / vid
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        lab = dict(label_of(m))
    ox, oy = new_face[0] - lab["face_px"][0], new_face[1] - lab["face_px"][1]
    old_face = lab.get("face_px_before_fix", lab["face_px"])
    lab["face_px"] = [float(new_face[0]), float(new_face[1])]
    lab["target_px"] = [float(lab["target_px"][0] + ox), float(lab["target_px"][1] + oy)]
    if vid == "orig":
        shape = cv2.imread(str(store.prep / name / "image.png")).shape
    else:
        shape = cv2.imread(str(gt_image(d)[0])).shape
    g = gaze_from_target(lab["face_px"], lab["target_px"], lab["face_depth"], lab["target_depth"], shape)
    lab["gaze"] = g
    lab["face_px_before_fix"] = old_face
    lab["face_fixed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if lab.get("source") == "requested":
        lab["source"] = "manual"
    if vid == "orig":
        info["source_label"] = lab
        (store.prep / name / "prep.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    else:
        m["label"] = lab
        store.write_meta(d, m)
    return lab


def cmd_face_fix(store, args):
    """
    Move face points flagged by face-check (or --ids / --all).
      yellow = current face point   green = detected eyes (suggestion)
      a = use the suggestion   left click = put the face point there
      space / Enter = save (if moved) + next   k = keep as is + next
      r = undo the move on this view   b = back   q = quit
    The target moves with the face, so the gaze arrow keeps its direction.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    p = store.root / "face_check.json"
    fc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    views = {k: (path, lab, kind) for k, path, lab, kind in _all_views(store)}
    if args.ids:
        keys = [k for k in args.ids if k in views]
    elif args.all:
        keys = list(views)
    else:
        keys = [k for k, r in fc.items() if r["flag"] and k in views]
    if not keys:
        print("[Face fix] nothing to do (run face-check, or pass --ids / --all).")
        return
    print(f"[Face fix] {len(keys)} view(s).")
    root = tk.Tk()
    root.geometry("+0+0")
    cw, ch = root.winfo_screenwidth() - 30, root.winfo_screenheight() - 200
    canvas = tk.Canvas(root, width=cw, height=ch, bg="black", highlightthickness=0, cursor="tcross")
    canvas.pack()
    status = tk.Label(root, font=("Arial", 12, "bold"))
    status.pack(pady=(4, 0))
    tk.Label(root, font=("Arial", 10), text=(
        "yellow = current face point   green = detected eyes   a = use suggestion   "
        "left click = set face point   space = save + next   k = keep + next   r = undo   "
        "b = back   q = quit")).pack(pady=(0, 4))
    S = {"pos": 0, "new": None, "img": None, "scale": 1, "off": (0, 0), "done": 0}

    def cur():
        k = keys[S["pos"]]
        path, lab, kind = views[k]
        return k, path, lab

    def load():
        k, path, lab = cur()
        S["img"] = cv2.imread(str(path))
        S["new"] = None
        # crop around the person: face point, suggestion and target
        pts = [lab["face_px"]]
        det = (fc.get(k) or {}).get("det")
        if det:
            pts += [det["eyes"], det["bbox"][:2], det["bbox"][2:]]
        pts = np.array(pts, float)
        h, w = S["img"].shape[:2]
        span = max(260.0, float(np.ptp(pts[:, 0])) + 300, float(np.ptp(pts[:, 1])) + 300)
        c = pts.mean(axis=0)
        x0 = int(np.clip(c[0] - span / 2, 0, max(0, w - span)))
        y0 = int(np.clip(c[1] - span / 2, 0, max(0, h - span)))
        S["off"] = (x0, y0)
        S["crop"] = S["img"][y0:y0 + int(span), x0:x0 + int(span)]
        S["scale"] = min(cw / S["crop"].shape[1], ch / S["crop"].shape[0])
        draw()

    def to_c(pt):
        return ((pt[0] - S["off"][0]) * S["scale"], (pt[1] - S["off"][1]) * S["scale"])

    def draw():
        k, path, lab = cur()
        sc = S["scale"]
        disp = cv2.resize(S["crop"], None, fx=sc, fy=sc,
                          interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas._img = img
        canvas.delete("all")
        canvas.create_image(0, 0, image=img, anchor="nw")
        det = (fc.get(k) or {}).get("det")
        if det:
            a, b = to_c(det["bbox"][:2]), to_c(det["bbox"][2:])
            canvas.create_rectangle(*a, *b, outline="#3c3", width=2, dash=(4, 3))
            ex, ey = to_c(det["eyes"])
            canvas.create_line(ex - 12, ey, ex + 12, ey, fill="#3f3", width=3)
            canvas.create_line(ex, ey - 12, ex, ey + 12, fill="#3f3", width=3)
        fx, fy = to_c(lab["face_px"])
        tx, ty = to_c(lab["target_px"])
        canvas.create_oval(fx - 9, fy - 9, fx + 9, fy + 9, outline="yellow", width=3)
        canvas.create_line(fx, fy, fx + (tx - fx) * 0.35, fy + (ty - fy) * 0.35, fill="yellow",
                           width=2, arrow="last", dash=(5, 3))
        if S["new"]:
            nx, ny = to_c(S["new"])
            canvas.create_oval(nx - 9, ny - 9, nx + 9, ny + 9, outline="#0ff", width=3)
            canvas.create_line(nx, ny, nx + (tx - fx) * 0.35, ny + (ty - fy) * 0.35, fill="#0ff",
                               width=3, arrow="last")
        r = fc.get(k) or {}
        status.config(text=f"[{S['pos'] + 1}/{len(keys)}]  {k}   {r.get('why', '')}"
                           f"{'   -> moved (space to save)' if S['new'] else ''}"
                           f"{'   [fixed before]' if lab.get('face_fixed') else ''}   saved {S['done']}")
        root.title(f"Face fix — {k}")

    def click(e):
        S["new"] = [S["off"][0] + e.x / S["scale"], S["off"][1] + e.y / S["scale"]]
        draw()

    def step(n):
        S["pos"] = min(max(0, S["pos"] + n), len(keys) - 1)
        load()

    def on_key(e):
        k = e.keysym.lower()
        key, path, lab = cur()
        if k == "a":
            det = (fc.get(key) or {}).get("det")
            if det:
                S["new"] = list(det["eyes"])
                draw()
        elif k in ("space", "return"):
            if S["new"]:
                views[key] = (path, _set_face(store, key, S["new"]), views[key][2])
                S["done"] += 1
            if S["pos"] + 1 < len(keys):
                step(1)
            else:
                load()
        elif k == "k":
            step(1)
        elif k == "r":
            if lab.get("face_fixed"):
                views[key] = (path, _set_face(store, key, lab["face_px_before_fix"]), views[key][2])
            S["new"] = None
            load()
        elif k == "b":
            step(-1)
        elif k == "q":
            root.destroy()

    canvas.bind("<Button-1>", click)
    root.bind("<Key>", on_key)
    load()
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()
    print(f"[Face fix] moved {S['done']} face point(s). Run `finalize` to update the dataset.")


# ══════════════════════════════════════════════════════════════════════════════
#  5. STATS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_stats(store, args):
    reviews = store.reviews()
    acc, pend, _ = bin_counts(store, reviews)
    metas = [m for _, m in store.candidates()]
    status = {}
    for m in metas:
        status[m["status"]] = status.get(m["status"], 0) + 1
    verdicts = {}
    for m in metas:
        v = reviews.get(cand_key(m), {}).get("verdict", "unreviewed")
        verdicts[v] = verdicts.get(v, 0) + 1

    print(f"\n[Stats] prepped images: {len(store.prepped_names())}   "
          f"candidates: {len(metas)} {status}   reviews: {verdicts}")
    print("\nAccepted per bin  (+pending = generated/planned, not reviewed yet)")
    head = "pitch\\yaw " + "".join(f"{yaw_bin_center(b):>7d}" for b in range(N_YAW_BINS))
    print(head)
    for p in reversed(range(len(PITCH_NAMES))):
        row = f"{PITCH_NAMES[p]:>9s} "
        for y in range(N_YAW_BINS):
            a, q = acc.get((y, p), 0), pend.get((y, p), 0)
            row += f"{a:>4d}" + (f"+{q:<2d}" if q else "   ")
        print(row)
    filled = [acc.get((y, p), 0) for y in range(N_YAW_BINS) for p in range(len(PITCH_NAMES))]
    print(f"\nAccepted total {sum(filled)}, min/bin {min(filled)}, max/bin {max(filled)}, "
          f"empty bins {sum(1 for x in filled if x == 0)}/{len(filled)}")

    per_cat, per_img = {}, {}
    for m in metas:
        if reviews.get(cand_key(m), {}).get("verdict") == "ok":
            per_cat[m["category"]] = per_cat.get(m["category"], 0) + 1
            per_img[m["image"]] = per_img.get(m["image"], 0) + 1
    print("Accepted per category: " + ", ".join(f"{k} {v}" for k, v in sorted(per_cat.items())))
    no_acc = [n for n in store.prepped_names() if n not in per_img]
    if no_acc:
        print(f"Images with no accepted sample yet ({len(no_acc)}): {', '.join(no_acc)}")


# ══════════════════════════════════════════════════════════════════════════════
#  GALLERY  (static HTML overview of everything generated)
# ══════════════════════════════════════════════════════════════════════════════

GALLERY_HTML = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Dataset</title>
<style>
:root {
  color-scheme: light;
  --bg:#f6f6f4; --surface:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --faint:#8a8984;
  --line:#deded8; --grid:#ebebe6;
  --s1:#2a78d6; --s2:#eb6834;
  --ok:#0ca30c; --bad:#d03b3b;
  --seq-1:#cde2fb; --seq-2:#9ec5f4; --seq-3:#6da7ec; --seq-4:#3987e5; --seq-5:#256abf;
  --seq-6:#184f95; --seq-7:#0d366b; --ink-lo:#0b0b0b; --ink-hi:#ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg:#141413; --surface:#1a1a19; --fg:#f0efec; --muted:#c3c2b7; --faint:#8f8e86;
    --line:#383835; --grid:#2a2a28;
    --s1:#3987e5; --s2:#d95926;
    --seq-1:#0d366b; --seq-2:#104281; --seq-3:#184f95; --seq-4:#1c5cab; --seq-5:#256abf;
    --seq-6:#3987e5; --seq-7:#6da7ec; --ink-lo:#f0efec; --ink-hi:#0b0b0b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg:#141413; --surface:#1a1a19; --fg:#f0efec; --muted:#c3c2b7; --faint:#8f8e86;
  --line:#383835; --grid:#2a2a28;
  --s1:#3987e5; --s2:#d95926;
  --seq-1:#0d366b; --seq-2:#104281; --seq-3:#184f95; --seq-4:#1c5cab; --seq-5:#256abf;
  --seq-6:#3987e5; --seq-7:#6da7ec; --ink-lo:#f0efec; --ink-hi:#0b0b0b;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, "Noto Sans TC", "Microsoft JhengHei", sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; }
header h1 { font-size:16px; margin:0 8px 0 0; }
header label { color:var(--muted); font-size:13px; }
select { font:inherit; padding:2px 4px; background:var(--surface); color:var(--fg);
         border:1px solid var(--line); border-radius:4px; }
#count { color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums; }
main { padding:12px 16px 40px; max-width:1800px; margin:0 auto; }
.overview { display:grid; grid-template-columns:minmax(0,1.5fr) minmax(0,1fr); gap:12px; margin-bottom:16px; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:12px 14px; min-width:0; }
.panel h2 { font-size:14px; margin:0 0 2px; }
.panel .sub { color:var(--muted); font-size:12px; margin:0 0 8px; }
.stack { display:grid; gap:12px; min-width:0; }
.legend { display:flex; gap:14px; font-size:12px; color:var(--muted); margin-bottom:4px; }
.legend i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
svg text { font:11px system-ui, sans-serif; fill:var(--muted); }
svg .axis { stroke:var(--line); } svg .gridl { stroke:var(--grid); }
svg .dot { stroke:var(--surface); stroke-width:2; cursor:pointer; }
svg .dot:hover { stroke:var(--fg); }
table.heat { border-collapse:separate; border-spacing:2px; width:100%; table-layout:fixed;
             font-variant-numeric:tabular-nums; }
table.heat th:first-child { width:38px; }
table.heat td.s1, table.heat td.s2, table.heat td.s3, table.heat td.s4 { color:var(--ink-lo); }
table.heat td.s5, table.heat td.s6, table.heat td.s7 { color:var(--ink-hi); }
table.heat th { font-weight:400; color:var(--muted); font-size:11px; padding:2px; }
table.heat td { text-align:center; font-size:12px; height:30px; border-radius:4px; cursor:pointer; }
table.heat td.sel { outline:2px solid var(--fg); }
table.heat td.zero { color:var(--faint); background:var(--bg); }
.bars { display:grid; grid-template-columns:auto 1fr auto; gap:4px 8px; align-items:center; font-size:12px; }
.bars .bar { height:12px; background:var(--s1); border-radius:0 4px 4px 0; }
.bars .n { color:var(--muted); font-variant-numeric:tabular-nums; }
#tip { position:fixed; pointer-events:none; z-index:20; background:var(--surface); color:var(--fg);
       border:1px solid var(--line); border-radius:4px; padding:5px 8px; font-size:12px;
       box-shadow:0 2px 8px rgba(0,0,0,.15); display:none; white-space:nowrap; }
#grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(560px, 1fr)); gap:12px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:8px; scroll-margin-top:70px; }
.card.bad { opacity:.55; }
.card.flash { outline:3px solid var(--s1); }
.imgs { display:grid; grid-template-columns:repeat(3, 1fr); gap:4px; }
.imgs a { display:block; background:#000; border-radius:3px; overflow:hidden; aspect-ratio:3/4; }
.imgs img { width:100%; height:100%; object-fit:contain; display:block; }
.lab { display:grid; grid-template-columns:repeat(3, 1fr); gap:4px; color:var(--muted);
       font-size:11px; text-align:center; margin-top:2px; }
.meta { margin-top:6px; display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; }
.meta b { font-size:14px; }
.tag { font-size:12px; color:var(--muted); }
.pill { font-size:11px; padding:0 6px; border-radius:8px; border:1px solid var(--line); color:var(--muted); }
.pill i { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:4px; vertical-align:middle; }
.v-ok { color:var(--fg); } .v-ok::before { content:"✓ "; color:var(--ok); }
.v-bad { color:var(--fg); } .v-bad::before { content:"✕ "; color:var(--bad); }
.dial { width:38px; height:38px; flex:none; color:var(--muted); }
@media (max-width:1100px) { .overview { grid-template-columns:1fr; } }
@media (max-width:620px) { #grid { grid-template-columns:1fr; } }
</style></head><body>
<header>
  <h1>Eval Dataset</h1>
  <label>Review <select id="f-verdict"></select></label>
  <label>圖片 <select id="f-image"></select></label>
  <label>類別 <select id="f-cat"></select></label>
  <label>Label <select id="f-src"></select></label>
  <label>補洞 <select id="f-repair"></select></label>
  <label>Yaw <select id="f-yaw"></select></label>
  <label>Pitch <select id="f-pitch"></select></label>
  <label>排序 <select id="f-sort"><option value="image">依圖片</option><option value="bin">依方向</option></select></label>
  <span id="count"></span>
</header>
<main>
  <div class="overview">
    <div class="panel">
      <h2>Gaze label 分布</h2>
      <p class="sub">每點一個 sample；格線為 yaw / pitch 分格邊界。點一下跳到該 sample。</p>
      <div class="legend"><span><i style="background:var(--s1)"></i>人工標註</span><span><i style="background:var(--s2)"></i>= 要求的方向</span></div>
      <svg id="scatter" viewBox="0 0 720 340" width="100%" role="img" aria-label="yaw pitch scatter"></svg>
    </div>
    <div class="stack">
      <div class="panel">
        <h2>各方向格數量</h2>
        <p class="sub">點格子篩選該方向，再點一次取消。pitch 超過 ±45° 併入往上 / 往下。</p>
        <table class="heat" id="heat"></table>
      </div>
      <div class="panel">
        <h2>各類別數量</h2>
        <div class="bars" id="bars"></div>
      </div>
    </div>
  </div>
  <div id="grid"></div>
</main>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const YAWS = __YAWS__, PITCHES = __PITCHES__, PITCH_EDGES = __PEDGES__;
const PITCH_ZH = {down: "往下", level: "水平", up: "往上"};
const $ = id => document.getElementById(id);
function opts(sel, items, all, def) {
  sel.innerHTML = `<option value="">${all}</option>` +
    items.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
  if (def !== undefined) sel.value = def;
}
const hasOk = DATA.some(d => d.verdict === "ok");
opts($("f-verdict"), [["ok", "OK（資料集）"], ["bad", "不 OK"], ["unreviewed", "未 review"]], "全部", hasOk ? "ok" : "");
opts($("f-image"), [...new Set(DATA.map(d => d.image))].map(v => [v, v]), "全部");
opts($("f-cat"), [...new Set(DATA.map(d => d.category))].sort().map(v => [v, v]), "全部");
opts($("f-src"), [["manual", "人工標註"], ["requested", "= 要求的方向"]], "全部");
opts($("f-repair"), [["gemini", "Gemini"], ["inpaint", "inpaint"]], "全部");
opts($("f-yaw"), YAWS.map((y, i) => [i, (y > 0 ? "+" : "") + y + "°"]), "全部");
opts($("f-pitch"), PITCHES.map((p, i) => [i, PITCH_ZH[p] || p]), "全部");

const tip = $("tip");
function showTip(e, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.display = "none"; }
const sgn = v => (v > 0 ? "+" : "") + v.toFixed(0) + "°";
const srcColor = s => s === "manual" ? "var(--s1)" : "var(--s2)";
const srcName = s => s === "manual" ? "人工標註" : "= 要求的方向";

function filters() {
  return { verdict: $("f-verdict").value, image: $("f-image").value, cat: $("f-cat").value,
           src: $("f-src").value, repair: $("f-repair").value,
           yaw: $("f-yaw").value, pitch: $("f-pitch").value };
}
function match(d, f, skipBin) {
  return (!f.verdict || (d.verdict || "unreviewed") === f.verdict) &&
    (!f.image || d.image === f.image) && (!f.cat || d.category === f.cat) &&
    (!f.src || d.label_source === f.src) && (!f.repair || d.repair === f.repair) &&
    (skipBin || ((f.yaw === "" || d.bin[0] == f.yaw) && (f.pitch === "" || d.bin[1] == f.pitch)));
}

function scatter(rows, f) {
  const W = 720, H = 340, L = 44, R = 12, T = 10, B = 34;
  const x = v => L + (v + 180) / 360 * (W - L - R);
  const y = v => T + (90 - Math.max(-90, Math.min(90, v))) / 180 * (H - T - B);
  let s = "";
  for (let b = 0; b <= YAWS.length; b++) {
    const v = -195 + b * 30;
    if (v >= -180 && v <= 180) s += `<line class="gridl" x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${H - B}"/>`;
  }
  PITCH_EDGES.forEach(v => s += `<line class="gridl" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/>`);
  if (f.yaw !== "" || f.pitch !== "") {
    const yc = YAWS[+f.yaw], x0 = f.yaw === "" ? -180 : Math.max(-180, yc - 15), x1 = f.yaw === "" ? 180 : Math.min(180, yc + 15);
    const lo = +f.pitch === 0 ? -90 : PITCH_EDGES[+f.pitch];
    const hi = +f.pitch === PITCHES.length - 1 ? 90 : PITCH_EDGES[+f.pitch + 1];
    const p0 = f.pitch === "" ? -90 : lo, p1 = f.pitch === "" ? 90 : hi;
    s += `<rect x="${x(x0)}" y="${y(p1)}" width="${x(x1) - x(x0)}" height="${y(p0) - y(p1)}" fill="var(--s1)" fill-opacity=".08" stroke="var(--s1)" stroke-opacity=".5"/>`;
  }
  s += `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  [-180, -90, 0, 90, 180].forEach(v => s += `<text x="${x(v)}" y="${H - B + 15}" text-anchor="${v === -180 ? "start" : v === 180 ? "end" : "middle"}">${v > 0 ? "+" : ""}${v}°</text>`);
  s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">yaw（0 = 朝向鏡頭，±180 = 背對）</text>`;
  [-90, -45, -15, 15, 45, 90].forEach(v => s += `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v > 0 ? "+" : ""}${v}°</text>`);
  s += `<text transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">pitch</text>`;
  rows.forEach((d, i) => s += `<circle class="dot" data-i="${i}" cx="${x(d.yaw)}" cy="${y(d.pitch)}" r="4.5" fill="${srcColor(d.label_source)}"/>`);
  const svg = $("scatter"); svg.innerHTML = s;
  svg.querySelectorAll(".dot").forEach(c => {
    const d = rows[+c.dataset.i];
    c.onmousemove = e => showTip(e, `<b>${d.image} / ${d.cand_id}</b><br>yaw ${sgn(d.yaw)} · pitch ${sgn(d.pitch)}<br>${d.desc}<br>${srcName(d.label_source)}`);
    c.onmouseleave = hideTip;
    c.onclick = () => jump(d);
  });
}

function heat(rows, f) {
  const n = {}; rows.forEach(d => { const k = d.bin.join(","); n[k] = (n[k] || 0) + 1; });
  const max = Math.max(1, ...Object.values(n));
  let h = "<tr><th></th>" + YAWS.map(v => `<th>${v > 0 ? "+" : ""}${v}</th>`).join("") + "</tr>";
  for (let p = PITCHES.length - 1; p >= 0; p--) {
    h += `<tr><th>${PITCH_ZH[PITCHES[p]] || PITCHES[p]}</th>`;
    YAWS.forEach((_, yb) => {
      const c = n[yb + "," + p] || 0;
      const sel = f.yaw === String(yb) && f.pitch === String(p) ? " sel" : "";
      if (!c) { h += `<td class="zero${sel}" data-y="${yb}" data-p="${p}" data-n="0">0</td>`; return; }
      const step = 1 + Math.min(6, Math.floor(c / max * 6.999));
      h += `<td class="s${step}${sel}" data-y="${yb}" data-p="${p}" data-n="${c}" style="background:var(--seq-${step})">${c}</td>`;
    });
    h += "</tr>";
  }
  const t = $("heat"); t.innerHTML = h;
  t.querySelectorAll("td").forEach(td => {
    const yv = YAWS[+td.dataset.y], pn = PITCH_ZH[PITCHES[+td.dataset.p]];
    td.onmousemove = e => showTip(e, `yaw ${yv > 0 ? "+" : ""}${yv}° ±15° · ${pn}<br>${td.dataset.n} 個 sample`);
    td.onmouseleave = hideTip;
    td.onclick = () => {
      const same = $("f-yaw").value === td.dataset.y && $("f-pitch").value === td.dataset.p;
      $("f-yaw").value = same ? "" : td.dataset.y; $("f-pitch").value = same ? "" : td.dataset.p;
      render();
    };
  });
}

function bars(rows) {
  const n = {}; rows.forEach(d => n[d.category] = (n[d.category] || 0) + 1);
  const cats = [...new Set(DATA.map(d => d.category))].sort();
  const max = Math.max(1, ...Object.values(n));
  $("bars").innerHTML = cats.map(c =>
    `<span>${c}</span><div><div class="bar" style="width:${(n[c] || 0) / max * 100}%"></div></div><span class="n">${n[c] || 0}</span>`).join("");
}

function dial(yaw) {
  const r = yaw * Math.PI / 180, x = 19 + 15 * Math.sin(r), y = 19 + 15 * Math.cos(r);
  return `<svg class="dial" viewBox="0 0 38 38" aria-label="top view">
    <circle cx="19" cy="19" r="17" fill="none" stroke="currentColor" stroke-opacity=".35"/>
    <rect x="15" y="34" width="8" height="4" fill="currentColor" fill-opacity=".6"/>
    <line x1="19" y1="19" x2="${x}" y2="${y}" stroke="${"var(--s1)"}" stroke-width="3" stroke-linecap="round"/></svg>`;
}
function card(d) {
  const v = d.verdict === "ok" ? '<span class="v-ok">OK</span>'
          : d.verdict === "bad" ? '<span class="v-bad">不 OK</span>' : '<span class="tag">未 review</span>';
  const im = (t, f) => `<a href="${f}" target="_blank"><img loading="lazy" src="${t}" alt=""></a>`;
  return `<div class="card ${d.verdict || ""}" id="c-${d.image}-${d.cand_id}">
    <div class="imgs">${im(d.t_src, d.f_src)}${im(d.t_fin, d.f_fin)}${im(d.t_lab, d.f_fin)}</div>
    <div class="lab"><span>原圖${d.src_yaw === null ? "（gaze 未標註）" : `（yaw ${sgn(d.src_yaw)} · pitch ${sgn(d.src_pitch)}）`}</span><span>生成結果</span><span>結果 + label 方向</span></div>
    <div class="meta">${dial(d.yaw)}
      <b>${d.image} / ${d.cand_id}</b><span class="tag">${d.category}</span>
      <span>yaw ${sgn(d.yaw)} · pitch ${sgn(d.pitch)}</span>
      <span class="tag">${d.desc}</span>
      <span class="pill"><i style="background:${srcColor(d.label_source)}"></i>${srcName(d.label_source)}</span>
      <span class="pill">${d.repair === "gemini" ? "Gemini 補洞" : "inpaint 補洞"}</span>${v}</div></div>`;
}
function jump(d) {
  const el = document.getElementById(`c-${d.image}-${d.cand_id}`);
  if (!el) return;
  el.scrollIntoView({behavior: "smooth", block: "start"});
  el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1500);
}
function render() {
  const f = filters();
  const base = DATA.filter(d => match(d, f, true));
  let rows = DATA.filter(d => match(d, f, false));
  scatter(rows, f); heat(base, f); bars(rows);
  if ($("f-sort").value === "bin")
    rows = [...rows].sort((a, b) => a.bin[0] - b.bin[0] || b.bin[1] - a.bin[1] || a.image.localeCompare(b.image));
  $("grid").innerHTML = rows.map(card).join("");
  $("count").textContent = `${rows.length} / ${DATA.length} 個 sample`;
}
document.querySelectorAll("header select").forEach(s => s.onchange = render);
render();
</script></body></html>
"""


def arrow_start(face, target, gap):
    """Start a face->target arrow outside the head so the eyes stay
    unobstructed; None if the target itself is within that radius."""
    v = np.subtract(target, face, dtype=float)
    n = float(np.hypot(*v))
    if n <= gap:
        return None
    return tuple(int(round(c)) for c in np.add(face, v / n * gap))


def draw_direction(bgr, face, target, gap):
    out = bgr.copy()
    lw = max(2, out.shape[1] // 250)
    a0 = arrow_start(face, target, gap)
    if a0:
        cv2.arrowedLine(out, a0, tuple(int(v) for v in target), (0, 230, 255), lw, tipLength=0.06)
    cv2.circle(out, tuple(int(v) for v in target), 4 * lw, (0, 0, 255), -1)
    return out


def cmd_gallery(store, args):
    """Write <data>/gallery/index.html: label distribution charts plus
    thumbnails of every generated candidate (defaults to the accepted set)."""
    out = store.root / "gallery"
    thumbs = out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    reviews = store.reviews()
    items = [(d, m) for d, m in store.candidates()
             if m["status"] == "done" and (d / "07_final_inpaint.png").exists()]
    print(f"[Gallery] {len(items)} candidate(s).")

    def thumb(bgr, path):
        h, w = bgr.shape[:2]
        s = args.thumb_height / h
        cv2.imwrite(str(path), cv2.resize(bgr, (int(w * s), args.thumb_height),
                                          interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    def one(d, m):
        stem = f"{m['image']}_{m['cand_id']}"
        t = {k: thumbs / f"{stem}_{k}.jpg" for k in ("src", "fin", "lab")}
        fin, repair = gt_image(d)
        newest = max(fin.stat().st_mtime, (d / "meta.json").stat().st_mtime,
                     (store.prep / m["image"] / "prep.json").stat().st_mtime)
        if args.redo or not all(p.exists() for p in t.values()) \
                or min(p.stat().st_mtime for p in t.values()) < newest:
            info = store.prep_info(m["image"]) or {}
            gap = max(25.0, 1.6 * float(info.get("head_r", 40)))
            lab = label_of(m)
            result = cv2.imread(str(fin))
            src = cv2.imread(str(store.prep / m["image"] / "image.png"))
            sl = info.get("source_label")
            if sl:
                src = draw_direction(src, sl["face_px"], sl["target_px"], gap)
            thumb(src, t["src"])
            thumb(result, t["fin"])
            thumb(draw_direction(result, lab["face_px"], lab["target_px"], gap), t["lab"])
        lab = label_of(m)
        g = lab["gaze"]
        sg = ((store.prep_info(m["image"]) or {}).get("source_label") or {}).get("gaze")
        rel = lambda p: Path(os.path.relpath(p, out)).as_posix()
        return {
            "repair": repair,
            "src_yaw": sg["yaw_deg"] if sg else None, "src_pitch": sg["pitch_deg"] if sg else None,
            "image": m["image"], "cand_id": m["cand_id"], "category": m["category"],
            "bin": list(label_bin(m)), "yaw": g["yaw_deg"], "pitch": g["pitch_deg"],
            "label_source": lab["source"],
            "desc": describe_gaze(g["yaw_deg"], g["pitch_deg"]),
            "verdict": reviews.get(cand_key(m), {}).get("verdict"),
            "t_src": rel(t["src"]), "t_fin": rel(t["fin"]), "t_lab": rel(t["lab"]),
            "f_src": rel(store.prep / m["image"] / "image.png"), "f_fin": rel(fin),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda dm: one(*dm), items))
    rows.sort(key=lambda r: (r["image"], r["cand_id"]))
    html = (GALLERY_HTML
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False))
            .replace("__YAWS__", json.dumps([yaw_bin_center(b) for b in range(N_YAW_BINS)]))
            .replace("__PITCHES__", json.dumps(PITCH_NAMES))
            .replace("__PEDGES__", json.dumps(PITCH_EDGES)))
    (out / "index.html").write_text(html, encoding="utf-8")
    for old in thumbs.glob("*_raw.jpg"):          # thumbnails from the previous layout
        old.unlink()
    print(f"[Gallery] -> {out / 'index.html'}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. FINALIZE
# ══════════════════════════════════════════════════════════════════════════════

def view_person_mask(image, raw, sam_mask):
    """
    Person mask (0/255) of a generated view, for export. Starts from the
    compositing foreground, then adds pixels inside the original person's area
    that Gemini drew but that fall under the compositing threshold (very dark
    clothing/hair, max channel 4-8) and fills enclosed holes, so a dark hoodie
    doesn't come out as a sieve. Other people cut out of the source mask stay
    excluded, since those regions open onto the background.
    """
    _, _, fg = composite_onto_background(image, raw, sam_mask, return_fg=True)
    h, w = image.shape[:2]
    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LANCZOS4)
    raw = remove_red_marker(raw)
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    dark = ((raw.max(axis=2) > 3) & (cv2.dilate(sam_mask, k(15)) > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k(5))
    m = cv2.morphologyEx(cv2.bitwise_or(fg, dark), cv2.MORPH_CLOSE, k(15))
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = cv2.bitwise_or(m, cv2.bitwise_not(flood))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(200, 0.02 * stats[1:, cv2.CC_STAT_AREA].max())
        m = np.where(keep[labels], 255, 0).astype(np.uint8)
    return m


def cmd_finalize(store, args):
    """
    Export the dataset to <data>/final/ as scenes, not (source, target) pairs:
    each scene is one photo and holds every view of its person — the original
    plus each accepted generated edit — with that view's gaze label and person
    mask. Any two views of a scene form a pair, in either direction.

      final/<image>/views/<view_id>.png   view_id = "orig" or the candidate id
      final/<image>/masks/<view_id>.png   person mask of that view
      final/manifest.json                 scenes -> views with labels
      final/views.csv                     one row per view
      final/pairs.csv                     every ordered pair within a scene

    Generated views use the Gemini gap repair if `repair` produced one, else
    the screening inpaint. No API calls.
    """
    import csv
    import shutil

    accepted = accepted_candidates(store)
    if args.require_gemini:
        before = len(accepted)
        accepted = [(d, m) for d, m in accepted if gt_image(d)[1] == "gemini"]
        print(f"[Finalize] --require-gemini: {before - len(accepted)} without Gemini repair left out.")
    by_image = {}
    for d, m in accepted:
        by_image.setdefault(m["image"], []).append((d, m))

    def gaze_fields(g):
        return {k: g[k] for k in ("dx", "dy", "dz", "yaw_deg", "pitch_deg")}

    def export_scene(name):
        info = store.prep_info(name)
        image, sam_mask, _, _ = store.load_prep_arrays(name)
        out = store.final / name
        if out.exists():
            shutil.rmtree(out)
        (out / "views").mkdir(parents=True)
        (out / "masks").mkdir()
        views = []
        sl = info.get("source_label")
        if sl:
            cv2.imwrite(str(out / "views" / "orig.png"), image)
            cv2.imwrite(str(out / "masks" / "orig.png"), sam_mask)
            views.append({
                "view_id": "orig", "kind": "original", "repair": None,
                **gaze_fields(sl["gaze"]), "label_source": "manual",
                "face_px": sl["face_px"], "target_px": sl["target_px"],
                "face_depth": sl["face_depth"], "target_depth": sl["target_depth"],
            })
        for d, m in sorted(by_image.get(name, []), key=lambda dm: dm[1]["cand_id"]):
            vid = m["cand_id"]
            tgt, repair = gt_image(d)
            raw = cv2.imread(str(d / "05_gemini_raw.png"))
            (out / "views" / f"{vid}.png").write_bytes(tgt.read_bytes())
            cv2.imwrite(str(out / "masks" / f"{vid}.png"), view_person_mask(image, raw, sam_mask))
            lab = label_of(m)
            views.append({
                "view_id": vid, "kind": "generated", "repair": repair,
                "fix": (load_issues(store).get(cand_key(m), {}).get("tags")
                        if (d / "09_final_fix.png").exists() else None),
                **gaze_fields(lab["gaze"]), "label_source": lab["source"],
                "face_px": lab["face_px"], "target_px": lab["target_px"],
                "face_depth": lab["face_depth"], "target_depth": lab["target_depth"],
                "requested": {"gaze": m["gaze"], "target_px": m["target_px"],
                              "target_depth": m["target_depth"]},
                "prompt": m["prompt"],
            })
        for v in views:
            v["file"] = f"{name}/views/{v['view_id']}.png"
            v["mask"] = f"{name}/masks/{v['view_id']}.png"
        return {"image": name, "category": info["category"], "size": info["size"],
                "views": views}

    names = sorted(set(by_image) | {n for n in store.prepped_names()
                                    if (store.prep_info(n) or {}).get("source_label")})
    no_src = [n for n in names if not (store.prep_info(n) or {}).get("source_label")]
    if no_src:
        print(f"[Finalize] WARNING: no source gaze label (orig view left out) for "
              f"{len(no_src)} image(s): {', '.join(no_src)} — run `label-source`")

    store.final.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        scenes = list(ex.map(export_scene, names))
    scenes = [sc for sc in scenes if len(sc["views"]) >= 2]
    keep = {sc["image"] for sc in scenes}
    for old in store.final.iterdir():                 # scenes from an earlier export
        if old.is_dir() and old.name not in keep and (old / "views").exists():
            shutil.rmtree(old)

    convention = ("dx>0 image-right, dy>0 up, dz>0 toward camera; yaw=atan2(dx,dz), "
                  "pitch=asin(dy). Each view's gaze is the direction from its face point to a "
                  "target point at a chosen depth (Z ~ 1/depth, f = 0.8*width), all in that "
                  "view's own image. kind: 'original' = the real photo, 'generated' = a Gemini "
                  "edit of it (training-free pipeline). label_source: 'manual' = set by a "
                  "reviewer to where the person looks, 'requested' = the target Gemini was "
                  "asked for and judged correct. repair = gap fill of a generated view. "
                  "Any two views of a scene form a pair, in either direction.")
    (store.final / "manifest.json").write_text(json.dumps(
        {"convention": convention, "scenes": scenes,
         "n_scenes": len(scenes), "n_views": sum(len(sc["views"]) for sc in scenes)},
        indent=2, ensure_ascii=False), encoding="utf-8")

    vcols = ["image", "category", "view_id", "kind", "file", "mask",
             "dx", "dy", "dz", "yaw_deg", "pitch_deg", "label_source", "repair"]
    with open(store.final / "views.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=vcols, extrasaction="ignore")
        wr.writeheader()
        for sc in scenes:
            for v in sc["views"]:
                wr.writerow({"image": sc["image"], "category": sc["category"], **v})

    n_pairs = 0
    with open(store.final / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "category", "view_a", "view_b", "kind_a", "kind_b",
                     "file_a", "file_b", "angle_deg"])
        for sc in scenes:
            for a in sc["views"]:
                for b in sc["views"]:
                    if a is b:
                        continue
                    va = np.array([a["dx"], a["dy"], a["dz"]])
                    vb = np.array([b["dx"], b["dy"], b["dz"]])
                    wr.writerow([sc["image"], sc["category"], a["view_id"], b["view_id"],
                                 a["kind"], b["kind"], a["file"], b["file"],
                                 round(angular_diff_deg(va, vb), 2)])
                    n_pairs += 1

    n_views = sum(len(sc["views"]) for sc in scenes)
    n_gen = sum(v["kind"] == "generated" for sc in scenes for v in sc["views"])
    n_gem = sum(v["repair"] == "gemini" for sc in scenes for v in sc["views"])
    print(f"[Finalize] {len(scenes)} scene(s), {n_views} view(s) "
          f"({n_views - n_gen} original, {n_gen} generated: {n_gem} Gemini-repaired, "
          f"{n_gen - n_gem} inpaint), {n_pairs} ordered pair(s) -> {store.final}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  7. EXPORT FOR HUGGING FACE
# ══════════════════════════════════════════════════════════════════════════════

HF_CARD = """---
license: cc-by-nc-4.0
pretty_name: __NAME__
task_categories:
- image-to-image
tags:
- gaze
- gaze-redirection
- image-editing
- evaluation
size_categories:
- n<1K
configs:
- config_name: views
  default: true
  data_dir: views
---

# __NAME__

An evaluation set for **gaze redirection / gaze-controlled image editing** on in-the-wild
photos. Each *scene* is one photograph with one target person, plus several edited
*views* of the same photo in which that person looks in a different, labelled 3D
direction. Any two views of a scene form an evaluation pair (source -> target), in
either direction.

| | |
|---|---|
| scenes (source photos) | __N_SCENES__ |
| views (incl. originals) | __N_VIEWS__ (__N_ORIG__ original + __N_GEN__ edited) |
| ordered pairs | __N_PAIRS__ |
| categories | __CATS__ |

## How it was made

1. 50 photos from [Pexels](https://www.pexels.com/), 10 categories x 5, not drawn from
   COCO, SHHQ or WIDER FACE. One person per photo was segmented (SAM ViT-H) and given a
   depth map (Depth-Anything-V2).
2. For each person, target gazes were planned to cover the yaw/pitch range evenly. A
   red dot (image position + depth) marked each 3D target; the person was edited with
   Gemini (`gemini-3-pro-image-preview`) to look at it and composited back into the photo.
3. Every edit was **reviewed by a person**. Edits were kept only if the result looked
   natural; when the person in the result looked somewhere other than the requested
   target, the label was **moved by hand to where the person actually looks** (2D target
   point + relative depth). The gaze of every original photo was labelled the same way.
4. Background holes left by the edit were filled (OpenCV inpainting, then a Gemini
   repair pass where needed, blended back with tone/grain matching), and a final
   targeted pass fixed views with implausible parts, artifacts or seams. Face points
   were checked against a face detector (insightface).

Labels are therefore **human judgements of the visible gaze**, not measurements.

## Gaze convention

Unit vector `(dx, dy, dz)` in camera space: `dx > 0` = image right, `dy > 0` = up,
`dz > 0` = toward the camera. `yaw = atan2(dx, dz)` (0 = looking at the camera,
+-180 = looking away), `pitch = asin(dy)`. The label is computed from the face point
and a 3D target (pixel + depth relative to the face) with a pinhole model, focal length
0.8 x image width.

__DIST__

## Files

```
views/<scene>/<view_id>.png     view_id = "orig" or an edit id (r1_00, ...)
views/metadata.csv              one row per view (labels; also drives the HF viewer)
masks/<scene>/<view_id>.png     person mask of that view (0/255)
pairs.csv                       every ordered pair within a scene + angle between gazes
manifest.json                   scenes -> views, with labels, face/target points,
                                requested target and the edit prompt
```

`views/metadata.csv` columns: `file_name, scene, category, view_id, kind
(original|generated), dx, dy, dz, yaw_deg, pitch_deg, face_x, face_y, target_x,
target_y, label_source (manual|requested), repair (null|gemini|inpaint), mask`.

```python
from datasets import load_dataset
views = load_dataset("__REPO__", split="train")            # images + labels
pairs = load_dataset("csv", data_files="hf://datasets/__REPO__/pairs.csv", split="train")
```

Or download the whole folder (`huggingface-cli download __REPO__ --repo-type dataset
--local-dir GazeCtrl-Eval`) and read `manifest.json`.

Images keep the resolution of the source (longest side 1500 px). Edited views may carry
Gemini's SynthID watermark.

## Limitations

- Labels are manual and approximate (the depth component especially); compare methods
  with an angular tolerance rather than exact agreement.
- The distribution is skewed toward side views; a few yaw/pitch bins are empty.
- Edited images are generated: hands, accessories and backgrounds may contain small
  inconsistencies that were judged acceptable.

## License and source images

The edited views, masks and labels are released under **CC BY-NC 4.0**. The source
photographs come from Pexels and remain under the
[Pexels license](https://www.pexels.com/license/); they are included only to make the
evaluation pairs usable. Do not use the images to identify, profile or portray the
people shown in a negative way, and do not redistribute the photographs as stock images.
If you appear in one of the photos and want it removed, please open a discussion on
this repository.
"""


def cmd_export_hf(store, args):
    """
    Build an upload-ready Hugging Face dataset folder from final/ (run
    `finalize` first). Layout: views/<scene>/<view>.png + views/metadata.csv,
    masks/, pairs.csv, manifest.json, README.md (dataset card). No upload.
    """
    import csv
    import shutil
    from collections import Counter

    man = json.loads((store.final / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else store.root / "release"
    if out.exists():
        shutil.rmtree(out)
    (out / "views").mkdir(parents=True)
    rows, n_pairs = [], 0
    for sc in man["scenes"]:
        name = sc["image"]
        (out / "views" / name).mkdir()
        (out / "masks" / name).mkdir(parents=True)
        for v in sc["views"]:
            vid = v["view_id"]
            shutil.copy2(store.final / v["file"], out / "views" / name / f"{vid}.png")
            shutil.copy2(store.final / v["mask"], out / "masks" / name / f"{vid}.png")
            v["file"] = f"views/{name}/{vid}.png"
            v["mask"] = f"masks/{name}/{vid}.png"
            rows.append({
                "file_name": f"{name}/{vid}.png", "scene": name, "category": sc["category"],
                "view_id": vid, "kind": v["kind"],
                **{k: v[k] for k in ("dx", "dy", "dz", "yaw_deg", "pitch_deg")},
                "face_x": round(v["face_px"][0], 2), "face_y": round(v["face_px"][1], 2),
                "target_x": round(v["target_px"][0], 2), "target_y": round(v["target_px"][1], 2),
                "label_source": v["label_source"], "repair": v.get("repair") or "",
                "mask": v["mask"],
            })
    (out / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "views" / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(store.final / "pairs.csv", encoding="utf-8") as fi, \
            open(out / "pairs.csv", "w", newline="", encoding="utf-8") as fo:
        rd = csv.DictReader(fi)
        wr = csv.DictWriter(fo, fieldnames=["scene", "category", "view_a", "view_b", "kind_a",
                                            "kind_b", "file_a", "file_b", "angle_deg"])
        wr.writeheader()
        for r in rd:
            wr.writerow({"scene": r["image"], "category": r["category"],
                         "view_a": r["view_a"], "view_b": r["view_b"],
                         "kind_a": r["kind_a"], "kind_b": r["kind_b"],
                         "file_a": f"views/{r['image']}/{r['view_a']}.png",
                         "file_b": f"views/{r['image']}/{r['view_b']}.png",
                         "angle_deg": r["angle_deg"]})
            n_pairs += 1

    # distribution table for the card: 12 yaw bins x 3 pitch bins
    grid = Counter()
    for r in rows:
        yb = yaw_bin(float(r["yaw_deg"]))
        p = float(r["pitch_deg"])
        pb = 0 if p < PITCH_EDGES[1] else (2 if p >= PITCH_EDGES[2] else 1)
        grid[(pb, yb)] += 1
    yaws = [yaw_bin_center(b) for b in range(N_YAW_BINS)]
    dist = ["Views per gaze bin (yaw bin centre x pitch; all views):", "",
            "| pitch \\ yaw | " + " | ".join(f"{y:g}" for y in yaws) + " |",
            "|---|" + "---|" * len(yaws)]
    for pb, pname in ((2, "up (>= 15)"), (1, "level"), (0, "down (< -15)")):
        dist.append(f"| {pname} | " + " | ".join(str(grid[(pb, yb)]) for yb in range(N_YAW_BINS)) + " |")
    cats = Counter(sc["category"] for sc in man["scenes"])
    n_orig = sum(r["kind"] == "original" for r in rows)
    card = (HF_CARD.replace("__NAME__", args.name).replace("__REPO__", args.repo or f"<user>/{args.name}")
            .replace("__N_SCENES__", str(len(man["scenes"]))).replace("__N_VIEWS__", str(len(rows)))
            .replace("__N_ORIG__", str(n_orig)).replace("__N_GEN__", str(len(rows) - n_orig))
            .replace("__N_PAIRS__", str(n_pairs))
            .replace("__CATS__", ", ".join(f"{c} ({n})" for c, n in sorted(cats.items())))
            .replace("__DIST__", "\n".join(dist)))
    (out / "README.md").write_text(card, encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2 ** 20
    print(f"[Export HF] {len(man['scenes'])} scenes, {len(rows)} views, {n_pairs} pairs, "
          f"{size:.0f} MB -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Eval-set GT generator (training-free pipeline)")
    ap.add_argument("--data", default=str(PROJECT_ROOT / "eval_dataset"),
                    help="Eval dataset root containing raw/ (default: ../eval_dataset)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="interactive mask + face point per image")
    p.add_argument("--images", nargs="+", metavar="STEM")
    p.add_argument("--redo", action="store_true", help="re-prep images that are already done")

    p = sub.add_parser("plan", help="choose gaze targets for a new round (no API calls)")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--per-image", type=int, default=10)
    p.add_argument("--images", nargs="+", metavar="STEM")
    p.add_argument("--seed", type=int)
    p.add_argument("--force", action="store_true", help="plan again even if this round exists")

    p = sub.add_parser("run", help="send planned/failed candidates to Gemini")
    p.add_argument("--round", type=int)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, help="only run the first N (for a trial run)")
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("recomposite", help="rebuild composites from saved Gemini output (no API)")
    p.add_argument("--round", type=int)
    p.add_argument("--workers", type=int, default=6)

    p = sub.add_parser("review", help="keyboard review UI")
    p.add_argument("--round", type=int)
    p.add_argument("--images", nargs="+", metavar="STEM")
    p.add_argument("--all", action="store_true", help="include already-reviewed candidates")

    sub.add_parser("stats", help="bin coverage summary")

    p = sub.add_parser("gallery", help="static HTML overview of all generated candidates")
    p.add_argument("--thumb-height", type=int, default=420)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--redo", action="store_true", help="rebuild all thumbnails")

    p = sub.add_parser("label-source", help="label the gaze of the person in each source photo")
    p.add_argument("--images", nargs="+", metavar="STEM")
    p.add_argument("--all", action="store_true", help="include already-labelled images")

    p = sub.add_parser("repair-select", help="inspect gap fills, mark views for Gemini repair")
    p.add_argument("--images", nargs="+", metavar="STEM")
    p.add_argument("--all", action="store_true", help="include already-inspected views")

    p = sub.add_parser("repair", help="Gemini gap repair for accepted candidates")
    p.add_argument("--selected", action="store_true",
                   help="only views marked 'fix' in repair-select")
    p.add_argument("--redo", action="store_true", help="repair again even if already repaired")
    p.add_argument("--merge", choices=["feather", "full", "gap"], default="feather",
                   help="how Gemini's response is combined (see merge_repair)")
    p.add_argument("--min-gap-pct", type=float, default=0.0,
                   help="keep inpaint when the gap is below this %% of the image")
    p.add_argument("--ids", nargs="+", metavar="ID",
                   help="only these (image/cand or image_cand), re-repairing if done")
    p.add_argument("--limit", type=int, help="only the first N (for a trial run)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("remerge", help="re-merge saved Gemini responses (no API calls)")
    p.add_argument("--merge", choices=["feather", "full", "gap"], required=True)
    p.add_argument("--ids", nargs="+", metavar="ID")

    p = sub.add_parser("repair-view", help="sent image | Gemini response | merged result")
    p.add_argument("--ids", nargs="+", metavar="ID")

    p = sub.add_parser("merge-select", help="pick Gemini full image or feather merge per view")
    p.add_argument("--ids", nargs="+", metavar="ID")
    p.add_argument("--all", action="store_true", help="start from the first view, not the first unpicked")

    p = sub.add_parser("issue-tag", help="tag problems (implausible / artifact / seam) + paint the area")
    p.add_argument("--ids", nargs="+", metavar="ID")
    p.add_argument("--unpicked", action="store_true",
                   help="add Gemini-repaired views not picked in merge-select")

    p = sub.add_parser("fix", help="targeted Gemini fix of the views tagged in issue-tag")
    p.add_argument("--ids", nargs="+", metavar="ID")
    p.add_argument("--redo", action="store_true", help="fix again even if unchanged")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--show-prompt", action="store_true", help="print the first prompt before asking")
    p.add_argument("--ref-original", action="store_true",
                   help="also send the original photo as a reference for lost objects")

    p = sub.add_parser("fix-view", help="before | Gemini response | result; pick region / full / revert")
    p.add_argument("--ids", nargs="+", metavar="ID")

    p = sub.add_parser("unfix", help="remove the targeted fix of some views")
    p.add_argument("ids", nargs="+", metavar="ID")

    p = sub.add_parser("face-check", help="flag label face points that are off the face (insightface)")
    p.add_argument("--rel", type=float, default=0.5,
                   help="flag if farther than this x face size from the detected eyes")

    p = sub.add_parser("face-fix", help="move flagged face points (target moves along)")
    p.add_argument("--ids", nargs="+", metavar="ID")
    p.add_argument("--all", action="store_true")

    p = sub.add_parser("export-hf", help="build an upload-ready Hugging Face dataset folder from final/")
    p.add_argument("--out", help="output folder (default: <data>/release)")
    p.add_argument("--name", default="GazeCtrl-Eval", help="dataset name in the card")
    p.add_argument("--repo", help="HF repo id for the loading example, e.g. user/GazeCtrl-Eval")

    p = sub.add_parser("unrepair", help="drop the Gemini repair of some samples (back to inpaint)")
    p.add_argument("ids", nargs="+", metavar="ID")

    p = sub.add_parser("finalize", help="export scenes of labelled views to final/ (no API calls)")
    p.add_argument("--require-gemini", action="store_true",
                   help="leave out generated views without a Gemini gap repair")
    p.add_argument("--workers", type=int, default=6)

    args = ap.parse_args()
    store = Store(args.data)
    {"prep": cmd_prep, "plan": cmd_plan, "run": cmd_run, "recomposite": cmd_recomposite,
     "review": cmd_review,
     "stats": cmd_stats, "gallery": cmd_gallery, "label-source": cmd_label_source,
     "repair-select": cmd_repair_select, "repair": cmd_repair, "remerge": cmd_remerge,
     "repair-view": cmd_repair_view, "merge-select": cmd_merge_select, "unrepair": cmd_unrepair,
     "face-check": cmd_face_check, "export-hf": cmd_export_hf, "face-fix": cmd_face_fix,
     "issue-tag": cmd_issue_tag, "fix": cmd_fix, "fix-view": cmd_fix_view, "unfix": cmd_unfix,
     "finalize": cmd_finalize}[args.cmd](store, args)


if __name__ == "__main__":
    main()
