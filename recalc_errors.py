"""
recalc_errors.py — Recalculate gaze error using corrected 2D vector method.

Reads all existing evaluation.json files from results_final/ and
recomputes angular error using the new method, comparing against the old.

Usage: python recalc_errors.py [--results-dir results_final]
"""
import json, math, argparse, pathlib
import numpy as np


def angular_error_2d(eye_center, gaze_target, predicted_gaze):
    """
    Compute angular error between eye→target 2D direction and
    L2CS-Net predicted gaze projected to 2D image plane.

    This avoids the virtual pinhole focal length problem.
    """
    # 1. Eye → Target direction (2D image space)
    dx = gaze_target[0] - eye_center[0]   # positive = right
    dy = gaze_target[1] - eye_center[1]   # positive = down (image coords)
    target_vec = np.array([dx, dy], dtype=np.float64)
    norm_t = np.linalg.norm(target_vec)
    if norm_t < 1e-6:
        return 0.0
    target_vec /= norm_t

    # 2. L2CS-Net gaze → 2D image direction
    #    L2CS-Net (Gaze360 convention):
    #      yaw > 0  = looking RIGHT (viewer's perspective)
    #      pitch > 0 = looking UP
    #    Image coords: x-right, y-down
    yaw = predicted_gaze["yaw_rad"]
    pitch = predicted_gaze["pitch_rad"]
    gaze_x = math.sin(yaw)         # positive = right (matches image x)
    gaze_y = -math.sin(pitch)      # negate: pitch>0=up, but image y>0=down
    gaze_vec = np.array([gaze_x, gaze_y], dtype=np.float64)
    norm_g = np.linalg.norm(gaze_vec)
    if norm_g < 1e-6:
        return 0.0
    gaze_vec /= norm_g

    # 3. Angle between the two 2D direction vectors
    cos_angle = np.clip(np.dot(target_vec, gaze_vec), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def angular_error_2d_yaw_flip(eye_center, gaze_target, predicted_gaze):
    """Same but with yaw sign flipped (in case L2CS yaw>0 = LEFT)."""
    dx = gaze_target[0] - eye_center[0]
    dy = gaze_target[1] - eye_center[1]
    target_vec = np.array([dx, dy], dtype=np.float64)
    norm_t = np.linalg.norm(target_vec)
    if norm_t < 1e-6:
        return 0.0
    target_vec /= norm_t

    yaw = predicted_gaze["yaw_rad"]
    pitch = predicted_gaze["pitch_rad"]
    gaze_x = -math.sin(yaw)        # FLIPPED: yaw>0 = left
    gaze_y = -math.sin(pitch)
    gaze_vec = np.array([gaze_x, gaze_y], dtype=np.float64)
    norm_g = np.linalg.norm(gaze_vec)
    if norm_g < 1e-6:
        return 0.0
    gaze_vec /= norm_g

    cos_angle = np.clip(np.dot(target_vec, gaze_vec), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def angular_error_2d_pitch_flip(eye_center, gaze_target, predicted_gaze):
    """Same but with pitch sign flipped (pitch>0 = DOWN)."""
    dx = gaze_target[0] - eye_center[0]
    dy = gaze_target[1] - eye_center[1]
    target_vec = np.array([dx, dy], dtype=np.float64)
    norm_t = np.linalg.norm(target_vec)
    if norm_t < 1e-6:
        return 0.0
    target_vec /= norm_t

    yaw = predicted_gaze["yaw_rad"]
    pitch = predicted_gaze["pitch_rad"]
    gaze_x = math.sin(yaw)
    gaze_y = math.sin(pitch)       # NO negate: pitch>0=down=same as image y
    gaze_vec = np.array([gaze_x, gaze_y], dtype=np.float64)
    norm_g = np.linalg.norm(gaze_vec)
    if norm_g < 1e-6:
        return 0.0
    gaze_vec /= norm_g

    cos_angle = np.clip(np.dot(target_vec, gaze_vec), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def angular_error_2d_both_flip(eye_center, gaze_target, predicted_gaze):
    """Both yaw and pitch flipped."""
    dx = gaze_target[0] - eye_center[0]
    dy = gaze_target[1] - eye_center[1]
    target_vec = np.array([dx, dy], dtype=np.float64)
    norm_t = np.linalg.norm(target_vec)
    if norm_t < 1e-6:
        return 0.0
    target_vec /= norm_t

    yaw = predicted_gaze["yaw_rad"]
    pitch = predicted_gaze["pitch_rad"]
    gaze_x = -math.sin(yaw)
    gaze_y = math.sin(pitch)
    gaze_vec = np.array([gaze_x, gaze_y], dtype=np.float64)
    norm_g = np.linalg.norm(gaze_vec)
    if norm_g < 1e-6:
        return 0.0
    gaze_vec /= norm_g

    cos_angle = np.clip(np.dot(target_vec, gaze_vec), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_final")
    args = parser.parse_args()

    base = pathlib.Path(args.results_dir)

    # Collect all evaluation.json files
    all_evals = []
    for eval_file in sorted(base.rglob("evaluation.json")):
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        if data.get("predicted_gaze") is None:
            continue
        data["_path"] = str(eval_file.relative_to(base))
        data["_experiment"] = eval_file.parent.parent.name
        all_evals.append(data)

    print(f"Found {len(all_evals)} evaluation files\n")

    # Test all 4 sign conventions to find which matches best
    methods = {
        "old_pinhole": lambda d: d["angular_error_deg"],
        "2d_yR_pU": lambda d: angular_error_2d(
            d["eye_center"], d["gaze_target"], d["predicted_gaze"]),
        "2d_yL_pU": lambda d: angular_error_2d_yaw_flip(
            d["eye_center"], d["gaze_target"], d["predicted_gaze"]),
        "2d_yR_pD": lambda d: angular_error_2d_pitch_flip(
            d["eye_center"], d["gaze_target"], d["predicted_gaze"]),
        "2d_yL_pD": lambda d: angular_error_2d_both_flip(
            d["eye_center"], d["gaze_target"], d["predicted_gaze"]),
    }

    # Group by experiment
    experiments = {}
    for d in all_evals:
        exp = d["_experiment"]
        if exp not in experiments:
            experiments[exp] = []
        experiments[exp].append(d)

    # Print per-experiment summary for each method
    print(f"{'Experiment':<45} {'old':>7} {'yR_pU':>7} {'yL_pU':>7} {'yR_pD':>7} {'yL_pD':>7}")
    print("=" * 85)

    method_totals = {m: [] for m in methods}

    for exp_name in sorted(experiments):
        items = experiments[exp_name]
        row = f"{exp_name:<45}"
        for method_name, fn in methods.items():
            errors = [fn(d) for d in items]
            mean_err = np.mean(errors)
            method_totals[method_name].extend(errors)
            row += f" {mean_err:>6.1f}°"
        print(row)

    print("-" * 85)
    row = f"{'OVERALL MEAN':<45}"
    for method_name in methods:
        row += f" {np.mean(method_totals[method_name]):>6.1f}°"
    print(row)

    # Show details for user-specified good/bad examples
    print("\n\n" + "=" * 70)
    print("  DETAILED: User-specified good & bad examples")
    print("=" * 70)

    highlight = [
        ("res_none_vector_contrastive", "image_000012", "GOOD"),
        ("res_none_vector_contrastive", "image_000003", "GOOD"),
        ("res_none_vector_p1_contrastive_gaze", "image_000014", "BAD"),
    ]

    for exp, img, label in highlight:
        matches = [d for d in all_evals if d["_experiment"] == exp and d["image"] == img]
        if not matches:
            print(f"\n  [{label}] {exp}/{img} — NOT FOUND")
            continue
        d = matches[0]
        eye = d["eye_center"]
        tgt = d["gaze_target"]
        pred = d["predicted_gaze"]
        dx, dy = tgt[0] - eye[0], tgt[1] - eye[1]

        print(f"\n  [{label}] {exp}/{img}")
        print(f"    Eye center:   ({eye[0]}, {eye[1]})")
        print(f"    Gaze target:  ({tgt[0]}, {tgt[1]})  →  Δx={dx:+d}, Δy={dy:+d}")
        print(f"    Target dir:   {'RIGHT' if dx>0 else 'LEFT'} {abs(dx)}px, {'DOWN' if dy>0 else 'UP'} {abs(dy)}px")
        print(f"    L2CS predict:  yaw={pred['yaw_deg']:.1f}° pitch={pred['pitch_deg']:.1f}°")
        print(f"    Old error:     {d['angular_error_deg']:.1f}°")

        for method_name, fn in methods.items():
            if method_name == "old_pinhole":
                continue
            err = fn(d)
            print(f"    {method_name}: {err:.1f}°")


if __name__ == "__main__":
    main()
