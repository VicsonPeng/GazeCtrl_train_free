"""
Phase 3 — Build Training Dataset
==================================
Reads all metadata.json files from dataset_pipeline.py output,
applies quality filtering, copies training-relevant files,
and generates a manifest.csv for use in model training.

Output structure
----------------
  <output>/
    {sample_id}/
      source_person.png     ← isolated person before Gemini edit  (04a_isolated_person.png)
      result_full.png       ← final composited image               (07_final.png)
      background.png        ← background without person            (03_background.png)
    manifest.csv            ← flat table with all training labels
    stats.json              ← dataset statistics

Usage
-----
  # Dry run — see stats without copying files
  python build_training_set.py --results-dir e_results --dry-run

  # Build with default threshold (angular error ≤ 20°)
  python build_training_set.py --results-dir e_results --output training_data

  # Custom threshold
  python build_training_set.py --results-dir e_results --output training_data --max-error 15.0

  # Include samples without gaze prediction (--model none was used)
  python build_training_set.py --results-dir e_results --output training_data --no-filter
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# CSV columns written to manifest.csv
CSV_COLUMNS = [
    "sample_id",
    "source_person",          # relative path inside output/
    "result_full",
    "background",
    "target_dx", "target_dy", "target_dz",
    "target_yaw_deg", "target_pitch_deg",
    "predicted_dx", "predicted_dy", "predicted_dz",
    "predicted_yaw_deg", "predicted_pitch_deg",
    "angular_error_deg",
    "quality_flag",
    "gaze_target_px_x", "gaze_target_px_y",
    "chosen_depth",
    "person_depth_median",
    "sam_mask_coverage_pct",
    "face_center_px_x", "face_center_px_y",
    "repair_mode",
]


def load_all_metadata(results_dir: Path):
    metas = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        mf = d / "metadata.json"
        if mf.exists():
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
                m["_sample_dir"] = d
                metas.append(m)
            except Exception as e:
                print(f"  [Warning] Cannot read {mf}: {e}")
    return metas


def is_good_sample(meta: dict, max_error: float, no_filter: bool) -> tuple[bool, str]:
    """Returns (accept, reason_if_rejected)."""
    # Must have a final image
    sample_dir = meta["_sample_dir"]
    if not (sample_dir / "07_final.png").exists():
        return False, "missing final image"

    if no_filter:
        return True, ""

    # Must have target gaze
    if not meta.get("target_gaze"):
        return False, "no target_gaze"

    # No face detected by gaze model
    if meta.get("quality_flag") == "no_face":
        return False, "no face in final image"

    # Angular error filter
    err = meta.get("angular_error_deg")
    if err is None:
        # No prediction yet — accept but flag
        return True, ""
    if err > max_error:
        return False, f"angular_error {err:.1f}° > {max_error}°"

    return True, ""


def build_row(meta: dict, output_dir: Path) -> dict:
    sid = meta["sample_id"]
    tg  = meta.get("target_gaze") or {}
    pg  = meta.get("predicted_gaze") or {}
    gtp = meta.get("gaze_target_px", [None, None])
    fcp = meta.get("face_center_px", [None, None])

    return {
        "sample_id":             sid,
        "source_person":         f"{sid}/source_person.png",
        "result_full":           f"{sid}/result_full.png",
        "background":            f"{sid}/background.png",
        "target_dx":             tg.get("dx", ""),
        "target_dy":             tg.get("dy", ""),
        "target_dz":             tg.get("dz", ""),
        "target_yaw_deg":        tg.get("yaw_deg", ""),
        "target_pitch_deg":      tg.get("pitch_deg", ""),
        "predicted_dx":          pg.get("dx", ""),
        "predicted_dy":          pg.get("dy", ""),
        "predicted_dz":          pg.get("dz", ""),
        "predicted_yaw_deg":     pg.get("yaw_deg", ""),
        "predicted_pitch_deg":   pg.get("pitch_deg", ""),
        "angular_error_deg":     meta.get("angular_error_deg", ""),
        "quality_flag":          meta.get("quality_flag", ""),
        "gaze_target_px_x":      gtp[0] if len(gtp) > 0 else "",
        "gaze_target_px_y":      gtp[1] if len(gtp) > 1 else "",
        "chosen_depth":          meta.get("chosen_depth", ""),
        "person_depth_median":   meta.get("person_depth_median", ""),
        "sam_mask_coverage_pct": meta.get("sam_mask_coverage_pct", ""),
        "face_center_px_x":      fcp[0] if len(fcp) > 0 else "",
        "face_center_px_y":      fcp[1] if len(fcp) > 1 else "",
        "repair_mode":           meta.get("repair_mode", ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Filter and package training dataset")
    parser.add_argument(
        "--results-dir", type=str, default="e_results",
        help="Directory from dataset_pipeline.py (default: e_results)")
    parser.add_argument(
        "--output", type=str, default="training_data",
        help="Output directory for the training dataset (default: training_data)")
    parser.add_argument(
        "--max-error", type=float, default=20.0,
        help="Maximum allowed angular error in degrees (default: 20.0)")
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Include all samples regardless of angular error")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without copying any files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[Error] Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir = Path(args.output)

    # ── Load ──────────────────────────────────────────────────────────────────
    all_meta = load_all_metadata(results_dir)
    print(f"[Phase 3] Loaded {len(all_meta)} samples from {results_dir}")

    # ── Filter ────────────────────────────────────────────────────────────────
    accepted, rejected = [], []
    for m in all_meta:
        ok, reason = is_good_sample(m, args.max_error, args.no_filter)
        if ok:
            accepted.append(m)
        else:
            rejected.append((m["sample_id"], reason))

    print(f"  Accepted: {len(accepted)}   Rejected: {len(rejected)}")
    if rejected:
        print("  Rejection breakdown:")
        from collections import Counter
        for reason, count in Counter(r for _, r in rejected).most_common():
            print(f"    {count:4d}  {reason}")

    # ── Statistics on accepted ────────────────────────────────────────────────
    errors = [m["angular_error_deg"] for m in accepted
              if m.get("angular_error_deg") is not None]
    stats = {
        "total_input":    len(all_meta),
        "accepted":       len(accepted),
        "rejected":       len(rejected),
        "max_error_threshold": None if args.no_filter else args.max_error,
    }
    if errors:
        sorted_e = sorted(errors)
        stats.update({
            "angular_error_mean":   round(sum(errors) / len(errors), 2),
            "angular_error_median": round(sorted_e[len(sorted_e) // 2], 2),
            "angular_error_p90":    round(sorted_e[int(len(sorted_e) * 0.9)], 2),
            "angular_error_max":    round(max(errors), 2),
        })
        print(f"\n  Angular error on accepted samples:")
        print(f"    mean   = {stats['angular_error_mean']:.1f}°")
        print(f"    median = {stats['angular_error_median']:.1f}°")
        print(f"    p90    = {stats['angular_error_p90']:.1f}°")
        print(f"    max    = {stats['angular_error_max']:.1f}°")

    if args.dry_run:
        print("\n[Dry run] No files copied.")
        return

    # ── Copy files ────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for meta in accepted:
        sid = meta["sample_id"]
        src_dir = meta["_sample_dir"]
        dst_dir = output_dir / sid
        dst_dir.mkdir(exist_ok=True)

        # source_person: isolated person before Gemini edit
        _copy(src_dir / "04a_isolated_person.png", dst_dir / "source_person.png")
        # result_full: final composited image
        _copy(src_dir / "07_final.png",             dst_dir / "result_full.png")
        # background: scene without person
        _copy(src_dir / "03_background.png",        dst_dir / "background.png")

        # Also copy metadata for reference
        _copy(src_dir / "metadata.json", dst_dir / "metadata.json")

        rows.append(build_row(meta, output_dir))

    # ── manifest.csv ──────────────────────────────────────────────────────────
    csv_path = output_dir / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ── stats.json ────────────────────────────────────────────────────────────
    stats["rejected_samples"] = [{"id": sid, "reason": r} for sid, r in rejected]
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[Phase 3] Done.  {len(rows)} training pairs in {output_dir.resolve()}")
    print(f"  manifest.csv  →  {csv_path}")


def _copy(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
    else:
        print(f"  [Warning] Missing: {src.name}")


if __name__ == "__main__":
    main()
