"""
aggregate_results.py — Parse all summary.json files and produce a ranked leaderboard.
Usage: python aggregate_results.py --results-dir results_final
"""
import json, argparse, pathlib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_final")
    args = parser.parse_args()

    base = pathlib.Path(args.results_dir)
    rows = []

    for d in sorted(base.iterdir()):
        s = d / "summary.json"
        if not s.exists():
            continue
        data = json.loads(s.read_text(encoding="utf-8"))
        parts = d.name.replace("res_", "").split("_", 2)
        warp = parts[0]
        vector = parts[1]
        prompt = parts[2] if len(parts) > 2 else "unknown"
        rows.append({
            "warp": warp,
            "vector": vector,
            "prompt": prompt,
            "mean": round(data.get("mean_error_deg", 999), 2),
            "median": round(data.get("median_error_deg", 999), 2),
            "std": round(data.get("std_error_deg", 999), 2),
            "min": round(data.get("min_error_deg", 999), 2),
            "max": round(data.get("max_error_deg", 999), 2),
            "valid": data.get("valid_results", 0),
            "failed": data.get("failed", 0),
        })

    if not rows:
        print("[Error] No summary.json files found.")
        return

    rows.sort(key=lambda x: x["mean"])

    # Save JSON leaderboard
    out_json = base / "leaderboard.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print table
    print(f"\n{'#':<4} {'Warp':<6} {'Vector':<9} {'Prompt':<20} "
          f"{'Mean':>6} {'Med':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'OK':>4} {'Fail':>4}")
    print("-" * 90)
    for i, r in enumerate(rows, 1):
        print(f"{i:<4} {r['warp']:<6} {r['vector']:<9} {r['prompt']:<20} "
              f"{r['mean']:>6.1f} {r['median']:>6.1f} {r['std']:>6.1f} "
              f"{r['min']:>6.1f} {r['max']:>6.1f} {r['valid']:>4} {r['failed']:>4}")

    # Summary stats
    best = rows[0]
    worst = rows[-1]
    means = [r["mean"] for r in rows]
    print(f"\n{'='*50}")
    print(f"  Total combinations: {len(rows)}")
    print(f"  Overall mean error: {np.mean(means):.1f}°")
    print(f"  🏆 BEST:   {best['warp']}/{best['vector']}/{best['prompt']} → {best['mean']}°")
    print(f"  💀 WORST:  {worst['warp']}/{worst['vector']}/{worst['prompt']} → {worst['mean']}°")
    print(f"{'='*50}")

    # Dimension-wise analysis
    print("\n── By Warp Mode ──")
    for w in sorted(set(r["warp"] for r in rows)):
        sub = [r["mean"] for r in rows if r["warp"] == w]
        print(f"  {w:<8} avg={np.mean(sub):.1f}°  best={min(sub):.1f}°")

    print("\n── By Vector ──")
    for v in ["novector", "vector"]:
        sub = [r["mean"] for r in rows if r["vector"] == v]
        print(f"  {v:<10} avg={np.mean(sub):.1f}°  best={min(sub):.1f}°")

    print("\n── By Prompt ──")
    for p in sorted(set(r["prompt"] for r in rows)):
        sub = [r["mean"] for r in rows if r["prompt"] == p]
        print(f"  {p:<22} avg={np.mean(sub):.1f}°  best={min(sub):.1f}°")

    print(f"\nLeaderboard saved: {out_json}")


if __name__ == "__main__":
    main()
