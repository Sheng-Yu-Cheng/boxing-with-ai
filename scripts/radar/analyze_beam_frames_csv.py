#!/usr/bin/env python3
"""Analyze RadarBox beam frames.csv by beam command.

Use this after a beam switching run. It groups stable segments by beam_cmd and
prints median/mean metrics. It can ignore a few frames after each beam change.

Examples:
  python analyze_beam_frames_csv.py --csv C:\temp\radarbox_beam_test\frames.csv
  python analyze_beam_frames_csv.py --csv C:\temp\radarbox_beam_test\frames.csv --range-min 5.3 --range-max 6.1
  python analyze_beam_frames_csv.py --csv C:\temp\radarbox_beam_test\frames.csv --metric static_roi_power_db
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median, mean


def to_float(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mark_stable(rows, discard_after_change: int):
    prev = None
    since_change = 10**9
    for r in rows:
        cmd = (r.get("beam_cmd") or "").strip()
        if cmd != prev:
            since_change = 0
            prev = cmd
        else:
            since_change += 1
        r["_stable"] = since_change >= discard_after_change
        r["_since_change"] = since_change
    return rows


def passes_filter(r, args):
    if (r.get("valid") or "").lower() not in ("true", "1", "yes"):
        return False
    if not r.get("beam_cmd"):
        return False
    if not r.get("_stable"):
        return False

    if args.range_min is not None or args.range_max is not None:
        rr = to_float(r.get(args.range_field))
        if rr is None:
            return False
        if args.range_min is not None and rr < args.range_min:
            return False
        if args.range_max is not None and rr > args.range_max:
            return False
    return True


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values_sorted = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "p10": values_sorted[max(0, int(0.10 * (len(values_sorted)-1)))],
        "p90": values_sorted[min(len(values_sorted)-1, int(0.90 * (len(values_sorted)-1)))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="frames.csv from logger")
    p.add_argument("--metric", default="rd_peak_snr_like_db", help="metric column to compare")
    p.add_argument("--discard-after-change", type=int, default=20, help="drop this many frames after each beam change")
    p.add_argument("--range-field", default="rd_peak_range_m", help="range column for optional filtering")
    p.add_argument("--range-min", type=float, default=None)
    p.add_argument("--range-max", type=float, default=None)
    p.add_argument("--out", default="", help="optional output summary csv")
    args = p.parse_args()

    rows = mark_stable(read_rows(Path(args.csv)), args.discard_after_change)
    grouped = defaultdict(list)
    segment_counts = defaultdict(int)
    prev_cmd = None
    for r in rows:
        cmd = (r.get("beam_cmd") or "").strip()
        if cmd and cmd != prev_cmd:
            segment_counts[cmd] += 1
            prev_cmd = cmd
        if passes_filter(r, args):
            grouped[cmd].append(to_float(r.get(args.metric)))

    print(f"CSV: {args.csv}")
    print(f"metric: {args.metric}")
    print(f"discard_after_change: {args.discard_after_change} frames")
    if args.range_min is not None or args.range_max is not None:
        print(f"range filter: {args.range_field} in [{args.range_min}, {args.range_max}] m")
    print()

    records = []
    for cmd in sorted(grouped.keys()):
        s = summarize(grouped[cmd])
        if not s:
            continue
        rec = {"beam_cmd": cmd, "segments": segment_counts.get(cmd, 0), **s}
        records.append(rec)
        print(
            f"beam={cmd:10s} segments={rec['segments']:2d} n={rec['n']:5d} "
            f"median={rec['median']:8.3f} mean={rec['mean']:8.3f} "
            f"p10={rec['p10']:8.3f} p90={rec['p90']:8.3f} min={rec['min']:8.3f} max={rec['max']:8.3f}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            fields = ["beam_cmd", "segments", "n", "mean", "median", "min", "max", "p10", "p90"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rec in records:
                w.writerow(rec)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
