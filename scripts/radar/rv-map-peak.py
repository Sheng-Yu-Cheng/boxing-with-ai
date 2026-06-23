#!/usr/bin/env python3
r"""
scripts/run_rv_peak_monitor.py

Standalone RV-map peak monitor for RadarBox.

Purpose
-------
Before doing full Vision + Fusion, verify directly on recent RV maps:

    "In the last N seconds, within range <= 3 m, what is the strongest
     negative-velocity Doppler peak?"

This answers the basic radar question:

    Does the RV map actually contain a punch-like approaching velocity
    such as -4 ~ -8 m/s during a straight punch?

Typical command:

    python .\scripts\radar\rv-map-peak.py `
      --window-s 5.0 `
      --range-max 3.0 `
      --min-abs-velocity 2.0 `
      --direction negative `
      --top-k 8 `
      --plot

Run order:
    1. Start this script.
    2. Start DCA1000 capture / StartFrame in mmWave Studio.
    3. Throw straight punches toward the radar.
    4. Watch peak velocity / top candidates.

Notes:
    - This script intentionally inspects RadarAgent's recent frame buffer.
      It is a diagnostic tool, not the final game API.
    - It uses time.perf_counter() timestamps, same as RadarAgent.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from core.radar_agent import RadarAgent, RadarConfig


def make_config(config_cls, **kwargs):
    if not dataclasses.is_dataclass(config_cls):
        return config_cls(**kwargs)
    allowed = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return config_cls(**filtered)


def pow_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def select_velocity_mask(velocity_axis, direction: str, min_abs_velocity: float, max_abs_velocity: float):
    v = np.asarray(velocity_axis)
    mask = np.abs(v) >= min_abs_velocity
    mask &= np.abs(v) <= max_abs_velocity

    if direction == "negative":
        mask &= v < 0
    elif direction == "positive":
        mask &= v > 0
    elif direction == "both":
        pass
    else:
        raise ValueError(f"bad direction: {direction}")

    return mask


def get_recent_frames(radar: RadarAgent, window_s: float):
    now = time.perf_counter()
    t0 = now - window_s

    # Diagnostic script: intentionally read recent processed RV frames.
    with radar._lock:
        frames = [f for f in radar._frame_buffer if f.valid and f.timestamp >= t0]
        frames = list(frames)

    return frames, now


def analyze_recent_rv(
    radar: RadarAgent,
    window_s: float,
    range_min: float,
    range_max: float,
    min_abs_velocity: float,
    max_abs_velocity: float,
    direction: str,
    top_k: int,
    velocity_strong: float,
):
    frames, now = get_recent_frames(radar, window_s)
    if not frames:
        return {
            "valid": False,
            "reason": "no_valid_frames_in_recent_window",
            "frames": 0,
            "top_candidates": [],
        }

    r_axis = radar.range_axis_m
    v_axis = radar.velocity_axis_mps

    r_mask = (r_axis >= range_min) & (r_axis <= range_max)
    v_mask = select_velocity_mask(v_axis, direction, min_abs_velocity, max_abs_velocity)

    r_idx = np.where(r_mask)[0]
    v_idx = np.where(v_mask)[0]

    if len(r_idx) == 0:
        return {"valid": False, "reason": "empty_range_roi", "frames": len(frames), "top_candidates": []}
    if len(v_idx) == 0:
        return {"valid": False, "reason": "empty_velocity_roi", "frames": len(frames), "top_candidates": []}

    # Max-over-time RV map for the recent window.
    # Shape: [range_roi, velocity_roi]
    roi_stack = []
    for f in frames:
        roi_stack.append(f.rd_power[np.ix_(r_idx, v_idx)].astype(np.float64))

    stack = np.stack(roi_stack, axis=0)  # [T, R, V]
    max_power = np.max(stack, axis=0)
    max_frame_idx = np.argmax(stack, axis=0)  # [R, V], local time index that produced max

    vals = max_power[np.isfinite(max_power) & (max_power > 0)]
    if vals.size == 0:
        return {"valid": False, "reason": "no_finite_power", "frames": len(frames), "top_candidates": []}

    noise = float(np.median(vals)) + 1e-12
    noise_db = float(pow_to_db(noise))

    # Candidate list for every ROI bin, sorted by power and by speed-weighted score.
    candidates = []
    for rr_local in range(max_power.shape[0]):
        for vv_local in range(max_power.shape[1]):
            p = float(max_power[rr_local, vv_local]) + 1e-12
            p_db = float(pow_to_db(p))
            snr_db = p_db - noise_db

            rr = int(r_idx[rr_local])
            vv = int(v_idx[vv_local])
            vel = float(v_axis[vv])
            abs_v = abs(vel)

            intensity = (abs_v - min_abs_velocity) / max(velocity_strong - min_abs_velocity, 1e-6)
            intensity = float(np.clip(intensity, 0.0, 1.0))

            # This score intentionally favors fast bins. Power-only often picks body sway.
            score = float(max(snr_db, 0.0) * (0.25 + 0.75 * intensity))

            fi = int(max_frame_idx[rr_local, vv_local])
            frame = frames[fi]

            candidates.append({
                "timestamp": float(frame.timestamp),
                "age_s": float(now - frame.timestamp),
                "range_m": float(r_axis[rr]),
                "velocity_mps": vel,
                "abs_velocity_mps": abs_v,
                "power_db": p_db,
                "snr_db": float(snr_db),
                "intensity_score": intensity,
                "score": score,
            })

    by_power = sorted(candidates, key=lambda c: c["power_db"], reverse=True)[:top_k]
    by_score = sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]

    best_power = by_power[0]
    best_score = by_score[0]

    return {
        "valid": True,
        "reason": "ok",
        "now": now,
        "frames": len(frames),
        "window_s": window_s,
        "range_min_m": range_min,
        "range_max_m": range_max,
        "direction": direction,
        "min_abs_velocity_mps": min_abs_velocity,
        "noise_floor_db": noise_db,
        "best_by_power": best_power,
        "best_by_score": best_score,
        "top_by_power": by_power,
        "top_by_score": by_score,
        "max_power_roi_db": pow_to_db(max_power),
        "range_axis_roi": r_axis[r_idx].copy(),
        "velocity_axis_roi": v_axis[v_idx].copy(),
    }


def print_candidate(prefix: str, c: dict):
    print(
        f"{prefix} "
        f"v={c['velocity_mps']:+.2f} m/s "
        f"|v|={c['abs_velocity_mps']:.2f} "
        f"r={c['range_m']:.2f} m "
        f"snr={c['snr_db']:.1f} dB "
        f"p={c['power_db']:.1f} dB "
        f"I={c['intensity_score']:.2f} "
        f"score={c['score']:.1f} "
        f"age={c['age_s']:.2f}s"
    )


def make_heatmap_image(result: dict, width: int = 900, height: int = 480):
    if cv2 is None:
        return None

    rv_db = result["max_power_roi_db"]
    r_axis = result["range_axis_roi"]
    v_axis = result["velocity_axis_roi"]

    if rv_db.size == 0:
        return None

    # Robust display range.
    finite = rv_db[np.isfinite(rv_db)]
    if finite.size == 0:
        return None

    lo = np.percentile(finite, 10)
    hi = np.percentile(finite, 99.5)
    if hi <= lo:
        hi = lo + 1.0

    img = np.clip((rv_db - lo) / (hi - lo), 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

    # rd is [range, velocity], display range vertical, velocity horizontal.
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    best = result["best_by_score"]
    # Draw simple labels.
    lines = [
        f"Recent RV max-over-time, window={result['window_s']:.1f}s, direction={result['direction']}",
        f"ROI: range {result['range_min_m']:.1f}-{result['range_max_m']:.1f} m, |v|>={result['min_abs_velocity_mps']:.1f} m/s",
        f"Best score: v={best['velocity_mps']:+.2f} m/s, r={best['range_m']:.2f} m, I={best['intensity_score']:.2f}, SNR={best['snr_db']:.1f} dB",
        f"Velocity axis: {float(v_axis[0]):+.2f} to {float(v_axis[-1]):+.2f} m/s | Range axis: {float(r_axis[0]):.2f} to {float(r_axis[-1]):.2f} m",
    ]
    y = 24
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 24

    return img


def main():
    parser = argparse.ArgumentParser(description="Monitor recent RV-map negative-speed peaks")
    parser.add_argument("--pc-ip", default="192.168.33.30")
    parser.add_argument("--dca-ip", default="192.168.33.180")
    parser.add_argument("--data-port", type=int, default=4098)

    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--buffer-s", type=float, default=6.0)
    parser.add_argument("--range-min", type=float, default=0.0)
    parser.add_argument("--range-max", type=float, default=3.0)
    parser.add_argument("--min-abs-velocity", type=float, default=2.0)
    parser.add_argument("--max-abs-velocity", type=float, default=15.5)
    parser.add_argument("--velocity-strong", type=float, default=8.0)
    parser.add_argument("--direction", choices=["negative", "positive", "both"], default="negative")
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--print-top", action="store_true")
    args = parser.parse_args()

    cfg = make_config(
        RadarConfig,
        pc_ip=args.pc_ip,
        dca_ip=args.dca_ip,
        data_port=args.data_port,
        radar_buffer_seconds=args.buffer_s,
        min_abs_velocity_mps=args.min_abs_velocity,
    )

    radar = RadarAgent(cfg)

    print()
    print("=== RV Peak Monitor ===")
    print(f"window     : {args.window_s:.1f} s")
    print(f"range ROI  : {args.range_min:.2f} ~ {args.range_max:.2f} m")
    print(f"velocity   : direction={args.direction}, |v|>={args.min_abs_velocity:.2f} m/s")
    print()
    print("Start mmWave Studio capture / StartFrame after this script is listening.")
    print("Press Ctrl+C to stop. If --plot, press q in the RV window to stop.")
    print()

    radar.start()

    try:
        last_print = 0.0
        while True:
            result = analyze_recent_rv(
                radar=radar,
                window_s=args.window_s,
                range_min=args.range_min,
                range_max=args.range_max,
                min_abs_velocity=args.min_abs_velocity,
                max_abs_velocity=args.max_abs_velocity,
                direction=args.direction,
                top_k=args.top_k,
                velocity_strong=args.velocity_strong,
            )

            now = time.perf_counter()
            if now - last_print >= args.period:
                last_print = now

                health = radar.get_health()
                print(
                    f"[RV] health={health.status} fps={health.fps:.1f}/{health.expected_fps:.1f} "
                    f"frames={getattr(health, 'valid_frame_count', '?')}"
                )

                if not result["valid"]:
                    print(f"[RV] invalid: {result['reason']} frames={result.get('frames', 0)}")
                else:
                    print(f"[RV] recent frames={result['frames']} noise={result['noise_floor_db']:.1f} dB")
                    print_candidate("[RV] best_by_power", result["best_by_power"])
                    print_candidate("[RV] best_by_score", result["best_by_score"])

                    if args.print_top:
                        print("[RV] top_by_score:")
                        for i, c in enumerate(result["top_by_score"], 1):
                            print_candidate(f"  #{i}", c)
                print()

            if args.plot and result.get("valid") and cv2 is not None:
                img = make_heatmap_image(result)
                if img is not None:
                    cv2.imshow("Recent RV Map Peak Monitor", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print()
        print("[Main] Ctrl+C")

    finally:
        print("[Main] stopping...")
        radar.stop()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[Main] stopped.")


if __name__ == "__main__":
    main()
