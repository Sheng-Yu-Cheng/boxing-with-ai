"""Shared recent Range-Velocity map analysis and OpenCV visualization."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _pow_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def analyze_recent_rv(
    radar: Any,
    window_s: float = 5.0,
    range_min: float = 0.0,
    range_max: float = 3.0,
    min_abs_velocity: float = 2.0,
    max_abs_velocity: float = 15.5,
    direction: str = "negative",
    velocity_strong: float = 8.0,
) -> dict:
    """Build the same max-over-time RV result used by rv-map-peak.py."""
    now = time.perf_counter()
    with radar._lock:
        frames = [
            frame
            for frame in radar._frame_buffer
            if frame.valid and frame.timestamp >= now - window_s
        ]

    if not frames:
        return {"valid": False, "reason": "no_valid_frames_in_recent_window", "frames": 0}

    range_axis = np.asarray(radar.range_axis_m)
    velocity_axis = np.asarray(radar.velocity_axis_mps)
    range_indices = np.where((range_axis >= range_min) & (range_axis <= range_max))[0]

    velocity_mask = (np.abs(velocity_axis) >= min_abs_velocity) & (
        np.abs(velocity_axis) <= max_abs_velocity
    )
    if direction == "negative":
        velocity_mask &= velocity_axis < 0
    elif direction == "positive":
        velocity_mask &= velocity_axis > 0
    elif direction != "both":
        raise ValueError(f"bad direction: {direction}")
    velocity_indices = np.where(velocity_mask)[0]

    if not len(range_indices):
        return {"valid": False, "reason": "empty_range_roi", "frames": len(frames)}
    if not len(velocity_indices):
        return {"valid": False, "reason": "empty_velocity_roi", "frames": len(frames)}

    stack = np.stack(
        [frame.rd_power[np.ix_(range_indices, velocity_indices)] for frame in frames],
        axis=0,
    ).astype(np.float64)
    max_power = np.max(stack, axis=0)
    max_frame_indices = np.argmax(stack, axis=0)
    finite_values = max_power[np.isfinite(max_power) & (max_power > 0)]
    if not finite_values.size:
        return {"valid": False, "reason": "no_finite_power", "frames": len(frames)}

    noise_db = float(_pow_to_db(float(np.median(finite_values)) + 1e-12))
    best = None
    for local_range in range(max_power.shape[0]):
        for local_velocity in range(max_power.shape[1]):
            power_db = float(_pow_to_db(float(max_power[local_range, local_velocity]) + 1e-12))
            snr_db = power_db - noise_db
            velocity = float(velocity_axis[velocity_indices[local_velocity]])
            intensity = float(
                np.clip(
                    (abs(velocity) - min_abs_velocity)
                    / max(velocity_strong - min_abs_velocity, 1e-6),
                    0.0,
                    1.0,
                )
            )
            candidate = {
                "timestamp": float(frames[int(max_frame_indices[local_range, local_velocity])].timestamp),
                "range_m": float(range_axis[range_indices[local_range]]),
                "velocity_mps": velocity,
                "power_db": power_db,
                "snr_db": snr_db,
                "intensity_score": intensity,
                "score": float(max(snr_db, 0.0) * (0.25 + 0.75 * intensity)),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return {
        "valid": True,
        "reason": "ok",
        "frames": len(frames),
        "window_s": window_s,
        "range_min_m": range_min,
        "range_max_m": range_max,
        "direction": direction,
        "min_abs_velocity_mps": min_abs_velocity,
        "noise_floor_db": noise_db,
        "best_by_score": best,
        "max_power_roi_db": _pow_to_db(max_power),
        "range_axis_roi": range_axis[range_indices].copy(),
        "velocity_axis_roi": velocity_axis[velocity_indices].copy(),
    }


def make_heatmap_image(result: dict, cv2, width: int = 900, height: int = 480):
    """Render a recent RV result using the rv-map-peak.py visual style."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if not result.get("valid"):
        health = result.get("radar_health", "UNKNOWN")
        cv2.putText(
            canvas,
            f"Radar health={health} | {result.get('reason', 'no data')}",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        return canvas

    rv_db = result["max_power_roi_db"]
    finite = rv_db[np.isfinite(rv_db)]
    if not finite.size:
        return canvas
    lo = float(np.percentile(finite, 10))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    image = np.clip((rv_db - lo) / (hi - lo), 0, 1)
    image = cv2.applyColorMap((image * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    best = result["best_by_score"]
    lines = [
        f"Recent RV max-over-time, window={result['window_s']:.1f}s, direction={result['direction']}",
        f"ROI: range {result['range_min_m']:.1f}-{result['range_max_m']:.1f} m, |v|>={result['min_abs_velocity_mps']:.1f} m/s",
        f"Best score: v={best['velocity_mps']:+.2f} m/s, r={best['range_m']:.2f} m, I={best['intensity_score']:.2f}, SNR={best['snr_db']:.1f} dB",
        f"Health: {result.get('radar_health', 'UNKNOWN')} | Frames: {result['frames']} | noise={result['noise_floor_db']:.1f} dB",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (12, 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return image
