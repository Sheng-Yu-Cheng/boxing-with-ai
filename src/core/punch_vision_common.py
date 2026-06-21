#!/usr/bin/env python3
r"""
punch_vision_common.py

Shared MediaPipe Tasks + trajectory feature utilities for RadarBox.

This file is used by:

    record_punch_dataset.py
    train_punch_classifier.py
    vision_agent_trajectory.py

Install:
    pip install mediapipe opencv-contrib-python numpy scipy scikit-learn joblib

Model:
    models/pose_landmarker_lite.task
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# MediaPipe Pose indices.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
]

REQUIRED_UPPER_BODY = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
]


@dataclass
class LandmarkFrame:
    timestamp: float
    frame_id: int
    landmarks: np.ndarray       # shape: (33, 4), x/y/z/visibility
    world_landmarks: Optional[np.ndarray] = None  # shape: (33, 3), x/y/z meters-ish
    pose_detected: bool = True
    confidence: float = 1.0


@dataclass
class PoseDetectorConfig:
    model_path: str = "models/pose_landmarker_lite.task"
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseTaskDetector:
    """
    Thin wrapper around MediaPipe Tasks PoseLandmarker VIDEO mode.
    """

    def __init__(self, config: PoseDetectorConfig):
        self.cfg = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"PoseLandmarker model not found: {model_path.resolve()}\n"
                "Download pose_landmarker_lite.task into ./models first."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass

    def detect(self, frame_bgr: np.ndarray, timestamp: float, frame_id: int) -> LandmarkFrame:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts_ms = int(timestamp * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        return pose_result_to_frame(result, timestamp, frame_id)


def pose_result_to_frame(result, timestamp: float, frame_id: int) -> LandmarkFrame:
    if result is None or not result.pose_landmarks:
        return LandmarkFrame(
            timestamp=timestamp,
            frame_id=frame_id,
            landmarks=np.zeros((33, 4), dtype=np.float32),
            world_landmarks=None,
            pose_detected=False,
            confidence=0.0,
        )

    lm = result.pose_landmarks[0]
    arr = np.zeros((33, 4), dtype=np.float32)

    for i, p in enumerate(lm[:33]):
        visibility = getattr(p, "visibility", 1.0)
        if visibility is None:
            visibility = 1.0
        arr[i] = [float(p.x), float(p.y), float(p.z), float(visibility)]

    world_arr = None
    if getattr(result, "pose_world_landmarks", None):
        wlm = result.pose_world_landmarks[0]
        world_arr = np.zeros((33, 3), dtype=np.float32)
        for i, p in enumerate(wlm[:33]):
            world_arr[i] = [float(p.x), float(p.y), float(p.z)]

    conf = float(np.mean(arr[REQUIRED_UPPER_BODY, 3]))
    pose_detected = bool(np.all(arr[REQUIRED_UPPER_BODY, 3] >= 0.35))

    return LandmarkFrame(
        timestamp=timestamp,
        frame_id=frame_id,
        landmarks=arr,
        world_landmarks=world_arr,
        pose_detected=pose_detected,
        confidence=conf,
    )


def draw_pose(image_bgr: np.ndarray, frame: LandmarkFrame, color=(0, 220, 255)) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]

    if not frame.pose_detected:
        return img

    def pt(idx: int):
        x = int(frame.landmarks[idx, 0] * w)
        y = int(frame.landmarks[idx, 1] * h)
        return x, y

    for a, b in POSE_CONNECTIONS:
        cv2.line(img, pt(a), pt(b), color, 2)

    for idx in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]:
        cv2.circle(img, pt(idx), 4, (0, 255, 0), -1)

    return img


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.degrees(math.acos(cosv)))


def _safe_stat(x: np.ndarray, fn, default: float = 0.0) -> float:
    if x.size == 0 or np.any(~np.isfinite(x)):
        return float(default)
    return float(fn(x))


def _interp_resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a sequence of shape (T, D) to (target_len, D).
    """
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[1] if seq.ndim == 2 else 1), dtype=np.float32)
    if len(seq) == 1:
        return np.repeat(seq.astype(np.float32), target_len, axis=0)

    t_old = np.linspace(0.0, 1.0, len(seq))
    t_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(t_new, t_old, seq[:, d])
    return out


def frames_to_arrays(frames: list[LandmarkFrame]) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    ts = np.array([f.timestamp for f in frames], dtype=np.float32)
    lm = np.stack([f.landmarks for f in frames], axis=0).astype(np.float32)
    if all(f.world_landmarks is not None for f in frames):
        wlm = np.stack([f.world_landmarks for f in frames], axis=0).astype(np.float32)
    else:
        wlm = None
    return ts, lm, wlm


def extract_punch_features(
    frames: list[LandmarkFrame],
    hand: str = "right",
    resample_len: int = 16,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract one fixed-size feature vector from a motion segment.

    This uses the whole trajectory, not a single peak frame.
    It combines:
      - geometric trajectory features
      - velocity features
      - elbow/extension features
      - a low-dimensional resampled wrist path

    Coordinates:
      image coordinates are normalized by torso length, relative to same-side shoulder.
      y grows downward, so upward motion is -dy.
    """
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")

    valid_frames = [f for f in frames if f.pose_detected]
    if len(valid_frames) < 3:
        names = base_feature_names(resample_len)
        return np.zeros(len(names), dtype=np.float32), names

    ts, lm, wlm = frames_to_arrays(valid_frames)

    if hand == "right":
        wrist_i, elbow_i, shoulder_i = RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER
    else:
        wrist_i, elbow_i, shoulder_i = LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER

    wrist = lm[:, wrist_i, :3]
    elbow = lm[:, elbow_i, :3]
    shoulder = lm[:, shoulder_i, :3]
    lsh = lm[:, LEFT_SHOULDER, :3]
    rsh = lm[:, RIGHT_SHOULDER, :3]
    lhip = lm[:, LEFT_HIP, :3]
    rhip = lm[:, RIGHT_HIP, :3]

    mid_sh = 0.5 * (lsh + rsh)
    mid_hip = 0.5 * (lhip + rhip)

    torso = np.linalg.norm(mid_sh[:, :2] - mid_hip[:, :2], axis=1)
    shoulder_width = np.linalg.norm(lsh[:, :2] - rsh[:, :2], axis=1)
    scale = np.maximum(torso, shoulder_width)
    scale = np.maximum(scale, 1e-4)

    wrist_rel = (wrist - shoulder) / scale[:, None]
    elbow_rel = (elbow - shoulder) / scale[:, None]

    t0 = float(ts[0])
    ts_rel = ts - ts[0]
    duration = float(max(ts[-1] - ts[0], 1e-4))
    dt = np.diff(ts)
    dt = np.maximum(dt, 1e-4)

    dw = np.diff(wrist_rel, axis=0)
    vel = dw / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    vx = vel[:, 0]
    vy = vel[:, 1]
    vz = vel[:, 2]
    up_v = -vy
    z_forward_v = -vz

    start = wrist_rel[0]
    end = wrist_rel[-1]
    disp = end - start
    dx, dy, dz = float(disp[0]), float(disp[1]), float(disp[2])
    up = -dy
    z_forward = -dz

    path_xy = float(np.sum(np.linalg.norm(np.diff(wrist_rel[:, :2], axis=0), axis=1)))
    disp_xy = float(np.linalg.norm(disp[:2]))
    straightness = float(disp_xy / max(path_xy, 1e-6))
    curvature = float(path_xy / max(disp_xy, 1e-6))

    extension = np.linalg.norm((wrist[:, :2] - shoulder[:, :2]) / scale[:, None], axis=1)
    ext_start = float(extension[0])
    ext_end = float(extension[-1])
    ext_max = float(np.max(extension))
    ext_min = float(np.min(extension))
    ext_gain_end = ext_end - ext_start
    ext_gain_peak = ext_max - ext_start
    ext_drop_min = ext_min - ext_start

    elbow_angles = np.array(
        [_angle_deg(shoulder[i, :2], elbow[i, :2], wrist[i, :2]) for i in range(len(valid_frames))],
        dtype=np.float32,
    )

    # Features from world landmarks if available. They are optional.
    world_feats = []
    world_names = []
    if wlm is not None:
        ww = wlm[:, wrist_i, :]
        ws = wlm[:, shoulder_i, :]
        wr = ww - ws
        wdisp = wr[-1] - wr[0]
        world_feats += [
            float(wdisp[0]), float(wdisp[1]), float(wdisp[2]),
            float(np.linalg.norm(wdisp)),
        ]
        world_names += [
            "world_dx", "world_dy", "world_dz", "world_disp",
        ]
    else:
        world_feats += [0.0, 0.0, 0.0, 0.0]
        world_names += ["world_dx", "world_dy", "world_dz", "world_disp"]

    # Resampled trajectory shape. Use wrist path and extension/elbow over time.
    traj = np.column_stack([
        wrist_rel[:, 0],
        wrist_rel[:, 1],
        wrist_rel[:, 2],
        extension,
        elbow_angles / 180.0,
    ]).astype(np.float32)
    traj_rs = _interp_resample(traj, resample_len).reshape(-1)

    feats = []
    names = []

    def add(name, value):
        names.append(name)
        feats.append(float(value))

    add("duration", duration)
    add("num_frames", len(valid_frames))
    add("dx", dx)
    add("dy", dy)
    add("up", up)
    add("dz", dz)
    add("z_forward", z_forward)
    add("abs_dx", abs(dx))
    add("abs_up", abs(up))
    add("disp_xy", disp_xy)
    add("path_xy", path_xy)
    add("straightness", straightness)
    add("curvature", min(curvature, 20.0))

    add("speed_max", _safe_stat(speed, np.max))
    add("speed_mean", _safe_stat(speed, np.mean))
    add("vx_max", _safe_stat(vx, np.max))
    add("vx_min", _safe_stat(vx, np.min))
    add("abs_vx_max", _safe_stat(np.abs(vx), np.max))
    add("up_v_max", _safe_stat(up_v, np.max))
    add("down_v_max", _safe_stat(-up_v, np.max))
    add("z_forward_v_max", _safe_stat(z_forward_v, np.max))

    add("horizontal_dominance", abs(dx) / max(abs(up), 1e-6))
    add("vertical_dominance", abs(up) / max(abs(dx), 1e-6))
    add("forward_dominance", abs(z_forward) / max(disp_xy, 1e-6))

    add("extension_start", ext_start)
    add("extension_end", ext_end)
    add("extension_max", ext_max)
    add("extension_min", ext_min)
    add("extension_gain_end", ext_gain_end)
    add("extension_gain_peak", ext_gain_peak)
    add("extension_drop_min", ext_drop_min)

    add("elbow_start", float(elbow_angles[0]))
    add("elbow_end", float(elbow_angles[-1]))
    add("elbow_max", float(np.max(elbow_angles)))
    add("elbow_min", float(np.min(elbow_angles)))
    add("elbow_mean", float(np.mean(elbow_angles)))
    add("elbow_gain_end", float(elbow_angles[-1] - elbow_angles[0]))
    add("elbow_gain_peak", float(np.max(elbow_angles) - elbow_angles[0]))

    for name, val in zip(world_names, world_feats):
        add(name, val)

    for i, val in enumerate(traj_rs):
        add(f"traj_{i:02d}", val)

    return np.array(feats, dtype=np.float32), names


def base_feature_names(resample_len: int = 16) -> list[str]:
    dummy = [
        LandmarkFrame(
            timestamp=float(i) * 0.033,
            frame_id=i,
            landmarks=np.zeros((33, 4), dtype=np.float32),
            pose_detected=True,
            confidence=1.0,
        )
        for i in range(4)
    ]
    for f in dummy:
        f.landmarks[:, 3] = 1.0
        f.landmarks[LEFT_SHOULDER, :2] = [0.4, 0.4]
        f.landmarks[RIGHT_SHOULDER, :2] = [0.6, 0.4]
        f.landmarks[LEFT_HIP, :2] = [0.45, 0.8]
        f.landmarks[RIGHT_HIP, :2] = [0.55, 0.8]
        f.landmarks[RIGHT_ELBOW, :2] = [0.65, 0.5]
        f.landmarks[RIGHT_WRIST, :2] = [0.7, 0.6]
        f.landmarks[LEFT_ELBOW, :2] = [0.35, 0.5]
        f.landmarks[LEFT_WRIST, :2] = [0.3, 0.6]
    _, names = extract_punch_features(dummy, "right", resample_len)
    return names


def save_segment_npz(
    path: str | Path,
    frames: list[LandmarkFrame],
    label: str,
    hand: str,
    meta: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ts, lm, wlm = frames_to_arrays(frames)
    payload = {
        "timestamps": ts.astype(np.float32),
        "landmarks": lm.astype(np.float32),
        "label": np.array(label),
        "hand": np.array(hand),
    }
    if wlm is not None:
        payload["world_landmarks"] = wlm.astype(np.float32)
    if meta is not None:
        payload["meta_json"] = np.array(__import__("json").dumps(meta, ensure_ascii=False))

    np.savez_compressed(path, **payload)


def load_segment_npz(path: str | Path) -> tuple[list[LandmarkFrame], str, str, dict]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    ts = data["timestamps"].astype(np.float32)
    lm = data["landmarks"].astype(np.float32)
    label = str(data["label"].item())
    hand = str(data["hand"].item())
    wlm = data["world_landmarks"].astype(np.float32) if "world_landmarks" in data.files else None

    meta = {}
    if "meta_json" in data.files:
        try:
            meta = __import__("json").loads(str(data["meta_json"].item()))
        except Exception:
            meta = {}

    frames = []
    for i in range(len(ts)):
        frames.append(
            LandmarkFrame(
                timestamp=float(ts[i]),
                frame_id=i,
                landmarks=lm[i],
                world_landmarks=None if wlm is None else wlm[i],
                pose_detected=True,
                confidence=float(np.mean(lm[i, REQUIRED_UPPER_BODY, 3])),
            )
        )

    return frames, label, hand, meta
