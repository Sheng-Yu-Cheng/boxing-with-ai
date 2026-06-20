#!/usr/bin/env python3
"""
vision_agent.py

RadarBox VisionAgent for current MediaPipe Tasks API.

Why this version exists:
    Newer MediaPipe packages, including the one you installed, may not expose
    the legacy `mp.solutions.pose` API. This script uses the current
    MediaPipe Tasks PoseLandmarker API instead.

Install:
    pip install opencv-python mediapipe numpy

You must download a MediaPipe Pose Landmarker task model, for example:
    pose_landmarker_lite.task

Run:
    python .\src\vision_agent.py --debug --model-path .\models\pose_landmarker_lite.task

Default assumptions:
    - laptop built-in webcam: camera_index = 0
    - player faces the laptop camera
    - upper body visible
    - output actions:
        left_straight / right_straight
        left_hook / right_hook
        left_uppercut / right_uppercut
        block / block_end

Timestamp convention:
    Uses time.perf_counter(), matching radar_agent.py.
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ============================================================
# Pose landmark indices, same 33-point MediaPipe Pose layout
# ============================================================

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


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class VisionConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror_image: bool = True

    model_path: str = "models/pose_landmarker_lite.task"

    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    pose_smooth_alpha: float = 0.45
    velocity_smooth_alpha: float = 0.35
    min_landmark_visibility: float = 0.45

    punch_cooldown_s: float = 0.35
    block_min_duration_s: float = 0.12

    # These are torso-normalized thresholds.
    wrist_speed_start_thresh: float = 1.15
    wrist_speed_impact_drop_ratio: float = 0.70
    min_extension_increase: float = 0.12

    straight_elbow_angle_deg: float = 145.0
    straight_min_z_forward: float = 0.035

    hook_min_horizontal_disp: float = 0.16
    hook_max_elbow_angle_deg: float = 155.0

    uppercut_min_upward_disp: float = 0.13
    uppercut_min_upward_speed: float = 0.90

    block_max_hand_face_dist: float = 0.95
    block_max_hand_midline_x_dist: float = 0.95
    block_min_hand_height_ratio: float = 0.35

    action_queue_size: int = 64
    debug_window_name: str = "RadarBox VisionAgent Debug"


@dataclass
class PlayerActionEvent:
    timestamp: float
    action_type: str
    hand: str
    phase: str
    start_time: float
    impact_time: Optional[float]
    end_time: Optional[float]
    confidence: float
    camera_speed: float
    reason: str


@dataclass
class VisionHealth:
    running: bool
    camera_opened: bool
    pose_detected: bool
    fps: float
    frame_count: int
    last_frame_time: Optional[float]
    status: str


@dataclass
class PoseDebugFrame:
    timestamp: float
    frame_id: int
    image_bgr: np.ndarray
    pose_detected: bool
    current_action_label: str
    left_state: str
    right_state: str
    block_active: bool
    status: str


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float

    def arr3(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class PoseFrame:
    timestamp: float
    frame_id: int
    landmarks: dict[str, LandmarkPoint]
    pose_detected: bool
    confidence: float


@dataclass
class HandFeatures:
    hand: str
    wrist: np.ndarray
    elbow: np.ndarray
    shoulder: np.ndarray
    nose: np.ndarray
    mid_shoulder: np.ndarray
    mid_hip: np.ndarray
    wrist_velocity: np.ndarray
    wrist_speed: float
    horizontal_speed: float
    vertical_speed: float
    extension: float
    elbow_angle_deg: float
    torso_len: float
    visibility: float
    valid: bool = True


@dataclass
class HandState:
    hand: str
    state: str = "idle"
    start_time: float = 0.0
    last_event_time: float = -1e9

    start_wrist: Optional[np.ndarray] = None
    start_extension: float = 0.0
    start_elbow_angle: float = 0.0

    peak_time: float = 0.0
    peak_speed: float = 0.0
    peak_extension: float = 0.0
    peak_wrist: Optional[np.ndarray] = None
    peak_elbow_angle: float = 0.0
    peak_features: Optional[HandFeatures] = None
    prev_speed: float = 0.0


# ============================================================
# Math helpers
# ============================================================

def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.degrees(math.acos(cosv)))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.copy()
    return alpha * cur + (1.0 - alpha) * prev


# ============================================================
# VisionAgent
# ============================================================

class VisionAgent:
    def __init__(self, config: Optional[VisionConfig] = None):
        self.cfg = config or VisionConfig()

        self._cap = None
        self._landmarker = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._action_queue: "queue.Queue[PlayerActionEvent]" = queue.Queue(maxsize=self.cfg.action_queue_size)
        self._latest_debug_frame: Optional[PoseDebugFrame] = None

        self._frame_count = 0
        self._last_frame_time: Optional[float] = None
        self._pose_detected = False
        self._camera_opened = False
        self._status = "not_started"
        self._recent_frame_times: deque[float] = deque(maxlen=120)

        self._smoothed_landmarks: dict[str, np.ndarray] = {}
        self._smoothed_velocities: dict[str, np.ndarray] = {}
        self._prev_pose_time: Optional[float] = None

        self._left_state = HandState(hand="left")
        self._right_state = HandState(hand="right")
        self._block_active = False
        self._block_start_time = 0.0
        self._current_action_label = "idle"

    # -----------------------------
    # Public API
    # -----------------------------

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="VisionAgent", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass

        try:
            cv2.destroyWindow(self.cfg.debug_window_name)
        except Exception:
            pass

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_next_action_event(self) -> Optional[PlayerActionEvent]:
        try:
            return self._action_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_pose_debug_frame(self) -> Optional[PoseDebugFrame]:
        with self._lock:
            return self._latest_debug_frame

    def get_health(self) -> VisionHealth:
        with self._lock:
            status = self._status
            if not self.is_running():
                status = "STOPPED"
            elif not self._camera_opened:
                status = "CAMERA_NOT_OPENED"
            elif not self._pose_detected:
                status = "POSE_LOST"

            return VisionHealth(
                running=self.is_running(),
                camera_opened=self._camera_opened,
                pose_detected=self._pose_detected,
                fps=self._fps_locked(),
                frame_count=self._frame_count,
                last_frame_time=self._last_frame_time,
                status=status,
            )

    # -----------------------------
    # Worker
    # -----------------------------

    def _worker(self) -> None:
        print("[VisionAgent] starting")
        print(f"[VisionAgent] model_path={self.cfg.model_path}")

        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            with self._lock:
                self._status = "MODEL_NOT_FOUND"
            print()
            print("[VisionAgent] ERROR: pose landmarker model not found.")
            print(f"Expected: {model_path.resolve()}")
            print("Download a MediaPipe Pose Landmarker model, for example pose_landmarker_lite.task,")
            print("then run:")
            print(f"  python .\\src\\vision_agent.py --debug --model-path {model_path}")
            print()
            return

        self._cap = cv2.VideoCapture(self.cfg.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        if not self._cap.isOpened():
            with self._lock:
                self._camera_opened = False
                self._status = "CAMERA_OPEN_FAILED"
            print("[VisionAgent] ERROR: camera open failed")
            return

        with self._lock:
            self._camera_opened = True
            self._status = "OK"

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.cfg.min_pose_detection_confidence,
            min_pose_presence_confidence=self.cfg.min_pose_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        print("[VisionAgent] camera opened. Press q in debug window to quit.")

        while not self._stop_event.is_set():
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                with self._lock:
                    self._status = "CAMERA_READ_FAILED"
                time.sleep(0.01)
                continue

            if self.cfg.mirror_image:
                frame_bgr = cv2.flip(frame_bgr, 1)

            now = time.perf_counter()
            frame_id = self._frame_count
            self._frame_count += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Needs monotonically increasing ms timestamps.
            timestamp_ms = int(now * 1000)
            try:
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            except ValueError:
                # If perf_counter ms produces duplicate timestamp on fast loops, force monotonic by frame id.
                timestamp_ms = int(frame_id * (1000.0 / max(self.cfg.fps, 1)))
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_frame = self._extract_pose_frame(result, now, frame_id)
            events = self._process_pose_frame(pose_frame)

            for ev in events:
                self._push_action_event(ev)

            debug_img = self._draw_debug(frame_bgr, result, pose_frame, events)
            debug_frame = PoseDebugFrame(
                timestamp=now,
                frame_id=frame_id,
                image_bgr=debug_img,
                pose_detected=pose_frame.pose_detected,
                current_action_label=self._current_action_label,
                left_state=self._left_state.state,
                right_state=self._right_state.state,
                block_active=self._block_active,
                status=self._status,
            )

            with self._lock:
                self._latest_debug_frame = debug_frame
                self._last_frame_time = now
                self._pose_detected = pose_frame.pose_detected
                self._recent_frame_times.append(now)
                self._status = "OK" if pose_frame.pose_detected else "POSE_LOST"

        print("[VisionAgent] stopped")

    def _extract_pose_frame(self, result, timestamp: float, frame_id: int) -> PoseFrame:
        if result is None or not result.pose_landmarks:
            return PoseFrame(timestamp, frame_id, {}, False, 0.0)

        # Single person only.
        lm = result.pose_landmarks[0]

        wanted = {
            "nose": NOSE,
            "left_shoulder": LEFT_SHOULDER,
            "right_shoulder": RIGHT_SHOULDER,
            "left_elbow": LEFT_ELBOW,
            "right_elbow": RIGHT_ELBOW,
            "left_wrist": LEFT_WRIST,
            "right_wrist": RIGHT_WRIST,
            "left_hip": LEFT_HIP,
            "right_hip": RIGHT_HIP,
        }

        landmarks: dict[str, LandmarkPoint] = {}
        vis_list = []

        for name, idx in wanted.items():
            p = lm[idx]
            # New Tasks landmarks usually include visibility/presence. Be robust.
            vis = float(getattr(p, "visibility", 1.0) or 1.0)
            landmarks[name] = LandmarkPoint(float(p.x), float(p.y), float(p.z), vis)
            vis_list.append(vis)

        required = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]
        required_ok = all(landmarks[k].visibility >= self.cfg.min_landmark_visibility for k in required)
        confidence = float(np.mean(vis_list)) if vis_list else 0.0

        return PoseFrame(timestamp, frame_id, landmarks, bool(required_ok), confidence)

    # -----------------------------
    # Pose processing
    # -----------------------------

    def _process_pose_frame(self, pose: PoseFrame) -> list[PlayerActionEvent]:
        if not pose.pose_detected:
            self._current_action_label = "pose_lost"
            return []

        self._update_smoothing(pose)

        left = self._compute_hand_features(pose, "left")
        right = self._compute_hand_features(pose, "right")

        events: list[PlayerActionEvent] = []

        block_event = self._update_block_state(pose, left, right)
        if block_event is not None:
            events.append(block_event)

        ev_l = self._update_hand_state(self._left_state, left, pose.timestamp)
        ev_r = self._update_hand_state(self._right_state, right, pose.timestamp)

        if ev_l is not None:
            events.append(ev_l)
        if ev_r is not None:
            events.append(ev_r)

        if events:
            self._current_action_label = events[-1].action_type
        elif self._block_active:
            self._current_action_label = "block"
        else:
            self._current_action_label = "idle"

        return events

    def _update_smoothing(self, pose: PoseFrame) -> None:
        dt = 0.0 if self._prev_pose_time is None else max(pose.timestamp - self._prev_pose_time, 1e-6)
        self._prev_pose_time = pose.timestamp

        for name, p in pose.landmarks.items():
            cur = p.arr3()
            prev = self._smoothed_landmarks.get(name)
            smoothed = _ema(prev, cur, self.cfg.pose_smooth_alpha)

            if prev is not None and dt > 0:
                raw_v = (smoothed - prev) / dt
            else:
                raw_v = np.zeros(3, dtype=np.float32)

            prev_v = self._smoothed_velocities.get(name)
            vel = _ema(prev_v, raw_v, self.cfg.velocity_smooth_alpha)

            self._smoothed_landmarks[name] = smoothed
            self._smoothed_velocities[name] = vel

    def _p(self, pose: PoseFrame, name: str) -> np.ndarray:
        return self._smoothed_landmarks.get(name, pose.landmarks[name].arr3())

    def _v(self, name: str) -> np.ndarray:
        return self._smoothed_velocities.get(name, np.zeros(3, dtype=np.float32))

    def _compute_hand_features(self, pose: PoseFrame, hand: str) -> HandFeatures:
        wrist_name = f"{hand}_wrist"
        elbow_name = f"{hand}_elbow"
        shoulder_name = f"{hand}_shoulder"

        wrist = self._p(pose, wrist_name)
        elbow = self._p(pose, elbow_name)
        shoulder = self._p(pose, shoulder_name)
        nose = self._p(pose, "nose")

        lsh = self._p(pose, "left_shoulder")
        rsh = self._p(pose, "right_shoulder")
        lhip = self._p(pose, "left_hip")
        rhip = self._p(pose, "right_hip")

        mid_shoulder = 0.5 * (lsh + rsh)
        mid_hip = 0.5 * (lhip + rhip)

        torso_len = _norm(mid_shoulder[:2] - mid_hip[:2])
        if torso_len < 1e-4:
            torso_len = _norm(lsh[:2] - rsh[:2])
        torso_len = max(torso_len, 1e-4)

        v = self._v(wrist_name).copy()
        v_norm = v.copy()
        v_norm[0] /= torso_len
        v_norm[1] /= torso_len
        v_norm[2] /= torso_len

        wrist_speed = _norm(v_norm)
        horizontal_speed = abs(float(v_norm[0]))
        vertical_speed = float(-v_norm[1])  # image y goes downward

        extension = _norm((wrist[:2] - shoulder[:2]) / torso_len)
        elbow_angle = _angle_deg(shoulder[:2], elbow[:2], wrist[:2])

        visibility = min(
            pose.landmarks[wrist_name].visibility,
            pose.landmarks[elbow_name].visibility,
            pose.landmarks[shoulder_name].visibility,
        )

        return HandFeatures(
            hand=hand,
            wrist=wrist,
            elbow=elbow,
            shoulder=shoulder,
            nose=nose,
            mid_shoulder=mid_shoulder,
            mid_hip=mid_hip,
            wrist_velocity=v_norm,
            wrist_speed=float(wrist_speed),
            horizontal_speed=float(horizontal_speed),
            vertical_speed=float(vertical_speed),
            extension=float(extension),
            elbow_angle_deg=float(elbow_angle),
            torso_len=float(torso_len),
            visibility=float(visibility),
            valid=visibility >= self.cfg.min_landmark_visibility,
        )

    # -----------------------------
    # Block
    # -----------------------------

    def _update_block_state(self, pose: PoseFrame, left: HandFeatures, right: HandFeatures) -> Optional[PlayerActionEvent]:
        now = pose.timestamp
        is_block = self._is_block_pose(left, right)

        if is_block and not self._block_active:
            self._block_active = True
            self._block_start_time = now
            return PlayerActionEvent(
                timestamp=now,
                action_type="block",
                hand="both",
                phase="block_start",
                start_time=now,
                impact_time=None,
                end_time=None,
                confidence=0.85,
                camera_speed=0.0,
                reason="hands_near_face_or_chest",
            )

        if not is_block and self._block_active:
            dur = now - self._block_start_time
            self._block_active = False
            if dur >= self.cfg.block_min_duration_s:
                return PlayerActionEvent(
                    timestamp=now,
                    action_type="block_end",
                    hand="both",
                    phase="block_end",
                    start_time=self._block_start_time,
                    impact_time=None,
                    end_time=now,
                    confidence=0.75,
                    camera_speed=0.0,
                    reason=f"block_released_after_{dur:.2f}s",
                )

        return None

    def _is_block_pose(self, left: HandFeatures, right: HandFeatures) -> bool:
        if not left.valid or not right.valid:
            return False

        nose = left.nose
        mid_shoulder = left.mid_shoulder
        mid_hip = left.mid_hip
        torso_len = max(left.torso_len, 1e-4)

        def hand_ok(f: HandFeatures) -> bool:
            wrist = f.wrist
            face_dist = _norm((wrist[:2] - nose[:2]) / torso_len)
            midline_dist = abs(float((wrist[0] - mid_shoulder[0]) / torso_len))

            shoulder_y = mid_shoulder[1]
            hip_y = mid_hip[1]
            denom = max(abs(hip_y - shoulder_y), 1e-4)
            height_ratio = float((hip_y - wrist[1]) / denom)

            return (
                face_dist <= self.cfg.block_max_hand_face_dist
                and midline_dist <= self.cfg.block_max_hand_midline_x_dist
                and height_ratio >= self.cfg.block_min_hand_height_ratio
            )

        return hand_ok(left) and hand_ok(right)

    # -----------------------------
    # Punch state
    # -----------------------------

    def _update_hand_state(self, state: HandState, f: HandFeatures, now: float) -> Optional[PlayerActionEvent]:
        if not f.valid:
            return None

        if state.state == "cooldown":
            if now - state.last_event_time >= self.cfg.punch_cooldown_s:
                state.state = "idle"
            else:
                return None

        if state.state == "idle":
            if now - state.last_event_time < self.cfg.punch_cooldown_s:
                return None
            if self._should_start_punch(f):
                state.state = "extension"
                state.start_time = now
                state.start_wrist = f.wrist.copy()
                state.start_extension = f.extension
                state.start_elbow_angle = f.elbow_angle_deg
                state.peak_time = now
                state.peak_speed = f.wrist_speed
                state.peak_extension = f.extension
                state.peak_wrist = f.wrist.copy()
                state.peak_elbow_angle = f.elbow_angle_deg
                state.peak_features = f
                state.prev_speed = f.wrist_speed
            return None

        if state.state == "extension":
            if f.wrist_speed > state.peak_speed:
                state.peak_speed = f.wrist_speed
                state.peak_time = now
                state.peak_wrist = f.wrist.copy()
                state.peak_extension = f.extension
                state.peak_elbow_angle = f.elbow_angle_deg
                state.peak_features = f

            speed_drop = f.wrist_speed < max(
                self.cfg.wrist_speed_start_thresh * 0.45,
                state.peak_speed * self.cfg.wrist_speed_impact_drop_ratio,
            )
            extension_peak = (
                state.peak_extension - state.start_extension >= self.cfg.min_extension_increase
                and f.extension < state.peak_extension - 0.03
            )
            timeout = now - state.start_time > 0.55

            if speed_drop or extension_peak or timeout:
                ev = self._make_punch_event(state, f, now)
                state.state = "cooldown"
                state.last_event_time = now
                return ev

        return None

    def _should_start_punch(self, f: HandFeatures) -> bool:
        if self._block_active:
            return False
        if f.wrist_speed < self.cfg.wrist_speed_start_thresh:
            return False
        if f.extension < 0.35 and f.elbow_angle_deg < 80:
            return False
        return True

    def _make_punch_event(self, state: HandState, current_f: HandFeatures, now: float) -> PlayerActionEvent:
        f = state.peak_features or current_f
        if state.start_wrist is None or state.peak_wrist is None:
            displacement = np.zeros(3, dtype=np.float32)
        else:
            displacement = (state.peak_wrist - state.start_wrist).copy()

        torso_len = max(float(f.torso_len), 1e-4)
        disp_norm = displacement.copy()
        disp_norm[0] /= torso_len
        disp_norm[1] /= torso_len
        disp_norm[2] /= torso_len

        dx = float(disp_norm[0])
        upward_disp = float(-disp_norm[1])
        dz = float(disp_norm[2])

        extension_gain = state.peak_extension - state.start_extension
        elbow_gain = state.peak_elbow_angle - state.start_elbow_angle

        action_type, reason, confidence = self._classify_punch(
            hand=state.hand,
            peak_features=f,
            dx=dx,
            upward_disp=upward_disp,
            dz=dz,
            extension_gain=extension_gain,
            elbow_gain=elbow_gain,
            peak_speed=state.peak_speed,
        )

        return PlayerActionEvent(
            timestamp=now,
            action_type=action_type,
            hand=state.hand,
            phase="impact",
            start_time=state.start_time,
            impact_time=state.peak_time,
            end_time=now,
            confidence=confidence,
            camera_speed=float(state.peak_speed),
            reason=reason,
        )

    def _classify_punch(
        self,
        hand: str,
        peak_features: HandFeatures,
        dx: float,
        upward_disp: float,
        dz: float,
        extension_gain: float,
        elbow_gain: float,
        peak_speed: float,
    ) -> tuple[str, str, float]:
        cfg = self.cfg
        elbow = peak_features.elbow_angle_deg
        abs_dx = abs(dx)
        z_forward = -dz

        if upward_disp >= cfg.uppercut_min_upward_disp or peak_features.vertical_speed >= cfg.uppercut_min_upward_speed:
            conf = _clip01(0.45 + 0.25 * upward_disp / max(cfg.uppercut_min_upward_disp, 1e-6) + 0.15 * peak_speed / 2.0)
            return f"{hand}_uppercut", f"uppercut: up={upward_disp:.2f}, speed={peak_speed:.2f}", conf

        straight_score = 0.0
        if elbow >= cfg.straight_elbow_angle_deg:
            straight_score += 0.35
        if extension_gain >= cfg.min_extension_increase:
            straight_score += 0.30
        if abs(z_forward) >= cfg.straight_min_z_forward:
            straight_score += 0.15
        if peak_speed >= cfg.wrist_speed_start_thresh:
            straight_score += 0.20

        hook_score = 0.0
        if abs_dx >= cfg.hook_min_horizontal_disp:
            hook_score += 0.45
        if elbow <= cfg.hook_max_elbow_angle_deg:
            hook_score += 0.25
        if abs_dx > max(abs(upward_disp), abs(extension_gain)):
            hook_score += 0.15
        if peak_speed >= cfg.wrist_speed_start_thresh:
            hook_score += 0.15

        if hook_score > straight_score and hook_score >= 0.45:
            return f"{hand}_hook", f"hook: dx={dx:.2f}, elbow={elbow:.0f}, speed={peak_speed:.2f}", _clip01(hook_score)

        if straight_score >= 0.45:
            return f"{hand}_straight", f"straight: ext={extension_gain:.2f}, elbow={elbow:.0f}, speed={peak_speed:.2f}", _clip01(straight_score)

        if peak_speed >= cfg.wrist_speed_start_thresh:
            if abs_dx >= abs(upward_disp):
                return f"{hand}_hook", f"fallback_hook: dx={dx:.2f}, speed={peak_speed:.2f}", 0.45
            return f"{hand}_uppercut", f"fallback_uppercut: up={upward_disp:.2f}, speed={peak_speed:.2f}", 0.45

        return "idle", "motion_too_weak", 0.0

    # -----------------------------
    # Helpers
    # -----------------------------

    def _push_action_event(self, ev: PlayerActionEvent) -> None:
        if ev.action_type == "idle":
            return

        try:
            self._action_queue.put_nowait(ev)
        except queue.Full:
            try:
                self._action_queue.get_nowait()
            except queue.Empty:
                pass
            self._action_queue.put_nowait(ev)

        print(
            f"[VisionAgent] event action={ev.action_type} hand={ev.hand} "
            f"phase={ev.phase} conf={ev.confidence:.2f} "
            f"speed={ev.camera_speed:.2f} reason={ev.reason}"
        )

    def _fps_locked(self) -> float:
        if len(self._recent_frame_times) < 2:
            return 0.0
        dt = self._recent_frame_times[-1] - self._recent_frame_times[0]
        return (len(self._recent_frame_times) - 1) / max(dt, 1e-9)

    def _draw_debug(self, image_bgr: np.ndarray, result, pose: PoseFrame, events: list[PlayerActionEvent]) -> np.ndarray:
        img = image_bgr.copy()
        h, w = img.shape[:2]

        # Draw skeleton manually because legacy mp.solutions drawing may not exist.
        if result is not None and result.pose_landmarks:
            lms = result.pose_landmarks[0]

            def pt(idx: int):
                p = lms[idx]
                return int(p.x * w), int(p.y * h)

            for a, b in POSE_CONNECTIONS:
                cv2.line(img, pt(a), pt(b), (0, 220, 255), 2)

            for idx in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]:
                cv2.circle(img, pt(idx), 4, (0, 255, 0), -1)

        label = events[-1].action_type if events else self._current_action_label
        color = (0, 255, 0) if pose.pose_detected else (0, 0, 255)

        cv2.putText(img, f"Action: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"L:{self._left_state.state} R:{self._right_state.state} Block:{self._block_active}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        health = self.get_health()
        cv2.putText(img, f"FPS:{health.fps:.1f} Status:{health.status}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if pose.pose_detected:
            try:
                lf = self._compute_hand_features(pose, "left")
                rf = self._compute_hand_features(pose, "right")
                lines = [
                    f"L speed={lf.wrist_speed:.2f} ext={lf.extension:.2f} elbow={lf.elbow_angle_deg:.0f}",
                    f"R speed={rf.wrist_speed:.2f} ext={rf.extension:.2f} elbow={rf.elbow_angle_deg:.0f}",
                ]
                y = h - 45
                for line in lines:
                    cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    y += 24
            except Exception:
                pass

        return img


# ============================================================
# Standalone runners
# ============================================================

def run_debug(agent: VisionAgent) -> None:
    print()
    print("VisionAgent debug mode. Press q in the webcam window to quit.")
    print()

    while agent.is_running():
        dbg = agent.get_latest_pose_debug_frame()
        if dbg is not None:
            cv2.imshow(agent.cfg.debug_window_name, dbg.image_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        time.sleep(0.001)

    agent.stop()


def run_console(agent: VisionAgent) -> None:
    print()
    print("VisionAgent console mode. Press Ctrl+C to stop.")
    print()

    while True:
        time.sleep(0.02)
        ev = agent.get_next_action_event()
        if ev is not None:
            print(
                f"event: {ev.action_type}, hand={ev.hand}, "
                f"impact={ev.impact_time}, conf={ev.confidence:.2f}, "
                f"speed={ev.camera_speed:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="RadarBox VisionAgent using MediaPipe Tasks PoseLandmarker")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--model-path", default="models/pose_landmarker_lite.task")
    parser.add_argument("--min-vis", type=float, default=0.45)
    args = parser.parse_args()

    cfg = VisionConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror_image=not args.no_mirror,
        model_path=args.model_path,
        min_landmark_visibility=args.min_vis,
    )

    agent = VisionAgent(cfg)
    agent.start()

    try:
        if args.debug:
            run_debug(agent)
        else:
            run_console(agent)
    except KeyboardInterrupt:
        print("\nStopping VisionAgent...")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
