#!/usr/bin/env python3
r"""
vision_agent.py

Trajectory-based VisionAgent for RadarBox.

This replaces the earlier single-frame / peak-rule classifier with:

    MediaPipe PoseLandmarker
    -> motion segmentation
    -> trajectory feature extraction
    -> trained classifier
    -> PlayerActionEvent

Train first:

    python .\scripts\record_punch_dataset.py --label right_straight --hand right --count 30
    python .\scripts\record_punch_dataset.py --label right_hook     --hand right --count 30
    python .\scripts\record_punch_dataset.py --label right_uppercut --hand right --count 30
    python .\scripts\record_punch_dataset.py --label negative      --hand right --count 30

    python .\scripts\train_punch_classifier.py --dataset .\data\punch_dataset --out .\models\punch_classifier.joblib --hand right

Run:

    python .\src\core\vision_agent.py --debug --classifier .\models\punch_classifier.joblib
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np

from core.punch_vision_common import (
    PoseDetectorConfig,
    PoseTaskDetector,
    LandmarkFrame,
    draw_pose,
    extract_punch_features,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)


@dataclass
class PlayerActionEvent:
    timestamp: float
    action_type: str
    hand: str
    phase: str
    start_time: float
    impact_time: float
    end_time: float
    confidence: float
    camera_speed: float
    reason: str


@dataclass
class TrajectoryVisionConfig:
    classifier_path: str = "models/punch_classifier.joblib"
    model_path: str = "models/pose_landmarker_lite.task"
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    active_hand: str = "right"  # right / left

    pre_seconds: float = 0.15
    min_segment_seconds: float = 0.25
    max_segment_seconds: float = 1.10

    motion_start_speed: float = 1.00
    motion_end_speed: float = 0.45
    motion_end_hold_frames: int = 5

    cooldown_s: float = 0.45
    confidence_threshold: float = 0.75

    negative_labels: tuple[str, ...] = ("negative", "idle", "unknown")

    queue_size: int = 64


class VisionAgent:
    """
    Importable trajectory-based agent with the same basic public API:

        start()
        stop()
        get_next_action_event()
    """

    def __init__(self, config: TrajectoryVisionConfig):
        self.cfg = config

        self.detector = PoseTaskDetector(PoseDetectorConfig(model_path=config.model_path))
        self.bundle = joblib.load(config.classifier_path)
        self.model = self.bundle["model"]
        self.feature_names = self.bundle["feature_names"]
        self.resample_len = int(self.bundle.get("resample_len", 16))

        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._event_queue: "queue.Queue[PlayerActionEvent]" = queue.Queue(maxsize=config.queue_size)

        self._latest_debug_image = None
        self._latest_status = "not_started"
        self._latest_prediction = "none"
        self._latest_confidence = 0.0

        self._rolling_frames: deque[LandmarkFrame] = deque(maxlen=int(config.fps * 3))
        self._segment_frames: list[LandmarkFrame] = []
        self._state = "idle"
        self._segment_start_time = 0.0
        self._end_hold = 0
        self._last_event_time = -1e9
        self._frame_id = 0

        print("[TrajectoryVisionAgent] classifier labels:", self.bundle.get("labels", "unknown"))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self.detector.close()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_next_action_event(self) -> Optional[PlayerActionEvent]:
        try:
            return self._event_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_debug_image(self):
        with self._lock:
            return None if self._latest_debug_image is None else self._latest_debug_image.copy()

    def _worker(self):
        self._cap = cv2.VideoCapture(self.cfg.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        if not self._cap.isOpened():
            self._latest_status = "camera_open_failed"
            print("[TrajectoryVisionAgent] camera open failed")
            return

        self._latest_status = "OK"
        print("[TrajectoryVisionAgent] camera opened")

        while not self._stop_event.is_set():
            ok, frame_bgr = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            pose_frame = self.detector.detect(frame_bgr, now, self._frame_id)
            self._frame_id += 1

            self._rolling_frames.append(pose_frame)
            speed = self._active_hand_speed(pose_frame)

            self._update_segmentation(pose_frame, speed, now)

            dbg = self._draw_debug(frame_bgr, pose_frame, speed)
            with self._lock:
                self._latest_debug_image = dbg

    def _active_hand_speed(self, frame: LandmarkFrame) -> float:
        if not frame.pose_detected or len(self._rolling_frames) < 2:
            return 0.0

        prev = self._rolling_frames[-2]
        if not prev.pose_detected:
            return 0.0

        wrist_i = RIGHT_WRIST if self.cfg.active_hand == "right" else LEFT_WRIST
        shoulder_i = RIGHT_SHOULDER if self.cfg.active_hand == "right" else LEFT_SHOULDER

        lhip = frame.landmarks[LEFT_HIP, :2]
        rhip = frame.landmarks[RIGHT_HIP, :2]
        lsh = frame.landmarks[LEFT_SHOULDER, :2]
        rsh = frame.landmarks[RIGHT_SHOULDER, :2]
        mid_hip = 0.5 * (lhip + rhip)
        mid_sh = 0.5 * (lsh + rsh)
        scale = float(np.linalg.norm(mid_sh - mid_hip))
        scale = max(scale, 1e-4)

        dt = max(frame.timestamp - prev.timestamp, 1e-4)

        p = (frame.landmarks[wrist_i, :3] - frame.landmarks[shoulder_i, :3]) / scale
        q = (prev.landmarks[wrist_i, :3] - prev.landmarks[shoulder_i, :3]) / scale
        speed = float(np.linalg.norm((p - q) / dt))
        return speed

    def _update_segmentation(self, frame: LandmarkFrame, speed: float, now: float) -> None:
        if not frame.pose_detected:
            return

        if self._state == "idle":
            if now - self._last_event_time < self.cfg.cooldown_s:
                return

            if speed >= self.cfg.motion_start_speed:
                pre = [
                    f for f in self._rolling_frames
                    if now - f.timestamp <= self.cfg.pre_seconds and f.pose_detected
                ]
                self._segment_frames = list(pre)
                self._segment_start_time = self._segment_frames[0].timestamp if self._segment_frames else now
                self._state = "recording"
                self._end_hold = 0
            return

        if self._state == "recording":
            self._segment_frames.append(frame)
            duration = now - self._segment_start_time

            if speed <= self.cfg.motion_end_speed:
                self._end_hold += 1
            else:
                self._end_hold = 0

            should_end = (
                (duration >= self.cfg.min_segment_seconds and self._end_hold >= self.cfg.motion_end_hold_frames)
                or duration >= self.cfg.max_segment_seconds
            )

            if should_end:
                self._finish_segment(now)

    def _finish_segment(self, now: float) -> None:
        frames = [f for f in self._segment_frames if f.pose_detected]
        self._state = "idle"
        self._segment_frames = []
        self._end_hold = 0

        if len(frames) < 5:
            return

        feats, names = extract_punch_features(frames, hand=self.cfg.active_hand, resample_len=self.resample_len)

        # Safety check.
        if names != self.feature_names:
            print("[TrajectoryVisionAgent] feature mismatch; ignored segment")
            return

        X = feats.reshape(1, -1)
        pred = str(self.model.predict(X)[0])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)
            conf = float(proba[classes.index(pred)])
        else:
            conf = 1.0

        self._latest_prediction = pred
        self._latest_confidence = conf

        if pred in self.cfg.negative_labels:
            print(f"[TrajectoryVisionAgent] negative/idle pred={pred} conf={conf:.2f}")
            return

        if conf < self.cfg.confidence_threshold:
            print(f"[TrajectoryVisionAgent] low_conf pred={pred} conf={conf:.2f}")
            return

        event = PlayerActionEvent(
            timestamp=now,
            action_type=pred,
            hand=self.cfg.active_hand,
            phase="impact",
            start_time=frames[0].timestamp,
            impact_time=frames[len(frames)//2].timestamp,
            end_time=frames[-1].timestamp,
            confidence=conf,
            camera_speed=self._active_hand_speed(frames[-1]),
            reason=f"trajectory_classifier: pred={pred}, conf={conf:.2f}, frames={len(frames)}",
        )

        self._last_event_time = now
        self._push_event(event)

    def _push_event(self, event: PlayerActionEvent) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
            self._event_queue.put_nowait(event)

        print(
            f"[TrajectoryVisionAgent] event action={event.action_type} "
            f"conf={event.confidence:.2f} reason={event.reason}"
        )

    def _draw_debug(self, frame_bgr, pose_frame: LandmarkFrame, speed: float):
        img = draw_pose(frame_bgr, pose_frame)
        cv2.putText(img, f"State: {self._state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(img, f"Hand: {self.cfg.active_hand}  speed={speed:.2f}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(img, f"Pred: {self._latest_prediction}  conf={self._latest_confidence:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        cv2.putText(img, f"Status: {self._latest_status}", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        return img


def run_debug(agent: VisionAgent):
    print("[TrajectoryVisionAgent] debug mode. Press q to quit.")
    while agent.is_running():
        img = agent.get_latest_debug_image()
        if img is not None:
            cv2.imshow("RadarBox Trajectory VisionAgent", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        time.sleep(0.001)
    agent.stop()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="RadarBox trajectory-based VisionAgent")
    parser.add_argument("--classifier", default="models/punch_classifier.joblib")
    parser.add_argument("--model-path", default="models/pose_landmarker_lite.task")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--active-hand", choices=["left", "right"], default="right")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--motion-start-speed", type=float, default=1.00)
    parser.add_argument("--motion-end-speed", type=float, default=0.45)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--cooldown", type=float, default=0.45)
    args = parser.parse_args()

    cfg = TrajectoryVisionConfig(
        classifier_path=args.classifier,
        model_path=args.model_path,
        camera_index=args.camera_index,
        active_hand=args.active_hand,
        motion_start_speed=args.motion_start_speed,
        motion_end_speed=args.motion_end_speed,
        confidence_threshold=args.confidence_threshold,
        cooldown_s=args.cooldown,
    )

    agent = VisionAgent(cfg)
    agent.start()

    try:
        if args.debug:
            run_debug(agent)
        else:
            while True:
                ev = agent.get_next_action_event()
                if ev:
                    print(ev)
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
