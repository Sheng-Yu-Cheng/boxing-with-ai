#!/usr/bin/env python3
r"""
record_punch_dataset.py

Manual MediaPipe trajectory recorder for RadarBox.

Recommended data collection:

    python .\src\record_punch_dataset.py --label right_straight --hand right --count 30
    python .\src\record_punch_dataset.py --label right_hook     --hand right --count 30
    python .\src\record_punch_dataset.py --label right_uppercut --hand right --count 30
    python .\src\record_punch_dataset.py --label negative      --hand right --count 30

How to record:
    - A webcam window opens.
    - Press SPACE to record one sample.
    - After countdown, throw exactly one punch during the recording window.
    - Press q to quit.

The saved files are compressed .npz trajectory segments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from punch_vision_common import (
    PoseDetectorConfig,
    PoseTaskDetector,
    LandmarkFrame,
    draw_pose,
    save_segment_npz,
)


def open_camera(index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")
    return cap


def draw_text(img, lines, x=20, y=35, scale=0.7):
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        y += int(32 * scale + 12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record RadarBox punch trajectory dataset")
    parser.add_argument("--label", required=True, help="right_straight / right_hook / right_uppercut / negative")
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--out-dir", default="data/punch_dataset")
    parser.add_argument("--model-path", default="models/pose_landmarker_lite.task")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--record-seconds", type=float, default=1.10)
    parser.add_argument("--countdown", type=float, default=0.60)
    parser.add_argument("--mirror-display", action="store_true", help="mirror display only; saved landmarks are from unmirrored frame")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PoseTaskDetector(PoseDetectorConfig(model_path=args.model_path))
    cap = open_camera(args.camera_index, args.width, args.height, args.fps)

    print()
    print("[Recorder] label:", args.label)
    print("[Recorder] hand :", args.hand)
    print("[Recorder] Press SPACE to record one sample. Press q to quit.")
    print()

    recorded = 0
    frame_id = 0
    last_pose = None

    try:
        while recorded < args.count:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[Recorder] camera read failed")
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            pose_frame = detector.detect(frame_bgr, now, frame_id)
            frame_id += 1
            last_pose = pose_frame

            display = draw_pose(frame_bgr, pose_frame)
            if args.mirror_display:
                display = cv2.flip(display, 1)

            draw_text(display, [
                f"Label: {args.label}    Hand: {args.hand}",
                f"Recorded: {recorded}/{args.count}",
                "SPACE: record sample    q: quit",
                "Throw exactly ONE punch during GO window.",
            ])

            cv2.imshow("RadarBox Punch Recorder", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key != ord(" "):
                continue

            # Countdown.
            start_countdown = time.perf_counter()
            while time.perf_counter() - start_countdown < args.countdown:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                remaining = args.countdown - (time.perf_counter() - start_countdown)
                pose_frame = detector.detect(frame_bgr, time.perf_counter(), frame_id)
                frame_id += 1
                display = draw_pose(frame_bgr, pose_frame)
                if args.mirror_display:
                    display = cv2.flip(display, 1)
                draw_text(display, [
                    f"Get ready: {remaining:.1f}s",
                    f"Label: {args.label}",
                ], scale=1.0)
                cv2.imshow("RadarBox Punch Recorder", display)
                cv2.waitKey(1)

            # Record window.
            frames: list[LandmarkFrame] = []
            record_start = time.perf_counter()
            while time.perf_counter() - record_start < args.record_seconds:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                now = time.perf_counter()
                pose_frame = detector.detect(frame_bgr, now, frame_id)
                frame_id += 1
                frames.append(pose_frame)

                display = draw_pose(frame_bgr, pose_frame)
                if args.mirror_display:
                    display = cv2.flip(display, 1)
                draw_text(display, [
                    "GO!",
                    f"Recording {time.perf_counter() - record_start:.2f}/{args.record_seconds:.2f}s",
                    f"Label: {args.label}",
                ], scale=1.0)
                cv2.imshow("RadarBox Punch Recorder", display)
                cv2.waitKey(1)

            valid = sum(1 for f in frames if f.pose_detected)
            if valid < max(5, len(frames) // 2):
                print(f"[Recorder] skipped sample, pose lost too often: valid={valid}/{len(frames)}")
                continue

            ts = int(time.time() * 1000)
            fname = f"{args.label}_{args.hand}_{recorded:03d}_{ts}.npz"
            path = out_dir / fname

            meta = {
                "label": args.label,
                "hand": args.hand,
                "record_seconds": args.record_seconds,
                "camera_index": args.camera_index,
                "created_unix": time.time(),
                "valid_frames": valid,
                "total_frames": len(frames),
            }
            save_segment_npz(path, frames, args.label, args.hand, meta)

            with open(Path(args.out_dir) / "metadata.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"file": str(path), **meta}, ensure_ascii=False) + "\n")

            recorded += 1
            print(f"[Recorder] saved {path}  valid={valid}/{len(frames)}")

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

    print(f"[Recorder] done: {recorded}/{args.count}")


if __name__ == "__main__":
    main()
