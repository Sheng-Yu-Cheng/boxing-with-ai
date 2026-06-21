#!/usr/bin/env python3
r"""
scripts/run_fusion_debug.py

Debug runner for integrating the three runtime components:

    1. VisionAgent        -> camera punch / block events
    2. RadarAgent         -> AWR2243 + DCA1000 Doppler burst events
    3. FusionCore         -> fused player events for GameEngine

Typical full test:

    python .\scripts\run_fusion_debug.py `
      --vision-debug `
      --classifier .\models\punch_classifier.joblib `
      --model-path .\models\pose_landmarker_lite.task `
      --active-hand right `
      --confidence-threshold 0.60

Camera-only test without radar:

    python .\scripts\run_fusion_debug.py `
      --no-radar `
      --vision-debug `
      --classifier .\models\punch_classifier.joblib `
      --model-path .\models\pose_landmarker_lite.task `
      --active-hand right `
      --confidence-threshold 0.60

Before running:
    pip install -e .

Notes:
    - Start this script first.
    - Then start mmWave Studio capture / StartFrame for radar streaming.
    - If radar is not streaming yet, straight punches will still be emitted as
      camera-only unless --require-radar-for-straight is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import asdict
from typing import Any, Optional

try:
    import cv2
except Exception:
    cv2 = None

from core.vision_agent import VisionAgent, TrajectoryVisionConfig
from core.fusion_core import FusionCore, FusionConfig

# Radar is optional because camera-only debug is useful.
try:
    from core.radar_agent import RadarAgent, RadarConfig
except Exception as e:
    RadarAgent = None
    RadarConfig = None
    _RADAR_IMPORT_ERROR = e
else:
    _RADAR_IMPORT_ERROR = None


# ============================================================
# Helpers
# ============================================================

def make_config(config_cls, **kwargs):
    """
    Create a dataclass config while ignoring unsupported kwargs.

    This makes the script robust if RadarConfig / TrajectoryVisionConfig changes
    slightly between versions.
    """
    if not dataclasses.is_dataclass(config_cls):
        return config_cls(**kwargs)

    allowed = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return config_cls(**filtered)


def fmt_float(x, nd=2, none="-"):
    if x is None:
        return none
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def print_fused_event(event):
    """
    Compact one-line fused event printout.
    """
    radar_part = ""
    if event.radar_valid:
        radar_part = (
            f" radar_v={fmt_float(event.radar_abs_velocity_mps)}m/s"
            f" snr={fmt_float(event.radar_snr_db)}dB"
            f" range={fmt_float(event.radar_peak_range_m)}m"
        )

    print(
        "[FUSED] "
        f"action={event.action_type:<16} "
        f"source={event.source:<28} "
        f"conf={event.final_confidence:.2f} "
        f"intensity={event.intensity_score:.2f} "
        f"damage={event.damage_scale:.2f} "
        f"vision_conf={event.vision_confidence:.2f} "
        f"radar_valid={event.radar_valid}"
        f"{radar_part} "
        f"reason={event.fusion_reason}"
    )


def print_radar_health(radar):
    if radar is None:
        print("[RadarHealth] radar disabled")
        return

    try:
        h = radar.get_health()
    except Exception as e:
        print(f"[RadarHealth] ERROR: {type(e).__name__}: {e}")
        return

    status = getattr(h, "status", "UNKNOWN")
    fps = getattr(h, "fps", None)
    expected_fps = getattr(h, "expected_fps", None)
    mbps = getattr(h, "mb_per_sec", None)
    pps = getattr(h, "packets_per_sec", None)
    gaps = getattr(h, "packet_gap_count", None)
    frames = getattr(h, "total_frame_count", None)
    valid_frames = getattr(h, "valid_frame_count", None)

    print(
        "[RadarHealth] "
        f"status={status} "
        f"fps={fmt_float(fps)}/{fmt_float(expected_fps)} "
        f"MBps={fmt_float(mbps)} "
        f"pps={fmt_float(pps)} "
        f"gaps={gaps} "
        f"frames={frames} valid={valid_frames}"
    )


def print_vision_hint():
    print()
    print("=== Fusion Debug Controls ===")
    print("Ctrl+C : stop")
    if cv2 is not None:
        print("q in vision debug window : stop")
    print()
    print("Expected behavior:")
    print("  right_straight -> source=vision+radar if radar burst is valid")
    print("  right_hook     -> source=vision")
    print("  right_uppercut -> source=vision")
    print("  block          -> source=block")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RadarBox FusionCore debug runner")

    # VisionAgent args.
    parser.add_argument("--classifier", default="models/punch_classifier.joblib")
    parser.add_argument("--model-path", default="models/pose_landmarker_lite.task")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--active-hand", choices=["left", "right"], default="right")
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--motion-start-speed", type=float, default=1.00)
    parser.add_argument("--motion-end-speed", type=float, default=0.45)
    parser.add_argument("--vision-cooldown", type=float, default=0.45)
    parser.add_argument("--vision-debug", action="store_true")

    # Block args, supported by the newer vision_agent_trajectory.py with block.
    parser.add_argument("--disable-block", action="store_true")
    parser.add_argument("--block-enter-frames", type=int, default=8)
    parser.add_argument("--block-exit-frames", type=int, default=6)
    parser.add_argument("--block-max-hand-speed", type=float, default=0.85)

    # RadarAgent args.
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--pc-ip", default="192.168.33.30")
    parser.add_argument("--dca-ip", default="192.168.33.180")
    parser.add_argument("--data-port", type=int, default=4098)
    parser.add_argument("--radar-range-min", type=float, default=0.6)
    parser.add_argument("--radar-range-max", type=float, default=2.5)
    parser.add_argument("--radar-min-abs-velocity", type=float, default=0.5)
    parser.add_argument("--radar-background-frames", type=int, default=0)

    # FusionCore args.
    parser.add_argument("--require-radar-for-straight", action="store_true")
    parser.add_argument("--radar-pre-impact", type=float, default=0.10)
    parser.add_argument("--radar-post-impact", type=float, default=0.15)
    parser.add_argument("--fusion-verbose", action="store_true")
    parser.add_argument("--status-period", type=float, default=2.0)
    parser.add_argument("--loop-sleep", type=float, default=0.005)

    args = parser.parse_args()

    print()
    print("=== RadarBox Fusion Debug ===")
    print("classifier :", args.classifier)
    print("pose model :", args.model_path)
    print("active hand:", args.active_hand)
    print("radar      :", "disabled" if args.no_radar else "enabled")
    print()

    # -----------------------------
    # Create VisionAgent
    # -----------------------------

    vision_cfg = make_config(
        TrajectoryVisionConfig,
        classifier_path=args.classifier,
        model_path=args.model_path,
        camera_index=args.camera_index,
        active_hand=args.active_hand,
        motion_start_speed=args.motion_start_speed,
        motion_end_speed=args.motion_end_speed,
        confidence_threshold=args.confidence_threshold,
        cooldown_s=args.vision_cooldown,
        enable_block=not args.disable_block,
        block_enter_frames=args.block_enter_frames,
        block_exit_frames=args.block_exit_frames,
        block_max_hand_speed=args.block_max_hand_speed,
    )

    vision = VisionAgent(vision_cfg)

    # -----------------------------
    # Create RadarAgent
    # -----------------------------

    radar = None

    if not args.no_radar:
        if RadarAgent is None or RadarConfig is None:
            print("[WARN] radar_agent import failed.")
            print("       Running camera-only fusion.")
            print(f"       Import error: {_RADAR_IMPORT_ERROR}")
        else:
            radar_cfg = make_config(
                RadarConfig,
                pc_ip=args.pc_ip,
                dca_ip=args.dca_ip,
                data_port=args.data_port,
                player_range_min=args.radar_range_min,
                player_range_max=args.radar_range_max,
                background_calibration_frames=args.radar_background_frames,
            )
            radar = RadarAgent(radar_cfg)

    # -----------------------------
    # Create FusionCore
    # -----------------------------

    fusion_cfg = FusionConfig(
        require_radar_for_straight=args.require_radar_for_straight,
        radar_pre_impact_s=args.radar_pre_impact,
        radar_post_impact_s=args.radar_post_impact,
        radar_range_min_m=args.radar_range_min,
        radar_range_max_m=args.radar_range_max,
        radar_min_abs_velocity_mps=args.radar_min_abs_velocity,
        verbose=args.fusion_verbose,
    )

    fusion = FusionCore(
        vision_agent=vision,
        radar_agent=radar,
        config=fusion_cfg,
    )

    # -----------------------------
    # Start agents
    # -----------------------------

    try:
        if radar is not None:
            radar.start()
            print("[Main] RadarAgent started.")
            print("[Main] Now start mmWave Studio DCA1000 capture / StartFrame if not already running.")
        else:
            print("[Main] RadarAgent disabled. Straight punches will be camera-only.")

        vision.start()
        print("[Main] VisionAgent started.")

        print_vision_hint()

        last_status = 0.0

        while True:
            # Synchronous fusion polling keeps debug behavior simple.
            fused = fusion.poll_once()
            if fused is not None:
                print_fused_event(fused)

            now = time.perf_counter()
            if now - last_status >= args.status_period:
                last_status = now
                print_radar_health(radar)

            if args.vision_debug:
                if cv2 is None:
                    print("[WARN] OpenCV not available; cannot show debug image.")
                    args.vision_debug = False
                else:
                    img = None
                    if hasattr(vision, "get_latest_debug_image"):
                        img = vision.get_latest_debug_image()
                    elif hasattr(vision, "get_latest_pose_debug_frame"):
                        dbg = vision.get_latest_pose_debug_frame()
                        img = getattr(dbg, "image_bgr", None) if dbg is not None else None

                    if img is not None:
                        cv2.imshow("RadarBox Fusion Debug - Vision", img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("[Main] q pressed")
                            break

            time.sleep(args.loop_sleep)

    except KeyboardInterrupt:
        print()
        print("[Main] Ctrl+C")

    finally:
        print("[Main] stopping...")
        try:
            vision.stop()
        except Exception as e:
            print(f"[Main] Vision stop error: {e}")

        if radar is not None:
            try:
                radar.stop()
            except Exception as e:
                print(f"[Main] Radar stop error: {e}")

        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        print("[Main] stopped.")


if __name__ == "__main__":
    main()
