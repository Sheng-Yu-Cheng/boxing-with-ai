#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.radar_agent import RadarAgent, RadarConfig  # noqa: E402


def atomic_write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="ascii")
    tmp.replace(p)


def parse_fixed_beam_codes(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--fixed-beam-codes must contain exactly three comma-separated integers")
    return tuple((int(float(p)) & 0x3F) for p in parts)  # type: ignore[return-value]


def fmt(value, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        if not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def get_canvas_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        rgba = np.asarray(rgba).reshape(-1, int(fig.bbox.width), 4)
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def nearest_index(axis: np.ndarray, value: Optional[float]) -> Optional[int]:
    if value is None or axis.size == 0:
        return None
    try:
        return int(np.argmin(np.abs(axis - float(value))))
    except (TypeError, ValueError):
        return None


def local_roi_peak_db(
    rv_db: np.ndarray,
    range_axis: np.ndarray,
    vel_axis: np.ndarray,
    range_m: Optional[float],
    velocity_mps: Optional[float],
    half_bins: int = 2,
) -> float:
    rr = nearest_index(range_axis, range_m)
    vv = nearest_index(vel_axis, velocity_mps)
    if rr is None or vv is None:
        return float("nan")
    r0 = max(0, rr - half_bins)
    r1 = min(rv_db.shape[0], rr + half_bins + 1)
    v0 = max(0, vv - half_bins)
    v1 = min(rv_db.shape[1], vv + half_bins + 1)
    roi = rv_db[r0:r1, v0:v1]
    if roi.size == 0 or not np.any(np.isfinite(roi)):
        return float("nan")
    return float(np.nanmax(roi))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RadarBox side-by-side camera + RV beam steering demo recorder")
    p.add_argument("--mode", choices=["fixed", "feedback"], default="fixed")
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--output", default="beam_demo.mp4")
    p.add_argument("--no-record", action="store_true", help="show the live demo without writing an MP4 video")
    p.add_argument("--metrics-csv", default=None)

    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--no-camera", action="store_true", help="disable live camera capture and show only the radar RV map")

    p.add_argument("--beam-cmd-file", default=r"C:\temp\radarbox_beam_cmd.txt")
    p.add_argument("--fixed-beam-codes", default="0,0,0")

    p.add_argument("--rv-vmin", type=float, default=0.0)
    p.add_argument("--rv-vmax", type=float, default=120.0)
    p.add_argument("--range-min-m", type=float, default=0.0)
    p.add_argument("--range-max-m", type=float, default=4.0)
    p.add_argument("--vel-min-mps", type=float, default=-16.0)
    p.add_argument("--vel-max-mps", type=float, default=16.0)

    p.add_argument("--figure-fps", type=int, default=20)
    p.add_argument("--radar-range-scale", type=float, default=1.0)
    p.add_argument("--title", default=None)
    return p


class BeamDemoRecorder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.start_time = time.perf_counter()
        self.last_camera_rgb = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
        self.latest_display_rv = None
        self.latest_display_range_axis = None
        self.latest_display_vel_axis = None
        self.latest_radar_frame_id = None
        self.latest_global_peak_db = float("nan")
        self.video_writer = None
        self.metrics_file = None
        self.metrics_writer = None
        self.closed = False
        self.record_metrics_this_update = False

        self.cap = None
        if not args.no_camera:
            self.cap = cv2.VideoCapture(args.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera index {args.camera_index}")

        if args.mode == "fixed":
            codes = parse_fixed_beam_codes(args.fixed_beam_codes)
            atomic_write_text(args.beam_cmd_file, f"{codes[0]},{codes[1]},{codes[2]}\n")
            enable_feedback = False
        else:
            enable_feedback = True

        self.radar = RadarAgent(RadarConfig(
            enable_aoa_feedback=enable_feedback,
            beam_cmd_file=args.beam_cmd_file,
        ))

        self.output_path = Path(args.output)
        if not args.no_record:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.metrics_csv:
            metrics_path = Path(args.metrics_csv)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
            self.metrics_writer = csv.DictWriter(
                self.metrics_file,
                fieldnames=[
                    "time_s",
                    "radar_frame_id",
                    "tracking_valid",
                    "range_m",
                    "velocity_mps",
                    "theta_raw_deg",
                    "theta_smooth_deg",
                    "snr_db",
                    "beam_tx0",
                    "beam_tx1",
                    "beam_tx2",
                    "roi_peak_db",
                    "global_peak_db",
                ],
            )
            self.metrics_writer.writeheader()

        print("[BeamDemo] Make sure mmWave Studio Lua polling script is running.")
        print(f"[BeamDemo] Beam command file: {args.beam_cmd_file}")
        print(f"[BeamDemo] Mode: {args.mode}")
        print(f"[BeamDemo] Recording: {'off' if args.no_record else str(self.output_path)}")
        print(f"[BeamDemo] Camera: {'off' if args.no_camera else f'index {args.camera_index}'}")
        print(f"[BeamDemo] RV color scale: vmin={args.rv_vmin}, vmax={args.rv_vmax}")
        print("[BeamDemo] Walk path: radar-left -> boresight center -> radar-right")

    def setup_figure(self):
        title = self.args.title
        if not title:
            title = "A: Fixed broadside, no AoA feedback" if self.args.mode == "fixed" else "B: AoA-feedback beam steering"

        if self.args.no_camera:
            self.fig, self.ax_rv = plt.subplots(1, 1, figsize=(7.6, 6.2), dpi=100)
            self.ax_cam = None
        else:
            self.fig, (self.ax_cam, self.ax_rv) = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=120)
        self.fig.suptitle(title)

        self.cam_im = None
        self.cam_text = None
        if self.ax_cam is not None:
            self.cam_im = self.ax_cam.imshow(self.last_camera_rgb)
            self.ax_cam.axis("off")
            self.cam_text = self.ax_cam.text(
                0.02,
                0.04,
                "",
                transform=self.ax_cam.transAxes,
                color="white",
                fontsize=10,
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
            )

        initial_rv = np.full((64, 64), self.args.rv_vmin, dtype=np.float32)
        self.rv_im = self.ax_rv.imshow(
            initial_rv,
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=self.args.rv_vmin,
            vmax=self.args.rv_vmax,
            extent=[self.args.vel_min_mps, self.args.vel_max_mps, self.args.range_min_m, self.args.range_max_m],
        )
        self.ax_rv.set_xlabel("Velocity (m/s)")
        self.ax_rv.set_ylabel("Range (m)")
        self.ax_rv.set_xlim(self.args.vel_min_mps, self.args.vel_max_mps)
        self.ax_rv.set_ylim(self.args.range_min_m, self.args.range_max_m)
        self.ax_rv.set_title("A: Fixed broadside beam, no AoA feedback" if self.args.mode == "fixed" else "B: AoA-feedback beam steering")
        self.tracking_marker, = self.ax_rv.plot([], [], "wo", markeredgecolor="black", markersize=8)
        self.tracking_text = self.ax_rv.text(
            0.02,
            0.98,
            "",
            transform=self.ax_rv.transAxes,
            color="white",
            fontsize=9,
            va="top",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )
        cbar = self.fig.colorbar(self.rv_im, ax=self.ax_rv, fraction=0.046, pad=0.04)
        cbar.set_label("Power (dB)")
        self.fig.tight_layout()

        # Keep a FuncAnimation object alive for a live animated matplotlib figure.
        self.ani = FuncAnimation(
            self.fig,
            self.update_live,
            interval=max(1, int(1000 / max(self.args.figure_fps, 1))),
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=False)

    def start(self):
        self.radar.start()
        self.setup_figure()

        frame_period = 1.0 / max(float(self.args.figure_fps), 1.0)
        next_frame_time = time.perf_counter()
        try:
            while True:
                now = time.perf_counter()
                if now - self.start_time >= self.args.duration_s:
                    break
                if now < next_frame_time:
                    plt.pause(min(0.01, next_frame_time - now))
                    continue

                self.record_metrics_this_update = True
                self.update(None)
                self.record_metrics_this_update = False
                if not self.args.no_record:
                    canvas_bgr = get_canvas_bgr(self.fig)
                    if self.video_writer is None:
                        h, w = canvas_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self.video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.args.figure_fps, (w, h))
                        if not self.video_writer.isOpened():
                            raise RuntimeError(f"Could not open video writer: {self.output_path}")
                    self.video_writer.write(canvas_bgr)
                plt.pause(0.001)
                next_frame_time += frame_period
        finally:
            self.close()

    def read_camera(self):
        if self.args.no_camera or self.cap is None:
            return self.last_camera_rgb
        ok, frame_bgr = self.cap.read()
        if ok:
            frame_bgr = cv2.resize(frame_bgr, (self.args.camera_width, self.args.camera_height))
            self.last_camera_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.last_camera_rgb

    def get_display_rv(self):
        dbg = self.radar.get_latest_debug_frame()
        if dbg is None or dbg.rd_db.size == 0:
            return None, None, None, None

        range_axis = np.asarray(dbg.range_axis_m, dtype=np.float64) * float(self.args.radar_range_scale)
        vel_axis = np.asarray(dbg.velocity_axis_mps, dtype=np.float64)
        r_mask = (range_axis >= self.args.range_min_m) & (range_axis <= self.args.range_max_m)
        v_mask = (vel_axis >= self.args.vel_min_mps) & (vel_axis <= self.args.vel_max_mps)
        r_idx = np.where(r_mask)[0]
        v_idx = np.where(v_mask)[0]
        if len(r_idx) == 0 or len(v_idx) == 0:
            return None, None, None, dbg.frame_id

        rv = np.asarray(dbg.rd_db[np.ix_(r_idx, v_idx)], dtype=np.float32)
        display_range_axis = range_axis[r_idx]
        display_vel_axis = vel_axis[v_idx]
        return rv, display_range_axis, display_vel_axis, dbg.frame_id

    def update(self, _frame):
        if self.closed:
            return tuple(a for a in (self.cam_im, self.rv_im, self.tracking_marker, self.cam_text, self.tracking_text) if a is not None)

        elapsed = time.perf_counter() - self.start_time
        if not self.args.no_camera and self.cam_im is not None and self.cam_text is not None:
            camera_rgb = self.read_camera()
            self.cam_im.set_data(camera_rgb)
            self.cam_text.set_text(
                f"mode: {self.args.mode}\n"
                f"elapsed: {elapsed:5.1f}s / {self.args.duration_s:.1f}s\n"
                f"output: {self.output_path}\n"
                "Walk left -> center -> right"
            )

        rv, range_axis, vel_axis, frame_id = self.get_display_rv()
        if rv is not None:
            self.latest_display_rv = rv
            self.latest_display_range_axis = range_axis
            self.latest_display_vel_axis = vel_axis
            self.latest_radar_frame_id = frame_id
            self.latest_global_peak_db = float(np.nanmax(rv)) if np.any(np.isfinite(rv)) else float("nan")
            self.rv_im.set_data(rv)
            self.rv_im.set_extent([vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]])

        tracking = self.radar.get_latest_tracking_state()
        tracking_range = getattr(tracking, "range_m", None)
        if tracking_range is not None:
            tracking_range = float(tracking_range) * float(self.args.radar_range_scale)
        tracking_velocity = getattr(tracking, "peak_velocity_mps", None)
        tracking_valid = bool(getattr(tracking, "valid", False))

        beam_codes = (
            getattr(tracking, "beam_tx0_code", None),
            getattr(tracking, "beam_tx1_code", None),
            getattr(tracking, "beam_tx2_code", None),
        )
        beam_text = "N/A" if all(x is None for x in beam_codes) else ",".join("N/A" if x is None else str(int(x)) for x in beam_codes)

        if tracking_valid and tracking_range is not None and tracking_velocity is not None:
            self.tracking_marker.set_data([float(tracking_velocity)], [float(tracking_range)])
        else:
            self.tracking_marker.set_data([], [])

        self.tracking_text.set_text(
            f"valid={tracking_valid}\n"
            f"R={fmt(tracking_range)} m  v={fmt(tracking_velocity)} m/s\n"
            f"theta_raw={fmt(getattr(tracking, 'theta_raw_deg', None))} deg\n"
            f"theta_smooth={fmt(getattr(tracking, 'theta_smooth_deg', None))} deg\n"
            f"SNR={fmt(getattr(tracking, 'snr_db', None))} dB\n"
            f"beam={beam_text}"
        )

        if self.record_metrics_this_update:
            self.write_metrics(elapsed, tracking, tracking_range, tracking_velocity)
        return tuple(a for a in (self.cam_im, self.rv_im, self.tracking_marker, self.cam_text, self.tracking_text) if a is not None)

    def update_live(self, frame):
        self.record_metrics_this_update = False
        return self.update(frame)

    def write_metrics(self, elapsed: float, tracking, tracking_range: Optional[float], tracking_velocity: Optional[float]) -> None:
        if self.closed or self.metrics_writer is None:
            return

        roi_peak = float("nan")
        if (
            bool(getattr(tracking, "valid", False))
            and self.latest_display_rv is not None
            and self.latest_display_range_axis is not None
            and self.latest_display_vel_axis is not None
        ):
            roi_peak = local_roi_peak_db(
                self.latest_display_rv,
                self.latest_display_range_axis,
                self.latest_display_vel_axis,
                tracking_range,
                tracking_velocity,
            )

        self.metrics_writer.writerow({
            "time_s": f"{elapsed:.3f}",
            "radar_frame_id": "" if self.latest_radar_frame_id is None else self.latest_radar_frame_id,
            "tracking_valid": bool(getattr(tracking, "valid", False)),
            "range_m": "" if tracking_range is None else f"{tracking_range:.6f}",
            "velocity_mps": "" if tracking_velocity is None else f"{float(tracking_velocity):.6f}",
            "theta_raw_deg": "" if getattr(tracking, "theta_raw_deg", None) is None else f"{float(tracking.theta_raw_deg):.6f}",
            "theta_smooth_deg": "" if getattr(tracking, "theta_smooth_deg", None) is None else f"{float(tracking.theta_smooth_deg):.6f}",
            "snr_db": "" if getattr(tracking, "snr_db", None) is None else f"{float(tracking.snr_db):.6f}",
            "beam_tx0": "" if getattr(tracking, "beam_tx0_code", None) is None else int(tracking.beam_tx0_code),
            "beam_tx1": "" if getattr(tracking, "beam_tx1_code", None) is None else int(tracking.beam_tx1_code),
            "beam_tx2": "" if getattr(tracking, "beam_tx2_code", None) is None else int(tracking.beam_tx2_code),
            "roi_peak_db": "" if not np.isfinite(roi_peak) else f"{roi_peak:.6f}",
            "global_peak_db": "" if not np.isfinite(self.latest_global_peak_db) else f"{self.latest_global_peak_db:.6f}",
        })

    def close(self):
        self.closed = True
        try:
            if getattr(self, "ani", None) is not None and self.ani.event_source is not None:
                self.ani.event_source.stop()
            self.radar.stop()
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            if self.metrics_file is not None:
                self.metrics_file.close()
            plt.close("all")
        if self.args.no_record:
            print("[BeamDemo] Video recording disabled (--no-record)")
        else:
            print(f"[BeamDemo] Wrote video: {self.output_path}")
        if self.args.metrics_csv:
            print(f"[BeamDemo] Wrote metrics: {self.args.metrics_csv}")


def main() -> int:
    args = build_arg_parser().parse_args()
    recorder = BeamDemoRecorder(args)
    recorder.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
