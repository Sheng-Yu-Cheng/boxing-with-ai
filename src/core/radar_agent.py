#!/usr/bin/env python3
"""
radar_agent.py

RadarAgent for RadarBox.

This module turns your AWR2243BOOST + DCA1000 UDP stream into a clean API
for the game/fusion layer:

    radar_agent.start()
    radar_agent.query_burst(t_start, t_end)
    radar_agent.get_health()
    radar_agent.get_latest_debug_frame()
    radar_agent.stop()

It is configured for your current RadarBox radar setup:

    AWR2243BOOST + DCA1000
    1Tx + 4Rx
    Tx0 only
    Complex1x 16-bit ADC
    256 ADC samples
    64 chirps/frame
    FrameConfig(0, 0, 0, 64, 20, 0, 1)
    frame period = 20 ms
    idle = 20 us
    ramp end = 40 us
    chirp period ~= 60 us
    expected stream rate ~= 13.11 MB/s

Important:
    This script DOES NOT configure mmWave Studio.
    This script DOES NOT start DCA1000 record/streaming.
    This script DOES NOT start radar frame.

Run order:
    1. Run your mmWave Studio Lua config.
    2. Start this RadarAgent / Python receiver.
    3. Start DCA1000 streaming / record with your own flow.
    4. Trigger StartFrame with your own flow.
    5. Stop StartFrame / DCA1000 with your own flow.

Timestamp convention:
    All event timestamps use time.perf_counter().
    VisionAgent should also use time.perf_counter() so FusionCore can query:
        radar_agent.query_burst(impact_time - 0.10, impact_time + 0.15)
"""

from __future__ import annotations

import argparse
import math
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import numpy as np


# ============================================================
# Dataclasses: public API
# ============================================================

@dataclass
class RadarConfig:
    # Network
    pc_ip: str = "192.168.33.30"
    dca_ip: str = "192.168.33.180"
    data_port: int = 4098
    socket_rcvbuf: int = 64 * 1024 * 1024
    recvfrom_bytes: int = 9000

    # Current RadarBox AWR2243 config
    num_loops: int = 64
    num_tx: int = 1
    num_rx: int = 4
    num_samples: int = 256
    iq: int = 2
    bytes_per_sample: int = 2

    frame_period_s: float = 0.020
    sample_rate_hz: float = 10e6
    slope_hz_per_s: float = 29.982e12
    start_freq_hz: float = 77e9
    idle_time_s: float = 20e-6
    ramp_end_time_s: float = 40e-6

    # Data format
    dca_header_bytes: int = 10

    # Processing
    range_fft_keep_half: bool = True
    enable_fast_time_dc_removal: bool = True
    enable_slow_time_mti: bool = True
    use_hanning_window: bool = True

    # Ring buffer
    radar_buffer_seconds: float = 2.0

    # Default punch search ROI
    player_range_min_m: float = 0.60
    player_range_max_m: float = 2.50
    min_abs_velocity_mps: float = 0.50
    max_abs_velocity_mps: float = 15.50

    # Burst detection thresholds
    burst_snr_min_db: float = 8.0
    burst_snr_full_score_db: float = 25.0
    punch_min_valid_velocity_mps: float = 3.0
    punch_full_scale_velocity_mps: float = 8.0

    # Intensity normalization for game score
    normal_punch_velocity_mps: float = 2.0
    critical_punch_velocity_mps: float = 8.0

    # Optional background subtraction
    background_scale: float = 1.05

    # Console logging
    health_report_interval_s: float = 2.0
    debug: bool = False
    debug_top_candidates: int = 5

    # Continuous UI/debug tracking from 4 physical Rx phase slope.
    # Important: this is for the current 3Tx simultaneous beamforming setup,
    # where the decoded ADC shape remains [loop, 1, sample, 4Rx].
    enable_rx_phase_aoa: bool = True
    aoa_min_snr_db: float = 2.0
    aoa_alpha: float = 0.25
    aoa_max_update_step_deg: float = 8.0
    aoa_max_abs_deg: float = 45.0
    aoa_sign: float = 1.0
    aoa_scale: float = 1.0
    aoa_offset_deg: float = 0.0
    rx_phase_offsets_rad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    tracking_range_min_m: float = 0.30
    tracking_range_max_m: float = 3.20
    tracking_min_abs_velocity_mps: float = 0.0
    tracking_max_abs_velocity_mps: float = 16.0
    tracking_enable_temporal_lock: bool = True
    tracking_range_gate_m: float = 0.45
    tracking_velocity_gate_mps: float = 3.0
    tracking_theta_gate_deg: float = 35.0
    tracking_snr_min_db: float = 6.0

    tracking_score_snr_weight: float = 1.0
    tracking_score_range_weight: float = 4.0
    tracking_score_velocity_weight: float = 0.5

    tracking_lost_timeout_s: float = 1.0

    # Optional AoA feedback to mmWave Studio Lua file poller.
    enable_aoa_feedback: bool = False
    beam_cmd_file: str = r"C:\temp\radarbox_beam_cmd.txt"
    beam_update_interval_s: float = 0.50
    phase_step_deg: float = 5.625
    beam_min_confidence: float = 0.20
    beam_min_snr_db: float = 6.0

    @property
    def chirp_period_s(self) -> float:
        return self.idle_time_s + self.ramp_end_time_s

    @property
    def frame_bytes(self) -> int:
        return (
            self.num_loops
            * self.num_tx
            * self.num_samples
            * self.num_rx
            * self.iq
            * self.bytes_per_sample
        )

    @property
    def expected_fps(self) -> float:
        return 1.0 / self.frame_period_s

    @property
    def c(self) -> float:
        return 299_792_458.0

    @property
    def wavelength_m(self) -> float:
        return self.c / self.start_freq_hz

    @property
    def v_max_mps(self) -> float:
        return self.wavelength_m / (4.0 * self.chirp_period_s)

    @property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.num_loops * self.chirp_period_s)


@dataclass
class RadarFrame:
    timestamp: float
    frame_id: int

    rd_power: np.ndarray          # [range, doppler], linear power after optional MTI/background
    rd_db: np.ndarray             # [range, doppler], dB
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    rd_complex: Optional[np.ndarray] = None  # [doppler, tx, range, rx]; tx stays 1 for 3Tx simultaneous

    valid: bool = True
    invalid_reason: Optional[str] = None
    invalid_bytes: int = 0


@dataclass
class RadarBurstEvent:
    valid: bool
    timestamp: Optional[float]
    t_start: float
    t_end: float

    peak_velocity_mps: Optional[float]
    abs_peak_velocity_mps: Optional[float]
    peak_range_m: Optional[float]

    peak_power_db: Optional[float]
    noise_floor_db: Optional[float]
    snr_db: Optional[float]

    burst_score: float
    intensity_score: float
    confidence: float

    frames_used: int
    valid_frames_used: int
    burst_frames: int

    reason: str
    top_candidates: list[dict] = field(default_factory=list)


@dataclass
class RadarHealth:
    running: bool
    streaming: bool

    fps: float
    expected_fps: float
    packets_per_sec: float
    mb_per_sec: float

    packet_gap_count: int
    invalid_frame_count: int
    total_frame_count: int
    valid_frame_count: int

    last_packet_time: Optional[float]
    last_frame_time: Optional[float]
    buffer_seconds: float

    status: str


@dataclass
class RadarSummary:
    timestamp: Optional[float]
    peak_velocity_mps: Optional[float]
    peak_range_m: Optional[float]
    peak_power_db: Optional[float]
    noise_floor_db: Optional[float]
    snr_db: Optional[float]
    frame_valid: bool
    reason: str


@dataclass
class RadarTrackingState:
    timestamp: Optional[float]
    valid: bool

    range_m: Optional[float]
    theta_deg: Optional[float]
    theta_raw_deg: Optional[float]
    theta_smooth_deg: Optional[float]

    snr_db: Optional[float]
    peak_velocity_mps: Optional[float]
    abs_peak_velocity_mps: Optional[float]
    peak_power_db: Optional[float]

    beam_tx0_code: Optional[int]
    beam_tx1_code: Optional[int]
    beam_tx2_code: Optional[int]
    beam_theta_deg: Optional[float]

    target_range_bin: Optional[int]
    target_doppler_bin: Optional[int]

    confidence: float
    reason: str
    track_range_m: Optional[float] = None
    track_velocity_mps: Optional[float] = None
    candidate_count: int = 0
    selected_score: Optional[float] = None


@dataclass
class RadarDebugFrame:
    timestamp: float
    frame_id: int
    rd_db: np.ndarray
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray

    peak_range_m: Optional[float]
    peak_velocity_mps: Optional[float]
    peak_power_db: Optional[float]

    valid: bool
    invalid_reason: Optional[str]


@dataclass
class RadarStats:
    total_packets: int = 0
    total_frames: int = 0
    valid_frames: int = 0
    invalid_frames: int = 0
    packet_gaps: int = 0

    inserted_missing_bytes: int = 0
    skipped_overlap_bytes: int = 0
    raw_bytes: int = 0

    first_seq: Optional[int] = None
    last_seq: Optional[int] = None

    start_time: float = field(default_factory=time.perf_counter)
    last_packet_time: Optional[float] = None
    last_frame_time: Optional[float] = None

    recent_packet_times: deque = field(default_factory=lambda: deque(maxlen=512))
    recent_frame_times: deque = field(default_factory=lambda: deque(maxlen=256))

    @property
    def raw_MB(self) -> float:
        return self.raw_bytes / 1e6

    @property
    def elapsed_s(self) -> float:
        return max(time.perf_counter() - self.start_time, 1e-9)

    @property
    def mbps(self) -> float:
        return (self.raw_bytes * 8.0 / 1e6) / self.elapsed_s

    @property
    def MBps(self) -> float:
        return (self.raw_bytes / 1e6) / self.elapsed_s

    @property
    def fps_recent(self) -> float:
        if len(self.recent_frame_times) < 2:
            return 0.0
        dt = self.recent_frame_times[-1] - self.recent_frame_times[0]
        return (len(self.recent_frame_times) - 1) / max(dt, 1e-9)

    @property
    def packets_per_sec_recent(self) -> float:
        if len(self.recent_packet_times) < 2:
            return 0.0
        dt = self.recent_packet_times[-1] - self.recent_packet_times[0]
        return (len(self.recent_packet_times) - 1) / max(dt, 1e-9)


# ============================================================
# RadarAgent
# ============================================================

class RadarAgent:
    """
    Game-facing radar API.

    Game/Fusion layer should call only:
        start()
        stop()
        query_burst(...)
        get_health()
        get_latest_summary()
        get_latest_debug_frame()

    It should NOT access ADC bytes, UDP packets, or FFT internals.
    """

    def __init__(self, config: Optional[RadarConfig] = None):
        self.cfg = config or RadarConfig()

        self.range_axis_m = self._make_range_axis()
        self.velocity_axis_mps = self._make_doppler_axis()

        buffer_frames = int(np.ceil(self.cfg.radar_buffer_seconds * self.cfg.expected_fps * 1.5))
        self._frame_buffer: deque[RadarFrame] = deque(maxlen=max(buffer_frames, 8))

        self._frame_queue: "deque[tuple[int, float, bytes]]" = deque()
        self._frame_queue_maxlen = 64

        self._lock = threading.RLock()
        self._queue_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._rx_thread: Optional[threading.Thread] = None
        self._proc_thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        self._stats = RadarStats()
        self._latest_debug_frame: Optional[RadarDebugFrame] = None
        self._latest_summary = RadarSummary(
            timestamp=None,
            peak_velocity_mps=None,
            peak_range_m=None,
            peak_power_db=None,
            noise_floor_db=None,
            snr_db=None,
            frame_valid=False,
            reason="not_started",
        )
        self._latest_tracking_state = RadarTrackingState(
            timestamp=None,
            valid=False,
            range_m=None,
            theta_deg=None,
            theta_raw_deg=None,
            theta_smooth_deg=None,
            snr_db=None,
            peak_velocity_mps=None,
            abs_peak_velocity_mps=None,
            peak_power_db=None,
            beam_tx0_code=None,
            beam_tx1_code=None,
            beam_tx2_code=None,
            beam_theta_deg=None,
            target_range_bin=None,
            target_doppler_bin=None,
            confidence=0.0,
            reason="not_started",
            track_range_m=None,
            track_velocity_mps=None,
            candidate_count=0,
            selected_score=None,
        )
        self._theta_smooth_deg: Optional[float] = None
        self._last_tracking_debug_print = 0.0
        self._track_range_m = None
        self._track_velocity_mps = None
        self._track_theta_deg = None
        self._track_last_update_time = None
        self._last_beam_update_time = 0.0
        self._last_beam_codes: Optional[tuple[int, int, int]] = None

        # Optional background model.
        self._background_power: Optional[np.ndarray] = None
        self._bg_collect_remaining = 0
        self._bg_collect_frames: list[np.ndarray] = []

        self._player_range_min_m = self.cfg.player_range_min_m
        self._player_range_max_m = self.cfg.player_range_max_m

    # -----------------------------
    # Lifecycle API
    # -----------------------------

    def start(self) -> None:
        if self.is_running():
            return

        self._stop_event.clear()
        self._stats = RadarStats()
        self._track_range_m = None
        self._track_velocity_mps = None
        self._track_theta_deg = None
        self._track_last_update_time = None
        self._theta_smooth_deg = None
        self._last_beam_update_time = 0.0
        self._last_beam_codes = None

        self._rx_thread = threading.Thread(
            target=self._receiver_worker,
            name="RadarAgentReceiver",
            daemon=True,
        )
        self._proc_thread = threading.Thread(
            target=self._processor_worker,
            name="RadarAgentProcessor",
            daemon=True,
        )

        self._rx_thread.start()
        self._proc_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

        if self._rx_thread is not None:
            self._rx_thread.join(timeout=2.0)

        if self._proc_thread is not None:
            self._proc_thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._rx_thread is not None and self._rx_thread.is_alive()

    # -----------------------------
    # Calibration / runtime config
    # -----------------------------

    def set_player_range(self, range_min_m: float, range_max_m: float) -> None:
        if range_min_m >= range_max_m:
            raise ValueError("range_min_m must be < range_max_m")
        with self._lock:
            self._player_range_min_m = float(range_min_m)
            self._player_range_max_m = float(range_max_m)

    def reset_background(self, frames: int = 50) -> None:
        """
        Collect a median background from the next N valid radar frames.

        For RadarBox, this is optional. The default processing already uses
        slow-time MTI. Use this if the RV map has stable environment clutter.
        """
        with self._lock:
            self._background_power = None
            self._bg_collect_frames = []
            self._bg_collect_remaining = int(frames)

    def estimate_player_range(self, duration_s: float = 1.0) -> Optional[float]:
        """
        Estimate rough player range from recent radar energy.

        This is intentionally simple. It is meant for calibration UI, not for
        precise tracking.
        """
        t1 = time.perf_counter()
        t0 = t1 - duration_s

        with self._lock:
            frames = [f for f in self._frame_buffer if f.valid and t0 <= f.timestamp <= t1]

        if not frames:
            return None

        energy_by_range = np.zeros_like(self.range_axis_m, dtype=np.float64)

        # Use low-to-medium speed bins; standing body + small motion.
        v_mask = np.abs(self.velocity_axis_mps) <= 2.0
        if not np.any(v_mask):
            v_mask = np.ones_like(self.velocity_axis_mps, dtype=bool)

        for f in frames:
            energy_by_range += np.sum(f.rd_power[:, v_mask], axis=1)

        idx = int(np.argmax(energy_by_range))
        return float(self.range_axis_m[idx])

    # -----------------------------
    # Game-facing API
    # -----------------------------

    def query_burst(
        self,
        t_start: float,
        t_end: float,
        range_min_m: Optional[float] = None,
        range_max_m: Optional[float] = None,
        min_abs_velocity_mps: Optional[float] = None,
        max_abs_velocity_mps: Optional[float] = None,
        prefer_direction: Optional[Literal["positive", "negative"]] = None,
    ) -> RadarBurstEvent:
        """
        Query strongest Doppler burst in a time window.

        FusionCore should call this after a camera impact event:

            burst = radar.query_burst(
                impact_time - 0.10,
                impact_time + 0.15,
            )

        prefer_direction:
            None       : search both velocity signs
            "positive" : search positive Doppler only
            "negative" : search negative Doppler only

        Sign convention depends on your radar data path. For the game, the
        absolute velocity is usually the useful quantity.
        """
        if t_end <= t_start:
            return self._invalid_burst(t_start, t_end, "invalid_time_window")

        range_min = self._player_range_min_m if range_min_m is None else float(range_min_m)
        range_max = self._player_range_max_m if range_max_m is None else float(range_max_m)
        min_v = self.cfg.min_abs_velocity_mps if min_abs_velocity_mps is None else float(min_abs_velocity_mps)
        max_v = self.cfg.max_abs_velocity_mps if max_abs_velocity_mps is None else float(max_abs_velocity_mps)

        with self._lock:
            frames = [f for f in self._frame_buffer if t_start <= f.timestamp <= t_end]
            valid_frames = [f for f in frames if f.valid]

        if not frames:
            return self._invalid_burst(t_start, t_end, "no_radar_frames_in_window")

        if not valid_frames:
            return RadarBurstEvent(
                valid=False,
                timestamp=None,
                t_start=t_start,
                t_end=t_end,
                peak_velocity_mps=None,
                abs_peak_velocity_mps=None,
                peak_range_m=None,
                peak_power_db=None,
                noise_floor_db=None,
                snr_db=None,
                burst_score=0.0,
                intensity_score=0.0,
                confidence=0.0,
                frames_used=len(frames),
                valid_frames_used=0,
                burst_frames=0,
                reason="only_invalid_radar_frames_in_window",
            )

        r_mask = (self.range_axis_m >= range_min) & (self.range_axis_m <= range_max)
        v_abs = np.abs(self.velocity_axis_mps)
        v_mask = (v_abs >= min_v) & (v_abs <= max_v)

        if prefer_direction == "positive":
            v_mask &= self.velocity_axis_mps > 0
        elif prefer_direction == "negative":
            v_mask &= self.velocity_axis_mps < 0

        r_idx = np.where(r_mask)[0]
        v_idx = np.where(v_mask)[0]

        if len(r_idx) == 0:
            return self._invalid_burst(t_start, t_end, "empty_range_roi")

        if len(v_idx) == 0:
            return self._invalid_burst(t_start, t_end, "empty_velocity_roi")

        candidates = []
        burst_frames = 0
        noise_db_values = []

        for f in valid_frames:
            roi = f.rd_power[np.ix_(r_idx, v_idx)]
            vals = roi[np.isfinite(roi) & (roi > 0)]

            if vals.size == 0:
                continue

            noise = float(np.median(vals)) + 1e-12
            noise_db = self._pow_to_db(noise)
            noise_db_values.append(noise_db)

            frame_has_burst = False
            for rr_local, vv_local in np.ndindex(roi.shape):
                power_db = float(self._pow_to_db(float(roi[rr_local, vv_local]) + 1e-12))
                snr_db = power_db - noise_db
                rr = int(r_idx[rr_local])
                vv = int(v_idx[vv_local])
                velocity = float(self.velocity_axis_mps[vv])
                abs_velocity = abs(velocity)

                snr_score = float(np.clip(
                    (snr_db - self.cfg.burst_snr_min_db)
                    / max(
                        self.cfg.burst_snr_full_score_db - self.cfg.burst_snr_min_db,
                        1e-6,
                    ),
                    0.0,
                    1.0,
                ))
                velocity_weight = float(np.clip(
                    (abs_velocity - min_v)
                    / max(self.cfg.punch_full_scale_velocity_mps - min_v, 1e-6),
                    0.0,
                    1.0,
                ))
                intensity_score = float(np.clip(
                    (abs_velocity - self.cfg.normal_punch_velocity_mps)
                    / max(
                        self.cfg.critical_punch_velocity_mps
                        - self.cfg.normal_punch_velocity_mps,
                        1e-6,
                    ),
                    0.0,
                    1.0,
                ))
                score = snr_score * velocity_weight

                if (
                    snr_db >= self.cfg.burst_snr_min_db
                    and abs_velocity >= self.cfg.punch_min_valid_velocity_mps
                ):
                    frame_has_burst = True

                candidates.append({
                    "timestamp": float(f.timestamp),
                    "range_m": float(self.range_axis_m[rr]),
                    "velocity_mps": velocity,
                    "abs_velocity_mps": abs_velocity,
                    "power_db": power_db,
                    "noise_floor_db": float(noise_db),
                    "snr_db": float(snr_db),
                    "snr_score": snr_score,
                    "velocity_weight": velocity_weight,
                    "intensity_score": intensity_score,
                    "score": float(score),
                })

            if frame_has_burst:
                burst_frames += 1

        candidates.sort(key=lambda c: (c["score"], c["snr_db"]), reverse=True)
        top_candidates = candidates[:max(1, int(self.cfg.debug_top_candidates))]
        best = top_candidates[0] if top_candidates else None

        if self.cfg.debug:
            print("[RadarAgent] top burst candidates:")
            for index, candidate in enumerate(top_candidates, 1):
                print(
                    f"  #{index} range={candidate['range_m']:.2f}m "
                    f"v={candidate['velocity_mps']:+.2f}m/s "
                    f"|v|={candidate['abs_velocity_mps']:.2f}m/s "
                    f"snr={candidate['snr_db']:.1f}dB "
                    f"intensity={candidate['intensity_score']:.2f} "
                    f"score={candidate['score']:.3f}"
                )

        if best is None:
            return RadarBurstEvent(
                valid=False,
                timestamp=None,
                t_start=t_start,
                t_end=t_end,
                peak_velocity_mps=None,
                abs_peak_velocity_mps=None,
                peak_range_m=None,
                peak_power_db=None,
                noise_floor_db=float(np.median(noise_db_values)) if noise_db_values else None,
                snr_db=None,
                burst_score=0.0,
                intensity_score=0.0,
                confidence=0.0,
                frames_used=len(frames),
                valid_frames_used=len(valid_frames),
                burst_frames=0,
                reason="no_finite_roi_power",
            )

        snr = best["snr_db"]
        abs_v = best["abs_velocity_mps"]
        if abs_v < self.cfg.punch_min_valid_velocity_mps:
            valid = False
            reason = "velocity_too_low_for_punch"
        elif snr < self.cfg.burst_snr_min_db:
            valid = False
            reason = "peak_below_snr_threshold"
        else:
            valid = True
            reason = "burst_found"

        burst_score = float(best["score"])
        intensity_score = float(best["intensity_score"])

        duration_score = np.clip(burst_frames / 3.0, 0.0, 1.0)
        confidence = float(np.clip(0.70 * burst_score + 0.30 * duration_score, 0.0, 1.0))

        event = RadarBurstEvent(
            valid=bool(valid),
            timestamp=best["timestamp"] if valid else None,
            t_start=t_start,
            t_end=t_end,
            peak_velocity_mps=best["velocity_mps"],
            abs_peak_velocity_mps=best["abs_velocity_mps"],
            peak_range_m=best["range_m"],
            peak_power_db=best["power_db"],
            noise_floor_db=best["noise_floor_db"],
            snr_db=best["snr_db"],
            burst_score=float(burst_score if valid else 0.0),
            intensity_score=float(intensity_score if valid else 0.0),
            confidence=confidence if valid else 0.0,
            frames_used=len(frames),
            valid_frames_used=len(valid_frames),
            burst_frames=burst_frames,
            reason=reason,
            top_candidates=top_candidates,
        )

        with self._lock:
            self._latest_summary = RadarSummary(
                timestamp=event.timestamp,
                peak_velocity_mps=event.peak_velocity_mps,
                peak_range_m=event.peak_range_m,
                peak_power_db=event.peak_power_db,
                noise_floor_db=event.noise_floor_db,
                snr_db=event.snr_db,
                frame_valid=event.valid,
                reason=event.reason,
            )

        return event

    def get_health(self) -> RadarHealth:
        with self._lock:
            stats = self._copy_stats_locked()
            last_frame_time = stats.last_frame_time
            last_packet_time = stats.last_packet_time
            buffer_seconds = self._buffer_seconds_locked()

        now = time.perf_counter()
        streaming = (
            last_packet_time is not None
            and (now - last_packet_time) < max(0.5, 3.0 * self.cfg.frame_period_s)
        )

        fps = stats.fps_recent
        status = "OK"

        if not self.is_running():
            status = "STOPPED"
        elif not streaming:
            status = "NO_STREAM"
        elif fps < 0.5 * self.cfg.expected_fps:
            status = "LOW_FPS"
        elif stats.invalid_frames > 0 and stats.valid_frames > 0:
            invalid_ratio = stats.invalid_frames / max(stats.total_frames, 1)
            if invalid_ratio > 0.10:
                status = "PACKET_LOSS"

        return RadarHealth(
            running=self.is_running(),
            streaming=streaming,
            fps=fps,
            expected_fps=self.cfg.expected_fps,
            packets_per_sec=stats.packets_per_sec_recent,
            mb_per_sec=stats.MBps,
            packet_gap_count=stats.packet_gaps,
            invalid_frame_count=stats.invalid_frames,
            total_frame_count=stats.total_frames,
            valid_frame_count=stats.valid_frames,
            last_packet_time=last_packet_time,
            last_frame_time=last_frame_time,
            buffer_seconds=buffer_seconds,
            status=status,
        )

    def get_latest_summary(self) -> RadarSummary:
        with self._lock:
            return self._latest_summary

    def get_latest_tracking_state(self) -> RadarTrackingState:
        with self._lock:
            return self._latest_tracking_state

    # -----------------------------
    # Debug-facing API
    # -----------------------------

    def get_latest_debug_frame(self) -> Optional[RadarDebugFrame]:
        with self._lock:
            return self._latest_debug_frame

    def get_stats(self) -> RadarStats:
        with self._lock:
            return self._copy_stats_locked()

    # ========================================================
    # Internal: UDP receiver / frame assembler
    # ========================================================

    def _receiver_worker(self) -> None:
        cfg = self.cfg

        print("[RadarAgent] starting UDP receiver")
        print(f"[RadarAgent] bind {cfg.pc_ip}:{cfg.data_port}, DCA={cfg.dca_ip}")
        print(f"[RadarAgent] frame_bytes={cfg.frame_bytes}, expected_fps={cfg.expected_fps:.1f}")
        print(f"[RadarAgent] v_max={cfg.v_max_mps:.2f} m/s, dv={cfg.velocity_resolution_mps:.2f} m/s")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock = sock

        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, cfg.socket_rcvbuf)
            sock.bind((cfg.pc_ip, cfg.data_port))
            sock.settimeout(0.2)

            actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(f"[RadarAgent] actual SO_RCVBUF={actual_buf}")
            print("[RadarAgent] listening for DCA1000 packets...")

            stream_buffer = bytearray()
            invalid_mask = bytearray()

            expected_byte_count: Optional[int] = None
            frame_id = 0
            last_report = time.perf_counter()

            while not self._stop_event.is_set():
                try:
                    data, addr = sock.recvfrom(cfg.recvfrom_bytes)
                except socket.timeout:
                    self._maybe_print_health(last_report)
                    continue
                except OSError:
                    break

                now = time.perf_counter()
                src_ip, _src_port = addr

                if cfg.dca_ip and src_ip != cfg.dca_ip:
                    continue

                parsed = self._parse_dca_packet(data)
                if parsed is None:
                    continue

                seq, byte_count, raw = parsed

                with self._lock:
                    st = self._stats
                    if st.first_seq is None:
                        st.first_seq = seq
                        print(f"[RadarAgent] first packet: seq={seq}, byte_count={byte_count}, raw_len={len(raw)}")

                    if st.last_seq is not None and seq != st.last_seq + 1:
                        gap = seq - st.last_seq - 1
                        if gap > 0:
                            st.packet_gaps += gap

                    st.last_seq = seq
                    st.total_packets += 1
                    st.raw_bytes += len(raw)
                    st.last_packet_time = now
                    st.recent_packet_times.append(now)

                # Align first packet to a frame boundary. If Python starts after
                # DCA/radar is already streaming, the first byte_count may be in
                # the middle of a frame. That partial frame must be discarded.
                if expected_byte_count is None:
                    offset = byte_count % cfg.frame_bytes
                    if offset != 0:
                        skip = cfg.frame_bytes - offset
                        if skip >= len(raw):
                            expected_byte_count = byte_count + len(raw)
                            continue
                        raw = raw[skip:]
                        byte_count += skip
                        print(f"[RadarAgent] first packet was mid-frame; skipped {skip} bytes to align.")
                    expected_byte_count = byte_count

                if byte_count > expected_byte_count:
                    missing_bytes = byte_count - expected_byte_count
                    stream_buffer.extend(b"\x00" * missing_bytes)
                    invalid_mask.extend(b"\x01" * missing_bytes)
                    with self._lock:
                        self._stats.inserted_missing_bytes += missing_bytes
                    expected_byte_count = byte_count

                elif byte_count < expected_byte_count:
                    # Duplicate / reordered / overlapping payload.
                    overlap = expected_byte_count - byte_count
                    if overlap >= len(raw):
                        with self._lock:
                            self._stats.skipped_overlap_bytes += len(raw)
                        continue
                    with self._lock:
                        self._stats.skipped_overlap_bytes += overlap
                    raw = raw[overlap:]
                    byte_count = expected_byte_count

                stream_buffer.extend(raw)
                invalid_mask.extend(b"\x00" * len(raw))
                expected_byte_count = byte_count + len(raw)

                while len(stream_buffer) >= cfg.frame_bytes:
                    frame_invalid_bytes = int(sum(invalid_mask[: cfg.frame_bytes]))
                    frame_bytes = bytes(stream_buffer[: cfg.frame_bytes])

                    del stream_buffer[: cfg.frame_bytes]
                    del invalid_mask[: cfg.frame_bytes]

                    frame_ts = time.perf_counter()
                    current_frame_id = frame_id
                    frame_id += 1

                    with self._lock:
                        self._stats.total_frames += 1

                    if frame_invalid_bytes > 0:
                        with self._lock:
                            self._stats.invalid_frames += 1
                        continue

                    self._push_frame_for_processing(current_frame_id, frame_ts, frame_bytes)

                if now - last_report >= cfg.health_report_interval_s:
                    self._print_health_line()
                    last_report = now

        finally:
            try:
                sock.close()
            except OSError:
                pass
            print("[RadarAgent] UDP receiver stopped")

    def _push_frame_for_processing(self, frame_id: int, timestamp: float, frame_bytes: bytes) -> None:
        with self._queue_lock:
            if len(self._frame_queue) >= self._frame_queue_maxlen:
                self._frame_queue.popleft()
            self._frame_queue.append((frame_id, timestamp, frame_bytes))

    def _pop_frame_for_processing(self) -> Optional[tuple[int, float, bytes]]:
        with self._queue_lock:
            if not self._frame_queue:
                return None
            return self._frame_queue.popleft()

    def _processor_worker(self) -> None:
        print("[RadarAgent] processor started")
        while not self._stop_event.is_set():
            item = self._pop_frame_for_processing()
            if item is None:
                time.sleep(0.001)
                continue

            frame_id, timestamp, frame_bytes = item

            try:
                radar_frame = self._process_valid_frame(frame_id, timestamp, frame_bytes)
            except Exception as e:
                with self._lock:
                    self._stats.invalid_frames += 1
                print(f"[RadarAgent] ERROR processing frame {frame_id}: {e}")
                continue

            with self._lock:
                self._frame_buffer.append(radar_frame)
                self._stats.valid_frames += 1
                self._stats.last_frame_time = timestamp
                self._stats.recent_frame_times.append(timestamp)

                self._latest_debug_frame = self._make_debug_frame_locked(radar_frame)
                self._latest_summary = self._make_summary_from_frame_locked(radar_frame)
                self._latest_tracking_state = self._estimate_tracking_state_from_frame_locked(radar_frame)
                tracking_state = self._latest_tracking_state

            self._maybe_print_tracking_debug(tracking_state)

        print("[RadarAgent] processor stopped")

    # ========================================================
    # Internal: signal processing
    # ========================================================

    def _process_valid_frame(self, frame_id: int, timestamp: float, frame_bytes: bytes) -> RadarFrame:
        adc = self._raw_frame_to_adc(frame_bytes)
        raw_rd_power, rd_complex = self._compute_range_doppler(adc)

        with self._lock:
            if self._bg_collect_remaining > 0:
                self._bg_collect_frames.append(raw_rd_power.copy())
                self._bg_collect_remaining -= 1

                if self._bg_collect_remaining == 0 and self._bg_collect_frames:
                    self._background_power = np.median(np.stack(self._bg_collect_frames, axis=0), axis=0)
                    self._bg_collect_frames = []
                    print("[RadarAgent] background ready")

            bg = self._background_power

        if bg is not None:
            rd_power = raw_rd_power - self.cfg.background_scale * bg
            rd_power = np.maximum(rd_power, 0.0)
        else:
            rd_power = raw_rd_power

        rd_db = self._pow_to_db(rd_power)

        return RadarFrame(
            timestamp=timestamp,
            frame_id=frame_id,
            rd_power=rd_power.astype(np.float32),
            rd_db=rd_db.astype(np.float32),
            range_axis_m=self.range_axis_m,
            velocity_axis_mps=self.velocity_axis_mps,
            rd_complex=rd_complex.astype(np.complex64) if self.cfg.enable_rx_phase_aoa else None,
            valid=True,
        )

    def _raw_frame_to_adc(self, frame_bytes: bytes) -> np.ndarray:
        """
        Output:
            adc.shape = [loop, tx, sample, rx]

        DCA1000 ordering assumed from your previous project:
            Rx0I Rx1I Rx2I Rx3I Rx0Q Rx1Q Rx2Q Rx3Q
        """
        cfg = self.cfg
        raw = np.frombuffer(frame_bytes, dtype=np.int16)

        expected_words = cfg.num_loops * cfg.num_tx * cfg.num_samples * cfg.num_rx * cfg.iq
        if raw.size != expected_words:
            raise ValueError(f"raw words={raw.size}, expected={expected_words}")

        data = raw.reshape(cfg.num_loops * cfg.num_tx, cfg.num_samples, cfg.iq, cfg.num_rx)

        i_data = data[:, :, 0, :].astype(np.float32)
        q_data = data[:, :, 1, :].astype(np.float32)
        adc = i_data + 1j * q_data

        return adc.reshape(cfg.num_loops, cfg.num_tx, cfg.num_samples, cfg.num_rx)

    def _compute_range_doppler(self, adc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg

        # adc: [loop, tx, sample, rx]
        if cfg.enable_fast_time_dc_removal:
            adc = adc - np.mean(adc, axis=2, keepdims=True)

        if cfg.use_hanning_window:
            range_win = np.hanning(cfg.num_samples)[None, None, :, None]
        else:
            range_win = 1.0

        range_fft = np.fft.fft(adc * range_win, axis=2)

        if cfg.range_fft_keep_half:
            range_fft = range_fft[:, :, : cfg.num_samples // 2, :]

        # Suppress static / very slow clutter before Doppler FFT.
        if cfg.enable_slow_time_mti:
            range_fft = range_fft - np.mean(range_fft, axis=0, keepdims=True)

        if cfg.use_hanning_window:
            doppler_win = np.hanning(cfg.num_loops)[:, None, None, None]
        else:
            doppler_win = 1.0

        rd = np.fft.fftshift(np.fft.fft(range_fft * doppler_win, axis=0), axes=0)
        # rd: [doppler, tx, range, rx]

        # Non-coherent sum over Tx/Rx. Tx dimension is 1 in current config.
        rd_power = np.sum(np.abs(rd) ** 2, axis=(1, 3))  # [doppler, range]
        rd_power = rd_power.T  # [range, doppler]

        return rd_power.astype(np.float64), rd

    def _compute_range_doppler_power(self, adc: np.ndarray) -> np.ndarray:
        """Compatibility wrapper preserving the previous public/internal behavior."""
        rd_power, _ = self._compute_range_doppler(adc)
        return rd_power

    # ========================================================
    # Internal: axes / helpers
    # ========================================================

    def _make_range_axis(self) -> np.ndarray:
        cfg = self.cfg
        freqs = np.fft.fftfreq(cfg.num_samples, d=1.0 / cfg.sample_rate_hz)

        if cfg.range_fft_keep_half:
            freqs = freqs[: cfg.num_samples // 2]

        return cfg.c * freqs / (2.0 * cfg.slope_hz_per_s)

    def _make_doppler_axis(self) -> np.ndarray:
        cfg = self.cfg
        fd = np.fft.fftshift(np.fft.fftfreq(cfg.num_loops, d=cfg.chirp_period_s))
        return fd * cfg.wavelength_m / 2.0

    def _parse_dca_packet(self, data: bytes) -> Optional[tuple[int, int, bytes]]:
        cfg = self.cfg
        if len(data) < cfg.dca_header_bytes:
            return None

        seq = int.from_bytes(data[0:4], byteorder="little", signed=False)
        byte_count = int.from_bytes(data[4:10], byteorder="little", signed=False)
        raw = data[cfg.dca_header_bytes :]
        return seq, byte_count, raw

    @staticmethod
    def _pow_to_db(power) -> np.ndarray | float:
        return 10.0 * np.log10(np.maximum(power, 1e-12))

    def _invalid_burst(self, t_start: float, t_end: float, reason: str) -> RadarBurstEvent:
        return RadarBurstEvent(
            valid=False,
            timestamp=None,
            t_start=t_start,
            t_end=t_end,
            peak_velocity_mps=None,
            abs_peak_velocity_mps=None,
            peak_range_m=None,
            peak_power_db=None,
            noise_floor_db=None,
            snr_db=None,
            burst_score=0.0,
            intensity_score=0.0,
            confidence=0.0,
            frames_used=0,
            valid_frames_used=0,
            burst_frames=0,
            reason=reason,
        )

    def _make_debug_frame_locked(self, frame: RadarFrame) -> RadarDebugFrame:
        peak_range = None
        peak_velocity = None
        peak_power_db = None

        if frame.valid and frame.rd_power.size > 0:
            idx = int(np.argmax(frame.rd_power))
            rr, vv = np.unravel_index(idx, frame.rd_power.shape)
            peak_range = float(frame.range_axis_m[rr])
            peak_velocity = float(frame.velocity_axis_mps[vv])
            peak_power_db = float(frame.rd_db[rr, vv])

        return RadarDebugFrame(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            rd_db=frame.rd_db,
            range_axis_m=frame.range_axis_m,
            velocity_axis_mps=frame.velocity_axis_mps,
            peak_range_m=peak_range,
            peak_velocity_mps=peak_velocity,
            peak_power_db=peak_power_db,
            valid=frame.valid,
            invalid_reason=frame.invalid_reason,
        )

    def _make_summary_from_frame_locked(self, frame: RadarFrame) -> RadarSummary:
        if not frame.valid or frame.rd_power.size == 0:
            return RadarSummary(
                timestamp=frame.timestamp,
                peak_velocity_mps=None,
                peak_range_m=None,
                peak_power_db=None,
                noise_floor_db=None,
                snr_db=None,
                frame_valid=False,
                reason=frame.invalid_reason or "invalid_frame",
            )

        # Summary uses default player ROI and velocity ROI.
        r_mask = (self.range_axis_m >= self._player_range_min_m) & (self.range_axis_m <= self._player_range_max_m)
        v_abs = np.abs(self.velocity_axis_mps)
        v_mask = (v_abs >= self.cfg.min_abs_velocity_mps) & (v_abs <= self.cfg.max_abs_velocity_mps)

        r_idx = np.where(r_mask)[0]
        v_idx = np.where(v_mask)[0]

        if len(r_idx) == 0 or len(v_idx) == 0:
            return RadarSummary(
                timestamp=frame.timestamp,
                peak_velocity_mps=None,
                peak_range_m=None,
                peak_power_db=None,
                noise_floor_db=None,
                snr_db=None,
                frame_valid=True,
                reason="empty_default_roi",
            )

        roi = frame.rd_power[np.ix_(r_idx, v_idx)]
        vals = roi[np.isfinite(roi) & (roi > 0)]
        if vals.size == 0:
            return RadarSummary(
                timestamp=frame.timestamp,
                peak_velocity_mps=None,
                peak_range_m=None,
                peak_power_db=None,
                noise_floor_db=None,
                snr_db=None,
                frame_valid=True,
                reason="no_roi_power",
            )

        noise = float(np.median(vals)) + 1e-12
        noise_db = float(self._pow_to_db(noise))
        flat_idx = int(np.argmax(roi))
        rr_local, vv_local = np.unravel_index(flat_idx, roi.shape)
        rr = int(r_idx[rr_local])
        vv = int(v_idx[vv_local])

        peak_power_db = float(frame.rd_db[rr, vv])
        snr_db = peak_power_db - noise_db

        return RadarSummary(
            timestamp=frame.timestamp,
            peak_velocity_mps=float(self.velocity_axis_mps[vv]),
            peak_range_m=float(self.range_axis_m[rr]),
            peak_power_db=peak_power_db,
            noise_floor_db=noise_db,
            snr_db=snr_db,
            frame_valid=True,
            reason="latest_frame_summary",
        )

    def _invalid_tracking_state(
        self,
        timestamp: Optional[float],
        reason: str,
        range_m: Optional[float] = None,
        snr_db: Optional[float] = None,
        peak_velocity_mps: Optional[float] = None,
        abs_peak_velocity_mps: Optional[float] = None,
        peak_power_db: Optional[float] = None,
        target_range_bin: Optional[int] = None,
        target_doppler_bin: Optional[int] = None,
        valid: bool = False,
        candidate_count: int = 0,
        selected_score: Optional[float] = None,
    ) -> RadarTrackingState:
        return RadarTrackingState(
            timestamp=timestamp,
            valid=valid,
            range_m=range_m,
            theta_deg=None,
            theta_raw_deg=None,
            theta_smooth_deg=self._theta_smooth_deg,
            snr_db=snr_db,
            peak_velocity_mps=peak_velocity_mps,
            abs_peak_velocity_mps=abs_peak_velocity_mps,
            peak_power_db=peak_power_db,
            beam_tx0_code=None,
            beam_tx1_code=None,
            beam_tx2_code=None,
            beam_theta_deg=None,
            target_range_bin=target_range_bin,
            target_doppler_bin=target_doppler_bin,
            confidence=0.0,
            reason=reason,
            track_range_m=self._track_range_m,
            track_velocity_mps=self._track_velocity_mps,
            candidate_count=candidate_count,
            selected_score=selected_score,
        )

    def _select_tracking_target_bin(
        self,
        frame: RadarFrame,
    ) -> dict:
        r_mask = (
            (self.range_axis_m >= self.cfg.tracking_range_min_m)
            & (self.range_axis_m <= self.cfg.tracking_range_max_m)
        )
        v_abs = np.abs(self.velocity_axis_mps)
        v_mask = (
            (v_abs >= self.cfg.tracking_min_abs_velocity_mps)
            & (v_abs <= self.cfg.tracking_max_abs_velocity_mps)
        )

        r_idx = np.where(r_mask)[0]
        v_idx = np.where(v_mask)[0]
        if len(r_idx) == 0 or len(v_idx) == 0:
            return {
                "valid": False,
                "reason": "empty_tracking_roi",
                "candidate_count": 0,
                "selected_score": None,
            }

        roi = frame.rd_power[np.ix_(r_idx, v_idx)]
        vals = roi[np.isfinite(roi) & (roi > 0)]
        if vals.size == 0:
            return {
                "valid": False,
                "reason": "no_tracking_roi_power",
                "candidate_count": 0,
                "selected_score": None,
            }

        noise = float(np.median(vals)) + 1e-12
        noise_floor_db = float(self._pow_to_db(noise))
        candidates = []

        for rr_local, vv_local in np.ndindex(roi.shape):
            power = float(roi[rr_local, vv_local])
            if not np.isfinite(power) or power <= 0.0:
                continue

            range_bin = int(r_idx[rr_local])
            doppler_bin = int(v_idx[vv_local])
            peak_power_db = float(frame.rd_db[range_bin, doppler_bin])
            snr_db = peak_power_db - noise_floor_db
            if snr_db < self.cfg.tracking_snr_min_db:
                continue

            range_m = float(self.range_axis_m[range_bin])
            velocity_mps = float(self.velocity_axis_mps[doppler_bin])
            candidates.append({
                "range_bin": range_bin,
                "doppler_bin": doppler_bin,
                "range_m": range_m,
                "velocity_mps": velocity_mps,
                "abs_velocity_mps": abs(velocity_mps),
                "peak_power_db": peak_power_db,
                "noise_floor_db": noise_floor_db,
                "snr_db": snr_db,
                "score": snr_db,
            })

        if not candidates:
            return {
                "valid": False,
                "reason": "no_tracking_candidate_above_snr",
                "candidate_count": 0,
                "selected_score": None,
            }

        strongest = max(candidates, key=lambda c: c["snr_db"])
        now = time.perf_counter()
        has_track = (
            self.cfg.tracking_enable_temporal_lock
            and self._track_range_m is not None
            and self._track_velocity_mps is not None
            and self._track_last_update_time is not None
        )
        track_lost = (
            has_track
            and now - float(self._track_last_update_time) > self.cfg.tracking_lost_timeout_s
        )

        if not has_track or track_lost or not self.cfg.tracking_enable_temporal_lock:
            selected = dict(strongest)
            selected["valid"] = True
            selected["reason"] = "tracking_reacquired" if track_lost else "tracking_acquired"
            selected["candidate_count"] = len(candidates)
            selected["selected_score"] = selected["score"]
            return selected

        gated = []
        for candidate in candidates:
            dr = abs(candidate["range_m"] - float(self._track_range_m))
            dv = abs(candidate["velocity_mps"] - float(self._track_velocity_mps))
            if dr > self.cfg.tracking_range_gate_m or dv > self.cfg.tracking_velocity_gate_mps:
                continue

            score = (
                self.cfg.tracking_score_snr_weight * candidate["snr_db"]
                - self.cfg.tracking_score_range_weight * dr
                - self.cfg.tracking_score_velocity_weight * dv
            )
            scored = dict(candidate)
            scored["score"] = float(score)
            gated.append(scored)

        if not gated:
            return {
                "valid": False,
                "reason": "tracking_candidate_out_of_gate",
                "candidate_count": len(candidates),
                "selected_score": None,
            }

        selected = max(gated, key=lambda c: c["score"])
        selected["valid"] = True
        selected["reason"] = "tracking_locked"
        selected["candidate_count"] = len(candidates)
        selected["selected_score"] = selected["score"]
        return selected

    def _estimate_aoa_from_frame_bin(
        self,
        frame: RadarFrame,
        range_bin: int,
        doppler_bin: int,
    ) -> tuple[Optional[float], str]:
        if frame.rd_complex is None:
            return None, "rd_complex_missing"

        rd_complex = frame.rd_complex
        if rd_complex.ndim != 4:
            return None, "rd_complex_bad_shape"
        if rd_complex.shape[1] != 1:
            return None, "unexpected_tx_dimension_for_3tx_simultaneous"
        if not (0 <= doppler_bin < rd_complex.shape[0] and 0 <= range_bin < rd_complex.shape[2]):
            return None, "target_bin_out_of_bounds"

        rx_complex = rd_complex[doppler_bin, 0, range_bin, :]
        if rx_complex.size < 4:
            return None, "not_enough_rx_channels"
        if not np.all(np.isfinite(rx_complex)):
            return None, "rx_complex_not_finite"

        mag = np.abs(rx_complex)
        if not np.any(mag > 1e-9):
            return None, "rx_complex_all_zero"

        phase = np.unwrap(np.angle(rx_complex))
        offsets = np.array(self.cfg.rx_phase_offsets_rad[: len(phase)], dtype=np.float64)
        if offsets.size < phase.size:
            offsets = np.pad(offsets, (0, phase.size - offsets.size), mode="constant")
        phase = phase - offsets

        x = np.arange(len(phase), dtype=np.float64)
        slope = float(np.polyfit(x, phase, 1)[0])
        sin_theta = float(np.clip(slope / np.pi, -1.0, 1.0))
        theta_raw = float(np.degrees(np.arcsin(sin_theta)))
        theta = (
            self.cfg.aoa_sign * self.cfg.aoa_scale * theta_raw
            + self.cfg.aoa_offset_deg
        )
        theta = float(np.clip(theta, -self.cfg.aoa_max_abs_deg, self.cfg.aoa_max_abs_deg))
        return theta, "rx_phase_slope"

    def _smooth_theta(self, theta_deg: float) -> float:
        alpha = float(np.clip(self.cfg.aoa_alpha, 0.0, 1.0))
        if self._theta_smooth_deg is None:
            self._theta_smooth_deg = float(theta_deg)
        else:
            previous = float(self._theta_smooth_deg)
            max_step = max(0.0, float(self.cfg.aoa_max_update_step_deg))
            delta = float(np.clip(theta_deg - previous, -max_step, max_step))
            theta_limited = previous + delta
            self._theta_smooth_deg = float(alpha * theta_limited + (1.0 - alpha) * previous)
        return self._theta_smooth_deg

    def _phase_deg_to_code(self, deg: float) -> int:
        return int(round((deg % 360.0) / self.cfg.phase_step_deg)) & 0x3F

    def _theta_to_beam_codes(self, theta_deg: float) -> tuple[int, int, int]:
        phi = -180.0 * math.sin(math.radians(theta_deg))
        return (
            self._phase_deg_to_code(0.0),
            self._phase_deg_to_code(phi),
            self._phase_deg_to_code(2.0 * phi),
        )

    def _atomic_write_text(self, path: str, text: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text, encoding="ascii")
        tmp.replace(p)

    def _maybe_update_beam_from_tracking_locked(self, state: RadarTrackingState) -> None:
        if not self.cfg.enable_aoa_feedback:
            return
        if not state.valid:
            return
        if state.theta_smooth_deg is None:
            return
        if state.confidence < self.cfg.beam_min_confidence:
            return
        if state.snr_db is None or state.snr_db < self.cfg.beam_min_snr_db:
            return

        codes = self._theta_to_beam_codes(float(state.theta_smooth_deg))
        state.beam_tx0_code, state.beam_tx1_code, state.beam_tx2_code = codes
        state.beam_theta_deg = float(state.theta_smooth_deg)

        now = time.perf_counter()
        if self._last_beam_codes == codes:
            return
        if now - self._last_beam_update_time < self.cfg.beam_update_interval_s:
            return

        text = f"{codes[0]},{codes[1]},{codes[2]}\n"
        try:
            self._atomic_write_text(self.cfg.beam_cmd_file, text)
        except Exception as exc:
            state.reason = f"{state.reason}; beam_write_failed:{type(exc).__name__}"
            return

        self._last_beam_codes = codes
        self._last_beam_update_time = now

    def _estimate_tracking_state_from_frame_locked(self, frame: RadarFrame) -> RadarTrackingState:
        if not frame.valid or frame.rd_power.size == 0:
            return self._invalid_tracking_state(
                frame.timestamp,
                frame.invalid_reason or "invalid_frame",
            )

        selected = self._select_tracking_target_bin(frame)
        if not selected.get("valid", False):
            return self._invalid_tracking_state(
                frame.timestamp,
                str(selected.get("reason", "no_tracking_target_in_roi")),
                valid=False,
                candidate_count=int(selected.get("candidate_count", 0) or 0),
                selected_score=selected.get("selected_score", None),
            )

        range_bin = int(selected["range_bin"])
        doppler_bin = int(selected["doppler_bin"])
        peak_power_db = float(selected["peak_power_db"])
        snr_db = float(selected["snr_db"])
        range_m = float(selected["range_m"])
        velocity_mps = float(selected["velocity_mps"])
        abs_velocity_mps = float(selected["abs_velocity_mps"])
        candidate_count = int(selected.get("candidate_count", 0) or 0)
        selected_score = selected.get("selected_score", selected.get("score", None))

        if snr_db < self.cfg.aoa_min_snr_db:
            return self._invalid_tracking_state(
                frame.timestamp,
                "snr_below_aoa_min",
                range_m=range_m,
                snr_db=snr_db,
                peak_velocity_mps=velocity_mps,
                abs_peak_velocity_mps=abs_velocity_mps,
                peak_power_db=peak_power_db,
                target_range_bin=range_bin,
                target_doppler_bin=doppler_bin,
                valid=False,
                candidate_count=candidate_count,
                selected_score=selected_score,
            )

        theta_raw = None
        theta_smooth = None
        confidence = float(np.clip((snr_db - self.cfg.aoa_min_snr_db) / 20.0, 0.0, 1.0))
        if not self.cfg.enable_rx_phase_aoa:
            reason = "rx_phase_aoa_disabled"
            confidence = 0.0
        else:
            theta_raw, reason = self._estimate_aoa_from_frame_bin(frame, range_bin, doppler_bin)
            if theta_raw is not None:
                if (
                    self._track_theta_deg is not None
                    and abs(theta_raw - float(self._track_theta_deg)) > self.cfg.tracking_theta_gate_deg
                ):
                    return self._invalid_tracking_state(
                        frame.timestamp,
                        "tracking_theta_out_of_gate",
                        range_m=range_m,
                        snr_db=snr_db,
                        peak_velocity_mps=velocity_mps,
                        abs_peak_velocity_mps=abs_velocity_mps,
                        peak_power_db=peak_power_db,
                        target_range_bin=range_bin,
                        target_doppler_bin=doppler_bin,
                        valid=False,
                        candidate_count=candidate_count,
                        selected_score=selected_score,
                    )
                theta_smooth = self._smooth_theta(theta_raw)
                self._track_range_m = range_m
                self._track_velocity_mps = velocity_mps
                self._track_theta_deg = theta_smooth
                self._track_last_update_time = time.perf_counter()
            else:
                confidence = 0.0

        state = RadarTrackingState(
            timestamp=frame.timestamp,
            valid=theta_raw is not None,
            range_m=range_m,
            theta_deg=theta_smooth,
            theta_raw_deg=theta_raw,
            theta_smooth_deg=theta_smooth,
            snr_db=snr_db,
            peak_velocity_mps=velocity_mps,
            abs_peak_velocity_mps=abs_velocity_mps,
            peak_power_db=peak_power_db,
            beam_tx0_code=None,
            beam_tx1_code=None,
            beam_tx2_code=None,
            beam_theta_deg=None,
            target_range_bin=range_bin,
            target_doppler_bin=doppler_bin,
            confidence=confidence,
            reason=reason,
            track_range_m=self._track_range_m,
            track_velocity_mps=self._track_velocity_mps,
            candidate_count=candidate_count,
            selected_score=selected_score,
        )
        self._maybe_update_beam_from_tracking_locked(state)
        return state

    @staticmethod
    def _fmt_tracking_value(value, precision: int = 2) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return str(value)

    def _maybe_print_tracking_debug(self, state: RadarTrackingState) -> None:
        now = time.perf_counter()
        if now - self._last_tracking_debug_print < 1.0:
            return
        self._last_tracking_debug_print = now

        print(
            "[RadarTracking] "
            f"valid={state.valid} "
            f"range={self._fmt_tracking_value(state.range_m)} "
            f"v={self._fmt_tracking_value(state.peak_velocity_mps)} "
            f"snr={self._fmt_tracking_value(state.snr_db)} "
            f"theta_raw={self._fmt_tracking_value(state.theta_raw_deg)} "
            f"theta_smooth={self._fmt_tracking_value(state.theta_smooth_deg)} "
            f"bins=({state.target_range_bin},{state.target_doppler_bin}) "
            f"track_r={self._fmt_tracking_value(state.track_range_m)} "
            f"track_v={self._fmt_tracking_value(state.track_velocity_mps)} "
            f"cand={state.candidate_count} "
            f"score={self._fmt_tracking_value(state.selected_score)} "
            f"reason={state.reason}"
        )

    def _buffer_seconds_locked(self) -> float:
        if len(self._frame_buffer) < 2:
            return 0.0
        return self._frame_buffer[-1].timestamp - self._frame_buffer[0].timestamp

    def _copy_stats_locked(self) -> RadarStats:
        s = self._stats
        copied = RadarStats()
        copied.total_packets = s.total_packets
        copied.total_frames = s.total_frames
        copied.valid_frames = s.valid_frames
        copied.invalid_frames = s.invalid_frames
        copied.packet_gaps = s.packet_gaps
        copied.inserted_missing_bytes = s.inserted_missing_bytes
        copied.skipped_overlap_bytes = s.skipped_overlap_bytes
        copied.raw_bytes = s.raw_bytes
        copied.first_seq = s.first_seq
        copied.last_seq = s.last_seq
        copied.start_time = s.start_time
        copied.last_packet_time = s.last_packet_time
        copied.last_frame_time = s.last_frame_time
        copied.recent_packet_times = deque(s.recent_packet_times, maxlen=s.recent_packet_times.maxlen)
        copied.recent_frame_times = deque(s.recent_frame_times, maxlen=s.recent_frame_times.maxlen)
        return copied

    def _print_health_line(self) -> None:
        h = self.get_health()
        print(
            "[RadarAgent] "
            f"status={h.status} "
            f"fps={h.fps:.1f}/{h.expected_fps:.1f} "
            f"pps={h.packets_per_sec:.0f} "
            f"MBps={h.mb_per_sec:.2f} "
            f"frames={h.valid_frame_count}/{h.total_frame_count} "
            f"bad={h.invalid_frame_count} "
            f"gaps={h.packet_gap_count} "
            f"buf={h.buffer_seconds:.2f}s"
        )

    def _maybe_print_health(self, last_report: float) -> None:
        # Kept for compatibility; actual report happens after received packets.
        return


# ============================================================
# Standalone debug runner
# ============================================================

def run_console(agent: RadarAgent, query_interval_s: float = 0.5) -> None:
    print()
    print("RadarAgent console mode.")
    print("Start DCA1000 streaming + StartFrame now.")
    print("Press Ctrl+C to stop.")
    print()

    while True:
        time.sleep(query_interval_s)

        now = time.perf_counter()
        burst = agent.query_burst(now - 0.25, now)

        h = agent.get_health()
        s = agent.get_latest_summary()

        if burst.valid:
            print(
                f"[query] v={burst.peak_velocity_mps:+.2f} m/s "
                f"|v|={burst.abs_peak_velocity_mps:.2f} "
                f"R={burst.peak_range_m:.2f} m "
                f"SNR={burst.snr_db:.1f} dB "
                f"I={burst.intensity_score:.2f} "
                f"conf={burst.confidence:.2f} "
                f"health={h.status}"
            )
        else:
            latest = ""
            if s.peak_velocity_mps is not None:
                latest = f" latest_v={s.peak_velocity_mps:+.2f} SNR={s.snr_db:.1f}dB"
            print(f"[query] no burst ({burst.reason}) health={h.status}{latest}")


def run_plot(agent: RadarAgent, interval_ms: int = 50) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as e:
        raise RuntimeError("matplotlib is required for --plot") from e

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("RadarBox RadarAgent Debug: Range–Velocity Map")
    ax.set_xlabel("Radial velocity (m/s)")
    ax.set_ylabel("Range (m)")

    # Initial dummy data.
    rd0 = np.zeros((len(agent.range_axis_m), len(agent.velocity_axis_mps)), dtype=np.float32)
    im = ax.imshow(
        rd0,
        origin="lower",
        aspect="auto",
        extent=[
            agent.velocity_axis_mps[0],
            agent.velocity_axis_mps[-1],
            agent.range_axis_m[0],
            agent.range_axis_m[-1],
        ],
        vmin=60,
        vmax=130,
    )
    marker, = ax.plot([], [], "o", fillstyle="none", markersize=10, markeredgewidth=2)
    txt = fig.text(0.02, 0.02, "Waiting for radar frames...", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power (dB)")

    def update(_):
        dbg = agent.get_latest_debug_frame()
        h = agent.get_health()

        if dbg is None:
            txt.set_text(f"Waiting... status={h.status}")
            return im, marker, txt

        im.set_data(dbg.rd_db)

        if dbg.peak_range_m is not None and dbg.peak_velocity_mps is not None:
            marker.set_data([dbg.peak_velocity_mps], [dbg.peak_range_m])
        else:
            marker.set_data([], [])

        now = time.perf_counter()
        burst = agent.query_burst(now - 0.25, now)

        if burst.valid:
            burst_text = (
                f"burst: |v|={burst.abs_peak_velocity_mps:.2f}m/s, "
                f"R={burst.peak_range_m:.2f}m, SNR={burst.snr_db:.1f}dB, "
                f"I={burst.intensity_score:.2f}"
            )
        else:
            burst_text = f"burst: {burst.reason}"

        txt.set_text(
            f"frame={dbg.frame_id} "
            f"status={h.status} fps={h.fps:.1f}/{h.expected_fps:.1f} "
            f"gaps={h.packet_gap_count} bad={h.invalid_frame_count} "
            f"peak_v={dbg.peak_velocity_mps:+.2f}m/s "
            f"peak_R={dbg.peak_range_m:.2f}m "
            f"peak={dbg.peak_power_db:.1f}dB | "
            f"{burst_text}"
        )

        return im, marker, txt

    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False)

    try:
        plt.show(block=True)
    finally:
        agent.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="RadarBox RadarAgent")
    parser.add_argument("--pc-ip", default="192.168.33.30")
    parser.add_argument("--dca-ip", default="192.168.33.180")
    parser.add_argument("--data-port", type=int, default=4098)
    parser.add_argument("--plot", action="store_true", help="show live Range-Velocity debug plot")
    parser.add_argument("--range-min", type=float, default=0.60)
    parser.add_argument("--range-max", type=float, default=2.50)
    parser.add_argument("--packet-snr-db", type=float, default=8.0, help="burst SNR threshold in dB")
    parser.add_argument("--background-frames", type=int, default=0, help="collect N frames for median background")
    args = parser.parse_args()

    cfg = RadarConfig(
        pc_ip=args.pc_ip,
        dca_ip=args.dca_ip,
        data_port=args.data_port,
        player_range_min_m=args.range_min,
        player_range_max_m=args.range_max,
        burst_snr_min_db=args.packet_snr_db,
        debug=args.plot,
    )

    agent = RadarAgent(cfg)

    print("=== RadarBox RadarAgent ===")
    print(f"Config: 1Tx+4Rx, frame_bytes={cfg.frame_bytes}, expected={cfg.expected_fps:.1f} Hz")
    print(f"Velocity: vmax={cfg.v_max_mps:.2f} m/s, dv={cfg.velocity_resolution_mps:.2f} m/s")
    print("This script only receives/processes UDP. It does not StartFrame/StopFrame.")
    print()

    agent.start()

    if args.background_frames > 0:
        print(f"Collecting background from next {args.background_frames} valid frames.")
        agent.reset_background(args.background_frames)

    try:
        if args.plot:
            run_plot(agent)
        else:
            run_console(agent)
    except KeyboardInterrupt:
        print()
        print("Stopping RadarAgent...")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
