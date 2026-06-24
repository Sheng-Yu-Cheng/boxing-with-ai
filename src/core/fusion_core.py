#!/usr/bin/env python3
r"""
fusion_core.py

RadarBox FusionCore.

Purpose
-------
FusionCore consumes events from VisionAgent and optionally queries RadarAgent
to produce one unified FusedPlayerEvent for the GameEngine.

Runtime flow:

    VisionAgent
        -> PlayerActionEvent
        -> FusionCore
        -> FusedPlayerEvent
        -> GameEngine

For straight punches:

    Vision event gives:
        action_type, impact_time, confidence

    Radar query gives:
        Doppler burst velocity, intensity_score, confidence

    FusionCore combines them into:
        right_straight + intensity + final_confidence

For hook / uppercut:

    Camera-only first version.
    Radar is not required because hook/uppercut motion is less radial.

For block:

    Pass through as camera-only state event:
        block / block_end

Timestamp convention
--------------------
Both vision_agent_trajectory.py and radar_agent.py should use time.perf_counter().
Therefore FusionCore can safely query:

    radar_agent.query_burst(action.impact_time - 0.10,
                            action.impact_time + 0.15)

Expected APIs
-------------
VisionAgent must provide:

    get_next_action_event() -> PlayerActionEvent | None

RadarAgent should provide:

    query_burst(t_start, t_end, ...) -> RadarBurstEvent
    get_health() -> RadarHealth

This file intentionally uses duck typing so it works with your current scripts
without forcing strict imports.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Any


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class FusionConfig:
    # Which camera actions should query radar.
    radar_actions: tuple[str, ...] = (
        "left_straight",
        "right_straight",
    )

    # Hook / uppercut are camera-only in the first stable game version.
    camera_only_actions: tuple[str, ...] = (
        "left_hook",
        "right_hook",
        "left_uppercut",
        "right_uppercut",
    )

    block_actions: tuple[str, ...] = (
        "block",
        "block_end",
    )

    # Radar query window around the vision-estimated impact time.
    radar_pre_impact_s: float = 0.10
    radar_post_impact_s: float = 0.15

    # Radar range of player. Keep consistent with RadarAgent defaults.
    radar_range_min_m: float = 0.6
    radar_range_max_m: float = 2.5

    # Minimum absolute velocity for punch burst search.
    radar_min_abs_velocity_mps: float = 0.5

    # If True, a straight punch is emitted only if radar burst is valid.
    # If False, camera still emits straight punch even when radar is unavailable,
    # but intensity will be weaker / camera-only.
    require_radar_for_straight: bool = False

    # Final confidence combination.
    vision_weight: float = 0.55
    radar_weight: float = 0.45

    # Camera-only intensity estimate.
    # Used for hook/uppercut and radar-missing straight.
    camera_speed_full_scale: float = 8.0
    default_camera_intensity: float = 0.45

    # Event queue / thread.
    poll_interval_s: float = 0.005
    queue_size: int = 128

    # Debug print.
    verbose: bool = True


@dataclass
class FusedPlayerEvent:
    timestamp: float
    action_type: str
    hand: str
    phase: str

    start_time: Optional[float]
    impact_time: Optional[float]
    end_time: Optional[float]

    # Main outputs for GameEngine.
    final_confidence: float
    intensity_score: float
    damage_scale: float

    # Where the decision came from.
    source: str  # "vision", "vision+radar", "vision_only_radar_missing", "block"

    # Vision info.
    vision_confidence: float
    camera_speed: float
    vision_reason: str

    # Radar info.
    radar_valid: bool
    radar_confidence: float
    radar_velocity_mps: Optional[float]
    radar_abs_velocity_mps: Optional[float]
    radar_snr_db: Optional[float]
    radar_peak_range_m: Optional[float]
    radar_reason: str

    # Human-readable explanation.
    fusion_reason: str


# ============================================================
# Helpers
# ============================================================

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def _is_action(action_type: str, actions: tuple[str, ...]) -> bool:
    return action_type in actions


# ============================================================
# FusionCore
# ============================================================

class FusionCore:
    def __init__(
        self,
        vision_agent: Any,
        radar_agent: Optional[Any] = None,
        config: Optional[FusionConfig] = None,
    ):
        self.vision_agent = vision_agent
        self.radar_agent = radar_agent
        self.cfg = config or FusionConfig()

        self._queue: "queue.Queue[FusedPlayerEvent]" = queue.Queue(maxsize=self.cfg.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._last_fused_event: Optional[FusedPlayerEvent] = None

    # -----------------------------
    # Lifecycle
    # -----------------------------

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="FusionCore", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -----------------------------
    # Public API
    # -----------------------------

    def get_next_fused_event(self) -> Optional[FusedPlayerEvent]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_fused_event(self) -> Optional[FusedPlayerEvent]:
        return self._last_fused_event

    def poll_once(self) -> Optional[FusedPlayerEvent]:
        """
        Synchronous one-step poll.

        Useful before using the background thread:

            fused = fusion.poll_once()
        """
        vision_event = self.vision_agent.get_next_action_event()
        if vision_event is None:
            return None

        fused = self.process_vision_event(vision_event)
        if fused is not None:
            self._push(fused)
        return fused

    def process_vision_event(self, vision_event: Any) -> Optional[FusedPlayerEvent]:
        action_type = str(_get(vision_event, "action_type", ""))
        if action_type in ("", "idle", "negative", "unknown"):
            return None

        if _is_action(action_type, self.cfg.block_actions):
            return self._fuse_block(vision_event)

        if _is_action(action_type, self.cfg.radar_actions):
            return self._fuse_straight_with_radar(vision_event)

        if _is_action(action_type, self.cfg.camera_only_actions):
            return self._fuse_camera_only(vision_event)

        # Unknown but non-idle action: pass camera-only with low confidence.
        return self._fuse_camera_only(
            vision_event,
            source="vision_unknown_action",
            reason=f"unknown action_type={action_type}; passed as camera-only",
        )

    # -----------------------------
    # Worker
    # -----------------------------

    def _worker(self) -> None:
        if self.cfg.verbose:
            print("[FusionCore] started")

        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:
                print(f"[FusionCore] ERROR: {type(e).__name__}: {e}")

            time.sleep(self.cfg.poll_interval_s)

        if self.cfg.verbose:
            print("[FusionCore] stopped")

    # -----------------------------
    # Fusion methods
    # -----------------------------

    def _fuse_block(self, ev: Any) -> FusedPlayerEvent:
        now = time.perf_counter()
        action_type = str(_get(ev, "action_type", "block"))
        conf = float(_get(ev, "confidence", 1.0) or 1.0)

        fused = FusedPlayerEvent(
            timestamp=now,
            action_type=action_type,
            hand=str(_get(ev, "hand", "both")),
            phase=str(_get(ev, "phase", action_type)),
            start_time=_get(ev, "start_time", None),
            impact_time=_get(ev, "impact_time", None),
            end_time=_get(ev, "end_time", None),
            final_confidence=conf,
            intensity_score=0.0,
            damage_scale=0.0,
            source="block",
            vision_confidence=conf,
            camera_speed=float(_get(ev, "camera_speed", 0.0) or 0.0),
            vision_reason=str(_get(ev, "reason", "")),
            radar_valid=False,
            radar_confidence=0.0,
            radar_velocity_mps=None,
            radar_abs_velocity_mps=None,
            radar_snr_db=None,
            radar_peak_range_m=None,
            radar_reason="radar_not_used_for_block",
            fusion_reason="block pass-through from VisionAgent",
        )
        return fused

    def _fuse_camera_only(
        self,
        ev: Any,
        source: str = "vision",
        reason: str = "camera-only action",
    ) -> FusedPlayerEvent:
        now = time.perf_counter()
        action_type = str(_get(ev, "action_type", "unknown"))
        vision_conf = float(_get(ev, "confidence", 0.0) or 0.0)
        camera_speed = float(_get(ev, "camera_speed", 0.0) or 0.0)

        # Camera-only damage is intentionally moderate.
        speed_intensity = _clip01(camera_speed / max(self.cfg.camera_speed_full_scale, 1e-6))
        intensity = max(self.cfg.default_camera_intensity, speed_intensity * 0.70)
        intensity = _clip01(intensity)

        return FusedPlayerEvent(
            timestamp=now,
            action_type=action_type,
            hand=str(_get(ev, "hand", "")),
            phase=str(_get(ev, "phase", "impact")),
            start_time=_get(ev, "start_time", None),
            impact_time=_get(ev, "impact_time", None),
            end_time=_get(ev, "end_time", None),
            final_confidence=vision_conf,
            intensity_score=intensity,
            damage_scale=intensity,
            source=source,
            vision_confidence=vision_conf,
            camera_speed=camera_speed,
            vision_reason=str(_get(ev, "reason", "")),
            radar_valid=False,
            radar_confidence=0.0,
            radar_velocity_mps=None,
            radar_abs_velocity_mps=None,
            radar_snr_db=None,
            radar_peak_range_m=None,
            radar_reason="radar_not_used_for_camera_only_action",
            fusion_reason=reason,
        )

    def _fuse_straight_with_radar(self, ev: Any) -> Optional[FusedPlayerEvent]:
        now = time.perf_counter()
        action_type = str(_get(ev, "action_type", "unknown"))
        vision_conf = float(_get(ev, "confidence", 0.0) or 0.0)
        camera_speed = float(_get(ev, "camera_speed", 0.0) or 0.0)

        impact_time = _get(ev, "impact_time", None)
        if impact_time is None:
            impact_time = _get(ev, "timestamp", now)

        # No radar available: fallback.
        if self.radar_agent is None:
            if self.cfg.require_radar_for_straight:
                if self.cfg.verbose:
                    print("[FusionCore] dropped straight: radar required but missing")
                return None

            return self._fuse_camera_only(
                ev,
                source="vision_only_radar_missing",
                reason="straight punch emitted without radar_agent",
            )

        # Optional health check.
        radar_health_status = "UNKNOWN"
        try:
            health = self.radar_agent.get_health()
            radar_health_status = str(_get(health, "status", "UNKNOWN"))
        except Exception:
            health = None

        # Still query radar even if health is LOW_FPS; skip only if stopped/no stream.
        if radar_health_status in ("STOPPED", "NO_STREAM", "CAMERA_NOT_OPENED"):
            if self.cfg.require_radar_for_straight:
                if self.cfg.verbose:
                    print(f"[FusionCore] dropped straight: radar status={radar_health_status}")
                return None

            return self._fuse_camera_only(
                ev,
                source="vision_only_radar_unhealthy",
                reason=f"radar unavailable: status={radar_health_status}",
            )

        t_start = float(impact_time) - self.cfg.radar_pre_impact_s
        t_end = float(impact_time) + self.cfg.radar_post_impact_s

        burst = None
        radar_error = None

        try:
            burst = self.radar_agent.query_burst(
                t_start,
                t_end,
                range_min_m=self.cfg.radar_range_min_m,
                range_max_m=self.cfg.radar_range_max_m,
                min_abs_velocity_mps=self.cfg.radar_min_abs_velocity_mps,
            )
        except TypeError as e:
            # Do not silently ignore parameter-name bugs.
            # If this branch is reached, FusionCore may be calling an older
            # RadarAgent API. Fall back to the simple signature, but preserve
            # the error string so the reason is visible when debugging.
            radar_error = f"{type(e).__name__}: {e}"
            try:
                burst = self.radar_agent.query_burst(t_start, t_end)
            except Exception as e2:
                radar_error = f"{radar_error}; fallback failed: {type(e2).__name__}: {e2}"
        except Exception as e:
            radar_error = f"{type(e).__name__}: {e}"

        if burst is None:
            if self.cfg.require_radar_for_straight:
                if self.cfg.verbose:
                    print(f"[FusionCore] dropped straight: radar query failed: {radar_error}")
                return None

            return self._fuse_camera_only(
                ev,
                source="vision_only_radar_query_failed",
                reason=f"radar query failed: {radar_error}",
            )

        radar_valid = bool(_get(burst, "valid", False))
        radar_conf = float(_get(burst, "confidence", 0.0) or 0.0)
        radar_intensity = float(_get(burst, "intensity_score", 0.0) or 0.0)

        if not radar_valid:
            if self.cfg.require_radar_for_straight:
                if self.cfg.verbose:
                    print(f"[FusionCore] dropped straight: radar invalid: {_get(burst, 'reason', '')}")
                return None

            fused = self._fuse_camera_only(
                ev,
                source="vision_only_radar_invalid",
                reason=f"radar invalid: {_get(burst, 'reason', '')}",
            )
            fused.radar_reason = str(_get(burst, "reason", "radar_invalid"))
            return fused

        if radar_intensity <= 0.05:
            fused = self._fuse_camera_only(
                ev,
                source="vision_only_low_radar_intensity",
                reason="radar valid but intensity was too low; used camera fallback",
            )
            fused.radar_reason = "low_radar_intensity"
            fused.radar_velocity_mps = _get(burst, "peak_velocity_mps", None)
            fused.radar_abs_velocity_mps = _get(burst, "abs_peak_velocity_mps", None)
            fused.radar_snr_db = _get(burst, "snr_db", None)
            fused.radar_peak_range_m = _get(burst, "peak_range_m", None)
            return fused

        final_conf = _clip01(
            self.cfg.vision_weight * vision_conf
            + self.cfg.radar_weight * radar_conf
        )

        intensity = _clip01(radar_intensity)
        damage_scale = intensity

        return FusedPlayerEvent(
            timestamp=now,
            action_type=action_type,
            hand=str(_get(ev, "hand", "")),
            phase=str(_get(ev, "phase", "impact")),
            start_time=_get(ev, "start_time", None),
            impact_time=float(impact_time),
            end_time=_get(ev, "end_time", None),
            final_confidence=final_conf,
            intensity_score=intensity,
            damage_scale=damage_scale,
            source="vision+radar",
            vision_confidence=vision_conf,
            camera_speed=camera_speed,
            vision_reason=str(_get(ev, "reason", "")),
            radar_valid=True,
            radar_confidence=radar_conf,
            radar_velocity_mps=_get(burst, "peak_velocity_mps", None),
            radar_abs_velocity_mps=_get(burst, "abs_peak_velocity_mps", None),
            radar_snr_db=_get(burst, "snr_db", None),
            radar_peak_range_m=_get(burst, "peak_range_m", None),
            radar_reason=str(_get(burst, "reason", "")),
            fusion_reason=(
                f"straight fused with radar window "
                f"[{t_start:.3f}, {t_end:.3f}]"
            ),
        )

    # -----------------------------
    # Queue
    # -----------------------------

    def _push(self, fused: FusedPlayerEvent) -> None:
        self._last_fused_event = fused

        try:
            self._queue.put_nowait(fused)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(fused)

        if self.cfg.verbose:
            radar_details = f" radar_reason={fused.radar_reason}"
            if fused.radar_valid:
                snr = (
                    f"{fused.radar_snr_db:.1f}dB"
                    if fused.radar_snr_db is not None
                    else "-"
                )
                peak_range = (
                    f"{fused.radar_peak_range_m:.2f}m"
                    if fused.radar_peak_range_m is not None
                    else "-"
                )
                if fused.radar_velocity_mps is None:
                    radar_details = " radar_v=- radar_speed=-"
                else:
                    radar_details = (
                        f" radar_v={fused.radar_velocity_mps:+.2f}m/s"
                        f" radar_speed={abs(fused.radar_velocity_mps):.2f}m/s"
                    )
                radar_details += f" radar_snr={snr} radar_range={peak_range}"
            print(
                f"[FusionCore] fused action={fused.action_type} "
                f"source={fused.source} conf={fused.final_confidence:.2f} "
                f"intensity={fused.intensity_score:.2f} "
                f"damage_scale={fused.damage_scale:.2f} "
                f"radar_valid={fused.radar_valid}"
                f"{radar_details}"
            )


# ============================================================
# Minimal manual smoke test with fake agents
# ============================================================

if __name__ == "__main__":
    print("fusion_core.py is intended to be imported by the game runtime.")
    print()
    print("Example:")
    print("    fusion = FusionCore(vision_agent, radar_agent)")
    print("    fusion.start()")
    print("    event = fusion.get_next_fused_event()")
