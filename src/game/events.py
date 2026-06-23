from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GameInputEvent:
    """
    Renderer-independent player input event.

    Bridge format:
        Keyboard / FusionCore -> GameCore -> Panda3D renderer
    """
    action_type: str
    source: str = "keyboard"
    hand: str = ""
    confidence: float = 1.0
    intensity_score: float = 0.5
    damage_scale: float = 0.5
    timestamp: float = field(default_factory=time.perf_counter)

    radar_valid: bool = False
    radar_abs_velocity_mps: Optional[float] = None
    fusion_reason: str = ""
    raw: Any = None


@dataclass
class RenderCommand:
    """
    Renderer command produced by GameCore.
    GameCore must not call Panda3D directly.
    """
    type: str
    payload: dict


def event_from_fused(fused: Any) -> GameInputEvent:
    """
    Convert core.fusion_core.FusedPlayerEvent into GameInputEvent.
    Uses duck typing so it keeps working if the dataclass changes slightly.
    """
    return GameInputEvent(
        action_type=str(getattr(fused, "action_type", "unknown")),
        source=str(getattr(fused, "source", "fusion")),
        hand=str(getattr(fused, "hand", "")),
        confidence=float(getattr(fused, "final_confidence", 0.0) or 0.0),
        intensity_score=float(getattr(fused, "intensity_score", 0.0) or 0.0),
        damage_scale=float(getattr(fused, "damage_scale", 0.0) or 0.0),
        timestamp=float(getattr(fused, "timestamp", time.perf_counter()) or time.perf_counter()),
        radar_valid=bool(getattr(fused, "radar_valid", False)),
        radar_abs_velocity_mps=getattr(fused, "radar_abs_velocity_mps", None),
        fusion_reason=str(getattr(fused, "fusion_reason", "")),
        raw=fused,
    )


def keyboard_event(action_type: str, hand: str = "", intensity: float = 0.55) -> GameInputEvent:
    return GameInputEvent(
        action_type=action_type,
        source="keyboard",
        hand=hand,
        confidence=1.0,
        intensity_score=float(intensity),
        damage_scale=float(intensity),
    )
