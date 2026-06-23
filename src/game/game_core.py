from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .events import GameInputEvent, RenderCommand


@dataclass
class CombatState:
    player_hp: int = 100
    opponent_hp: int = 100

    player_blocking: bool = False
    opponent_blocking: bool = False
    opponent_stunned: bool = False

    last_action: str = "idle"
    last_source: str = "none"
    last_damage: int = 0


def parse_action(action_type: str) -> Tuple[str, str]:
    """
    Parse actions like:
        right_straight, left_hook, left_uppercut, block, block_end
    into:
        hand, kind
    """
    action_type = (action_type or "").lower().strip()

    if action_type in ("block", "block_start"):
        return "both", "block"
    if action_type in ("block_end", "guard_end"):
        return "both", "block_end"

    parts = action_type.split("_", 1)
    if len(parts) == 2:
        hand, kind = parts
        return hand, kind

    return "", action_type


class GameCore:
    """
    Renderer-independent boxing game logic.

    It receives GameInputEvent from keyboard or FusionCore and emits RenderCommand.
    It does not import Panda3D.
    """

    BASE_DAMAGE = {
        "straight": 10,
        "hook": 13,
        "uppercut": 16,
    }

    def __init__(self):
        self.state = CombatState()

    def reset(self) -> list[RenderCommand]:
        self.state = CombatState()
        return [
            RenderCommand("reset", {}),
            RenderCommand("opponent_idle", {}),
            RenderCommand("player_block", {"enabled": False}),
        ]

    def toggle_opponent_block(self) -> list[RenderCommand]:
        self.state.opponent_blocking = not self.state.opponent_blocking
        self.state.last_action = "opponent_block_on" if self.state.opponent_blocking else "opponent_block_off"
        return [RenderCommand("opponent_block", {"enabled": self.state.opponent_blocking})]

    def force_stun(self) -> list[RenderCommand]:
        self.state.opponent_stunned = True
        self.state.last_action = "force_stunned"
        return [RenderCommand("opponent_reaction", {"reaction": "stunned"})]

    def handle_player_event(self, ev: GameInputEvent) -> list[RenderCommand]:
        action_type = (ev.action_type or "").lower().strip()
        hand, kind = parse_action(action_type)

        if action_type in ("", "idle", "negative", "unknown"):
            return []

        if kind == "block":
            self.state.player_blocking = True
            self.state.last_action = "player_block"
            self.state.last_source = ev.source
            return [RenderCommand("player_block", {"enabled": True})]

        if kind == "block_end":
            self.state.player_blocking = False
            self.state.last_action = "player_block_end"
            self.state.last_source = ev.source
            return [RenderCommand("player_block", {"enabled": False})]

        if kind not in self.BASE_DAMAGE:
            self.state.last_action = f"ignored:{action_type}"
            self.state.last_source = ev.source
            return []

        return self._resolve_player_attack(ev, hand=hand, kind=kind)

    def _resolve_player_attack(self, ev: GameInputEvent, hand: str, kind: str) -> list[RenderCommand]:
        cmds: List[RenderCommand] = [
            RenderCommand(
                "player_punch",
                {
                    "hand": hand,
                    "kind": kind,
                    "intensity": ev.intensity_score,
                    "source": ev.source,
                },
            )
        ]

        if self.state.opponent_hp <= 0:
            self.state.last_action = "opponent_already_down"
            self.state.last_source = ev.source
            return cmds

        base = self.BASE_DAMAGE[kind]

        # Confidence soft-gate. A low-confidence vision event still gives some damage,
        # but the damage is reduced.
        conf = max(0.20, min(1.0, float(ev.confidence)))

        # Radar/vision intensity scaling. FusionCore uses damage_scale 0~1.
        # Keep a floor so camera-only hook/uppercut still feels playable.
        intensity = max(0.25, min(1.0, float(ev.damage_scale)))
        damage_float = base * conf * (0.65 + 0.75 * intensity)

        if self.state.opponent_blocking:
            damage = max(1, int(round(damage_float * 0.22)))
            reaction = "block"
            outcome = "blocked"
        else:
            damage = max(1, int(round(damage_float)))
            if kind == "uppercut":
                reaction = "receive_uppercut"
            else:
                reaction = "light_hit"
            outcome = "hit"

        self.state.opponent_hp = max(0, self.state.opponent_hp - damage)
        self.state.last_damage = damage
        self.state.last_source = ev.source
        self.state.last_action = (
            f"{hand}_{kind} -> {outcome} ({damage}) "
            f"conf={ev.confidence:.2f} intensity={ev.intensity_score:.2f}"
        )

        if ev.radar_abs_velocity_mps is not None:
            self.state.last_action += f" radar_v={ev.radar_abs_velocity_mps:.2f}m/s"

        if self.state.opponent_hp <= 0:
            self.state.opponent_stunned = True
            reaction = "stunned"
            self.state.last_action += " KO"

        cmds.append(
            RenderCommand(
                "opponent_reaction",
                {
                    "reaction": reaction,
                    "damage": damage,
                    "outcome": outcome,
                    "source": ev.source,
                    "action_type": ev.action_type,
                },
            )
        )
        return cmds
