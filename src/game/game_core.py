from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from .events import GameInputEvent, PlayerDefenseState, RenderCommand


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


@dataclass
class EnemyAIConfig:
    enabled: bool = True
    attack_min_s: float = 0.8
    attack_max_s: float = 1.6
    telegraph_s: float = 0.45
    recover_s: float = 0.85
    guard_chance: float = 0.30
    damage_scale: float = 1.0


@dataclass
class EnemyAIState:
    phase: str = "idle"
    timer_s: float = 0.8
    next_decision_s: float = 0.8
    current_attack: str = "none"
    ai_enabled: bool = True
    difficulty: str = "demo"


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

    ENEMY_BASE_DAMAGE = {
        "straight": 8,
        "hook": 11,
        "uppercut": 14,
    }

    def __init__(self):
        self.state = CombatState()
        self.enemy_ai_cfg = EnemyAIConfig()
        self.enemy_ai_state = EnemyAIState()
        self._enemy_rng = random.Random(7)

    def configure_enemy_ai(
        self,
        *,
        enabled: bool,
        attack_min_s: float = 0.8,
        attack_max_s: float = 1.6,
        telegraph_s: float = 0.45,
        recover_s: float = 0.85,
        guard_chance: float = 0.30,
        damage_scale: float = 1.0,
    ) -> None:
        self.enemy_ai_cfg = EnemyAIConfig(
            enabled=bool(enabled),
            attack_min_s=max(0.05, float(attack_min_s)),
            attack_max_s=max(0.05, float(attack_max_s)),
            telegraph_s=max(0.05, float(telegraph_s)),
            recover_s=max(0.05, float(recover_s)),
            guard_chance=max(0.0, min(1.0, float(guard_chance))),
            damage_scale=max(0.0, float(damage_scale)),
        )
        if self.enemy_ai_cfg.attack_max_s < self.enemy_ai_cfg.attack_min_s:
            self.enemy_ai_cfg.attack_max_s = self.enemy_ai_cfg.attack_min_s
        self.enemy_ai_state.ai_enabled = self.enemy_ai_cfg.enabled
        self._reset_enemy_ai()

    def reset(self) -> list[RenderCommand]:
        self.state = CombatState()
        self._reset_enemy_ai()
        return [
            RenderCommand("reset", {}),
            RenderCommand("opponent_idle", {}),
            RenderCommand("player_block", {"enabled": False}),
        ]

    def _reset_enemy_ai(self) -> None:
        self.enemy_ai_state = EnemyAIState(
            phase="idle",
            timer_s=self._enemy_idle_delay(),
            next_decision_s=0.0,
            current_attack="none",
            ai_enabled=self.enemy_ai_cfg.enabled,
            difficulty="demo",
        )

    def _enemy_idle_delay(self) -> float:
        cfg = self.enemy_ai_cfg
        return self._enemy_rng.uniform(cfg.attack_min_s, cfg.attack_max_s)

    def _choose_enemy_attack(self) -> str:
        r = self._enemy_rng.random()
        if r < 0.50:
            return "straight"
        if r < 0.80:
            return "hook"
        return "uppercut"

    def update_enemy_ai(self, dt: float, defense_state: PlayerDefenseState) -> list[RenderCommand]:
        cfg = self.enemy_ai_cfg
        ai = self.enemy_ai_state

        if not cfg.enabled:
            return []
        if self.state.player_hp <= 0 or self.state.opponent_hp <= 0:
            return []
        if self.state.opponent_stunned:
            return []

        ai.ai_enabled = cfg.enabled
        ai.timer_s -= max(0.0, float(dt))
        if ai.timer_s > 0.0:
            return []

        if ai.phase == "idle":
            if self._enemy_rng.random() < cfg.guard_chance:
                ai.phase = "guard"
                ai.current_attack = "none"
                ai.timer_s = self._enemy_rng.uniform(0.45, 0.90)
                self.state.opponent_blocking = True
                self.state.last_action = "enemy_guard"
                self.state.last_source = "enemy_ai"
                return [RenderCommand("opponent_block", {"enabled": True, "source": "enemy_ai"})]

            ai.phase = "telegraph"
            ai.current_attack = self._choose_enemy_attack()
            jitter = self._enemy_rng.uniform(-0.10, 0.10)
            ai.timer_s = max(0.15, cfg.telegraph_s + jitter)
            self.state.opponent_blocking = False
            self.state.last_action = f"enemy_telegraph_{ai.current_attack}"
            self.state.last_source = "enemy_ai"
            return [
                RenderCommand("opponent_block", {"enabled": False, "source": "enemy_ai"}),
                RenderCommand("opponent_attack", {"kind": ai.current_attack, "source": "enemy_ai"}),
            ]

        if ai.phase == "guard":
            ai.phase = "idle"
            ai.current_attack = "none"
            ai.timer_s = self._enemy_idle_delay()
            self.state.opponent_blocking = False
            self.state.last_action = "enemy_guard_end"
            self.state.last_source = "enemy_ai"
            return [RenderCommand("opponent_block", {"enabled": False, "source": "enemy_ai"})]

        if ai.phase == "telegraph":
            kind = ai.current_attack if ai.current_attack in self.ENEMY_BASE_DAMAGE else "straight"
            blocked = bool(defense_state.blocking)
            base_damage = self.ENEMY_BASE_DAMAGE[kind]
            block_multiplier = 0.20 if blocked else 1.0
            damage = max(0, int(round(base_damage * cfg.damage_scale * block_multiplier)))
            self.state.player_hp = max(0, self.state.player_hp - damage)
            self.state.last_damage = damage
            self.state.last_source = "enemy_ai"
            self.state.last_action = (
                f"enemy_{kind} -> {'blocked' if blocked else 'hit'} ({damage}) "
                f"guard={defense_state.guard_score:.2f}"
            )
            ai.phase = "recover"
            ai.timer_s = cfg.recover_s
            return [
                RenderCommand(
                    "player_damaged",
                    {
                        "kind": kind,
                        "damage": damage,
                        "blocked": blocked,
                        "guard_score": defense_state.guard_score,
                        "source": "enemy_ai",
                    },
                )
            ]

        if ai.phase == "recover":
            ai.phase = "idle"
            ai.current_attack = "none"
            ai.timer_s = self._enemy_idle_delay()
            self.state.last_action = "enemy_recover"
            self.state.last_source = "enemy_ai"
            return [RenderCommand("opponent_idle", {"source": "enemy_ai"})]

        self._reset_enemy_ai()
        return []

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
        # Preserve the full radar range so slow and fast punches feel distinct;
        # camera-only actions retain a floor so they remain playable.
        raw_intensity = max(0.0, min(1.0, float(ev.damage_scale)))
        intensity = raw_intensity if ev.radar_valid else max(0.25, raw_intensity)
        # Make Doppler speed materially visible in the demo. Radar intensity is
        # normalized from punch velocity by RadarAgent; a valid fast punch can
        # now reach a 2.40x multiplier, while camera-only actions keep the
        # original, more conservative curve.
        if ev.radar_valid:
            damage_multiplier = 0.65 + 1.75 * intensity
        else:
            damage_multiplier = 0.65 + 0.75 * intensity
        damage_float = base * conf * damage_multiplier

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
            f"conf={ev.confidence:.2f} intensity={ev.intensity_score:.2f} "
            f"x{damage_multiplier:.2f}"
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
