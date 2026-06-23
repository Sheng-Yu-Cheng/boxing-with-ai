#!/usr/bin/env python3
r"""
scripts/game/alpha_game.py

RadarBox alpha prototype.

Features:
    1. Load assets/scene.glb
    2. Load assets/opponent.glb
    3. Load assets/boxing_glove.glb + assets/boxing_glove.png
    4. Keyboard-controlled first-person punches
    5. Opponent plays hit reaction / block / stunned animations
    6. Opponent HP decreases

Install:
    pip install panda3d panda3d-gltf

Run from project root:
    python .\scripts\game\alpha_game.py

Controls:
    J : right straight
    H : right hook
    U : right uppercut
    F : left straight
    G : left hook
    T : left uppercut
    B : player block / guard
    O : opponent block toggle
    K : force opponent stunned
    R : reset HP / reset state
    Esc : quit
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import panda3d_gltf  # noqa: F401
except Exception:
    pass

from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    CardMaker,
    AmbientLight,
    DirectionalLight,
    Filename,
    NodePath,
    TextNode,
    Vec3,
    Vec4,
    WindowProperties,
)

# =============================================================================
# Easy edit defaults
# =============================================================================

# Scene / arena settings
SCENE_RESIZE = 0.11
SCENE_POS_X = 0.0
SCENE_POS_Y = -2.0
SCENE_POS_Z = 0.0
SCENE_HEADING = 0.0

# Opponent settings
# Rotate opponent by 180 degrees so it faces the player/camera by default.
OPPONENT_POS_X = 0.0
OPPONENT_POS_Y = 0.0
OPPONENT_POS_Z = 0.0
OPPONENT_SCALE = 1.3
OPPONENT_HEADING = 0.0

# Simple environment
SKY_COLOR = (0.45, 0.72, 0.98, 1.0)
GROUND_COLOR = (0.22, 0.60, 0.22, 1.0)
GROUND_Z = -2.0
GROUND_SIZE = 80.0

def resolve_path(path_str: str, must_exist: bool = True) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            f"Current working directory: {Path.cwd()}"
        )
    return Filename.fromOsSpecific(str(p)).getFullpath()


def norm_name(s: str) -> str:
    return (
        s.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .replace("|", " ")
        .strip()
    )


def find_anim(available: list[str], candidates: list[str]) -> str | None:
    available_norm = [(name, norm_name(name)) for name in available]

    for cand in candidates:
        c = norm_name(cand)
        for name, n in available_norm:
            if n == c:
                return name

    for cand in candidates:
        tokens = [t for t in norm_name(cand).split() if t]
        for name, n in available_norm:
            if all(t in n for t in tokens):
                return name

    return None


@dataclass
class GloveMotion:
    active: bool = False
    kind: str = "idle"
    t: float = 0.0
    duration: float = 0.32


@dataclass
class CombatState:
    player_hp: int = 100
    opponent_hp: int = 100
    opponent_blocking: bool = False
    opponent_stunned: bool = False
    last_action: str = "idle"


class RadarBoxAlphaGame(ShowBase):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args
        self.clock = self.taskMgr.globalClock

        self.scene_path = resolve_path(args.scene, must_exist=True)
        self.opponent_path = resolve_path(args.opponent, must_exist=True)
        self.glove_path = resolve_path(args.glove, must_exist=True)
        self.glove_texture_path = (
            resolve_path(args.glove_texture, must_exist=False)
            if args.glove_texture
            else None
        )

        self.state = CombatState()
        self.right_motion = GloveMotion()
        self.left_motion = GloveMotion()
        self.player_blocking = False

        self.disableMouse()
        self.set_background_color(*SKY_COLOR)

        props = WindowProperties()
        props.setTitle("RadarBox Alpha")
        props.setSize(args.width, args.height)
        self.win.requestProperties(props)

        self._setup_camera()
        self._setup_lights()
        self._setup_ground()
        self._load_scene()
        self._load_opponent()
        self._load_player_gloves()
        self._setup_ui()
        self._setup_keys()

        self.taskMgr.add(self._update, "alpha_game_update")

    def _setup_camera(self) -> None:
        # Panda3D camera looks toward +Y when HPR is 0.
        self.camera.setPos(0, self.args.camera_y, self.args.camera_z)
        self.camera.setHpr(0, 0, 0)
        self.camLens.setFov(self.args.fov)

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.55, 0.55, 0.60, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        key = DirectionalLight("key")
        key.setColor(Vec4(1.0, 0.95, 0.86, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-30, -55, 0)
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.35, 0.45, 0.75, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(55, -25, 0)
        self.render.setLight(fill_np)


    def _setup_ground(self) -> None:
        # Simple green ground plane at z = GROUND_Z
        cm = CardMaker("ground")
        cm.setFrame(-GROUND_SIZE, GROUND_SIZE, -GROUND_SIZE, GROUND_SIZE)
        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setP(-90)  # make it horizontal
        self.ground.setPos(0, 0, GROUND_Z)
        self.ground.setColor(*GROUND_COLOR)
        self.ground.setTwoSided(True)

    def _load_scene(self) -> None:
        print(f"[Alpha] loading scene: {self.scene_path}")
        self.scene_model = self.loader.loadModel(self.scene_path)
        if self.scene_model.isEmpty():
            raise RuntimeError(f"Failed to load scene: {self.scene_path}")
        self.scene_model.reparentTo(self.render)
        self.scene_model.setPos(self.args.scene_x, self.args.scene_y, self.args.scene_z)
        self.scene_model.setScale(self.args.scene_scale)
        self.scene_model.setHpr(self.args.scene_heading, 0, 0)

    def _load_opponent(self) -> None:
        print(f"[Alpha] loading opponent: {self.opponent_path}")
        self.opponent = Actor(self.opponent_path)
        self.opponent.reparentTo(self.render)
        self.opponent.setPos(self.args.opponent_x, self.args.opponent_y, self.args.opponent_z)
        self.opponent.setScale(self.args.opponent_scale)
        self.opponent.setHpr(self.args.opponent_heading, 0, 0)
        print(f"[Alpha] opponent heading: {self.args.opponent_heading} deg")

        self.available_anims = sorted(list(self.opponent.getAnimNames()))
        print("[Alpha] opponent animations:")
        if not self.available_anims:
            print("  No animations found. Re-export opponent.glb with Animation + NLA Strips + All Actions.")
        else:
            for anim in self.available_anims:
                print(f"  - {anim}")

        self.anim = {
            "idle": find_anim(self.available_anims, ["Idle", "idle", "anim_00"]),
            "block": find_anim(self.available_anims, ["Block", "block"]),
            "light_hit": find_anim(self.available_anims, ["Light Hit To Head", "light hit", "hit head"]),
            "receive_uppercut": find_anim(self.available_anims, ["Receive Uppercut To The Face", "receive uppercut"]),
            "stunned": find_anim(self.available_anims, ["Stunned", "stunned"]),
        }

        print("[Alpha] animation mapping:")
        for k, v in self.anim.items():
            print(f"  {k}: {v}")

        self._play_opponent_idle()

    def _apply_texture_if_available(self, model: NodePath) -> None:
        if not self.glove_texture_path:
            return
        texture_file = Filename(self.glove_texture_path)
        if not texture_file.exists():
            print(f"[Alpha] glove texture not found, skip: {self.glove_texture_path}")
            return
        tex = self.loader.loadTexture(texture_file)
        if tex is None:
            print(f"[Alpha] failed to load glove texture: {self.glove_texture_path}")
            return
        tex.setWrapU(tex.WM_repeat)
        tex.setWrapV(tex.WM_repeat)
        model.setTexture(tex, 1)
        for np in model.findAllMatches("**"):
            np.setTexture(tex, 1)
        print(f"[Alpha] applied glove texture: {self.glove_texture_path}")

    def _make_centered_glove(self, name: str, parent: NodePath, base_model: NodePath) -> NodePath:
        root = parent.attachNewNode(name)
        visual = base_model.copyTo(root)
        visual.setName(name + "_visual")
        visual.setTwoSided(True)

        bounds = visual.getTightBounds()
        if bounds is None:
            raise RuntimeError("Could not compute glove bounds. Model may contain no visible geometry.")

        bmin, bmax = bounds
        center = (bmin + bmax) * 0.5
        size_vec = bmax - bmin
        max_dim = max(size_vec.x, size_vec.y, size_vec.z)
        if max_dim <= 1e-9:
            raise RuntimeError(f"Invalid glove bounds: bmin={bmin}, bmax={bmax}")

        normalize_scale = self.args.glove_visual_size / max_dim
        visual.setPos(-center)
        visual.setScale(normalize_scale)
        print(f"[Alpha] {name}: center={center}, max_dim={max_dim:.6f}, scale={normalize_scale:.6f}")
        return root

    def _load_player_gloves(self) -> None:
        print(f"[Alpha] loading glove: {self.glove_path}")
        base = self.loader.loadModel(self.glove_path)
        if base.isEmpty():
            raise RuntimeError(f"Failed to load glove model: {self.glove_path}")

        self._apply_texture_if_available(base)

        self.glove_root = self.camera.attachNewNode("player_gloves_root")
        self.right_glove = self._make_centered_glove("right_glove", self.glove_root, base)
        self.left_glove = self._make_centered_glove("left_glove", self.glove_root, base)
        base.removeNode()

        self.idle_right_pos = Vec3(self.args.hand_x, self.args.hand_y, self.args.hand_z)
        self.idle_left_pos = Vec3(-self.args.hand_x, self.args.hand_y, self.args.hand_z)
        self.guard_right_pos = Vec3(0.30, self.args.hand_y + 0.10, -0.12)
        self.guard_left_pos = Vec3(-0.30, self.args.hand_y + 0.10, -0.12)

        self.right_glove.setPos(self.idle_right_pos)
        self.left_glove.setPos(self.idle_left_pos)
        self.right_glove.setHpr(self.args.glove_heading, self.args.glove_pitch, self.args.glove_roll)
        if self.args.mirror_left_glove:
            self.left_glove.setScale(-1, 1, 1)
            self.left_glove.setHpr(-self.args.glove_heading, self.args.glove_pitch, -self.args.glove_roll)
        else:
            self.left_glove.setHpr(self.args.glove_heading, self.args.glove_pitch, self.args.glove_roll)

    def _setup_ui(self) -> None:
        help_text = (
            "RadarBox Alpha\n"
            "J/H/U right straight/hook/uppercut | F/G/T left straight/hook/uppercut\n"
            "B player block | O opponent block toggle | K stun opponent | R reset | Esc quit"
        )
        self.help_text = OnscreenText(
            text=help_text,
            pos=(-1.32, 0.93),
            scale=0.043,
            align=TextNode.ALeft,
            fg=(1, 1, 1, 1),
            mayChange=False,
        )
        self.hp_text = OnscreenText(
            text="",
            pos=(-1.32, -0.88),
            scale=0.060,
            align=TextNode.ALeft,
            fg=(1, 0.9, 0.25, 1),
            mayChange=True,
        )
        self.action_text = OnscreenText(
            text="",
            pos=(-1.32, -0.96),
            scale=0.050,
            align=TextNode.ALeft,
            fg=(0.75, 0.95, 1.0, 1),
            mayChange=True,
        )
        self._update_ui()

    def _setup_keys(self) -> None:
        self.accept("escape", sys.exit)
        self.accept("r", self.reset_game)
        self.accept("o", self.toggle_opponent_block)
        self.accept("k", self.force_stun)
        self.accept("b", self.set_player_blocking, [True])
        self.accept("b-up", self.set_player_blocking, [False])
        self.accept("j", self.player_punch, ["right", "straight"])
        self.accept("h", self.player_punch, ["right", "hook"])
        self.accept("u", self.player_punch, ["right", "uppercut"])
        self.accept("f", self.player_punch, ["left", "straight"])
        self.accept("g", self.player_punch, ["left", "hook"])
        self.accept("t", self.player_punch, ["left", "uppercut"])

    def _play_opponent_idle(self) -> None:
        if self.state.opponent_hp <= 0:
            return
        if self.state.opponent_blocking and self.anim.get("block"):
            self.opponent.loop(self.anim["block"])
            return
        if self.anim.get("idle"):
            self.opponent.loop(self.anim["idle"])
        elif self.available_anims:
            self.opponent.loop(self.available_anims[0])

    def _play_opponent_once_then_idle(self, anim_name: str | None) -> None:
        if not anim_name:
            self._play_opponent_idle()
            return
        self.opponent.stop()
        self.opponent.play(anim_name)
        try:
            duration = float(self.opponent.getDuration(anim_name))
        except Exception:
            duration = 0.8
        self.taskMgr.remove("opponent_return_idle")
        self.taskMgr.doMethodLater(duration, self._return_opponent_idle_task, "opponent_return_idle")

    def _return_opponent_idle_task(self, task):
        self._play_opponent_idle()
        return task.done

    def set_player_blocking(self, value: bool) -> None:
        self.player_blocking = value
        self.state.last_action = "player_block" if value else "idle"
        self._update_ui()

    def toggle_opponent_block(self) -> None:
        self.state.opponent_blocking = not self.state.opponent_blocking
        self.state.last_action = "opponent_block_on" if self.state.opponent_blocking else "opponent_block_off"
        self._play_opponent_idle()
        self._update_ui()

    def force_stun(self) -> None:
        self.state.opponent_stunned = True
        self.state.last_action = "force_stunned"
        self._play_opponent_once_then_idle(self.anim.get("stunned"))
        self._update_ui()

    def player_punch(self, side: str, kind: str) -> None:
        if self.state.opponent_hp <= 0:
            self.state.last_action = "opponent_already_down"
            self._update_ui()
            return

        motion = self.right_motion if side == "right" else self.left_motion
        motion.active = True
        motion.kind = kind
        motion.t = 0.0
        motion.duration = {"straight": 0.30, "hook": 0.36, "uppercut": 0.34}.get(kind, 0.32)
        self.state.last_action = f"{side}_{kind}"
        self.resolve_hit(kind)
        self._update_ui()

    def resolve_hit(self, kind: str) -> None:
        if self.state.opponent_blocking:
            damage = 3
            self.state.opponent_hp = max(0, self.state.opponent_hp - damage)
            self.state.last_action += f" -> blocked ({damage})"
            self._play_opponent_once_then_idle(self.anim.get("block"))
            return

        base_damage = {"straight": 10, "hook": 13, "uppercut": 16}.get(kind, 8)
        damage = max(1, base_damage + random.randint(-2, 3))
        self.state.opponent_hp = max(0, self.state.opponent_hp - damage)
        self.state.last_action += f" -> hit ({damage})"

        if self.state.opponent_hp <= 0:
            self.state.opponent_stunned = True
            self.state.last_action += " KO"
            self._play_opponent_once_then_idle(self.anim.get("stunned"))
            return

        if kind == "uppercut" and self.anim.get("receive_uppercut"):
            self._play_opponent_once_then_idle(self.anim["receive_uppercut"])
        elif self.anim.get("light_hit"):
            self._play_opponent_once_then_idle(self.anim["light_hit"])
        else:
            self._play_opponent_once_then_idle(self.anim.get("stunned"))

    def reset_game(self) -> None:
        self.state = CombatState()
        self.right_motion = GloveMotion()
        self.left_motion = GloveMotion()
        self.player_blocking = False
        self.right_glove.setPos(self.idle_right_pos)
        self.left_glove.setPos(self.idle_left_pos)
        self._play_opponent_idle()
        self._update_ui()

    def _pose_for_motion(self, side: str, motion: GloveMotion, dt: float) -> Vec3:
        idle = self.idle_right_pos if side == "right" else self.idle_left_pos
        if not motion.active:
            return idle

        motion.t += dt
        p = min(1.0, motion.t / max(motion.duration, 1e-6))
        punch = math.sin(math.pi * p)
        sign = 1.0 if side == "right" else -1.0

        if motion.kind == "straight":
            offset = Vec3(0.0, self.args.straight_depth * punch, 0.08 * punch)
        elif motion.kind == "hook":
            offset = Vec3(-sign * self.args.hook_width * punch, 0.55 * self.args.straight_depth * punch, 0.14 * punch)
        elif motion.kind == "uppercut":
            offset = Vec3(-sign * 0.10 * punch, 0.50 * self.args.straight_depth * punch, self.args.uppercut_height * punch)
        else:
            offset = Vec3(0, 0, 0)

        if p >= 1.0:
            motion.active = False
            motion.t = 0.0
        return idle + offset

    def _smooth_set_pos(self, node: NodePath, target: Vec3, alpha: float = 0.35) -> None:
        cur = node.getPos()
        node.setPos(cur + (target - cur) * alpha)

    def _update_gloves(self, dt: float) -> None:
        if self.player_blocking:
            right_target = self.guard_right_pos
            left_target = self.guard_left_pos
        else:
            right_target = self._pose_for_motion("right", self.right_motion, dt)
            left_target = self._pose_for_motion("left", self.left_motion, dt)

        self._smooth_set_pos(self.right_glove, right_target)
        self._smooth_set_pos(self.left_glove, left_target)

        if self.player_blocking:
            self.right_glove.setHpr(self.args.glove_heading - 15, self.args.glove_pitch, self.args.glove_roll + 10)
            if self.args.mirror_left_glove:
                self.left_glove.setHpr(-self.args.glove_heading + 15, self.args.glove_pitch, -self.args.glove_roll - 10)
            else:
                self.left_glove.setHpr(self.args.glove_heading + 15, self.args.glove_pitch, self.args.glove_roll - 10)
        else:
            self.right_glove.setHpr(self.args.glove_heading, self.args.glove_pitch, self.args.glove_roll)
            if self.args.mirror_left_glove:
                self.left_glove.setHpr(-self.args.glove_heading, self.args.glove_pitch, -self.args.glove_roll)
            else:
                self.left_glove.setHpr(self.args.glove_heading, self.args.glove_pitch, self.args.glove_roll)

    def _update_ui(self) -> None:
        block = "ON" if self.state.opponent_blocking else "OFF"
        pblock = "ON" if self.player_blocking else "OFF"
        self.hp_text.setText(
            f"Player HP: {self.state.player_hp:3d}    "
            f"Opponent HP: {self.state.opponent_hp:3d}    "
            f"Player Block: {pblock}    Opponent Block: {block}"
        )
        self.action_text.setText(f"Action: {self.state.last_action}")

    def _update(self, task):
        dt = self.clock.getDt()
        self._update_gloves(dt)
        self._update_ui()
        return task.cont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RadarBox alpha prototype")
    parser.add_argument("--scene", default="assets/scene.glb")
    parser.add_argument("--opponent", default="assets/opponent.glb")
    parser.add_argument("--glove", default="assets/boxing_glove.glb")
    parser.add_argument("--glove-texture", default="assets/boxing_glove.png")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=72.0)

    parser.add_argument("--camera-y", type=float, default=-6.0)
    parser.add_argument("--camera-z", type=float, default=1.65)

    parser.add_argument("--scene-x", type=float, default=SCENE_POS_X)
    parser.add_argument("--scene-y", type=float, default=SCENE_POS_Y)
    parser.add_argument("--scene-z", type=float, default=SCENE_POS_Z)
    parser.add_argument("--scene-scale", "--scene-resize", dest="scene_scale", type=float, default=SCENE_RESIZE)
    parser.add_argument("--scene-heading", type=float, default=SCENE_HEADING)

    parser.add_argument("--opponent-x", type=float, default=OPPONENT_POS_X)
    parser.add_argument("--opponent-y", type=float, default=OPPONENT_POS_Y)
    parser.add_argument("--opponent-z", type=float, default=OPPONENT_POS_Z)
    parser.add_argument("--opponent-scale", type=float, default=OPPONENT_SCALE)
    parser.add_argument("--opponent-heading", type=float, default=OPPONENT_HEADING)

    parser.add_argument("--glove-visual-size", type=float, default=0.38)
    parser.add_argument("--glove-heading", type=float, default=0.0)
    parser.add_argument("--glove-pitch", type=float, default=0.0)
    parser.add_argument("--glove-roll", type=float, default=0.0)
    parser.add_argument("--no-mirror-left-glove", dest="mirror_left_glove", action="store_false")
    parser.set_defaults(mirror_left_glove=True)

    parser.add_argument("--hand-x", type=float, default=0.42)
    parser.add_argument("--hand-y", type=float, default=1.25)
    parser.add_argument("--hand-z", type=float, default=-0.45)
    parser.add_argument("--straight-depth", type=float, default=1.05)
    parser.add_argument("--hook-width", type=float, default=0.75)
    parser.add_argument("--uppercut-height", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = RadarBoxAlphaGame(args)
    app.run()


if __name__ == "__main__":
    main()
