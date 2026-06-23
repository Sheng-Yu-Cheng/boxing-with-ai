#!/usr/bin/env python3
r"""
scripts/game/test_player_gloves_v2.py

A simple robust Panda3D viewer for assets/boxing_glove.glb + assets/boxing_glove.png.

Purpose:
    Load one boxing glove model, apply texture manually if needed, duplicate it
    as left/right first-person gloves, auto-center, auto-scale, and test simple
    punch motions by keyboard.

Install:
    pip install panda3d panda3d-gltf

Run:
    python .\scripts\game\test_player_gloves_v2.py

Or explicitly:
    python .\scripts\game\test_player_gloves_v2.py --model .\assets\boxing_glove.glb --texture .\assets\boxing_glove.png

Keys:
    J : right straight
    H : right hook
    U : right uppercut

    F : left straight
    G : left hook
    T : left uppercut

    B : hold block / guard while pressed
    R : reset hands
    Esc : quit

Useful tuning:
    python .\scripts\game\test_player_gloves_v2.py --visual-size 0.45
    python .\scripts\game\test_player_gloves_v2.py --heading 90
    python .\scripts\game\test_player_gloves_v2.py --pitch 90
    python .\scripts\game\test_player_gloves_v2.py --roll 90
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import panda3d_gltf  # noqa: F401
except Exception:
    pass

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Filename,
    NodePath,
    TextNode,
    Vec3,
    Vec4,
    WindowProperties,
)


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


@dataclass
class GloveMotion:
    active: bool = False
    kind: str = "idle"
    t: float = 0.0
    duration: float = 0.32


class PlayerGlovesV2(ShowBase):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args
        self.model_path = resolve_path(args.model, must_exist=True)
        self.texture_path = resolve_path(args.texture, must_exist=False) if args.texture else None
        self.clock = self.taskMgr.globalClock

        self.disableMouse()
        self.set_background_color(0.035, 0.035, 0.045, 1)

        props = WindowProperties()
        props.setTitle("RadarBox Player Gloves V2")
        props.setSize(args.width, args.height)
        self.win.requestProperties(props)

        self._setup_camera()
        self._setup_lights()
        self._load_gloves()
        self._setup_ui()
        self._setup_keys()

        self.right_motion = GloveMotion()
        self.left_motion = GloveMotion()
        self.blocking = False

        self.taskMgr.add(self._update, "update_player_gloves_v2")

    def _setup_camera(self) -> None:
        # First-person camera. In Panda3D, camera looks toward +Y.
        self.camera.setPos(0, 0, 0)
        self.camera.setHpr(0, 0, 0)
        self.camLens.setFov(self.args.fov)

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.75, 0.75, 0.78, 1))
        ambient_np = self.camera.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        key = DirectionalLight("key")
        key.setColor(Vec4(1.0, 0.95, 0.88, 1))
        key_np = self.camera.attachNewNode(key)
        key_np.setHpr(35, -55, 0)
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.40, 0.48, 0.70, 1))
        fill_np = self.camera.attachNewNode(fill)
        fill_np.setHpr(-45, -25, 0)
        self.render.setLight(fill_np)

    def _apply_texture_if_available(self, model: NodePath) -> None:
        if not self.texture_path:
            return

        texture_file = Filename(self.texture_path)
        if not texture_file.exists():
            print(f"[GloveV2] texture not found, skip: {self.texture_path}")
            return

        tex = self.loader.loadTexture(texture_file)
        if tex is None:
            print(f"[GloveV2] failed to load texture: {self.texture_path}")
            return

        tex.setWrapU(tex.WM_repeat)
        tex.setWrapV(tex.WM_repeat)

        # Force-apply texture to the model and all children.
        model.setTexture(tex, 1)
        for np in model.findAllMatches("**"):
            np.setTexture(tex, 1)

        print(f"[GloveV2] applied texture: {self.texture_path}")

    def _make_centered_visual(self, name: str, parent: NodePath, base_model: NodePath) -> NodePath:
        """
        Create a container node. The visible model is shifted so its tight-bounds
        center is at the container origin, then normalized to visual_size.
        """
        root = parent.attachNewNode(name)

        visual = base_model.copyTo(root)
        visual.setName(name + "_visual")
        visual.setTwoSided(True)

        bounds = visual.getTightBounds()
        if bounds is None:
            raise RuntimeError(
                "Could not compute model bounds. "
                "The model may contain no visible geometry."
            )

        bmin, bmax = bounds
        center = (bmin + bmax) * 0.5
        size_vec = bmax - bmin
        max_dim = max(size_vec.x, size_vec.y, size_vec.z)

        if max_dim <= 1e-9:
            raise RuntimeError(f"Invalid model bounds: bmin={bmin}, bmax={bmax}")

        normalize_scale = self.args.visual_size / max_dim

        visual.setPos(-center)
        visual.setScale(normalize_scale)

        print(f"[GloveV2] {name} bounds:")
        print(f"  bmin = {bmin}")
        print(f"  bmax = {bmax}")
        print(f"  center = {center}")
        print(f"  max_dim = {max_dim:.6f}")
        print(f"  normalize_scale = {normalize_scale:.6f}")

        return root

    def _load_gloves(self) -> None:
        print(f"[GloveV2] loading model: {self.model_path}")

        base = self.loader.loadModel(self.model_path)
        if base.isEmpty():
            raise RuntimeError(f"Failed to load model: {self.model_path}")

        self._apply_texture_if_available(base)

        # Parent to camera so these are first-person gloves.
        self.glove_root = self.camera.attachNewNode("player_gloves_root")

        self.right_glove = self._make_centered_visual("right_glove", self.glove_root, base)
        self.left_glove = self._make_centered_visual("left_glove", self.glove_root, base)

        base.removeNode()

        self.idle_right_pos = Vec3(self.args.hand_x, self.args.hand_y, self.args.hand_z)
        self.idle_left_pos = Vec3(-self.args.hand_x, self.args.hand_y, self.args.hand_z)

        self.guard_right_pos = Vec3(0.28, self.args.hand_y + 0.08, -0.12)
        self.guard_left_pos = Vec3(-0.28, self.args.hand_y + 0.08, -0.12)

        self.right_glove.setPos(self.idle_right_pos)
        self.left_glove.setPos(self.idle_left_pos)

        self.right_glove.setHpr(self.args.heading, self.args.pitch, self.args.roll)

        if self.args.mirror_left:
            # Mirror left glove by flipping container X.
            self.left_glove.setScale(-1, 1, 1)
            self.left_glove.setHpr(-self.args.heading, self.args.pitch, -self.args.roll)
        else:
            self.left_glove.setHpr(self.args.heading, self.args.pitch, self.args.roll)

        print("[GloveV2] final scene tree:")
        self.glove_root.ls()

    def _setup_ui(self) -> None:
        text = (
            "RadarBox Player Gloves V2\n"
            "Right: J straight | H hook | U uppercut\n"
            "Left : F straight | G hook | T uppercut\n"
            "B hold block | R reset | Esc quit\n"
            "Tune: --visual-size, --hand-x, --hand-y, --hand-z, --heading, --pitch, --roll"
        )

        self.help_text = OnscreenText(
            text=text,
            pos=(-1.32, 0.92),
            scale=0.043,
            align=TextNode.ALeft,
            fg=(1, 1, 1, 1),
            mayChange=False,
        )

        self.status = OnscreenText(
            text="idle",
            pos=(-1.32, -0.92),
            scale=0.055,
            align=TextNode.ALeft,
            fg=(1, 0.9, 0.25, 1),
            mayChange=True,
        )

    def _setup_keys(self) -> None:
        self.accept("escape", sys.exit)
        self.accept("r", self.reset_hands)

        self.accept("j", self.trigger_motion, ["right", "straight"])
        self.accept("h", self.trigger_motion, ["right", "hook"])
        self.accept("u", self.trigger_motion, ["right", "uppercut"])

        self.accept("f", self.trigger_motion, ["left", "straight"])
        self.accept("g", self.trigger_motion, ["left", "hook"])
        self.accept("t", self.trigger_motion, ["left", "uppercut"])

        self.accept("b", self.set_blocking, [True])
        self.accept("b-up", self.set_blocking, [False])

    def reset_hands(self) -> None:
        self.right_motion = GloveMotion()
        self.left_motion = GloveMotion()
        self.blocking = False
        self.right_glove.setPos(self.idle_right_pos)
        self.left_glove.setPos(self.idle_left_pos)
        self.status.setText("reset")

    def set_blocking(self, value: bool) -> None:
        self.blocking = value
        self.status.setText("block" if value else "idle")

    def trigger_motion(self, side: str, kind: str) -> None:
        motion = self.right_motion if side == "right" else self.left_motion
        motion.active = True
        motion.kind = kind
        motion.t = 0.0
        motion.duration = {
            "straight": 0.30,
            "hook": 0.36,
            "uppercut": 0.34,
        }.get(kind, 0.32)

        self.status.setText(f"{side}_{kind}")
        print(f"[GloveV2] {side}_{kind}")

    def _pose_for_motion(self, side: str, motion: GloveMotion, dt: float) -> Vec3:
        idle = self.idle_right_pos if side == "right" else self.idle_left_pos

        if not motion.active:
            return idle

        motion.t += dt
        p = min(1.0, motion.t / max(motion.duration, 1e-6))

        # Smooth out-and-back: 0 -> 1 -> 0.
        punch = math.sin(math.pi * p)
        sign = 1.0 if side == "right" else -1.0

        if motion.kind == "straight":
            offset = Vec3(
                0.0,
                self.args.straight_depth * punch,
                0.08 * punch,
            )

        elif motion.kind == "hook":
            offset = Vec3(
                -sign * self.args.hook_width * punch,
                0.55 * self.args.straight_depth * punch,
                0.14 * punch,
            )

        elif motion.kind == "uppercut":
            offset = Vec3(
                -sign * 0.10 * punch,
                0.50 * self.args.straight_depth * punch,
                self.args.uppercut_height * punch,
            )

        else:
            offset = Vec3(0, 0, 0)

        if p >= 1.0:
            motion.active = False
            motion.t = 0.0

        return idle + offset

    def _smooth_set_pos(self, node: NodePath, target: Vec3, alpha: float = 0.35) -> None:
        cur = node.getPos()
        node.setPos(cur + (target - cur) * alpha)

    def _update(self, task):
        dt = self.clock.getDt()

        if self.blocking:
            right_target = self.guard_right_pos
            left_target = self.guard_left_pos
        else:
            right_target = self._pose_for_motion("right", self.right_motion, dt)
            left_target = self._pose_for_motion("left", self.left_motion, dt)

        self._smooth_set_pos(self.right_glove, right_target)
        self._smooth_set_pos(self.left_glove, left_target)

        if self.blocking:
            self.right_glove.setHpr(self.args.heading - 15, self.args.pitch, self.args.roll + 10)
            if self.args.mirror_left:
                self.left_glove.setHpr(-self.args.heading + 15, self.args.pitch, -self.args.roll - 10)
            else:
                self.left_glove.setHpr(self.args.heading + 15, self.args.pitch, self.args.roll - 10)
        else:
            self.right_glove.setHpr(self.args.heading, self.args.pitch, self.args.roll)
            if self.args.mirror_left:
                self.left_glove.setHpr(-self.args.heading, self.args.pitch, -self.args.roll)
            else:
                self.left_glove.setHpr(self.args.heading, self.args.pitch, self.args.roll)

        return task.cont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test first-person boxing gloves v2")
    parser.add_argument("--model", default="assets/boxing_glove.glb")
    parser.add_argument("--texture", default="assets/boxing_glove.png")

    # Target visible size after auto-centering and auto-scaling.
    parser.add_argument("--visual-size", type=float, default=0.38)

    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)

    parser.add_argument("--no-mirror-left", dest="mirror_left", action="store_false")
    parser.set_defaults(mirror_left=True)

    # First-person hand placement in camera coordinates.
    parser.add_argument("--hand-x", type=float, default=0.42)
    parser.add_argument("--hand-y", type=float, default=1.25)
    parser.add_argument("--hand-z", type=float, default=-0.45)

    # Motion amplitudes.
    parser.add_argument("--straight-depth", type=float, default=1.05)
    parser.add_argument("--hook-width", type=float, default=0.75)
    parser.add_argument("--uppercut-height", type=float, default=0.75)

    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = PlayerGlovesV2(args)
    app.run()


if __name__ == "__main__":
    main()
