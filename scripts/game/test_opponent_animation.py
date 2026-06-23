#!/usr/bin/env python3
r"""
scripts/game/test_opponent_animation.py

Panda3D test viewer for assets/opponent.glb.

Install:
    pip install panda3d panda3d-gltf

Run from project root:
    python .\scripts\game\test_opponent_animation.py --model .\assets\opponent.glb

If Panda3D cannot find the model, this script resolves the path to an absolute
OS path and converts it to Panda3D's Filename format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Importing panda3d_gltf registers the glTF/GLB loader when the package is installed.
try:
    import panda3d_gltf  # noqa: F401
except Exception:
    pass

from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Filename,
    Vec4,
    TextNode,
    WindowProperties,
    getModelPath,
)


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


def resolve_model_path(path_str: str) -> str:
    p = Path(path_str)

    if not p.is_absolute():
        # Assume current working directory is project root.
        p = Path.cwd() / p

    p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(
            f"Model not found: {p}\n"
            f"Current working directory: {Path.cwd()}\n"
            "Pass the real GLB path, for example:\n"
            r"  --model C:\Users\user\Desktop\boxing-with-ai\assets\opponent.glb"
        )

    panda_path = Filename.fromOsSpecific(str(p)).getFullpath()
    return panda_path


class OpponentAnimationTester(ShowBase):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args
        self.model_path = resolve_model_path(args.model)

        self.disableMouse()
        self.set_background_color(0.05, 0.05, 0.06, 1)

        props = WindowProperties()
        props.setTitle("RadarBox Opponent Animation Test")
        props.setSize(args.width, args.height)
        self.win.requestProperties(props)

        self._setup_camera()
        self._setup_lights()
        self._load_actor()
        self._setup_keymap()
        self._setup_ui()

        self.current_anim: str | None = None
        self.idle_anim = self.key_to_anim.get("1", None)

        if self.idle_anim:
            self.play_anim(self.idle_anim, loop=True)
        elif self.available_anims:
            self.play_anim(self.available_anims[0], loop=True)

    def _setup_camera(self) -> None:
        self.camera.setPos(0, -self.args.camera_distance, self.args.camera_height)
        self.camera.lookAt(0, 0, self.args.look_height)

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.45, 0.45, 0.45, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        key = DirectionalLight("key")
        key.setColor(Vec4(0.95, 0.95, 0.90, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-35, -55, 0)
        self.render.setLight(key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.35, 0.40, 0.55, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(45, -25, 0)
        self.render.setLight(fill_np)

    def _load_actor(self) -> None:
        print(f"[Viewer] loading: {self.model_path}")
        print(f"[Viewer] Panda model path: {getModelPath()}")

        # Use absolute Panda Filename path here. Relative Windows paths often fail because
        # Panda3D searches only its model-path, not necessarily the shell cwd.
        self.actor = Actor(self.model_path)
        self.actor.reparentTo(self.render)
        self.actor.setPos(0, 0, self.args.z)
        self.actor.setScale(self.args.scale)
        self.actor.setH(self.args.heading)

        self.available_anims = sorted(list(self.actor.getAnimNames()))
        print()
        print("[Viewer] available animations:")
        if not self.available_anims:
            print("  No animations found.")
            print("  Check Blender export settings:")
            print("    Animation: ON")
            print("    NLA Strips: ON")
            print("    All Actions: ON")
        else:
            for name in self.available_anims:
                try:
                    dur = self.actor.getDuration(name)
                    print(f"  - {name}  ({dur:.2f}s)")
                except Exception:
                    print(f"  - {name}")

        print()

    def _setup_keymap(self) -> None:
        aliases = {
            "1": ["Idle"],
            "2": ["Block"],
            "3": ["Cross Punch"],
            "4": ["Hook Punch"],
            "5": ["Uppercut"],
            "6": ["Cross Punch Mirror"],
            "7": ["Hook Punch Mirror"],
            "8": ["Uppercut Mirror"],
            "9": ["Light Hit To Head"],
            "0": ["Stunned"],
            "r": ["Receive Uppercut To The Face"],
        }

        self.key_to_anim: dict[str, str] = {}
        for key, candidates in aliases.items():
            anim = find_anim(self.available_anims, candidates)
            if anim is not None:
                self.key_to_anim[key] = anim

        fallback_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        for key, anim in zip(fallback_keys, self.available_anims):
            if key not in self.key_to_anim:
                self.key_to_anim[key] = anim

        print("[Viewer] key mapping:")
        for key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "r"]:
            if key in self.key_to_anim:
                print(f"  {key}: {self.key_to_anim[key]}")
        print()

        for key in self.key_to_anim:
            self.accept(key, self.play_from_key, [key])

        self.accept("space", self.replay_current)
        self.accept("escape", sys.exit)

    def _setup_ui(self) -> None:
        lines = [
            "RadarBox Opponent Animation Test",
            "1 Idle | 2 Block | 3 Cross | 4 Hook | 5 Uppercut",
            "6 Cross Mirror | 7 Hook Mirror | 8 Uppercut Mirror",
            "9 Light Hit | 0 Stunned | r Receive Uppercut | Space Replay | Esc Quit",
        ]

        self.text = OnscreenText(
            text="\n".join(lines),
            pos=(-1.32, 0.92),
            scale=0.045,
            align=TextNode.ALeft,
            fg=(1, 1, 1, 1),
            mayChange=True,
        )

        self.status = OnscreenText(
            text="Current: none",
            pos=(-1.32, -0.92),
            scale=0.055,
            align=TextNode.ALeft,
            fg=(1, 0.9, 0.25, 1),
            mayChange=True,
        )

    def play_from_key(self, key: str) -> None:
        anim = self.key_to_anim.get(key)
        if anim is None:
            print(f"[Viewer] no animation mapped to key {key}")
            return

        loop = key in ("1", "2")
        self.play_anim(anim, loop=loop)

    def replay_current(self) -> None:
        if self.current_anim:
            self.play_anim(self.current_anim, loop=False)

    def play_anim(self, anim: str, loop: bool = False) -> None:
        if anim not in self.available_anims:
            print(f"[Viewer] animation not found: {anim}")
            return

        self.current_anim = anim
        self.actor.stop()

        if loop:
            self.actor.loop(anim)
            self.status.setText(f"Current: {anim} [loop]")
            print(f"[Viewer] loop: {anim}")
            return

        self.actor.play(anim)
        self.status.setText(f"Current: {anim}")
        print(f"[Viewer] play: {anim}")

        if self.args.auto_return_idle and self.idle_anim and anim != self.idle_anim:
            try:
                duration = float(self.actor.getDuration(anim))
            except Exception:
                duration = 1.0
            self.taskMgr.remove("return_to_idle")
            self.taskMgr.doMethodLater(duration, self._return_to_idle_task, "return_to_idle")

    def _return_to_idle_task(self, task):
        if self.idle_anim:
            self.play_anim(self.idle_anim, loop=True)
        return task.done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test opponent.glb animations with keyboard")
    parser.add_argument("--model", default="assets/opponent.glb")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--heading", type=float, default=180.0)
    parser.add_argument("--camera-distance", type=float, default=6.0)
    parser.add_argument("--camera-height", type=float, default=1.7)
    parser.add_argument("--look-height", type=float, default=1.2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-auto-return-idle", dest="auto_return_idle", action="store_false")
    parser.set_defaults(auto_return_idle=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = OpponentAnimationTester(args)
    app.run()


if __name__ == "__main__":
    main()
