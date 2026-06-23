from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import panda3d_gltf  # noqa: F401
except Exception:
    pass

from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    Filename,
    NodePath,
    TextNode,
    Vec3,
    Vec4,
    WindowProperties,
)

from .events import RenderCommand, keyboard_event
from .game_core import GameCore
from .input_sources import KeyboardInputBuffer, build_fusion_input_source


SCENE_RESIZE = 0.11
SCENE_POS_X = 0.0
SCENE_POS_Y = -2.0
SCENE_POS_Z = 0.0
SCENE_HEADING = 0.0

OPPONENT_POS_X = 0.0
OPPONENT_POS_Y = 0.0
OPPONENT_POS_Z = 0.0
OPPONENT_SCALE = 1.0
OPPONENT_HEADING = 0.0

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
            f"File not found: {p}\nCurrent working directory: {Path.cwd()}"
        )

    return Filename.fromOsSpecific(str(p)).getFullpath()


def norm_name(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").replace(".", " ").replace("|", " ").strip()


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
    intensity: float = 0.5


class OpponentController:
    def __init__(self, app: ShowBase, model_path: str, args):
        self.app = app
        self.args = args
        print(f"[Opponent] loading: {model_path}")
        self.actor = Actor(model_path)
        self.actor.reparentTo(app.render)
        self.actor.setPos(args.opponent_x, args.opponent_y, args.opponent_z)
        self.actor.setScale(args.opponent_scale)
        self.actor.setHpr(args.opponent_heading, 0, 0)
        print(f"[Opponent] heading: {args.opponent_heading} deg")
        self.available_anims = sorted(list(self.actor.getAnimNames()))
        print("[Opponent] available animations:")
        for a in self.available_anims:
            print(f"  - {a}")
        self.anim = {
            "idle": find_anim(self.available_anims, ["Idle", "idle", "anim_00"]),
            "block": find_anim(self.available_anims, ["Block", "block"]),
            "light_hit": find_anim(self.available_anims, ["Light Hit To Head", "light hit", "hit head"]),
            "receive_uppercut": find_anim(self.available_anims, ["Receive Uppercut To The Face", "receive uppercut"]),
            "stunned": find_anim(self.available_anims, ["Stunned", "stunned"]),
            "cross": find_anim(self.available_anims, ["Cross Punch", "cross punch"]),
            "hook": find_anim(self.available_anims, ["Hook Punch", "hook punch"]),
            "uppercut": find_anim(self.available_anims, ["Uppercut", "uppercut"]),
        }
        print("[Opponent] animation mapping:")
        for k, v in self.anim.items():
            print(f"  {k}: {v}")
        self.blocking = False
        self.play_idle()

    def play_idle(self):
        if self.blocking and self.anim.get("block"):
            self.actor.loop(self.anim["block"])
        elif self.anim.get("idle"):
            self.actor.loop(self.anim["idle"])
        elif self.available_anims:
            self.actor.loop(self.available_anims[0])

    def set_blocking(self, enabled: bool):
        self.blocking = bool(enabled)
        self.play_idle()

    def play_reaction(self, reaction: str):
        if reaction == "block":
            self.set_blocking(True)
            return
        if reaction == "receive_uppercut":
            anim = self.anim.get("receive_uppercut")
        elif reaction == "stunned":
            anim = self.anim.get("stunned")
        else:
            anim = self.anim.get("light_hit")
        if anim is None:
            self.play_idle()
            return
        self.actor.stop()
        self.actor.play(anim)
        try:
            duration = float(self.actor.getDuration(anim))
        except Exception:
            duration = 0.8
        self.app.taskMgr.remove("opponent_return_idle")
        self.app.taskMgr.doMethodLater(duration, self._return_idle_task, "opponent_return_idle")

    def _return_idle_task(self, task):
        self.play_idle()
        return task.done

    def reset(self):
        self.blocking = False
        self.play_idle()


class PlayerHandsController:
    def __init__(self, app: ShowBase, glove_path: str, texture_path: Optional[str], args):
        self.app = app
        self.args = args
        base = app.loader.loadModel(glove_path)
        if base.isEmpty():
            raise RuntimeError(f"Failed to load glove model: {glove_path}")
        self._apply_texture_if_available(base, texture_path)
        self.root = app.camera.attachNewNode("player_gloves_root")
        self.right_glove = self._make_centered_glove("right_glove", base)
        self.left_glove = self._make_centered_glove("left_glove", base)
        base.removeNode()
        self.idle_right_pos = Vec3(args.hand_x, args.hand_y, args.hand_z)
        self.idle_left_pos = Vec3(-args.hand_x, args.hand_y, args.hand_z)
        self.guard_right_pos = Vec3(0.30, args.hand_y + 0.10, -0.12)
        self.guard_left_pos = Vec3(-0.30, args.hand_y + 0.10, -0.12)
        self.reset()

    def _apply_texture_if_available(self, model: NodePath, texture_path: Optional[str]) -> None:
        if not texture_path:
            return
        texture_file = Filename(texture_path)
        if not texture_file.exists():
            print(f"[Hands] texture not found, skip: {texture_path}")
            return
        tex = self.app.loader.loadTexture(texture_file)
        if tex is None:
            print(f"[Hands] failed to load texture: {texture_path}")
            return
        tex.setWrapU(tex.WM_repeat)
        tex.setWrapV(tex.WM_repeat)
        model.setTexture(tex, 1)
        for np in model.findAllMatches("**"):
            np.setTexture(tex, 1)
        print(f"[Hands] applied texture: {texture_path}")

    def _make_centered_glove(self, name: str, base_model: NodePath) -> NodePath:
        root = self.root.attachNewNode(name)
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
        normalize_scale = self.args.glove_visual_size / max(max_dim, 1e-9)
        visual.setPos(-center)
        visual.setScale(normalize_scale)
        print(f"[Hands] {name}: center={center}, max_dim={max_dim:.4f}, scale={normalize_scale:.6f}")
        return root

    def reset(self):
        self.blocking = False
        self.right_motion = GloveMotion()
        self.left_motion = GloveMotion()
        self.right_glove.setPos(self.idle_right_pos)
        self.left_glove.setPos(self.idle_left_pos)
        self.right_glove.setHpr(self.args.glove_heading, self.args.glove_pitch, self.args.glove_roll)
        if self.args.mirror_left_glove:
            self.left_glove.setScale(-1, 1, 1)
            self.left_glove.setHpr(-self.args.glove_heading, self.args.glove_pitch, -self.args.glove_roll)
        else:
            self.left_glove.setHpr(self.args.glove_heading, self.args.glove_pitch, self.args.glove_roll)

    def set_blocking(self, enabled: bool):
        self.blocking = bool(enabled)

    def punch(self, hand: str, kind: str, intensity: float = 0.5):
        motion = self.right_motion if hand == "right" else self.left_motion
        motion.active = True
        motion.kind = kind
        motion.t = 0.0
        motion.intensity = max(0.0, min(1.0, float(intensity)))
        motion.duration = {"straight": 0.30, "hook": 0.36, "uppercut": 0.34}.get(kind, 0.32)

    def _pose_for_motion(self, hand: str, motion: GloveMotion, dt: float) -> Vec3:
        idle = self.idle_right_pos if hand == "right" else self.idle_left_pos
        if not motion.active:
            return idle
        motion.t += dt
        p = min(1.0, motion.t / max(motion.duration, 1e-6))
        punch = math.sin(math.pi * p)
        sign = 1.0 if hand == "right" else -1.0
        amp = 0.70 + 0.60 * motion.intensity
        if motion.kind == "straight":
            offset = Vec3(0.0, self.args.straight_depth * amp * punch, 0.08 * punch)
        elif motion.kind == "hook":
            offset = Vec3(-sign * self.args.hook_width * amp * punch, 0.55 * self.args.straight_depth * punch, 0.14 * punch)
        elif motion.kind == "uppercut":
            offset = Vec3(-sign * 0.10 * punch, 0.50 * self.args.straight_depth * punch, self.args.uppercut_height * amp * punch)
        else:
            offset = Vec3(0, 0, 0)
        if p >= 1.0:
            motion.active = False
            motion.t = 0.0
        return idle + offset

    def _smooth_set_pos(self, node: NodePath, target: Vec3, alpha: float = 0.35):
        cur = node.getPos()
        node.setPos(cur + (target - cur) * alpha)

    def update(self, dt: float):
        if self.blocking:
            right_target = self.guard_right_pos
            left_target = self.guard_left_pos
        else:
            right_target = self._pose_for_motion("right", self.right_motion, dt)
            left_target = self._pose_for_motion("left", self.left_motion, dt)
        self._smooth_set_pos(self.right_glove, right_target)
        self._smooth_set_pos(self.left_glove, left_target)
        if self.blocking:
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


class RadarBoxGameApp(ShowBase):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.clock = self.taskMgr.globalClock
        self.game_core = GameCore()
        self.keyboard = KeyboardInputBuffer()
        self.fusion_input = None
        self.sensor_objects = []
        self.scene_path = resolve_path(args.scene)
        self.opponent_path = resolve_path(args.opponent)
        self.glove_path = resolve_path(args.glove)
        self.glove_texture_path = resolve_path(args.glove_texture, must_exist=False) if args.glove_texture else None
        self.disableMouse()
        self.set_background_color(*SKY_COLOR)
        props = WindowProperties()
        props.setTitle("RadarBox Game System")
        props.setSize(args.width, args.height)
        self.win.requestProperties(props)
        self._setup_camera()
        self._setup_lights()
        self._setup_ground()
        self._load_scene()
        self.opponent = OpponentController(self, self.opponent_path, args)
        self.hands = PlayerHandsController(self, self.glove_path, self.glove_texture_path, args)
        self._setup_ui()
        self._setup_keys()
        if args.input in ("fusion", "hybrid"):
            self._setup_fusion_input()
        self.taskMgr.add(self._update, "game_update")

    def _setup_camera(self):
        self.camera.setPos(0, self.args.camera_y, self.args.camera_z)
        self.camera.setHpr(0, 0, 0)
        self.camLens.setFov(self.args.fov)

    def _setup_lights(self):
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

    def _setup_ground(self):
        cm = CardMaker("ground")
        cm.setFrame(-GROUND_SIZE, GROUND_SIZE, -GROUND_SIZE, GROUND_SIZE)
        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setP(-90)
        self.ground.setPos(0, 0, GROUND_Z)
        self.ground.setColor(*GROUND_COLOR)
        self.ground.setTwoSided(True)

    def _load_scene(self):
        print(f"[Scene] loading: {self.scene_path}")
        self.scene_model = self.loader.loadModel(self.scene_path)
        if self.scene_model.isEmpty():
            raise RuntimeError(f"Failed to load scene: {self.scene_path}")
        self.scene_model.reparentTo(self.render)
        self.scene_model.setPos(self.args.scene_x, self.args.scene_y, self.args.scene_z)
        self.scene_model.setScale(self.args.scene_scale)
        self.scene_model.setHpr(self.args.scene_heading, 0, 0)

    def _setup_fusion_input(self):
        try:
            self.fusion_input, self.sensor_objects = build_fusion_input_source(self.args)
            self.fusion_input.start()
            print("[Game] Fusion input started")
        except Exception as e:
            if self.args.input == "fusion":
                raise
            print(f"[Game] Fusion input failed; continuing with keyboard only: {type(e).__name__}: {e}")
            self.fusion_input = None

    def _setup_ui(self):
        help_text = (
            "RadarBox Game System\n"
            "Keyboard: J/H/U right straight/hook/uppercut | F/G/T left straight/hook/uppercut\n"
            "B block | O opponent block | K stun | R reset | Esc quit\n"
            f"Input mode: {self.args.input}"
        )
        self.help_text = OnscreenText(text=help_text, pos=(-1.32, 0.93), scale=0.041, align=TextNode.ALeft, fg=(1, 1, 1, 1), mayChange=False)
        self.hp_text = OnscreenText(text="", pos=(-1.32, -0.86), scale=0.056, align=TextNode.ALeft, fg=(1, 0.9, 0.25, 1), mayChange=True)
        self.action_text = OnscreenText(text="", pos=(-1.32, -0.95), scale=0.045, align=TextNode.ALeft, fg=(0.75, 0.95, 1.0, 1), mayChange=True)

    def _setup_keys(self):
        self.accept("escape", self.quit)
        self.accept("r", self.reset_game)
        self.accept("o", self.toggle_opponent_block)
        self.accept("k", self.force_stun)
        self.accept("b", lambda: self.keyboard.push(keyboard_event("block", hand="both", intensity=0.0)))
        self.accept("b-up", lambda: self.keyboard.push(keyboard_event("block_end", hand="both", intensity=0.0)))
        self.accept("j", lambda: self.keyboard.push(keyboard_event("right_straight", hand="right")))
        self.accept("h", lambda: self.keyboard.push(keyboard_event("right_hook", hand="right")))
        self.accept("u", lambda: self.keyboard.push(keyboard_event("right_uppercut", hand="right")))
        self.accept("f", lambda: self.keyboard.push(keyboard_event("left_straight", hand="left")))
        self.accept("g", lambda: self.keyboard.push(keyboard_event("left_hook", hand="left")))
        self.accept("t", lambda: self.keyboard.push(keyboard_event("left_uppercut", hand="left")))

    def quit(self):
        if self.fusion_input is not None:
            self.fusion_input.stop()
        sys.exit()

    def reset_game(self):
        for cmd in self.game_core.reset():
            self.apply_command(cmd)
        self.hands.reset()
        self.opponent.reset()

    def toggle_opponent_block(self):
        for cmd in self.game_core.toggle_opponent_block():
            self.apply_command(cmd)

    def force_stun(self):
        for cmd in self.game_core.force_stun():
            self.apply_command(cmd)

    def apply_command(self, cmd: RenderCommand):
        if cmd.type == "reset":
            self.hands.reset()
            self.opponent.reset()
        elif cmd.type == "player_block":
            self.hands.set_blocking(bool(cmd.payload.get("enabled", False)))
        elif cmd.type == "player_punch":
            self.hands.punch(hand=str(cmd.payload.get("hand", "right")), kind=str(cmd.payload.get("kind", "straight")), intensity=float(cmd.payload.get("intensity", 0.5)))
        elif cmd.type == "opponent_block":
            self.opponent.set_blocking(bool(cmd.payload.get("enabled", False)))
        elif cmd.type == "opponent_idle":
            self.opponent.play_idle()
        elif cmd.type == "opponent_reaction":
            self.opponent.play_reaction(str(cmd.payload.get("reaction", "light_hit")))

    def _consume_events(self):
        events = self.keyboard.poll()
        if self.fusion_input is not None:
            events.extend(self.fusion_input.poll())
        for ev in events:
            cmds = self.game_core.handle_player_event(ev)
            for cmd in cmds:
                self.apply_command(cmd)

    def _update_ui(self):
        s = self.game_core.state
        self.hp_text.setText(
            f"Player HP: {s.player_hp:3d}    Opponent HP: {s.opponent_hp:3d}    "
            f"Player Block: {'ON' if s.player_blocking else 'OFF'}    Opponent Block: {'ON' if s.opponent_blocking else 'OFF'}"
        )
        self.action_text.setText(f"Action: {s.last_action}    Source: {s.last_source}")

    def _update(self, task):
        dt = self.clock.getDt()
        self._consume_events()
        self.hands.update(dt)
        self._update_ui()
        return task.cont


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RadarBox modular game system")
    parser.add_argument("--scene", default="assets/scene.glb")
    parser.add_argument("--opponent", default="assets/opponent.glb")
    parser.add_argument("--glove", default="assets/boxing_glove.glb")
    parser.add_argument("--glove-texture", default="assets/boxing_glove.png")
    parser.add_argument("--input", choices=["keyboard", "fusion", "hybrid"], default="keyboard")
    parser.add_argument("--fusion-module", default="core.fusion_core")
    parser.add_argument("--vision-module", default="core.vision_agent")
    parser.add_argument("--vision-class", default=None)
    parser.add_argument("--radar-module", default="core.radar_agent")
    parser.add_argument("--radar-class", default=None)
    parser.add_argument("--enable-radar", action="store_true")
    parser.add_argument("--pose-model", default="models/pose_landmarker_lite.task")
    parser.add_argument("--classifier", default="models/punch_classifier.joblib")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--vision-debug", action="store_true")
    parser.add_argument("--active-hand", default="right")
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--radar-pc-ip", default="192.168.33.30")
    parser.add_argument("--radar-data-port", type=int, default=4098)
    parser.add_argument("--radar-min-abs-velocity", type=float, default=2.0)
    parser.add_argument("--require-radar-for-straight", action="store_true")
    parser.add_argument("--fusion-verbose", action="store_true")
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
    return parser


def main():
    args = build_arg_parser().parse_args()
    app = RadarBoxGameApp(args)
    app.run()
