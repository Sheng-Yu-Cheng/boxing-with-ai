from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.vision_agent import PlayerPoseState

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

try:
    import cv2
except Exception:
    cv2 = None

from core.rv_map_debug import analyze_recent_rv, make_heatmap_image


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

RADAR_RANGE_MIN_M = 0.6
RADAR_RANGE_MAX_M = 2.5
RADAR_MARKER_Y_MIN = 0.5
RADAR_MARKER_Y_MAX = 3.0


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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
        self.last_pose_x_range_gain = 1.0
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

    def _pose_x_range_gain(self, radar_range_m: Optional[float]) -> float:
        gain = 1.0
        if (
            self.args.pose_use_radar_range_x
            and radar_range_m is not None
            and radar_range_m > 0.0
            and self.args.pose_x_reference_range_m > 0.0
        ):
            gain = float(radar_range_m) / float(self.args.pose_x_reference_range_m)
            gain = clamp(
                gain,
                float(self.args.pose_x_range_gain_min),
                float(self.args.pose_x_range_gain_max),
            )
        self.last_pose_x_range_gain = gain
        return gain

    def update_from_pose_state(self, pose_state: "PlayerPoseState", dt: float, radar_range_m: Optional[float] = None) -> None:
        """Drive glove visuals from the continuous VisionAgent pose stream."""
        del dt  # Reserved for time-based smoothing/tuning later.
        range_gain = self._pose_x_range_gain(radar_range_m)

        def target_for_hand(pose_hand, idle: Vec3, outward: float) -> Vec3:
            if not pose_state.pose_detected or not pose_hand.detected:
                return idle
            return Vec3(
                pose_hand.x * self.args.pose_scale_x * range_gain + outward,
                self.args.hand_y + pose_hand.y * self.args.pose_depth_scale,
                pose_hand.z * self.args.pose_scale_z + self.args.pose_z_offset,
            )

        alpha = max(0.0, min(1.0, float(self.args.pose_smoothing_alpha)))
        # VisionAgent mirrors camera input by default for natural display/training.
        # MediaPipe's landmark labels then arrive swapped relative to the user's
        # physical hands, so swap only the continuous visual pose stream here.
        right_pose_hand = pose_state.left_hand if self.args.mirror_input else pose_state.right_hand
        left_pose_hand = pose_state.right_hand if self.args.mirror_input else pose_state.left_hand
        right_target = target_for_hand(
            right_pose_hand,
            self.idle_right_pos,
            self.args.pose_hand_spacing,
        )
        left_target = target_for_hand(
            left_pose_hand,
            self.idle_left_pos,
            -self.args.pose_hand_spacing,
        )

        self._smooth_set_pos(self.right_glove, right_target, alpha)
        self._smooth_set_pos(self.left_glove, left_target, alpha)

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
        self.vision_agent = None
        self.radar_agent = None
        self.sensor_objects = []
        self.pose_control_enabled = (
            args.pose_control
            if getattr(args, "pose_control", None) is not None
            else args.input in ("fusion", "hybrid")
        )
        self.debug_enabled = bool(args.debug)
        self._last_rv_debug_update = 0.0
        self._latest_rv_debug_image = None
        self.radar_origin_x = 0.0
        self.radar_origin_y = 0.0
        self.radar_origin_z = 0.0
        self.radar_world_scale = 1.2
        self.radar_forward_sign = -1.0
        self.radar_theta_sign = -1.0
        self.enable_radar_player_motion = True
        self.player_camera_height = 1.65
        self.player_camera_look_at_z = 1.45
        self.player_motion_smoothing_alpha = 0.35
        self._radar_player_world_xy = None
        self.latest_radar_range_for_pose_m = None
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
        self._setup_player_marker()
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

    def _setup_player_marker(self):
        marker = self.loader.loadModel("models/misc/sphere")
        if marker.isEmpty():
            cm = CardMaker("player_marker_card")
            cm.setFrame(-0.08, 0.08, -0.08, 0.08)
            marker = self.render.attachNewNode(cm.generate())
            marker.setP(-90)
        else:
            marker.reparentTo(self.render)
        marker.setName("radar_player_marker")
        marker.setScale(0.09)
        marker.setColor(0.1, 0.35, 1.0, 1.0)
        marker.setPos(0.0, RADAR_MARKER_Y_MIN, 0.0)
        marker.setLightOff(1)
        self.player_marker = marker

    def _setup_fusion_input(self):
        try:
            self.fusion_input, self.sensor_objects = build_fusion_input_source(self.args)
            self.vision_agent = self.fusion_input.vision_agent
            self.radar_agent = self.fusion_input.radar_agent
            self.fusion_input.start()
            print("[Game] Fusion input started")
        except Exception as e:
            if self.args.input == "fusion":
                raise
            print(f"[Game] Fusion input failed; continuing with keyboard only: {type(e).__name__}: {e}")
            self.fusion_input = None
            self.vision_agent = None
            self.radar_agent = None

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
        self.radar_text = OnscreenText(text="", pos=(0.38, 0.92), scale=0.038, align=TextNode.ALeft, fg=(0.45, 0.75, 1.0, 1), mayChange=True)

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
        if cv2 is not None:
            cv2.destroyAllWindows()
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

    def _update_player_marker_from_range(self, range_m) -> None:
        if range_m is None:
            world_y = RADAR_MARKER_Y_MIN
        else:
            range_t = (
                (float(range_m) - RADAR_RANGE_MIN_M)
                / max(RADAR_RANGE_MAX_M - RADAR_RANGE_MIN_M, 1e-6)
            )
            range_t = clamp(range_t, 0.0, 1.0)
            world_y = RADAR_MARKER_Y_MIN + range_t * (RADAR_MARKER_Y_MAX - RADAR_MARKER_Y_MIN)
        self.player_marker.setPos(0.0, world_y, 0.0)

    def _radar_polar_to_world(self, range_m: float, theta_deg: float):
        theta = math.radians(theta_deg * self.radar_theta_sign)
        x = self.radar_origin_x + self.radar_world_scale * range_m * math.sin(theta)
        y = self.radar_origin_y + self.radar_forward_sign * self.radar_world_scale * range_m * math.cos(theta)
        z = self.radar_origin_z
        return x, y, z

    def _update_player_from_tracking(self, range_m, theta_deg):
        if range_m is None:
            return None
        theta = 0.0 if theta_deg is None else float(theta_deg)
        target_x, target_y, _ = self._radar_polar_to_world(float(range_m), theta)

        alpha = clamp(float(self.player_motion_smoothing_alpha), 0.0, 1.0)
        if self._radar_player_world_xy is None:
            current = self.camera.getPos()
            current_x = float(current.x)
            current_y = float(current.y)
        else:
            current_x, current_y = self._radar_player_world_xy

        world_x = current_x + alpha * (target_x - current_x)
        world_y = current_y + alpha * (target_y - current_y)
        self._radar_player_world_xy = (world_x, world_y)

        marker_pos = (world_x, world_y, 0.08)
        self.player_marker.setPos(*marker_pos)

        if self.enable_radar_player_motion:
            self.camera.setPos(world_x, world_y, self.player_camera_height)
            self.camera.lookAt(0.0, 0.0, self.player_camera_look_at_z)

        return marker_pos

    def update_radar_ui(self):
        def fmt(value, suffix: str = "") -> str:
            if value is None:
                return "N/A"
            try:
                return f"{float(value):.2f}{suffix}"
            except (TypeError, ValueError):
                return str(value)

        def fmt_beam(tracking) -> str:
            tx0 = getattr(tracking, "beam_tx0_code", None)
            tx1 = getattr(tracking, "beam_tx1_code", None)
            tx2 = getattr(tracking, "beam_tx2_code", None)
            theta = getattr(tracking, "beam_theta_deg", None)
            if tx0 is None and tx1 is None and tx2 is None and theta is None:
                return "N/A"
            codes = ",".join("N/A" if x is None else str(int(x)) for x in (tx0, tx1, tx2))
            return f"[{codes}] theta={fmt(theta, ' deg')}"

        if self.radar_agent is not None and hasattr(self.radar_agent, "get_latest_tracking_state"):
            try:
                tracking = self.radar_agent.get_latest_tracking_state()
            except Exception as exc:
                self.radar_text.setText(f"Radar Tracking: state error ({type(exc).__name__})")
                return

            range_m = getattr(tracking, "range_m", None)
            theta_smooth = getattr(tracking, "theta_smooth_deg", None)
            velocity = getattr(tracking, "peak_velocity_mps", None)
            player_world = None
            if getattr(tracking, "valid", False) and range_m is not None:
                self.latest_radar_range_for_pose_m = float(range_m)
                player_world = self._update_player_from_tracking(range_m, theta_smooth)
            player_x = player_world[0] if player_world is not None else None
            player_y = player_world[1] if player_world is not None else None
            glove_range_gain = getattr(self.hands, "last_pose_x_range_gain", 1.0)
            self.radar_text.setText(
                "Radar Tracking\n"
                f"valid: {getattr(tracking, 'valid', False)}\n"
                f"range: {fmt(range_m, ' m')}\n"
                f"AoA raw: {fmt(getattr(tracking, 'theta_raw_deg', None), ' deg')}\n"
                f"AoA smooth: {fmt(theta_smooth, ' deg')}\n"
                f"SNR: {fmt(getattr(tracking, 'snr_db', None), ' dB')}\n"
                f"velocity: {fmt(velocity, ' m/s')}\n"
                f"Player world: x={fmt(player_x)} y={fmt(player_y)} range={fmt(range_m, ' m')} theta={fmt(theta_smooth, ' deg')}\n"
                f"Radar origin: x={fmt(self.radar_origin_x)} y={fmt(self.radar_origin_y)} forward={fmt(self.radar_forward_sign)} theta_sign={fmt(self.radar_theta_sign)}\n"
                f"Radar player motion: enabled={self.enable_radar_player_motion}\n"
                f"Glove X range comp: enabled={self.args.pose_use_radar_range_x} radar={fmt(self.latest_radar_range_for_pose_m, ' m')} ref={fmt(self.args.pose_x_reference_range_m, ' m')} gain={fmt(glove_range_gain)} scale_x={fmt(self.args.pose_scale_x)} spacing={fmt(self.args.pose_hand_spacing)} dx=[{fmt(self.args.pose_glove_min_dx)},{fmt(self.args.pose_glove_max_dx)}]\n"
                f"track: r={fmt(getattr(tracking, 'track_range_m', None), ' m')} v={fmt(getattr(tracking, 'track_velocity_mps', None), ' m/s')}\n"
                f"bins: ({getattr(tracking, 'target_range_bin', None)}, {getattr(tracking, 'target_doppler_bin', None)}) cand={getattr(tracking, 'candidate_count', 0)}\n"
                f"beam: {fmt_beam(tracking)}\n"
                f"reason: {getattr(tracking, 'reason', 'unknown')}"
            )
            return

        summary = None
        if self.radar_agent is not None and hasattr(self.radar_agent, "get_latest_summary"):
            try:
                summary = self.radar_agent.get_latest_summary()
            except Exception as exc:
                self.radar_text.setText(f"Radar: summary error ({type(exc).__name__})")
                return

        if summary is None:
            self.radar_text.setText(
                "Radar\n"
                "peak_range_m: N/A\n"
                "peak_velocity_mps: N/A\n"
                "snr_db: N/A\n"
                "reason: no_radar_agent\n"
                "AoA: N/A\n"
                "Beam: N/A"
            )
            self._update_player_marker_from_range(None)
            return

        peak_range_m = getattr(summary, "peak_range_m", None)
        peak_velocity_mps = getattr(summary, "peak_velocity_mps", None)
        snr_db = getattr(summary, "snr_db", None)
        reason = getattr(summary, "reason", "unknown")

        self.radar_text.setText(
            "Radar\n"
            f"peak_range_m: {fmt(peak_range_m)}\n"
            f"peak_velocity_mps: {fmt(peak_velocity_mps)}\n"
            f"snr_db: {fmt(snr_db)}\n"
            f"reason: {reason}\n"
            "AoA: N/A\n"
            "Beam: N/A"
        )

        self._update_player_marker_from_range(peak_range_m)

    def _update_debug_windows(self) -> None:
        if not self.debug_enabled or cv2 is None:
            return

        try:
            vision_image = None
            if self.vision_agent is not None:
                vision_image = self.vision_agent.get_latest_debug_image()
            if vision_image is not None:
                cv2.imshow("RadarBox Debug - Vision", vision_image)

            now = time.perf_counter()
            if self.radar_agent is not None and (
                self._latest_rv_debug_image is None
                or now - self._last_rv_debug_update >= self.args.debug_rv_period
            ):
                self._last_rv_debug_update = now
                result = analyze_recent_rv(
                    self.radar_agent,
                    window_s=self.args.debug_rv_window_s,
                    range_min=self.args.debug_rv_range_min,
                    range_max=self.args.debug_rv_range_max,
                    min_abs_velocity=self.args.debug_rv_min_abs_velocity,
                    max_abs_velocity=self.args.debug_rv_max_abs_velocity,
                    direction=self.args.debug_rv_direction,
                    velocity_strong=self.args.debug_rv_velocity_strong,
                )
                health = self.radar_agent.get_health()
                result["radar_health"] = getattr(health, "status", "UNKNOWN")
                self._latest_rv_debug_image = make_heatmap_image(result, cv2)
            if self._latest_rv_debug_image is not None:
                cv2.imshow("RadarBox Debug - Recent RV Map", self._latest_rv_debug_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.quit()
        except Exception as exc:
            print(f"[Game] debug windows disabled: {type(exc).__name__}: {exc}")
            self.debug_enabled = False

    def _update(self, task):
        dt = self.clock.getDt()
        self._consume_events()
        pose_state = None
        if (
            self.pose_control_enabled
            and self.vision_agent is not None
            and hasattr(self.vision_agent, "get_latest_player_pose_state")
        ):
            pose_state = self.vision_agent.get_latest_player_pose_state()

        if pose_state is not None:
            self.hands.update_from_pose_state(pose_state, dt, radar_range_m=self.latest_radar_range_for_pose_m)
        else:
            self.hands.update(dt)
        self._update_ui()
        self.update_radar_ui()
        self._update_debug_windows()
        return task.cont


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RadarBox modular game system")
    parser.add_argument("--scene", default="assets/scene.glb")
    parser.add_argument("--opponent", default="assets/opponent.glb")
    parser.add_argument("--glove", default="assets/boxing_glove.glb")
    parser.add_argument("--glove-texture", default="assets/boxing_glove.png")
    parser.add_argument("--input", choices=["keyboard", "fusion", "hybrid"], default="keyboard")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show Vision and recent RV-map windows beside the game (fusion/hybrid only)",
    )
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
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--mirror-input", dest="mirror_input", action="store_true")
    mirror.add_argument("--no-mirror-input", dest="mirror_input", action="store_false")
    parser.set_defaults(mirror_input=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    pose_control = parser.add_mutually_exclusive_group()
    pose_control.add_argument("--pose-control", dest="pose_control", action="store_true")
    pose_control.add_argument("--no-pose-control", dest="pose_control", action="store_false")
    parser.set_defaults(pose_control=None)
    parser.add_argument("--pose-scale-x", type=float, default=0.75)
    parser.add_argument("--pose-scale-z", type=float, default=0.75)
    parser.add_argument("--pose-depth-scale", type=float, default=0.80)
    parser.add_argument("--pose-z-offset", type=float, default=-0.35)
    parser.add_argument("--pose-smoothing-alpha", type=float, default=0.35)
    parser.add_argument(
        "--pose-use-radar-range-x",
        action="store_true",
        help="scale MediaPipe pose hand X by latest radar range to stabilize glove spacing",
    )
    parser.add_argument("--pose-x-reference-range-m", type=float, default=2.0)
    parser.add_argument("--pose-x-range-gain-min", type=float, default=0.65)
    parser.add_argument("--pose-x-range-gain-max", type=float, default=1.6)
    parser.add_argument("--pose-glove-min-dx", type=float, default=1.05)
    parser.add_argument("--pose-glove-max-dx", type=float, default=1.60)
    parser.add_argument(
        "--pose-hand-spacing",
        type=float,
        default=0.12,
        help="extra outward offset for each pose-controlled glove",
    )
    parser.add_argument("--radar-pc-ip", default="192.168.33.30")
    parser.add_argument("--radar-dca-ip", default="192.168.33.180")
    parser.add_argument("--radar-data-port", type=int, default=4098)
    parser.add_argument("--radar-min-abs-velocity", type=float, default=2.0)
    parser.add_argument(
        "--enable-aoa-feedback",
        action="store_true",
        help="write AoA-smoothed 3Tx phase codes to the mmWave Studio beam command file",
    )
    parser.add_argument("--beam-cmd-file", default=r"C:\temp\radarbox_beam_cmd.txt")
    parser.add_argument("--beam-update-interval-s", type=float, default=0.50)
    parser.add_argument("--beam-min-confidence", type=float, default=0.20)
    parser.add_argument("--beam-min-snr-db", type=float, default=6.0)
    parser.add_argument("--debug-rv-window-s", type=float, default=5.0)
    parser.add_argument("--debug-rv-period", type=float, default=0.25)
    parser.add_argument("--debug-rv-range-min", type=float, default=0.0)
    parser.add_argument("--debug-rv-range-max", type=float, default=3.0)
    parser.add_argument("--debug-rv-min-abs-velocity", type=float, default=2.0)
    parser.add_argument("--debug-rv-max-abs-velocity", type=float, default=15.5)
    parser.add_argument(
        "--debug-rv-direction",
        choices=["negative", "positive", "both"],
        default="negative",
    )
    parser.add_argument("--debug-rv-velocity-strong", type=float, default=8.0)
    parser.add_argument("--require-radar-for-straight", action="store_true")
    parser.add_argument(
        "--camera-only-straight-damage",
        action="store_true",
        help="allow straight punches to fall back to camera-only damage when radar is enabled",
    )
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
    parser.add_argument("--hand-x", type=float, default=0.52)
    parser.add_argument("--hand-y", type=float, default=1.25)
    parser.add_argument("--hand-z", type=float, default=-0.45)
    parser.add_argument("--straight-depth", type=float, default=1.05)
    parser.add_argument("--hook-width", type=float, default=0.75)
    parser.add_argument("--uppercut-height", type=float, default=0.75)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.debug:
        if args.input not in ("fusion", "hybrid"):
            parser.error("--debug requires --input fusion or --input hybrid")
        args.enable_radar = True
        args.vision_debug = True
        args.fusion_verbose = True
        print("[Game] --debug enabled: radar input + Vision/RV windows")
    app = RadarBoxGameApp(args)
    app.run()
