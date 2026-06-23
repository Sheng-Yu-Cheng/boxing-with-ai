from __future__ import annotations

import importlib
import inspect
import queue
from dataclasses import is_dataclass, fields
from typing import Any, Optional

from .events import GameInputEvent, event_from_fused


class KeyboardInputBuffer:
    def __init__(self):
        self._queue: "queue.Queue[GameInputEvent]" = queue.Queue()

    def push(self, ev: GameInputEvent) -> None:
        self._queue.put(ev)

    def poll(self) -> list[GameInputEvent]:
        out: list[GameInputEvent] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out


class FusionInputSource:
    def __init__(self, fusion_core: Any):
        self.fusion_core = fusion_core

    def start(self) -> None:
        if hasattr(self.fusion_core, "start"):
            self.fusion_core.start()

    def stop(self) -> None:
        if hasattr(self.fusion_core, "stop"):
            self.fusion_core.stop()

    def poll(self) -> list[GameInputEvent]:
        events: list[GameInputEvent] = []
        while True:
            fused = self.fusion_core.get_next_fused_event()
            if fused is None:
                break
            events.append(event_from_fused(fused))
        return events


def _import_module(module_name: str):
    return importlib.import_module(module_name)


def _find_class_with_method(module, method_name: str, preferred_name: Optional[str] = None):
    if preferred_name:
        cls = getattr(module, preferred_name, None)
        if cls is None:
            raise RuntimeError(f"Cannot find class {preferred_name} in {module.__name__}")
        return cls

    candidates = []
    for _, obj in vars(module).items():
        if inspect.isclass(obj) and hasattr(obj, method_name):
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"Cannot find a class with method {method_name} in {module.__name__}")

    candidates.sort(key=lambda c: (c.__module__ != module.__name__, c.__name__))
    return candidates[0]


def _filter_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in kwargs.items() if v is not None}
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return clean
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return clean
    return {k: v for k, v in clean.items() if k in params}


def _find_config_class(module, agent_cls):
    agent_name = agent_cls.__name__.lower()
    preferred = []
    if "vision" in agent_name:
        preferred = ["TrajectoryVisionConfig", "VisionConfig", "PoseVisionConfig"]
    elif "radar" in agent_name:
        preferred = ["RadarConfig"]

    for name in preferred:
        obj = getattr(module, name, None)
        if inspect.isclass(obj):
            return obj

    configs = [obj for _, obj in vars(module).items() if inspect.isclass(obj) and obj.__name__.endswith("Config")]
    if not configs:
        return None
    configs.sort(key=lambda c: (not is_dataclass(c), c.__module__ != module.__name__, c.__name__))
    return configs[0]


def _build_config(config_cls, kwargs: dict[str, Any]):
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if is_dataclass(config_cls):
        names = {f.name for f in fields(config_cls)}
        return config_cls(**{k: v for k, v in clean.items() if k in names})
    return config_cls(**_filter_kwargs(config_cls, clean))


def _construct_agent(module, cls, kwargs: dict[str, Any]):
    clean = {k: v for k, v in kwargs.items() if v is not None}
    sig = inspect.signature(cls)
    required = [
        name for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect._empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]

    # Handles VisionAgent(config: TrajectoryVisionConfig) and RadarAgent(config: RadarConfig).
    if len(required) == 1 and (required[0] == "config" or required[0] not in clean):
        config_cls = _find_config_class(module, cls)
        if config_cls is not None:
            config = _build_config(config_cls, clean)
            try:
                return cls(config=config)
            except TypeError:
                return cls(config)

    try:
        return cls(**_filter_kwargs(cls, clean))
    except Exception as e1:
        try:
            return cls()
        except Exception as e2:
            raise RuntimeError(
                f"Failed to construct {cls.__module__}.{cls.__name__}. "
                f"kwargs error={type(e1).__name__}: {e1}; "
                f"no-arg error={type(e2).__name__}: {e2}"
            ) from e2


def _start_if_possible(agent: Any) -> None:
    if agent is None:
        return
    for method in ("start", "run_async", "open"):
        fn = getattr(agent, method, None)
        if callable(fn):
            fn()
            return


def build_fusion_input_source(args) -> tuple[FusionInputSource, list[Any]]:
    fusion_mod = _import_module(args.fusion_module)
    FusionCore = getattr(fusion_mod, "FusionCore")
    FusionConfig = getattr(fusion_mod, "FusionConfig")

    vision_mod = _import_module(args.vision_module)
    vision_cls = _find_class_with_method(
        vision_mod,
        method_name="get_next_action_event",
        preferred_name=args.vision_class,
    )

    vision_kwargs = {
        # TrajectoryVisionConfig names
        "classifier_path": args.classifier,
        "model_path": args.pose_model,
        "camera_index": args.camera_index,
        "active_hand": args.active_hand,
        "confidence_threshold": args.confidence_threshold,
        # Optional future names
        "debug": args.vision_debug,
    }
    vision_agent = _construct_agent(vision_mod, vision_cls, vision_kwargs)

    radar_agent = None
    if args.enable_radar:
        radar_mod = _import_module(args.radar_module)
        radar_cls = _find_class_with_method(
            radar_mod,
            method_name="query_burst",
            preferred_name=args.radar_class,
        )
        radar_kwargs = {
            "pc_ip": args.radar_pc_ip,
            "data_port": args.radar_data_port,
        }
        radar_agent = _construct_agent(radar_mod, radar_cls, radar_kwargs)

    _start_if_possible(vision_agent)
    _start_if_possible(radar_agent)

    cfg = FusionConfig(
        radar_min_abs_velocity_mps=args.radar_min_abs_velocity,
        require_radar_for_straight=args.require_radar_for_straight,
        verbose=args.fusion_verbose,
    )
    fusion_core = FusionCore(vision_agent=vision_agent, radar_agent=radar_agent, config=cfg)

    return FusionInputSource(fusion_core), [vision_agent, radar_agent, fusion_core]
