# RadarBox

**A Camera-Radar Fusion Interactive Boxing System**

RadarBox is a real-time interactive boxing game that combines:

- **Webcam + MediaPipe Pose** for boxing action recognition.
- **TI AWR2243 + DCA1000 mmWave radar** for Doppler velocity and punch intensity estimation.
- **FusionCore** for combining camera events and radar bursts.
- **Panda3D** for a playable 3D alpha game with scene, opponent model, first-person gloves, animations, HP, block, hit reactions, and keyboard / fusion input.

The current project goal is no longer only “sensor demo.” It is now an **interactive boxing alpha game**:

```text
Player webcam/radar input
        ↓
VisionAgent / RadarAgent
        ↓
FusionCore
        ↓
GameCore
        ↓
Panda3D renderer
        ↓
Opponent animation + player gloves + HP UI
```

---

## Current Status

### Working / mostly implemented

- Trajectory-based `VisionAgent`
- Radar UDP receiver / `RadarAgent`
- `FusionCore`
- Trained punch classifier workflow
- Panda3D game prototype
- `opponent.glb` with Mixamo animations
- First-person `boxing_glove.glb` + texture
- `scene.glb`
- Keyboard-controlled alpha game
- Planned FusionCore-to-game integration

### Current important issue

The current `VisionAgent` constructor is config-style:

```python
VisionAgent(config: TrajectoryVisionConfig)
```

Therefore game integration code must **not** instantiate it as:

```python
VisionAgent(model_path=..., classifier_path=...)
```

Instead, it must build:

```python
TrajectoryVisionConfig(
    classifier_path=args.classifier,
    model_path=args.pose_model,
    camera_index=args.camera_index,
    active_hand=args.active_hand,
    confidence_threshold=args.confidence_threshold,
)
```

and then:

```python
vision_agent = VisionAgent(config)
```

A patch script was created for this:

```text
patch_game_system_v2.py
```

Codex should make sure this fix is applied directly inside:

```text
src/game/input_sources.py
```

---

## Current Project Layout

Current intended layout:

```text
boxing-with-ai/
├── assets/
│   ├── scene.glb
│   ├── opponent.glb
│   ├── boxing_glove.glb
│   └── boxing_glove.png
├── data/
│   └── punch_dataset/
├── models/
│   ├── pose_landmarker_lite.task
│   ├── punch_classifier.joblib
│   └── punch_classifier.json
├── scripts/
│   ├── game/
│   │   ├── run_game.py
│   │   ├── alpha_game.py                  # older monolithic prototype
│   │   ├── test_opponent_animation.py
│   │   └── test_player_gloves_v2.py
│   ├── radar/
│   └── vision/
├── src/
│   ├── core/
│   │   ├── punch_vision_common.py
│   │   ├── vision_agent.py
│   │   ├── radar_agent.py
│   │   └── fusion_core.py
│   └── game/
│       ├── __init__.py
│       ├── events.py
│       ├── game_core.py
│       ├── input_sources.py
│       └── panda_renderer.py
├── setup.py
├── environment.yml
└── README.md
```

The current recommended import paths are:

```python
from core.vision_agent import VisionAgent, TrajectoryVisionConfig
from core.radar_agent import RadarAgent
from core.fusion_core import FusionCore, FusionConfig
```

and:

```python
from game.game_core import GameCore
from game.events import GameInputEvent, RenderCommand
```

---

## Installation

### 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

or Conda:

```powershell
conda env create -f environment.yml
conda activate radarbox
```

### 2. Install dependencies

At minimum:

```powershell
pip install -e .
pip install panda3d panda3d-gltf
```

The project also uses:

```text
opencv-python
mediapipe
numpy
scipy
scikit-learn
joblib
matplotlib
```

### 3. Verify imports

From project root:

```powershell
python -c "import core.vision_agent; import core.fusion_core; print('core ok')"
python -c "import game.panda_renderer; print('game ok')"
```

Expected:

```text
core ok
game ok
```

---

## Assets

The alpha game expects:

```text
assets/scene.glb
assets/opponent.glb
assets/boxing_glove.glb
assets/boxing_glove.png
```

### Opponent model

`assets/opponent.glb` is a Mixamo character exported from Blender as GLB.

Current animation clips observed:

```text
Block
Cross Punch
Cross Punch Mirror
Hook Punch
Hook Punch Mirror
Idle
Light Hit To Head
Receive Uppercut To The Face
Stunned
Uppercut
Uppercut Mirror
```

The game maps these to:

```python
idle              -> "Idle"
block             -> "Block"
light_hit         -> "Light Hit To Head"
receive_uppercut  -> "Receive Uppercut To The Face"
stunned           -> "Stunned"
cross             -> "Cross Punch"
hook              -> "Hook Punch"
uppercut          -> "Uppercut"
```

### Player model

The player currently uses first-person gloves:

```text
assets/boxing_glove.glb
assets/boxing_glove.png
```

The glove model is duplicated into:

```text
right_glove
left_glove
```

The left glove is currently mirrored programmatically.

### Scene

The scene is:

```text
assets/scene.glb
```

Current preferred scene / opponent defaults:

```python
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
```

The renderer also adds a simple environment:

```python
SKY_COLOR = (0.45, 0.72, 0.98, 1.0)
GROUND_COLOR = (0.22, 0.60, 0.22, 1.0)
GROUND_Z = -2.0
GROUND_SIZE = 80.0
```

---

## Quick Start: Game

### 1. Keyboard-only game

Run this first to verify assets, renderer, UI, opponent animations, and gloves:

```powershell
python .\scripts\game\run_game.py --input keyboard
```

Controls:

```text
J : right straight
H : right hook
U : right uppercut

F : left straight
G : left hook
T : left uppercut

B : player block
O : opponent block toggle
K : force opponent stunned
R : reset
Esc : quit
```

Expected behavior:

- Scene loads.
- Opponent loads.
- Player gloves appear in first-person view.
- Punch keys move the gloves.
- Opponent HP decreases.
- Opponent plays hit / block / stunned reactions.
- UI shows HP and last action.

### 2. Fusion mode, camera only

After keyboard mode works:

```powershell
python .\scripts\game\run_game.py `
  --input fusion `
  --pose-model .\models\pose_landmarker_lite.task `
  --classifier .\models\punch_classifier.joblib `
  --confidence-threshold 0.60
```

Expected flow:

```text
VisionAgent
    ↓ PlayerActionEvent
FusionCore
    ↓ FusedPlayerEvent
GameInputEvent
    ↓
GameCore
    ↓ RenderCommand
Panda3D renderer
```

### 3. Fusion mode, camera + radar

Start AWR2243 / DCA1000 streaming first, then run:

```powershell
python .\scripts\game\run_game.py `
  --input fusion `
  --enable-radar `
  --pose-model .\models\pose_landmarker_lite.task `
  --classifier .\models\punch_classifier.joblib `
  --confidence-threshold 0.60 `
  --radar-min-abs-velocity 2.0
```

Straight punches use radar if available. Hook and uppercut are currently camera-only.

---

## Game Architecture

The current game system is intentionally **not** a web frontend + backend split.

It is one Python process, but internally separated into:

```text
Input layer
    keyboard / FusionCore

Game backend / logic layer
    GameCore
    CombatState
    damage / block / KO / reactions

Renderer frontend layer
    Panda3D
    scene.glb
    opponent.glb
    boxing gloves
    UI
```

### Data flow

```text
KeyboardInputBuffer / FusionInputSource
        ↓
GameInputEvent
        ↓
GameCore.handle_player_event()
        ↓
RenderCommand
        ↓
Panda3D renderer
```

### Important rule

Keep these boundaries:

```text
FusionCore does not know Panda3D.
GameCore does not know Panda3D.
Panda3D renderer does not decide damage.
RadarAgent does not touch game state.
VisionAgent does not touch game state.
```

---

## Game Modules

### `src/game/events.py`

Defines common game-facing dataclasses:

```python
GameInputEvent
RenderCommand
```

Also converts:

```python
FusedPlayerEvent -> GameInputEvent
```

### `src/game/game_core.py`

Renderer-independent game logic.

Responsibilities:

- player HP
- opponent HP
- block state
- damage calculation
- KO / stunned state
- choosing opponent reaction type

It should output `RenderCommand`, not call Panda3D directly.

### `src/game/input_sources.py`

Input adapters.

Responsibilities:

- collect keyboard input
- construct `VisionAgent`
- optionally construct `RadarAgent`
- construct `FusionCore`
- poll `FusionCore.get_next_fused_event()`
- convert fused events into `GameInputEvent`

Important implementation note:

`VisionAgent` must be constructed using `TrajectoryVisionConfig`.

Correct pattern:

```python
from core.vision_agent import VisionAgent, TrajectoryVisionConfig

cfg = TrajectoryVisionConfig(
    classifier_path=args.classifier,
    model_path=args.pose_model,
    camera_index=args.camera_index,
    active_hand=args.active_hand,
    confidence_threshold=args.confidence_threshold,
)

vision_agent = VisionAgent(cfg)
vision_agent.start()
```

### `src/game/panda_renderer.py`

Panda3D rendering and visual controllers.

Responsibilities:

- load scene
- add blue sky and green ground
- load opponent actor
- map opponent animations
- load first-person gloves
- apply glove texture
- render HP / action UI
- apply `RenderCommand`

### `scripts/game/run_game.py`

Entry point.

Expected command:

```powershell
python .\scripts\game\run_game.py --input keyboard
```

or:

```powershell
python .\scripts\game\run_game.py --input fusion
```

---

## Vision System

### Dataset recording

Record right-hand punch samples:

```powershell
python .\scripts\record_punch_dataset.py --label right_straight --hand right --count 30
python .\scripts\record_punch_dataset.py --label right_hook     --hand right --count 30
python .\scripts\record_punch_dataset.py --label right_uppercut --hand right --count 30
python .\scripts\record_punch_dataset.py --label negative      --hand right --count 30
```

During recording:

```text
SPACE = record one sample
q     = quit
```

For each sample, wait for `GO!`, then throw exactly one punch.

For `negative`, record non-punch motions:

```text
standing still
guard pose
small body movement
hands moving slowly
adjusting posture
fake punch / incomplete punch
```

### Train classifier

```powershell
python .\scripts\train_punch_classifier.py `
  --dataset .\data\punch_dataset `
  --out .\models\punch_classifier.joblib `
  --hand right
```

Expected output:

```text
classification report
confusion matrix
CV accuracy
models/punch_classifier.joblib
models/punch_classifier.json
```

Current observed performance was around:

```text
holdout accuracy ≈ 0.90
CV accuracy ≈ 0.93
```

### Run VisionAgent directly

Current implementation path:

```text
src/core/vision_agent.py
```

Run:

```powershell
python .\src\core\vision_agent.py `
  --debug `
  --classifier .\models\punch_classifier.joblib `
  --model-path .\models\pose_landmarker_lite.task `
  --active-hand right `
  --confidence-threshold 0.60
```

Expected startup:

```text
[TrajectoryVisionAgent] classifier labels: [...]
[TrajectoryVisionAgent] camera opened
```

Expected runtime events:

```text
[TrajectoryVisionAgent] event action=right_straight conf=...
[TrajectoryVisionAgent] event action=right_hook conf=...
[TrajectoryVisionAgent] event action=right_uppercut conf=...
```

---

## Radar System

Start the Python radar UDP receiver before starting radar frame capture:

```powershell
python .\src\core\radar_agent.py --plot
```

Then use mmWave Studio to configure AWR2243 + DCA1000 and start capture.

Current expected radar configuration:

```text
Radar: TI AWR2243 + DCA1000
Mode: 1Tx + 4Rx
ADC samples: 256
Loops per frame: 64
Frame period: 20 ms
UDP data port: 4098
PC IP: 192.168.33.30
DCA1000 IP: 192.168.33.180
```

Healthy radar stream:

```text
[RadarAgent] first packet: seq=...
[RadarAgent] status=OK fps=50.0/50.0
```

Radar is mainly used for:

```text
left_straight
right_straight
```

Hook and uppercut are camera-only in the first stable version.

---

## FusionCore

Current implementation path:

```text
src/core/fusion_core.py
```

Expected runtime flow:

```text
VisionAgent
    ↓ PlayerActionEvent
FusionCore
    ↓ query RadarAgent if straight punch
RadarAgent
    ↓ RadarBurstEvent
FusionCore
    ↓ FusedPlayerEvent
GameEngine
```

For straight punches:

```python
radar_agent.query_burst(
    action.impact_time - 0.10,
    action.impact_time + 0.15,
    range_min_m=0.6,
    range_max_m=2.5,
    min_abs_velocity_mps=2.0,
)
```

For hook / uppercut:

```text
camera-only
```

For block:

```text
pass-through
```

Important config:

```python
FusionConfig(
    radar_min_abs_velocity_mps=2.0,
    require_radar_for_straight=False,
    verbose=True,
)
```

---

## Known Bugs / Notes for Codex

### 1. VisionAgent constructor bug

If this appears:

```text
VisionAgent.__init__() missing 1 required positional argument: 'config'
```

Fix `src/game/input_sources.py`.

Do not instantiate:

```python
VisionAgent(model_path=..., classifier_path=...)
```

Instead:

```python
cfg = TrajectoryVisionConfig(
    classifier_path=args.classifier,
    model_path=args.pose_model,
    camera_index=args.camera_index,
    active_hand=args.active_hand,
    confidence_threshold=args.confidence_threshold,
)
vision_agent = VisionAgent(cfg)
```

### 2. Opponent heading

Current preferred default:

```python
OPPONENT_HEADING = 0.0
```

The earlier `180.0` value was tested but later changed by user preference.

### 3. Scene heading

Current preferred default:

```python
SCENE_HEADING = 0.0
```

Earlier `45.0` was used temporarily, but current user preference is `0.0`.

### 4. Scene position / scale

Current preferred default:

```python
SCENE_RESIZE = 0.11
SCENE_POS_Y = -2.0
```

### 5. Monolithic alpha vs modular game

There may still be older files:

```text
scripts/game/alpha_game.py
scripts/game/test_player_gloves_v2.py
scripts/game/test_opponent_animation.py
```

These are useful test scripts, but the current main entry point should be:

```text
scripts/game/run_game.py
```

### 6. Always test in this order

1. Keyboard game:
   ```powershell
   python .\scripts\game\run_game.py --input keyboard
   ```

2. VisionAgent alone:
   ```powershell
   python .\src\core\vision_agent.py --debug --classifier .\models\punch_classifier.joblib --model-path .\models\pose_landmarker_lite.task --confidence-threshold 0.60
   ```

3. Fusion game, camera only:
   ```powershell
   python .\scripts\game\run_game.py --input fusion --pose-model .\models\pose_landmarker_lite.task --classifier .\models\punch_classifier.joblib --confidence-threshold 0.60
   ```

4. RadarAgent alone:
   ```powershell
   python .\src\core\radar_agent.py --plot
   ```

5. Fusion game, camera + radar:
   ```powershell
   python .\scripts\game\run_game.py --input fusion --enable-radar --pose-model .\models\pose_landmarker_lite.task --classifier .\models\punch_classifier.joblib --confidence-threshold 0.60 --radar-min-abs-velocity 2.0
   ```

---

## Tuning

### Vision confidence threshold

If valid punches are ignored:

```powershell
--confidence-threshold 0.50
```

If wrong events appear too often:

```powershell
--confidence-threshold 0.70
```

Recommended starting value:

```powershell
--confidence-threshold 0.60
```

### Motion segmentation

If punches are not detected:

```powershell
--motion-start-speed 0.80
```

If random small movements start too many segments:

```powershell
--motion-start-speed 1.20
```

### Radar minimum velocity

For straight punch Doppler burst:

```powershell
--radar-min-abs-velocity 2.0
```

Earlier values around `0.5` were too permissive and could select slow body motion.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'core'`

Run:

```powershell
pip install -e .
```

Then verify:

```powershell
python -c "import core.vision_agent; print('ok')"
```

### `ModuleNotFoundError: No module named 'game'`

Run:

```powershell
pip install -e .
```

or make sure `scripts/game/run_game.py` inserts `src/` into `sys.path`.

### `PoseLandmarker model not found`

Make sure this exists:

```text
models/pose_landmarker_lite.task
```

Download:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" `
  -OutFile ".\models\pose_landmarker_lite.task"
```

### `NO_STREAM` from RadarAgent

Check:

```text
DCA1000 Ethernet IP settings
Windows firewall
mmWave Studio capture status
DCA1000 data port 4098
PC IP 192.168.33.30
```

Start Python radar receiver before starting radar frame capture.

### Panda3D cannot load GLB

Install:

```powershell
pip install panda3d-gltf
```

Make sure the file exists:

```powershell
Test-Path .\assets\opponent.glb
Test-Path .\assets\scene.glb
Test-Path .\assets\boxing_glove.glb
```

### Opponent animation names do not match

Run:

```powershell
python .\scripts\game\test_opponent_animation.py --model .\assets\opponent.glb
```

Check printed animation names and update mapping in `OpponentController`.

---

## Recommended Codex Tasks

### Immediate

1. Apply the `VisionAgent(config)` fix in `src/game/input_sources.py`.
2. Verify keyboard game mode.
3. Verify fusion camera-only mode.
4. Keep current defaults:
   ```python
   SCENE_RESIZE = 0.11
   SCENE_POS_Y = -2.0
   SCENE_HEADING = 0.0
   OPPONENT_HEADING = 0.0
   ```
5. Make sure `run_game.py` is the main entry point.

### Next

1. Add optional VisionAgent debug preview window in fusion mode.
2. Add cleaner HP bars instead of plain text.
3. Add hit spark / impact effect.
4. Add opponent AI attack loop.
5. Add player HP damage when opponent attacks and player is not blocking.
6. Replace keyboard glove animation with continuous `VisionAgent` hand pose later.
7. Add replay/logging of fused events for debugging.

---

## Project Highlights

- Real-time boxing action recognition
- Personalized trajectory-based punch classifier
- Camera-guided radar Doppler burst estimation
- Radar intensity scoring for straight punches
- Modular GameCore / Panda3D renderer architecture
- First-person player gloves
- Animated Mixamo opponent
- Fusion-ready interactive boxing alpha game

RadarBox demonstrates how human-understandable visual information and physically meaningful radar measurements can be combined into a playable real-time interactive boxing system.
