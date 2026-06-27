# RadarBox Boxing Game

RadarBox is a first-person boxing demo that combines:

- webcam / MediaPipe pose tracking for player glove control and punch classification
- TI mmWave radar range-Doppler tracking for punch validation and AoA beam feedback
- Panda3D rendering for the game UI, opponent animation, HP bars, leaderboard, and enemy AI

## 1. 硬體配置

目前專案假設使用以下硬體：

- Radar board: AWR2243BOOST
- Capture board: DCA1000EVM
- Camera: USB webcam
- Host PC: Windows PC running mmWave Studio + Python game

Radar network defaults used by the Python side:

```text
PC IP:  192.168.33.30
DCA IP: 192.168.33.180
UDP data port: 4098
```

Radar runtime files:

```text
C:\temp\radar_back.bin
C:\temp\radarbox_beam_cmd.txt
C:\temp\radarbox_radar_state.txt
C:\temp\radarbox_stop.txt
```

TODO: fill in exact hardware wiring / power / Ethernet setup.

```text
TODO:
- AWR2243BOOST power setup:
- DCA1000 switch / SOP setting:
- Ethernet adapter static IP setup:
- Camera mounting position:
- Radar-to-player recommended distance:
```

## 2. 韌體 / mmWave Studio 執行

This project uses AWR2243 3Tx simultaneous beamforming + 4Rx.

Important constraints:

- Do not convert the radar setup to TDM-MIMO.
- Runtime ADC shape is `loops x 1 x samples x 4Rx`.
- Beam control is done by writing phase codes to:

```text
C:\temp\radarbox_beam_cmd.txt
```

The beam command format is:

```text
tx0Code,tx1Code,tx2Code
```

Example:

```text
0,16,32
```

Recommended mmWave Studio flow:

1. Open mmWave Studio.
2. Run the radar setup Lua script:

```text
scripts/radar/AWR2243BOOST-DCA1000EVM-beam-tracking-setup.lua
```

3. Run the runtime beam-tracking Lua script:

```text
scripts/radar/RUN-beam-tracking.lua
```

The runtime script watches:

```text
C:\temp\radarbox_radar_state.txt
```

Behavior:

- `active` starts radar frame + beam polling.
- `inactive` stops radar frame and waits.
- Creating `C:\temp\radarbox_stop.txt` exits the Lua loop.

The Python game writes `active` on startup and `inactive` on exit.

TODO: fill in exact mmWave Studio version / device connection steps.

```text
TODO:
- mmWave Studio version:
- Required firmware / BSS / MSS files:
- Exact connection sequence:
- Lua script run order screenshots / notes:
```

## 3. 軟體環境建置

Recommended environment: Windows + PowerShell + Python 3.10+.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project and runtime dependencies:

```powershell
pip install -e .
pip install panda3d panda3d-gltf
```

Download / place the MediaPipe pose model here:

```text
models/pose_landmarker_lite.task
```

If needed:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" `
  -OutFile ".\models\pose_landmarker_lite.task"
```

Expected model / classifier files:

```text
models/pose_landmarker_lite.task
models/punch_classifier.joblib
models/punch_classifier.json
```

## 4. 資料錄製

Punch dataset is stored under:

```text
data/punch_dataset
```

Record right-hand punch classes:

```powershell
python .\scripts\vision\record_punch_dataset.py --label right_straight --hand right --count 30
python .\scripts\vision\record_punch_dataset.py --label right_hook --hand right --count 30
python .\scripts\vision\record_punch_dataset.py --label right_uppercut --hand right --count 30
python .\scripts\vision\record_punch_dataset.py --label negative --hand right --count 30
```

Recording notes:

- Press `SPACE` to record one sample.
- Throw exactly one punch during the GO window.
- Press `q` to quit.
- Default recording uses mirrored input, matching the game command `--mirror-input`.

Recommended negative samples:

- guard / hands in front
- blocking posture
- small hand movement without punching
- body sway / stepping
- returning to guard
- fake motion that should not count as a punch

Train the classifier:

```powershell
python .\scripts\vision\train_punch_classifier.py `
  --dataset .\data\punch_dataset `
  --out .\models\punch_classifier.joblib `
  --hand right
```

Useful quick check:

```powershell
python .\src\core\vision_agent.py `
  --debug `
  --mirror-input `
  --classifier .\models\punch_classifier.joblib `
  --model-path .\models\pose_landmarker_lite.task `
  --confidence-threshold 0.40
```

## 5. 程式執行

### Keyboard-only smoke test

Use this when testing rendering, HP bars, opponent animations, enemy AI, and leaderboard without sensors:

```powershell
python .\scripts\game\run_game.py --input keyboard --enemy-ai
```

Debug keyboard mode:

```powershell
python .\scripts\game\run_game.py --input keyboard --debug --enemy-ai
```

### Fusion game demo

Run this after mmWave Studio setup and `RUN-beam-tracking.lua` are running:

```powershell
python .\scripts\game\run_game.py `
  --input fusion `
  --mirror-input `
  --enable-radar `
  --enable-aoa-feedback `
  --pose-model .\models\pose_landmarker_lite.task `
  --classifier .\models\punch_classifier.joblib `
  --confidence-threshold 0.40 `
  --beam-cmd-file C:\temp\radarbox_beam_cmd.txt `
  --pose-scale-x 0.75 `
  --pose-hand-spacing 0.10 `
  --pose-use-radar-range-x `
  --pose-x-reference-range-m 2.0 `
  --pose-x-range-gain-min 0.65 `
  --pose-x-range-gain-max 1.6 `
  --enemy-ai `
  --enemy-ai-attack-min-s 0.8 `
  --enemy-ai-attack-max-s 1.6 `
  --enemy-ai-telegraph-s 0.45 `
  --enemy-ai-recover-s 0.85 `
  --enemy-ai-guard-chance 0.30 `
  --enemy-ai-damage-scale 1.0 `
  --enemy-ai-guard-threshold 0.60
```

### Fusion game with debug windows

This opens the game UI plus extra vision / RV-map debug windows:

```powershell
python .\scripts\game\run_game.py `
  --input fusion `
  --debug `
  --mirror-input `
  --enable-radar `
  --enable-aoa-feedback `
  --pose-model .\models\pose_landmarker_lite.task `
  --classifier .\models\punch_classifier.joblib `
  --confidence-threshold 0.40 `
  --beam-cmd-file C:\temp\radarbox_beam_cmd.txt `
  --pose-scale-x 0.75 `
  --pose-hand-spacing 0.10 `
  --pose-use-radar-range-x `
  --pose-x-reference-range-m 2.0 `
  --pose-x-range-gain-min 0.65 `
  --pose-x-range-gain-max 1.6 `
  --enemy-ai
```

Game behavior:

- Initial screen waits for `Right Straight Punch to Start Game`.
- Win condition: opponent HP reaches 0.
- Lose condition: player HP reaches 0.
- Score is the number of seconds used to defeat the opponent.
- Top 5 fastest KO scores are stored in:

```text
data/game_leaderboard.json
```

Common tuning:

```text
--confidence-threshold 0.40~0.45
--pose-scale-x
--pose-hand-spacing
--pose-use-radar-range-x
--enemy-ai-damage-scale
--enemy-ai-guard-threshold
```
