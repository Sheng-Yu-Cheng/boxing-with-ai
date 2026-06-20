# RadarBox

**A Camera-Radar Fusion Interactive Boxing System**

RadarBox is a real-time interactive boxing platform that combines human pose estimation from a webcam with Doppler-based motion sensing from a TI AWR2243 mmWave radar.

The system uses:

- **Webcam + MediaPipe** to recognize boxing actions such as straight punches, hooks, uppercuts, and block.
- **TI AWR2243 + DCA1000 mmWave radar** to estimate Doppler velocity and punch intensity, especially for forward straight punches.
- **Camera-guided radar fusion** to search for radar Doppler bursts only during the camera-detected punch impact window.

---

## Quick Start

### 1. Create and activate Python environment

Using an existing `.venv` on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Or use Conda:

```powershell
conda env create -f environment.yml
conda activate radarbox
```

---

### 2. Install this project as an editable package

From the project root:

```powershell
pip install -e .
```

This makes files in `src/` importable from scripts in `scripts/`.

For example, after editable install, this should work:

```powershell
python -c "import punch_vision_common; print('ok')"
```

Expected:

```text
ok
```

---

### 3. Download the MediaPipe PoseLandmarker model

Create the model folder:

```powershell
mkdir models
```

Download the official MediaPipe Pose Landmarker Lite task model:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" `
  -OutFile ".\models\pose_landmarker_lite.task"
```

Expected file:

```text
models/pose_landmarker_lite.task
```

---

### 4. Record a punch dataset

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

For `negative`, record non-punch motions such as:

```text
standing still
guard pose
small body movement
hands moving slowly
adjusting posture
fake punch / incomplete punch
```

The dataset will be saved under:

```text
data/punch_dataset/
```

---

### 5. Train the punch classifier

```powershell
python .\scripts\train_punch_classifier.py `
  --dataset .\data\punch_dataset `
  --out .\models\punch_classifier.joblib `
  --hand right
```

Expected output includes:

```text
classification report
confusion matrix
CV accuracy
models/punch_classifier.joblib
models/punch_classifier.json
```

A reasonable first result is around:

```text
accuracy ≈ 0.90
CV accuracy ≈ 0.90+
```

More samples usually improve stability.

---

### 6. Run the trajectory-based VisionAgent

```powershell
python .\src\vision_agent_trajectory.py `
  --debug `
  --classifier .\models\punch_classifier.joblib `
  --model-path .\models\pose_landmarker_lite.task `
  --active-hand right `
  --confidence-threshold 0.60
```

Expected runtime events:

```text
[TrajectoryVisionAgent] event action=right_straight conf=...
[TrajectoryVisionAgent] event action=right_hook conf=...
[TrajectoryVisionAgent] event action=right_uppercut conf=...
[TrajectoryVisionAgent] event action=block conf=...
[TrajectoryVisionAgent] event action=block_end conf=...
```

If many predictions are shown as `low_conf`, lower the threshold:

```powershell
--confidence-threshold 0.50
```

If the agent outputs too many wrong events, raise the threshold:

```powershell
--confidence-threshold 0.70
```

---

### 7. Run the RadarAgent

Start the Python radar UDP receiver before starting radar frame capture:

```powershell
python .\src\radar_agent.py --plot
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

Healthy radar stream should show approximately:

```text
[RadarAgent] first packet: seq=...
[RadarAgent] status=OK fps=50.0/50.0
```

---

## Project Layout

Recommended layout:

```text
boxing-with-ai/
├── setup.py
├── environment.yml
├── README.md
├── models/
│   ├── pose_landmarker_lite.task
│   └── punch_classifier.joblib
├── data/
│   └── punch_dataset/
├── scripts/
│   ├── record_punch_dataset.py
│   └── train_punch_classifier.py
└── src/
    ├── punch_vision_common.py
    ├── vision_agent_trajectory.py
    └── radar_agent.py
```

### Important files

| File | Purpose |
|------|---------|
| `setup.py` | Makes `src/` importable through `pip install -e .` |
| `src/punch_vision_common.py` | Shared MediaPipe detector, landmark utilities, trajectory feature extraction, dataset save/load |
| `scripts/record_punch_dataset.py` | Records MediaPipe trajectory samples into `.npz` dataset files |
| `scripts/train_punch_classifier.py` | Trains a scikit-learn punch classifier from recorded samples |
| `src/vision_agent_trajectory.py` | Runtime vision agent for punch recognition and block detection |
| `src/radar_agent.py` | Runtime radar UDP receiver and Doppler burst query service |

---

## System Overview

### Webcam Module

The vision subsystem performs:

- MediaPipe human pose estimation
- Punch trajectory segmentation
- Trajectory feature extraction
- Punch classification
- Guard / block detection

Recognized actions:

```text
right_straight
right_hook
right_uppercut
block
block_end
negative / idle
```

The current recommended implementation is trajectory-based. It uses the full motion segment instead of a single-frame wrist velocity peak.

---

### Radar Module

The radar subsystem performs:

- FMCW signal acquisition
- Range FFT
- Doppler FFT
- Range-Doppler map generation
- Doppler burst extraction
- Punch intensity estimation

The radar is primarily used for estimating the intensity of forward straight punches, where Doppler measurements are most reliable.

---

## Core Contribution

### Camera-Guided Doppler Burst Estimation

A key challenge in radar-based motion sensing is that the strongest Doppler peak is not always generated by the punching hand. Body sway, arm motion, and environmental reflections may also contribute to strong Doppler signatures.

RadarBox addresses this issue through a camera-guided fusion strategy:

```text
Webcam / MediaPipe
    ↓
Punch trajectory recognition
    ↓
Impact time window
    ↓
Radar Doppler burst query
    ↓
Punch intensity score
```

The camera determines the punch type and approximate impact window. The radar then searches for Doppler bursts only inside that window.

This reduces false Doppler detections and improves punch intensity estimation.

---

## Motion Phase and Trajectory Recognition

Each punch is treated as a short trajectory segment:

```text
Preparation
    ↓
Extension / Swing
    ↓
Impact
    ↓
Recovery
```

The trajectory classifier uses features such as:

```text
dx / dy / dz
upward displacement
path length
straightness
curvature
maximum speed
mean speed
extension gain
elbow angle change
horizontal dominance
vertical dominance
resampled wrist trajectory
```

This is more reliable than classifying from a single frame.

---

## Punch Intensity Estimation

After radar Range-Doppler processing:

```text
ADC Samples
    ↓
Range FFT
    ↓
Doppler FFT
    ↓
Range-Doppler Map
    ↓
Velocity Burst Extraction
```

The instantaneous Doppler velocity is estimated by:

```text
v_hat(t) = argmax_v Σ P(r, v, t)
```

where `P(r, v, t)` denotes Doppler power at range `r`, velocity `v`, and time `t`.

The punch intensity score is:

```text
I_punch = max |v_hat(t)|
```

inside the impact window given by the vision subsystem.

---

## Integration Plan

The intended runtime architecture is:

```text
VisionAgent
    ↓ PlayerActionEvent
FusionCore
    ↓ query radar burst if needed
RadarAgent
    ↓ RadarBurstEvent
FusionCore
    ↓ FusedPlayerEvent
GameEngine
    ↓
Hit / Miss / Blocked / Critical Hit
```

For example:

```python
action = vision_agent.get_next_action_event()

if action and action.action_type == "right_straight":
    burst = radar_agent.query_burst(
        action.impact_time - 0.10,
        action.impact_time + 0.15,
    )
```

Straight punches can use both camera recognition and radar intensity. Hook and uppercut can initially be camera-only.

---

## Expected VisionAgent Behavior

### Healthy startup

Expected:

```text
[TrajectoryVisionAgent] classifier labels: [...]
[TrajectoryVisionAgent] camera opened
```

The debug window should show:

```text
State: idle / recording
Hand: right
Pred: ...
Block: True / False
Status: OK
```

### Good runtime behavior

For clean right-hand punches:

```text
one punch → one event
```

Expected examples:

```text
event action=right_straight conf=0.83
event action=right_hook conf=0.84
event action=right_uppercut conf=0.88
```

For block:

```text
event action=block hand=both phase=block_start
event action=block_end hand=both phase=block_end
```

For idle or random non-punch movement:

```text
negative/idle pred=negative
```

---

## Tuning

### Confidence threshold

If too many valid punches are ignored:

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

---

### Motion segmentation

If punches are not detected at all:

```powershell
--motion-start-speed 0.80
```

If random small movements start too many segments:

```powershell
--motion-start-speed 1.20
```

---

### Block detection

If block is too hard to trigger:

```powershell
--block-enter-frames 5 --block-max-hand-speed 1.20
```

If block triggers too easily:

```powershell
--block-enter-frames 12 --block-exit-frames 8
```

To disable block during debugging:

```powershell
--disable-block
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'punch_vision_common'`

Run editable install from the project root:

```powershell
pip install -e .
```

Then test:

```powershell
python -c "import punch_vision_common; print('ok')"
```

---

### `PoseLandmarker model not found`

Make sure this file exists:

```text
models/pose_landmarker_lite.task
```

Download it again if needed:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" `
  -OutFile ".\models\pose_landmarker_lite.task"
```

---

### MediaPipe warning about XNNPACK or feedback tensors

These messages are usually harmless:

```text
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Feedback manager requires a model with a single signature inference.
```

They do not normally indicate a runtime failure.

---

### `Failed to send to clearcut`

This MediaPipe telemetry-related warning is also usually harmless:

```text
Failed to send to clearcut
```

It does not prevent pose detection.

---

### Too many `low_conf` predictions

Try:

```powershell
--confidence-threshold 0.50
```

Also record more examples for the confusing classes.

---

### Classifier accuracy is not high enough

Record more samples:

```text
right_straight: 50+
right_hook: 50+
right_uppercut: 50+
negative: 50+
```

Add difficult negative examples:

```text
guard movement
slow hand extension
body sway
hands down
fake punches
partial punches
```

---

### RadarAgent shows `NO_STREAM`

Check:

```text
DCA1000 Ethernet IP settings
Windows firewall
mmWave Studio capture status
DCA1000 data port 4098
PC IP 192.168.33.30
```

Also make sure the Python radar receiver is started before radar frame capture.

---

## Hardware

### Radar

- TI AWR2243 mmWave Radar
- DCA1000 Data Capture Card

### Vision

- Laptop built-in webcam or USB webcam

### Processing

- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- scikit-learn
- joblib
- Matplotlib

---

## Future Work

Potential extensions:

- FusionCore for combining vision events with radar bursts
- GameEngine finite-state machine
- Left-hand punch support
- Two-hand free boxing mode
- Larger personalized punch dataset
- More robust classifier calibration
- Multi-radar velocity reconstruction
- Adaptive difficulty AI
- Reinforcement-learning-based opponent behavior

---

## Project Highlights

- Real-time boxing action recognition
- Personalized trajectory-based punch classification
- Radar-based punch intensity estimation
- Camera-guided Doppler burst extraction
- Multi-sensor decision fusion
- Interactive AI boxing gameplay

RadarBox demonstrates how human-understandable visual information and physically meaningful radar measurements can be combined to create a richer interactive gaming experience.
