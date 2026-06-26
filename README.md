1. run mmwavestudio setup lua
2. run mmwave studio start frame script
3. run python script
```bash
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
    --pose-x-range-gain-max 1.6
```

```bash
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
    --pose-x-range-gain-max 1.6
```

TBD:
1. restart game
2. enemy AI
3. 血條，不用數字
3. frame lua, stop frame with command from python with file exchange
4. retrain classifier
5. no-debug mode