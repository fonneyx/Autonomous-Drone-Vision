# Models

Weights are not stored in git. Fetch the released ones with:

```bash
python scripts/tools/download_models.py
```

| File | What it is | Source |
|---|---|---|
| `avoidance-pilot.zip` | PPO obstacle-avoidance pilot (Stable-Baselines3), 99 % success at 40 obstacles in the MuJoCo course | [Model files release](https://github.com/fonneyx/autonomous-drone-vision/releases/tag/models), SHA-256 checked |
| `yolo11s-pose.pt`, `yolo11s.pt` | Person / object detection | downloaded by Ultralytics on first use |
| `gesture_recognizer.task` | Hand gestures (optional) | downloaded by MediaPipe on first use |
| Vosk small English model | Offline voice commands | downloaded by Vosk on first use (user cache, not models/) |
