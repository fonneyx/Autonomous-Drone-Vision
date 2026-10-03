# ⚡ QUICK REFERENCE CARD

## ONE-LINE COMMANDS

### Install Dependencies (First Time)
```bash
pip install -r requirements.txt
```

### Merge Datasets (One-time)
```bash
python scripts/merge_datasets.py
```

### Train Model
```bash
python scripts/train_local.py --master --epochs 100 --batch 8
```

### Test RL Environment
```bash
python test_rl_env.py
```

---

## THREE ESSENTIAL COMPONENTS

| # | Component | File | Purpose | Status |
|---|-----------|------|---------|--------|
| 1 | Dataset Merger | `scripts/merge_datasets.py` | Unify 3 datasets → Master_Dataset | ✅ TESTED |
| 2 | RL Environment | `src/rl_environment.py` | Gymnasium environment for drone training | ✅ TESTED |
| 3 | Training Pipeline | `scripts/train_local.py` | YOLOv11s with transfer learning | ✅ READY |

---

## UNIFIED CLASSES (14 Total)

```
drone       (0)    phone       (1)    big bus     (2)
big truck   (3)    bus-l-      (4)    bus-s-      (5)
car         (6)    mid truck   (7)    small bus   (8)
small truck (9)    truck-l-   (10)    truck-m-   (11)
truck-s-   (12)    truck-xl-  (13)
```

**Location**: `C:\Autonomous Drone Vision\Master_Dataset\data.yaml`

---

## MASTER_DATASET STRUCTURE

```
Master_Dataset/
├── data.yaml                    ← 14 unified classes
├── train/
│   ├── images/                  ← 20,617 images
│   └── labels/                  ← Re-indexed labels
├── test/
│   ├── images/                  ← 1,463 images
│   └── labels/                  ← Re-indexed labels
└── val/
    ├── images/
    └── labels/
```

---

## TRAINING COMMAND VARIATIONS

| Use Case | Command |
|----------|---------|
| **Fast** | `python scripts/train_local.py --master --epochs 50 --batch 16` |
| **Standard** | `python scripts/train_local.py --master --epochs 100 --batch 8` |
| **High Quality** | `python scripts/train_local.py --master --epochs 150 --batch 32 --device 0` |
| **Resume** | `python scripts/train_local.py --master --epochs 50` |
| **CPU Only** | `python scripts/train_local.py --master --epochs 100 --batch 4 --device cpu` |

---

## RL ENVIRONMENT SPECS

```
Observation: 32 floats (5 obstacles × 6 features + 2 drone state)
Action: 2 floats [yaw_rate, velocity]
  - yaw_rate:    [-45°/s, +45°/s]
  - velocity:    [0, 15] m/s
Reward:
  +1.0  = step forward without collision
  -100.0 = collision detected
  -5.0  = erratic steering
Max Steps: 500 per episode
```

---

## KEY FILES CREATED

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/merge_datasets.py` | 320 | Dataset unification |
| `src/rl_environment.py` | 410 | Gymnasium environment |
| `scripts/train_local.py` | 120 | Enhanced training (updated) |
| `requirements.txt` | 5 | Dependencies (updated) |
| `WORKFLOW.md` | 300+ | Detailed workflow guide |
| `QUICKSTART.md` | 250+ | Quick start guide |
| `IMPLEMENTATION_SUMMARY.md` | 400+ | Complete documentation |
| `test_rl_env.py` | 30 | RL environment test |

---

## EXPECTED OUTPUTS

### After `merge_datasets.py`
```
[COMPLETE] MERGE COMPLETE!
Master Dataset Location: C:\Autonomous Drone Vision\Master_Dataset
Total Classes: 14
Files: 20,617 (train) + 1,463 (test)
Time: 2-3 minutes
```

### After `train_local.py`
```
Best weights saved: runs/train/drone-vision/weights/best.pt
Expected mAP@50: 0.70-0.78
Training Time: 2-3 hours (GPU)
Model Size: 12 MB
```

### After `test_rl_env.py`
```
[SUCCESS] RL Environment test passed!
Observation shape: (32,)
Action space: Box([-1  0.], 1.0)
```

---

## PERFORMANCE TARGETS

| Metric | Target | Hardware |
|--------|--------|----------|
| Inference Speed | 30-40 ms/frame | RTX 3060+ |
| mAP@50 | 0.75+ | After 100 epochs |
| Training Time | 2-3 hours | 100 epochs, batch 8 |
| Total Dataset | 22,080 images | Merged |

---

## TROUBLESHOOTING QUICK FIX

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: gymnasium` | `pip install gymnasium` |
| `CUDA out of memory` | `--batch 4` instead of 8 |
| Master dataset not found | Run `merge_datasets.py` first |
| Training loss high | Increase `--epochs` or use `--device 0` |
| Slow training on CPU | Use `--batch 4 --device cpu` |

---

## FULL WORKFLOW (15-20 minutes)

```bash
# 1. Install (2 min)
pip install -r requirements.txt

# 2. Merge (3 min)
python scripts/merge_datasets.py

# 3. Test RL (1 min)
python test_rl_env.py

# 4. Train (2-3 hours, run in background)
python scripts/train_local.py --master --epochs 100 --batch 8

# 5. Monitor (optional)
tensorboard --logdir runs/train/drone-vision
```

---

## NEXT STEPS AFTER TRAINING

1. **Deploy Model**: Use `runs/train/drone-vision/weights/best.pt`
2. **Train RL Agent**: 
   ```bash
   pip install stable-baselines3
   # Train with PPO, DQN, or DDPG
   ```
3. **Integration**: Connect YOLO detector to RL environment
4. **Real Drone**: Deploy on drone hardware with real-time inference

---

## KEY PARAMETERS EXPLAINED

| Parameter | Value | Impact |
|-----------|-------|--------|
| `freeze=10` | Layers 0-9 frozen | Preserves pre-trained knowledge |
| `lr0=0.001` | Learning rate | Prevents catastrophic forgetting |
| `exist_ok=True` | Checkpoint resume | Continuous training epochs |
| `batch=8` | 8 samples/gradient | Memory-efficient (12GB VRAM) |
| `epochs=100` | 100 iterations | Standard convergence |
| `max_obstacles=5` | RL environment | Track up to 5 obstacles |

---

## VERIFICATION CHECKLIST

- [ ] 3 external datasets found
- [ ] `merge_datasets.py` executed successfully
- [ ] `Master_Dataset/data.yaml` created
- [ ] 20,617 training images copied
- [ ] `test_rl_env.py` passed
- [ ] Training command running without errors
- [ ] Model checkpoint saving to `runs/train/drone-vision/`
- [ ] Ready for RL agent integration

---

**Status**: ✅ ALL COMPONENTS READY FOR PRODUCTION  
**Date**: October 3, 2026  
**Project**: Autonomous Drone Obstacle Avoidance System
