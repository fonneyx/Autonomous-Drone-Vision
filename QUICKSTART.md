# 🚀 QUICK START GUIDE - Autonomous Drone Obstacle Avoidance

## Executive Summary
Three components working together:
1. **Dataset Merger** → Combine multiple YOLO datasets into one
2. **RL Environment** → Gymnasium-based drone control training
3. **Enhanced Training** → YOLOv11s with transfer learning

---

## 🎯 IMMEDIATE EXECUTION COMMANDS

### Step 1: Install Dependencies
```bash
cd "C:\Autonomous Drone Vision\Autonomous-Drone-Vision"
pip install -r requirements.txt
```

### Step 2: Merge Datasets (One-time Setup)
```bash
python scripts/merge_datasets.py
```
- **Input**: 3 external YOLO datasets from parent directory
- **Output**: `C:\Autonomous Drone Vision\Master_Dataset` with unified classes
- **Duration**: 2-5 minutes
- **Verification**: Check for `Master_Dataset/data.yaml` and unified class mapping

### Step 3: Train on Master Dataset
```bash
# Standard training (100 epochs, batch size 8)
python scripts/train_local.py --master --epochs 100 --batch 8

# Or with GPU ID specification
python scripts/train_local.py --master --epochs 100 --batch 16 --device 0

# Or resume previous training
python scripts/train_local.py --master --epochs 50
```

**Key Features**:
- ✅ `freeze=10` - Preserves pre-trained backbone
- ✅ `lr0=0.001` - Stable learning rate
- ✅ `exist_ok=True` - Resume from checkpoints
- 📊 Results saved to: `runs/train/drone-vision/weights/best.pt`

---

## 📦 COMPONENT DETAILS

### A. merge_datasets.py
**Purpose**: Unify class labels across datasets

**Input Datasets**:
- `C:\Autonomous Drone Vision\Drone.v1i.yolov11`
- `C:\Autonomous Drone Vision\phone.v1i.yolov11`
- `C:\Autonomous Drone Vision\vehicles.v2-release.yolov11`

**Processing**:
1. Parses `data.yaml` from each dataset
2. Builds unified `names` dictionary (eliminates duplicate classes)
3. Re-indexes class IDs in label `.txt` files
4. Prepends dataset name to filenames (prevents collisions)

**Output Structure**:
```
Master_Dataset/
├── data.yaml                    # Unified class mapping
├── train/
│   ├── images/                  # All training images
│   └── labels/                  # Re-indexed label files
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Example Unified Mapping** (if merged successfully):
```yaml
nc: 6
names: {0: 'drone', 1: 'person', 2: 'bird', 3: 'phone', 4: 'car', 5: 'truck'}
```

---

### B. rl_environment.py
**Purpose**: Gymnasium environment for drone RL training

**Key Classes**:
- `DroneAvoidanceEnv` - Main environment
- `DroneDetectionWrapper` - Bridges YOLO output to RL input

**Observation Space** (32 floats):
- 5 obstacles × 6 features: `[x, y, w, h, confidence, class_id]`
- 2 drone state: `[velocity, yaw]`

**Action Space** (2 floats):
- `[yaw_rate, forward_velocity]`
- Range: `yaw_rate ∈ [-45°/s, +45°/s]`, `velocity ∈ [0, 15 m/s]`

**Rewards**:
```
if collision detected:
    reward = -100
else:
    reward = +1 (base step reward)
    if high steering rate:
        reward -= 5
```

**Test the Environment**:
```python
from src.rl_environment import DroneAvoidanceEnv

env = DroneAvoidanceEnv(max_obstacles=5, max_steps=100)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (32,)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.2f}, Terminated: {terminated}")
```

---

### C. train_local.py (ENHANCED)
**New Features**:

| Feature | Before | After |
|---------|--------|-------|
| Default Dataset | `dataset/data.yaml` | Same or `--master` |
| Transfer Learning | ❌ | ✅ `freeze=10` |
| Learning Rate | Auto | ✅ `lr0=0.001` |
| Checkpoint Resume | ❌ | ✅ `exist_ok=True` |
| Master Dataset | ❌ | ✅ `--master` flag |

**Training Parameters Explained**:

| Parameter | Value | Impact |
|-----------|-------|--------|
| `freeze=10` | Layers 0-9 | Preserves pre-trained YOLO backbone |
| `lr0=0.001` | Learning rate | Prevents catastrophic forgetting on multi-dataset |
| `exist_ok=True` | Checkpoint resumption | Allows continuous training epochs |
| `epochs` | 100 (default) | Convergence iterations |
| `batch` | 8 (default) | Samples per gradient update |

---

## 🔗 INTEGRATION WORKFLOW

### Real-Time Drone Inference with RL
```python
from ultralytics import YOLO
from src.rl_environment import DroneAvoidanceEnv, DroneDetectionWrapper
import cv2

# Load trained model
model = YOLO('runs/train/drone-vision/weights/best.pt')

# Initialize environment and wrapper
env = DroneAvoidanceEnv(max_obstacles=5)
wrapper = DroneDetectionWrapper(env)

# Camera loop
cap = cv2.VideoCapture(0)
obs, _ = env.reset()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # YOLO inference
    results = model(frame, verbose=False)
    
    # Update environment with detections
    wrapper.update_from_yolo_detections(results[0])
    
    # Your agent policy (e.g., trained DQN, PPO, etc.)
    action = env.action_space.sample()  # Replace with policy
    
    # Step environment
    obs, reward, terminated, truncated, _ = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 EXPECTED OUTPUTS

### After merge_datasets.py
```
✅ MERGE COMPLETE!
Master Dataset Location: C:\Autonomous Drone Vision\Master_Dataset
Total Classes: 6
Class Mapping: {'drone': 0, 'person': 1, 'bird': 2, 'phone': 3, 'car': 4, 'truck': 5}
Files copied: ~1,530 images + labels
```

### After train_local.py --master
```
Ultralytics YOLOv11s training started...

Epoch 1/100
  Train: loss=0.45, obj_loss=0.12, cls_loss=0.33
  Val: mAP@50=0.32

Epoch 50/100
  Train: loss=0.15, obj_loss=0.04, cls_loss=0.11
  Val: mAP@50=0.65

Epoch 100/100
  Train: loss=0.08, obj_loss=0.02, cls_loss=0.06
  Val: mAP@50=0.78

✅ Training complete!
Best weights saved: runs/train/drone-vision/weights/best.pt
```

---

## ⚠️ TROUBLESHOOTING

### Issue: `FileNotFoundError: Master_Dataset/data.yaml not found`
**Solution**: Run `python scripts/merge_datasets.py` first

### Issue: `CUDA out of memory` during training
**Solution**: Reduce batch size
```bash
python scripts/train_local.py --master --batch 4 --epochs 100
```

### Issue: RL environment import error
**Solution**: Install gymnasium
```bash
pip install gymnasium>=0.29.0
```

### Issue: Dataset split paths don't exist
**Solution**: Verify source datasets have `train/`, `val/`, and `labels/` folders

---

## 🎓 LEARNING PATH

### For Computer Vision:
1. Run `merge_datasets.py` → Understand dataset unification
2. Inspect `Master_Dataset/data.yaml` → See unified class mapping
3. Run training with `--master` → See YOLO training on combined data
4. Visualize training: `tensorboard --logdir runs/train/drone-vision`

### For Reinforcement Learning:
1. Test `rl_environment.py` standalone
2. Study `DroneAvoidanceEnv` state/action/reward structure
3. Integrate with YOLO detections via `DroneDetectionWrapper`
4. Train RL agent with your favorite framework (stable-baselines3, etc.)

---

## 📈 PERFORMANCE TARGETS

| Metric | Target | Hardware |
|--------|--------|----------|
| Inference | 30-40ms/frame | RTX 3060+ |
| mAP@50 | 0.75+ | After 100 epochs |
| Training Time | 2-3 hours | 100 epochs, batch 8 |
| Dataset Size | 1,530 images | After merge |

---

## ✅ VERIFICATION CHECKLIST

Before executing, confirm:
- [ ] All 3 external datasets present in `C:\Autonomous Drone Vision\`
- [ ] `requirements.txt` updated (gymnasium, numpy added)
- [ ] `merge_datasets.py` created in `scripts/`
- [ ] `rl_environment.py` created in `src/`
- [ ] `train_local.py` updated with `--master` flag and training enhancements
- [ ] Python 3.8+ installed
- [ ] CUDA/GPU drivers installed (optional but recommended)

---

## 🚀 RUN NOW!

```bash
# Full pipeline (total time: 15-20 minutes)
cd "C:\Autonomous Drone Vision\Autonomous-Drone-Vision"
pip install -r requirements.txt
python scripts/merge_datasets.py
python scripts/train_local.py --master --epochs 100 --batch 8
```

**Next Steps**: Deploy with your RL agent, track metrics, iterate! 🎯
