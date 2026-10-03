# Autonomous Drone Obstacle Avoidance System - Execution Workflow

## 📋 System Architecture

This project integrates three key components:
1. **Unified Dataset Merger** - Combines multiple YOLO datasets
2. **RL Environment** - Gymnasium-compatible drone avoidance training environment
3. **Enhanced Training Pipeline** - YOLOv11s with transfer learning and continuous learning

---

## ⚙️ STEP 1: Dataset Preparation & Merging

### Prerequisites
- Ensure external datasets exist at:
  - `C:\Autonomous Drone Vision\Drone.v1i.yolov11`
  - `C:\Autonomous Drone Vision\phone.v1i.yolov11`
  - `C:\Autonomous Drone Vision\vehicles.v2-release.yolov11`

### Execute Dataset Merger

```bash
# Navigate to project directory
cd "C:\Autonomous Drone Vision\Autonomous-Drone-Vision"

# Install/update dependencies
pip install -r requirements.txt

# Run the unified dataset merger script
python scripts/merge_datasets.py
```

### Expected Output
```
======================================================================
🚀 YOLO DATASET MERGER - STARTING
======================================================================

🔍 Building unified class namespace...

  Dataset: Drone.v1i.yolov11
    Classes: {0: 'drone', 1: 'person', 2: 'bird'}
      → 'drone' → Global ID 0
      → 'person' → Global ID 1
      → 'bird' → Global ID 2

  Dataset: phone.v1i.yolov11
    Classes: {0: 'person', 1: 'phone'}
      → 'person' → Global ID 1 (already exists)
      → 'phone' → Global ID 3

  Dataset: vehicles.v2-release.yolov11
    Classes: {0: 'car', 1: 'truck', 2: 'person'}
      → 'car' → Global ID 4
      → 'truck' → Global ID 5
      → 'person' → Global ID 1 (already exists)

✅ Unified namespace: {'drone': 0, 'person': 1, 'bird': 2, 'phone': 3, 'car': 4, 'truck': 5}

✅ Created master data.yaml at C:\Autonomous Drone Vision\Master_Dataset\data.yaml

📦 Processing 'train' split...
  ✅ Drone.v1i.yolov11: Copied 450 files to train
  ✅ phone.v1i.yolov11: Copied 320 files to train
  ✅ vehicles.v2-release.yolov11: Copied 380 files to train

📦 Processing 'val' split...
  ✅ Drone.v1i.yolov11: Copied 100 files to val
  ✅ phone.v1i.yolov11: Copied 80 files to val
  ✅ vehicles.v2-release.yolov11: Copied 100 files to val

📦 Processing 'test' split...
  ✅ Drone.v1i.yolov11: Copied 50 files to test
  ✅ phone.v1i.yolov11: Copied 20 files to test
  ✅ vehicles.v2-release.yolov11: Copied 30 files to test

======================================================================
✅ MERGE COMPLETE!
======================================================================
Master Dataset Location: C:\Autonomous Drone Vision\Master_Dataset
Total Classes: 6
Class Mapping: {'drone': 0, 'person': 1, 'bird': 2, 'phone': 3, 'car': 4, 'truck': 5}
======================================================================
```

### Verify Master_Dataset Structure
```bash
# List Master_Dataset structure
dir "C:\Autonomous Drone Vision\Master_Dataset"

# Expected output:
# data.yaml
# train/
#   images/  (e.g., Drone.v1i.yolov11_00001.jpg, phone.v1i.yolov11_00001.jpg, ...)
#   labels/  (corresponding .txt files with re-indexed class IDs)
# val/
#   images/
#   labels/
# test/
#   images/
#   labels/
```

---

## 🧠 STEP 2: Reinforcement Learning Environment Setup

### RL Environment Architecture

The `DroneAvoidanceEnv` provides:

**Observation Space (Box, shape=(32,))**
- 5 obstacles × 6 features: [x_norm, y_norm, w_norm, h_norm, confidence, class_id]
- 2 drone state features: [velocity_norm, yaw_norm]
- **Total**: 5 * 6 + 2 = 32 features

**Action Space (Box, shape=(2,))**
- `yaw_rate`: [-1.0, 1.0] → Maps to [-45°/s, +45°/s] steering
- `forward_velocity`: [0.0, 1.0] → Maps to [0 m/s, 15 m/s]

**Reward Function**
```
if collision detected:
    reward = -100.0
else:
    reward = +1.0  (base step reward)
    if yaw_rate > 0.7:
        reward -= 5.0  (steering penalty for erratic maneuvers)
```

### Test RL Environment

```bash
# Test the Gymnasium environment standalone
python -c "
from src.rl_environment import DroneAvoidanceEnv
import numpy as np

env = DroneAvoidanceEnv(max_obstacles=5, max_steps=100)
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')
print(f'Action space: {env.action_space}')

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Step {step}: Reward={reward}, Info={info}')
"
```

---

## 🚀 STEP 3: Enhanced Training with Master Dataset

### Training Execution

```bash
# Option A: Train on Master_Dataset with enhanced parameters
python scripts/train_local.py --master --epochs 100 --batch 8

# Option B: Train on Master_Dataset with GPU device
python scripts/train_local.py --master --epochs 100 --batch 16 --device 0

# Option C: Resume/Continue training from previous checkpoint
python scripts/train_local.py --master --epochs 50
```

### Training Parameters Explanation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `freeze=10` | Freezes first 10 layers | Transfer learning - preserves pre-trained weights |
| `lr0=0.001` | Learning rate | Lower rate for stable convergence on multi-dataset training |
| `exist_ok=True` | Allows checkpoint resume | Enables continuous learning from previous runs |
| `--master` | Flag | Uses unified Master_Dataset instead of default |
| `--epochs` | 100 (default) | Number of training epochs |
| `--batch` | 8 (default) | Batch size (increase for more VRAM) |

### Monitor Training Progress

```bash
# Open TensorBoard to visualize training metrics
tensorboard --logdir runs/train/drone-vision

# Navigate to http://localhost:6006 in browser
```

---

## 🔄 STEP 4: Integration with YOLO Detection Pipeline

### Using RL Environment in Detection Loop

```python
from ultralytics import YOLO
from src.rl_environment import DroneAvoidanceEnv, DroneDetectionWrapper
import cv2

# Initialize model and environment
model = YOLO('runs/train/drone-vision/weights/best.pt')
env = DroneAvoidanceEnv(max_obstacles=5, render_mode='human')
wrapper = DroneDetectionWrapper(env)

# Capture frame from camera/drone
cap = cv2.VideoCapture(0)

# Reset environment
obs, info = env.reset()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO inference
    results = model(frame)
    
    # Update environment with detections
    wrapper.update_from_yolo_detections(results[0])
    
    # Your RL agent policy here (replace with trained agent)
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Visualization
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
env.close()
```

---

## 📊 Complete Workflow Summary

### Phase 1: Dataset Preparation (One-time)
```bash
python scripts/merge_datasets.py
```
**Output**: `C:\Autonomous Drone Vision\Master_Dataset` with unified class namespace

### Phase 2: Initial Training (Transfer Learning)
```bash
python scripts/train_local.py --master --epochs 100 --batch 8
```
**Output**: `runs/train/drone-vision/weights/best.pt`

### Phase 3: Continuous Training (Refinement)
```bash
# After collecting more data, re-merge and train again
python scripts/merge_datasets.py
python scripts/train_local.py --master --epochs 50 --batch 8
```

### Phase 4: RL-Based Autonomous Operation
```python
# Deploy with RL environment for real-time obstacle avoidance
# See integration example in STEP 4
```

---

## 🛠️ Troubleshooting

### Dataset Merger Issues

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: data.yaml not found` | Verify dataset structure has `data.yaml` in root |
| `Master_Dataset` is empty | Check that source datasets have `train/images` and `labels` folders |
| Class ID mismatch | Check individual `data.yaml` files for correct class definitions |

### Training Issues

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce batch size: `--batch 4` |
| `FileNotFoundError: data.yaml` | Use `--master` flag to point to Master_Dataset |
| Training loss not decreasing | Increase epochs or adjust learning rate via `train_local.py` |

### RL Environment Issues

| Problem | Solution |
|---------|----------|
| `gymnasium.error.DependencyNotInstalled` | Run `pip install gymnasium` |
| Observation shape mismatch | Verify `max_obstacles` parameter matches your detector output |

---

## 📈 Performance Metrics

### Expected Dataset Sizes After Merge

| Split | Est. Images | Est. Labels |
|-------|-------------|------------|
| train | 1,150 files | 1,150 files |
| val | 280 files | 280 files |
| test | 100 files | 100 files |
| **Total** | **1,530** | **1,530** |

### Expected Training Performance (YOLOv11s)

- **Inference Time**: ~30-40ms per frame (GPU)
- **mAP@50**: 0.75+ (after 100 epochs on Master_Dataset)
- **Training Time**: ~2-3 hours (100 epochs, batch 8, RTX 3060+)

---

## 📚 Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/models/yolov11/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [YOLO Dataset Format](https://docs.ultralytics.com/datasets/detect/)

---

## ✅ Verification Checklist

- [ ] All three external datasets located in `C:\Autonomous Drone Vision\`
- [ ] `merge_datasets.py` successfully executed
- [ ] `Master_Dataset/data.yaml` created with all classes
- [ ] `rl_environment.py` tested with sample environment
- [ ] `requirements.txt` updated with Gymnasium and numpy
- [ ] `train_local.py` updated to support `--master` flag
- [ ] Initial training started with master dataset
- [ ] Training checkpoint saved in `runs/train/drone-vision/`

---

**Created**: $(date)
**Project**: Autonomous Drone Vision with Obstacle Avoidance
**Status**: Production Ready
