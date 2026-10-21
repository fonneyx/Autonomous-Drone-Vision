# Autonomous Drone Vision

Teach a drone to fly through obstacles, entirely in simulation and on a laptop.

The project has two layers, matching how real autonomous drones work:

| Layer | Question it answers | How |
|---|---|---|
| **Avoidance (main)** | "Is there free space ahead, and how far is the nearest thing?" | A depth camera feeds an RL policy (PPO) that outputs speed and turn commands. Trained in the MuJoCo obstacle course in `src/sim_env.py`. |
| **Recognition (YOLO)** | "What is that object?" (drone, car, person) | YOLOv11s detector in `src/detector.py`. Useful for giving moving objects extra room, or for mission targets. |

Obstacle avoidance is built on depth, not YOLO, because YOLO only sees the
classes it was trained on. A tree, wall, pole or branch that is not in the class
list is invisible to it. Depth sees everything.

## Setup

Python 3.10 or newer. From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` installs a CPU-only PyTorch. That is fine for the drone
simulator, but YOLO training needs the GPU build (NVIDIA driver 580+):

```powershell
python -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu130
python scripts\test_gpu.py
```

If `test_gpu.py` says CUDA is not available, the laptop has probably switched the
NVIDIA GPU off. Plug in the charger, set the GPU mode to Standard/Hybrid in the
vendor app (Armoury Crate, Lenovo Vantage, MSI Center, ...), and restart.

## Part 1: Obstacle avoidance in the simulator

### 1. Check the simulator

```powershell
python test_sim_env.py
```

It checks the environment API, flies an empty course, confirms the depth camera
sees a pole ahead, confirms crashes are detected, and prints the simulation speed.

### 2. Watch an untrained drone (baseline)

```powershell
python scripts\fly_avoidance.py --random
```

A 3D window opens with the drone, plus a small window showing its depth camera
(bright = close). Random flying reaches the finish about 10% of the time.

### 3. Train

```powershell
python scripts\train_avoidance.py
```

- Default 3 million steps, about 1 hour on a 6-core laptop CPU. No GPU needed.
- Runs 8 simulators in parallel. Use `--envs 4` if the laptop gets too hot or slow.
- Ctrl+C stops early and still saves the model.
- Results go to `runs\avoidance\avoidance\`: `best_model.zip` (best evaluation
  score) and `final_model.zip`.
- To keep training a saved model:
  `python scripts\train_avoidance.py --resume runs\avoidance\avoidance\final_model.zip --timesteps 2000000`

Watch the learning curves live in another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir runs\avoidance\tensorboard
```

Open http://localhost:6006. The key curves are `eval/success_rate` (share of
evaluation flights that reach the finish) and `rollout/ep_rew_mean`.

### 4. Watch and score the trained drone

```powershell
python scripts\fly_avoidance.py                              # 3D viewer, newest best model
python scripts\fly_avoidance.py --obstacles 50               # harder course
python scripts\fly_avoidance.py --headless --episodes 100    # success rate, no windows
python scripts\fly_avoidance.py --record flight.mp4          # save a video
```

### How the simulator works

| | |
|---|---|
| Course | 40 m long and 16 m wide, with side walls. A new random layout every episode: 10–40 trees, poles and boxes. |
| Sensor | Forward depth camera, 64×48 px, 90° field of view, 10 m range, 2% noise and 1% dropped pixels (like a real stereo camera). |
| Drone state | Heading, sideways position, distance left, speed, turn rate. |
| Actions | Forward speed 0–4 m/s and yaw rate ±90°/s, at 10 Hz. Altitude is held at 1.5 m. |
| Rewards | +1 per metre of progress, +20 for crossing the finish line, −20 for a crash, a penalty for getting closer than 1.5 m to anything, and a small penalty for jerky turning. |

The drone takes velocity commands, the same interface PX4/ArduPilot flight
controllers accept from a companion computer in offboard/guided mode. A policy
trained here therefore plugs into a real autopilot the same way.

Settings to change are the constants at the top of `src/sim_env.py` (speed,
course size, camera, rewards) and the `DroneAvoidanceEnv(...)` arguments
(obstacle count, noise).

### Simplifications and next steps

The simulator is fast because it keeps things simple. In rough order, the
upgrades that make it more realistic are:

1. **Harder worlds:** more obstacles, thinner poles, moving obstacles, wind
   (random velocity disturbances).
2. **3D flight:** add a climb/descend action and obstacles of different heights.
3. **Real flight stack:** run the same policy against PX4 SITL with Gazebo
   (both free; run them under WSL2 on Windows). There it controls a full
   quadrotor model through MAVLink exactly as it would a real drone.
4. **Real hardware (later, when budget allows):** a small PX4/ArduPilot drone, a
   companion computer (Raspberry Pi 5 or Jetson Orin Nano) and a depth camera
   (Luxonis OAK-D Lite or Intel RealSense).

## Part 2: Object recognition (YOLO)

### Datasets

Three Roboflow datasets sit next to this folder: `Drone.v1i.yolov11`,
`phone.v1i.yolov11` and `vehicles.v2-release.yolov11`. Merge them into
`..\Master_Dataset` with unified class IDs:

```powershell
python scripts\merge_datasets.py
```

The merged dataset has 14 classes: 20,617 train, 2,964 validation and 1,463
test images.

### Train

```powershell
python scripts\train_local.py --master --epochs 50
```

On the RTX 3050 Laptop (4 GB), one epoch over the merged dataset takes about
8–9 minutes, so 50 epochs take about 7 hours. Leave it running overnight, or
rent a cloud RTX 4090 for a couple of hours (roughly $0.25–0.70/hr on Vast.ai or
RunPod). Each run gets its own folder, `runs\train\drone-vision-N`.

To train on any other YOLO dataset: `python scripts\train_local.py --data path\to\data.yaml`.
Training defaults live in `config\params.yaml`.

### Run detection

```powershell
python -m src.detector --source 0                         # webcam
python -m src.detector --source path\to\video.mp4 --no-display
python scripts\test_camera.py                             # newest trained best.pt on the webcam
```

Frames with uncertain detections are saved under `dataset\collected\images`.
Label them and add them to the training set to improve the model.

### Dataset label format

Each image needs a same-named `.txt` label file. Each line contains
`class_id x_center y_center width height`, with coordinates normalized to
0–1. The order of `names` in `data.yaml` must match the class IDs.

## Safety note

`src/controller.py` turns YOLO detections into `CLEAR` / `CAUTION` / `HOLD`
advisories based only on image position. A 2D detection box does not give
reliable distance. Never use these advisories, or a simulator-only policy, as
the sole obstacle avoidance on a real aircraft.
