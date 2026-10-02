# Autonomous Drone Vision

YOLO-based object detection and obstacle-awareness scaffolding for drone
experiments. The controller in this repository produces visual advisories only;
it is not connected to a flight controller and is not a substitute for
depth-aware, fail-safe navigation.

## Setup

Use Python 3.10 or newer. From the repository root, create and activate a
virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately if you want GPU inference. The
correct PyTorch package depends on your CUDA and driver versions; see the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Run detection

```powershell
py -m src.detector --source 0
py -m src.detector --source path\to\video.mp4 --no-display
```

The first run downloads Ultralytics' `yolo11s.pt` weights if a local
`models\yolo11s.pt` is not present. Model name, thresholds, and output paths can
be adjusted in `config\params.yaml`. Detection results are written under
`runs\detect`; frames containing low-confidence detections are saved under
`dataset\collected\images`.

## Train a model

Put images and matching YOLO-format label files in `dataset\train` and
`dataset\val`. Update `dataset\data.yaml` with the class names and run:

```powershell
py scripts\train_local.py --data dataset\data.yaml
```

The training script uses the model and training defaults in
`config\params.yaml`. Training artifacts are written under `runs\train`.

## GPU check

```powershell
py scripts\test_gpu.py
```

## Dataset label format

Each image needs a same-named `.txt` label file. Each line contains
`class_id x_center y_center width height`, with coordinates normalized to the
range 0–1. Update the `names` list in `dataset\data.yaml` so its order matches
the class IDs in the labels.

## Safety note

The controller only classifies detections by their image position and returns
`CLEAR`, `CAUTION`, or `HOLD` advisories. A 2D image does not provide reliable
distance or collision prediction. Do not use these advisories as direct flight
commands or rely on them as the sole obstacle-avoidance system.
