"""Train an Ultralytics YOLO model using the local dataset or Master Dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_MODEL_NAME = "yolo11s.pt"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Override dataset YAML path.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--batch", type=int, help="Override the configured batch size.")
    parser.add_argument("--device", help="Override device, e.g. '0' or 'cpu'.")
    parser.add_argument(
        "--master", 
        action="store_true",
        help="Use Master_Dataset from parent directory instead of default dataset."
    )
    args = parser.parse_args()

    config = load_config(ROOT / "config" / "params.yaml")
    training = config["training"]
    
    # Support Master_Dataset training
    if args.master:
        master_dataset_dir = Path("C:/Autonomous Drone Vision/Master_Dataset")
        data_path = master_dataset_dir / "data.yaml"
        print(f"✓ Using Master_Dataset: {data_path}")
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Master_Dataset not found at {data_path}. "
                "Run 'python scripts/merge_datasets.py' first."
            )
    else:
        data_path = args.data or ROOT / training["data"]
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    dataset_config = load_config(data_path)
    for split in ("train", "val"):
        split_path = Path(dataset_config[split])
        if not split_path.is_absolute():
            split_path = data_path.parent / split_path
        if not split_path.exists():
            raise FileNotFoundError(
                f"Dataset {split} path does not exist: {split_path}. "
                "Add images and YOLO label files before training."
            )

        image_files = [
            item
            for item in split_path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not image_files:
            raise FileNotFoundError(
                f"Dataset {split} folder contains no images: {split_path}. "
                "Add labeled training and validation images before running training."
            )

    epochs = args.epochs if args.epochs is not None else int(training["epochs"])
    batch = args.batch if args.batch is not None else int(training["batch"])
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if batch <= 0:
        raise ValueError("batch must be a positive integer.")

    model_setting = Path(str(config["model"]))
    if not model_setting.is_absolute():
        model_setting = ROOT / model_setting
    if model_setting.is_file():
        model_path = str(model_setting)
    elif model_setting.name == SUPPORTED_MODEL_NAME:
        model_path = SUPPORTED_MODEL_NAME
    else:
        raise ValueError(
            f"Only YOLOv11s is supported in this project. Received '{model_setting.name}'. "
            f"Use '{SUPPORTED_MODEL_NAME}' or keep the default config."
        )
    if model_path != SUPPORTED_MODEL_NAME and Path(model_path).name != SUPPORTED_MODEL_NAME:
        raise ValueError(f"Unsupported model: {model_path}. Only YOLOv11s is allowed.")
    
    model = YOLO(model_path)
    
    # Enhanced training with freeze, learning rate, and exist_ok
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=int(training["image_size"]),
        batch=batch,
        device=args.device
        if args.device is not None
        else (None if training["device"] == "auto" else training["device"]),
        project=str(ROOT / training["project"]),
        name="drone-vision",
        freeze=10,                  # Freeze first 10 layers for transfer learning
        lr0=0.001,                  # Lower learning rate for stable convergence
        exist_ok=True,              # Allow resuming from previous training
    )


if __name__ == "__main__":
    main()
