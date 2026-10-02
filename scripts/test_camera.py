"""Run live camera detection with the trained phone-detection model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CANDIDATES = [
    ROOT / "runs" / "train" / "drone-vision-2" / "weights" / "best.pt",
    ROOT / "runs" / "train" / "drone-vision" / "weights" / "best.pt",
    ROOT / "models" / "yolo11s.pt",
]


def find_best_model(model_path: str | None) -> str:
    if model_path:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Model file not found: {candidate}")

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "No trained model found. Train a model first or pass --model <path/to/best.pt>."
    )


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL.")
    parser.add_argument("--model", default=None, help="Path to the trained weights file (best.pt).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="CUDA device like 0 or cpu.")
    parser.add_argument("--no-display", action="store_true", help="Do not open the OpenCV window.")
    args = parser.parse_args()

    model = YOLO(find_best_model(args.model))
    source = parse_source(args.source)
    device = "cpu" if args.device.lower() == "cpu" else args.device

    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        stream=True,
        verbose=False,
    )

    window_name = "Phone Detection"
    display = not args.no_display

    try:
        for index, result in enumerate(results):
            annotated = result.plot()
            print(f"Frame {index}: {len(result.boxes)} detections")
            if display:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
