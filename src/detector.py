"""Run YOLO detection on a camera, video, or image source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.controller import Detection, assess_detections
    from src.data_collector import LowConfidenceFrameCollector
else:
    from .controller import Detection, assess_detections
    from .data_collector import LowConfidenceFrameCollector

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "params.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def resolve_model(model_setting: str) -> str:
    configured_path = Path(model_setting)
    if not configured_path.is_absolute():
        configured_path = ROOT / configured_path
    if configured_path.is_file():
        return str(configured_path)

    # This project is locked to YOLOv11s only.
    if configured_path.name == "yolo11s.pt":
        return "yolo11s.pt"
    raise ValueError(
        f"Unsupported model: {configured_path.name}. This project only supports YOLOv11s "
        "('yolo11s.pt')."
    )


def parse_source(source: str) -> int | str:
    return int(source) if source.isdecimal() else source


def _detections_from_result(result: Any) -> list[Detection]:
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    return [
        Detection(
            x1=coordinates[0],
            y1=coordinates[1],
            x2=coordinates[2],
            y2=coordinates[3],
            confidence=confidence,
            class_name=str(names[int(class_id)]),
        )
        for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids)
    ]


def run(source: int | str, config_path: Path, show: bool | None = None) -> None:
    config = load_config(config_path)
    detection_config = config["detection"]
    output_config = config["output"]
    collector_config = config["data_collector"]
    controller_config = config["controller"]
    reporting_threshold = float(detection_config["confidence_threshold"])

    model = YOLO(resolve_model(str(config["model"])))
    collector = (
        LowConfidenceFrameCollector(
            ROOT / collector_config["directory"],
            float(collector_config["minimum_confidence"]),
        )
        if collector_config["enabled"]
        else None
    )
    display = bool(output_config["display"]) if show is None else show
    output_dir = ROOT / output_config["directory"]
    save_results = bool(output_config["save"])
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=source,
        stream=True,
        conf=float(detection_config["inference_threshold"]),
        iou=float(detection_config["iou_threshold"]),
        imgsz=int(detection_config["image_size"]),
        device=None if detection_config["device"] == "auto" else detection_config["device"],
        verbose=False,
    )

    try:
        for index, result in enumerate(results):
            frame = result.orig_img
            detections = _detections_from_result(result)
            guidance = assess_detections(
                detections,
                frame.shape[1],
                frame.shape[0],
                confidence_threshold=float(controller_config["confidence_threshold"]),
                center_zone_ratio=float(controller_config["center_zone_ratio"]),
            )

            if collector is not None:
                collected = collector.save_if_low_confidence(
                    frame,
                    (detection.confidence for detection in detections),
                    reporting_threshold,
                )
                if collected is not None:
                    print(f"Collected low-confidence frame: {collected}")

            reported = [
                detection
                for detection in detections
                if detection.confidence >= reporting_threshold
            ]
            print(f"Frame {index}: {len(reported)} detections; advisory={guidance.value}")

            annotated = result.plot()
            if save_results:
                output_path = output_dir / f"frame_{index:06d}.jpg"
                if not cv2.imwrite(str(output_path), annotated):
                    raise OSError(f"OpenCV could not write detection output: {output_path}")
            if display:
                cv2.imshow("Autonomous Drone Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, image/video path, or stream URL (default: 0).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an OpenCV display window.",
    )
    args = parser.parse_args()
    run(parse_source(args.source), args.config, show=False if args.no_display else None)


if __name__ == "__main__":
    main()
