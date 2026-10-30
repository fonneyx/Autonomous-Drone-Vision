"""Render an automatically labelled YOLO dataset (person / vehicle / drone) from the simulator.

Every image is a random scene seen from the drone's camera: random objects,
colours, poses, obstacles, lighting and camera height. Labels come from the
renderer's segmentation, so they are exact and need no hand drawing.

Some real photos from the Roboflow datasets next to this repository are mixed
in (drones -> 'drone', all vehicle types -> 'vehicle') so the detector also
works on real camera images.

Example:
    python scripts/make_sim_dataset.py                       # 3000 train / 400 val sim images
    python scripts/make_sim_dataset.py --train 6000 --real 0  # more sim, no real photos
Then train:
    python scripts/train_local.py --data ../SimVision_Dataset/data.yaml --epochs 30 --name sim-vision
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.mission_env import CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, DroneMissionEnv  # noqa: E402

REAL_SOURCES = {
    # dataset folder next to the repo -> class every label in it becomes
    "Drone.v1i.yolov11": "drone",
    "vehicles.v2-release.yolov11": "vehicle",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def render_split(env: DroneMissionEnv, renderer: mujoco.Renderer, out: Path, split: str, count: int,
                 background_share: float) -> int:
    images, labels = out / split / "images", out / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    rng = env.np_random
    written = boxes = 0
    while written < count:
        env.randomize_for_dataset()
        renderer.update_scene(env.data, camera="fpv")
        rgb = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera="fpv")
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()
        found = env.boxes_from_segmentation(segmentation)
        if not found and rng.random() > background_share:
            continue  # keep only some empty images (they teach "nothing here")

        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if rng.random() < 0.3:
            # Camera sensor noise
            image = np.clip(image + rng.normal(0, rng.uniform(2, 8), image.shape), 0, 255).astype(np.uint8)
        if rng.random() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        name = f"sim_{split}_{written:05d}"
        cv2.imwrite(str(images / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (labels / f"{name}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in found), encoding="utf-8"
        )
        written += 1
        boxes += len(found)
        if written % 500 == 0:
            print(f"  {split}: {written}/{count} images")
    return boxes


def to_box(values: list[float]) -> tuple[float, float, float, float]:
    """YOLO box from either a box (4 values) or a polygon (x y x y ...)."""
    if len(values) == 4:
        return tuple(values)
    xs, ys = values[0::2], values[1::2]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys))


def copy_real(source_root: Path, class_name: str, out: Path, split: str, count: int, rng: random.Random) -> int:
    source_split = {"train": ["train"], "val": ["valid", "val"]}[split]
    folder = next((source_root / s for s in source_split if (source_root / s / "images").is_dir()), None)
    if folder is None or count <= 0:
        return 0
    images = sorted(p for p in (folder / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    rng.shuffle(images)
    class_id = CLASSES.index(class_name)
    copied = 0
    for image in images:
        label = folder / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            continue
        lines = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                x, y, w, h = to_box([float(v) for v in parts[1:]])
                lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        if not lines:
            continue
        name = f"real_{class_name}_{image.stem}"[:150]
        shutil.copy2(image, out / split / "images" / f"{name}{image.suffix.lower()}")
        (out / split / "labels" / f"{name}.txt").write_text("".join(lines), encoding="utf-8")
        copied += 1
        if copied >= count:
            break
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "SimVision_Dataset")
    parser.add_argument("--train", type=int, default=3000, help="Simulated training images.")
    parser.add_argument("--val", type=int, default=400, help="Simulated validation images.")
    parser.add_argument("--real", type=int, default=750,
                        help="Real photos per class for training (validation gets a seventh of that).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    env = DroneMissionEnv(world_seed=args.seed)
    env.reset(seed=args.seed)
    renderer = mujoco.Renderer(env.model, IMAGE_HEIGHT, IMAGE_WIDTH)

    rng = random.Random(args.seed)
    for split, count in (("train", args.train), ("val", args.val)):
        boxes = render_split(env, renderer, args.out, split, count, background_share=0.1)
        print(f"{split}: {count} simulated images, {boxes} labelled objects")
        real_count = args.real if split == "train" else args.real // 7
        for folder, class_name in REAL_SOURCES.items():
            copied = copy_real(ROOT.parent / folder, class_name, args.out, split, real_count, rng)
            if copied:
                print(f"{split}: + {copied} real '{class_name}' photos from {folder}")
    renderer.close()

    data_yaml = {
        "path": str(args.out),
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(CLASSES)),
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"Dataset ready: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
