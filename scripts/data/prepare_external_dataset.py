"""Link or copy an external YOLO-style dataset into the project dataset folder.

This avoids filling the system SSD by default: the script uses filesystem links
when possible. If a link cannot be created, it falls back to copying the files.

Example:
    python scripts/data/prepare_external_dataset.py --source D:/datasets/my_dataset --mode link
    python scripts/data/prepare_external_dataset.py --source D:/datasets/my_dataset --mode copy --split train,val
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_split_name(value: str) -> str:
    value = value.strip().lower()
    if value in {"train", "val", "test"}:
        return value
    raise ValueError(f"Unsupported split '{value}'. Supported splits: train, val, test.")


def choose_source_dirs(source_root: Path, split: str) -> tuple[Path, Path]:
    candidates = [
        source_root / split / "images",
        source_root / "images",
        source_root / split,
    ]
    for images_dir in candidates:
        if images_dir.exists():
            labels_dir = images_dir.parent / "labels"
            if labels_dir.exists():
                return images_dir, labels_dir
            return images_dir, source_root / split / "labels"
    raise FileNotFoundError(
        f"Could not find a YOLO image folder for split '{split}' under {source_root}. "
        "Expected a structure like '<root>/train/images' and '<root>/train/labels'."
    )


def link_or_copy_file(src: Path, dst: Path, mode: str, dry_run: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] {mode}: {src} -> {dst}")
        return
    if mode == "link":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            return
        except (OSError, NotImplementedError):
            print(f"Symlink failed for {src}; falling back to copy.")
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            return
        dst.unlink()
    shutil.copy2(src, dst)


def prepare_split(source_root: Path, target_root: Path, split: str, mode: str, dry_run: bool = False) -> int:
    source_images_dir, source_labels_dir = choose_source_dirs(source_root, split)
    target_images_dir = target_root / split / "images"
    target_labels_dir = target_root / split / "labels"
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        item for item in source_images_dir.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No supported image files found in {source_images_dir}.")

    count = 0
    for image_file in image_files:
        dst_image = target_images_dir / image_file.name
        if dry_run:
            print(f"[dry-run] prepare {split}: {image_file} -> {dst_image}")
        else:
            if not dst_image.exists():
                link_or_copy_file(image_file, dst_image, mode, dry_run=False)
        count += 1

        label_name = image_file.with_suffix(".txt").name
        src_label = source_labels_dir / label_name
        dst_label = target_labels_dir / label_name
        if src_label.exists():
            if dry_run:
                print(f"[dry-run] prepare {split}: {src_label} -> {dst_label}")
            else:
                if not dst_label.exists():
                    link_or_copy_file(src_label, dst_label, mode, dry_run=False)

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="External dataset root folder.")
    parser.add_argument("--target", type=Path, default=Path("dataset"), help="Project dataset root to populate.")
    parser.add_argument("--split", default="train,val", help="Comma-separated splits to prepare: train,val,test")
    parser.add_argument("--mode", choices=("link", "copy"), default="link", help="Use filesystem links when possible; otherwise copy.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without creating files.")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"External dataset root not found: {args.source}")

    project_root = Path(__file__).resolve().parents[2]
    target_root = (project_root / args.target).resolve() if not args.target.is_absolute() else args.target

    for raw_split in args.split.split(","):
        split = normalize_split_name(raw_split)
        try:
            count = prepare_split(args.source, target_root, split, args.mode, dry_run=args.dry_run)
            print(f"Prepared {split}: {count} image(s) -> {target_root / split}")
        except FileNotFoundError as exc:
            print(f"Skipping {split}: {exc}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
