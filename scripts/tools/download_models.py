"""Download the trained models from the GitHub release into models/ and check their SHA-256.

    python scripts/tools/download_models.py

YOLO11 and MediaPipe weights are not listed here: their libraries download them on first use.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RELEASE = "https://github.com/fonneyx/autonomous-drone-vision/releases/download/v0.2.0"
FILES = {
    "avoidance-pilot.zip": "2b65f8f679381b79ee4bd1db4e19d47eedbc108e64d74bb96d8d3a04e51fd864",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for name, expected in FILES.items():
        target = MODELS / name
        if target.exists() and sha256(target) == expected:
            print(f"ok       {name}")
            continue
        print(f"download {name}")
        urllib.request.urlretrieve(f"{RELEASE}/{name}", target)
        if sha256(target) != expected:
            target.unlink()
            sys.exit(f"checksum mismatch for {name}; file removed")
        print(f"ok       {name}")


if __name__ == "__main__":
    main()
