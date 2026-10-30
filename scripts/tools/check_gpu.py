"""Report CUDA availability and run a small GPU timing check."""

from __future__ import annotations

import sys
import time

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch. CPU execution remains possible.")
        return 1

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    tensor = torch.randn((2048, 2048), device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    _ = tensor @ tensor
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"2048x2048 GPU matrix multiplication: {elapsed:.4f} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
