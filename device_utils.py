"""Device utilities (CUDA/MPS/CPU)."""

from __future__ import annotations
import torch


def print_cuda_info() -> None:
    """Print basic CUDA availability information."""
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {cuda_id}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on availability."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
