"""Runtime helpers shared by train scripts."""
from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Select a PyTorch device.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. Explicit requests fail fast
    if the backend is unavailable so long experiments do not silently run on CPU.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    if requested == "mps":
        ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not ok:
            raise RuntimeError("--device mps requested, but Apple MPS is not available")
    if requested != "cpu" and requested not in {"cuda", "mps"}:
        raise ValueError(f"unknown device={requested!r}")
    return torch.device(requested)
