"""Measurement utilities: FLOPs, VRAM, parameter count."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Measure FLOPs via fvcore.nn.FlopCountAnalysis.

    Single forward pass, batch size 1, actual input shape.
    Returns total FLOPs as an integer.
    """
    from fvcore.nn import FlopCountAnalysis

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape, device=device)

    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input)
        total = flops.total()

    return int(total)


def get_peak_vram_mb() -> float:
    """Get peak VRAM usage in MB for the current CUDA device."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_vram_tracking() -> None:
    """Reset VRAM peak tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
