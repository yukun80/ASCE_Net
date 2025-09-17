from typing import List, Tuple

import torch


def argmax2d(activation: torch.Tensor) -> Tuple[int, int]:
    """Return row, col index of the global maximum."""
    if activation.ndim != 2:
        raise ValueError("Expected 2D tensor")
    flat_index = torch.argmax(activation)
    row = int(flat_index // activation.shape[1])
    col = int(flat_index % activation.shape[1])
    return row, col


def topk_peaks(activation: torch.Tensor, k: int = 5, threshold: float = 0.0) -> List[Tuple[int, int, float]]:
    """Return top-k peak positions (row, col, value) above threshold."""
    if activation.ndim != 2:
        raise ValueError("Expected 2D tensor")
    values, indices = torch.topk(activation.view(-1), k)
    peaks: List[Tuple[int, int, float]] = []
    width = activation.shape[1]
    for val, idx in zip(values.tolist(), indices.tolist()):
        if val < threshold:
            continue
        row = idx // width
        col = idx % width
        peaks.append((row, col, val))
    return peaks
