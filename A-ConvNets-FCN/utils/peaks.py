from typing import List, Tuple

import torch
import torch.nn.functional as F


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


def nms_topk_peaks(
    activation: torch.Tensor,
    k: int = 5,
    threshold: float = 0.0,
    window_size: int = 5,
    percentile: float = 99.0,
    scale: float = 1.0,
) -> List[Tuple[int, int, float]]:
    """
    Find top-k peaks after 2D NMS, with adaptive thresholding.

    - NMS: max-pooling with stride=1 and padding to keep shape
    - Threshold: if `threshold` > 0, use it; otherwise use
      torch.quantile(activation, percentile/100) * scale
    """
    if activation.ndim != 2:
        raise ValueError("Expected 2D tensor")

    if window_size <= 1:
        window_size = 3

    flat = activation.reshape(-1).float()
    if threshold <= 0.0:
        if flat.numel() == 0:
            thr_val = 0.0
        else:
            q = torch.quantile(flat, min(max(percentile, 0.0), 100.0) / 100.0)
            thr_val = float(q.item()) * float(scale)
    else:
        thr_val = float(threshold)

    x = activation.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(x, kernel_size=window_size, stride=1, padding=window_size // 2)
    pooled = pooled[0, 0]

    # Local maxima mask and thresholding
    mask = (activation == pooled) & (activation >= thr_val)
    if not torch.any(mask):
        return []

    coords = mask.nonzero(as_tuple=False)
    values = activation[mask]

    vals_sorted, order = torch.sort(values.reshape(-1), descending=True)
    coords_sorted = coords[order]

    k_eff = min(k, coords_sorted.shape[0])
    peaks: List[Tuple[int, int, float]] = []
    for i in range(k_eff):
        row = int(coords_sorted[i, 0].item())
        col = int(coords_sorted[i, 1].item())
        val = float(vals_sorted[i].item())
        peaks.append((row, col, val))
    return peaks
