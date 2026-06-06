from __future__ import annotations

import numpy as np


def structural_hamming_distance(
    pred: np.ndarray, target: np.ndarray, ignore_diag: bool = True
) -> int:
    """Return directed SHD as the number of mismatched binary adjacency entries."""
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    if pred_bool.shape != target_bool.shape:
        raise ValueError(f"shape mismatch: {pred_bool.shape} != {target_bool.shape}")
    diff = pred_bool != target_bool
    if ignore_diag and diff.ndim == 2:
        diag_len = min(diff.shape)
        diff[np.arange(diag_len), np.arange(diag_len)] = False
    return int(diff.sum())


def batch_shd(pred: np.ndarray, target: np.ndarray, ignore_diag: bool = True) -> float:
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    if pred_arr.ndim == 2:
        return float(structural_hamming_distance(pred_arr, target_arr, ignore_diag=ignore_diag))
    return float(
        np.mean(
            [
                structural_hamming_distance(p, t, ignore_diag=ignore_diag)
                for p, t in zip(pred_arr, target_arr, strict=True)
            ]
        )
    )


def percent_change(base: np.ndarray, changed: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    base_arr = np.asarray(base, dtype=np.float64)
    changed_arr = np.asarray(changed, dtype=np.float64)
    return np.abs(changed_arr - base_arr) / np.maximum(np.abs(base_arr), eps) * 100.0
