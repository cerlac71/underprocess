"""Pixel-based contingency metrics for inundation / rainfall event validation."""

from __future__ import annotations

import numpy as np


def binary_contingency_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute standard verification scores for binary inundation masks.

    Parameters
    ----------
    predicted, observed : 2D bool or {0,1} arrays (same shape)
    valid_mask : optional bool mask excluding ocean/permanent water/etc.
    """
    pred = predicted.astype(bool)
    obs = observed.astype(bool)
    if valid_mask is None:
        valid_mask = np.ones(pred.shape, dtype=bool)
    else:
        valid_mask = valid_mask.astype(bool)

    pred = pred & valid_mask
    obs = obs & valid_mask

    hits = int(np.sum(pred & obs))
    misses = int(np.sum(~pred & obs))
    false_alarms = int(np.sum(pred & ~obs))
    correct_negatives = int(np.sum(~pred & ~obs))

    pod = hits / (hits + misses) if (hits + misses) else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else np.nan
    precision = hits / (hits + false_alarms) if (hits + false_alarms) else np.nan
    recall = pod
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    denom_hss = (hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (
        false_alarms + correct_negatives
    )
    hss = (
        ((hits * correct_negatives) - (misses * false_alarms)) / denom_hss
        if denom_hss
        else np.nan
    )
    accuracy = (hits + correct_negatives) / valid_mask.sum() if valid_mask.sum() else np.nan

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "pod": float(pod),
        "far": float(far),
        "csi": float(csi),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "hss": float(hss),
        "accuracy": float(accuracy),
        "sample_size": int(valid_mask.sum()),
        "observed_fraction": float(obs.sum() / valid_mask.sum()) if valid_mask.sum() else np.nan,
        "predicted_fraction": float(pred.sum() / valid_mask.sum()) if valid_mask.sum() else np.nan,
    }
