"""PCC_DEG, the evaluation metric: how well a predicted response matches the measured one, gene by gene."""
from __future__ import annotations

import numpy as np


def pcc_deg(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pearson correlation of measured and predicted response across genes, one value per sample (row).

    The PCC_DEG of a set of samples is the mean of these values.
    """
    y = y - y.mean(1, keepdims=True)
    pred = pred - pred.mean(1, keepdims=True)
    return (y * pred).sum(1) / np.sqrt((y * y).sum(1) * (pred * pred).sum(1) + 1e-12)


def score(y: np.ndarray, pred: np.ndarray) -> float:
    """PCC_DEG of a set of samples."""
    return float(pcc_deg(y, pred).mean())
