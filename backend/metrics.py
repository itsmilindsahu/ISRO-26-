"""
metrics.py
==========
Thin wrappers around scikit-image's PSNR / SSIM for comparing an
interpolated frame against a held-out ground-truth frame.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Compute PSNR (dB) and SSIM between ``pred`` and ``truth``.

    Both arrays are expected to be float images in [0, 1] of identical
    shape. Returns a dict with keys ``psnr`` and ``ssim``.
    """
    pred = np.clip(pred, 0.0, 1.0).astype(np.float64)
    truth = np.clip(truth, 0.0, 1.0).astype(np.float64)
    psnr = peak_signal_noise_ratio(truth, pred, data_range=1.0)
    ssim = structural_similarity(truth, pred, data_range=1.0)
    return {"psnr": float(psnr), "ssim": float(ssim)}
