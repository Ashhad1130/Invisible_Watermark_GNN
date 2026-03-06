"""
Evaluation metrics for watermark detection experiments.

All functions accept lists (or arrays) of binary ground-truth labels
and predicted labels/scores.

tpr               – True Positive Rate (Recall / Sensitivity)
fpr               – False Positive Rate (Fall-out)
balanced_accuracy – Mean of TPR and (1 - FPR)
ber               – Bit Error Rate for multi-bit watermarks
"""

from __future__ import annotations

import numpy as np


def _to_binary_array(x) -> np.ndarray:
    """Coerce input to a flat numpy array of dtype bool."""
    return np.asarray(x, dtype=bool).ravel()


def tpr(y_true, y_pred) -> float:
    """True Positive Rate: TP / (TP + FN).

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (1 = watermarked, 0 = clean).
    y_pred:
        Predicted binary labels.

    Returns
    -------
    float in [0, 1].  Returns 0.0 if there are no positive ground-truth
    samples.
    """
    y_true = _to_binary_array(y_true)
    y_pred = _to_binary_array(y_pred)

    positives = y_true.sum()
    if positives == 0:
        return 0.0
    tp = (y_true & y_pred).sum()
    return float(tp / positives)


def fpr(y_true, y_pred) -> float:
    """False Positive Rate: FP / (FP + TN).

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (1 = watermarked, 0 = clean).
    y_pred:
        Predicted binary labels.

    Returns
    -------
    float in [0, 1].  Returns 0.0 if there are no negative ground-truth
    samples.
    """
    y_true = _to_binary_array(y_true)
    y_pred = _to_binary_array(y_pred)

    negatives = (~y_true).sum()
    if negatives == 0:
        return 0.0
    fp = (~y_true & y_pred).sum()
    return float(fp / negatives)


def balanced_accuracy(y_true, y_pred) -> float:
    """Balanced accuracy: (TPR + TNR) / 2 = (TPR + (1 - FPR)) / 2.

    Robust to class imbalance; 0.5 corresponds to random chance.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.

    Returns
    -------
    float in [0, 1].
    """
    sensitivity = tpr(y_true, y_pred)
    specificity = 1.0 - fpr(y_true, y_pred)
    return (sensitivity + specificity) / 2.0


def ber(bits_true, bits_pred) -> float:
    """Bit Error Rate: fraction of incorrectly decoded bits.

    Parameters
    ----------
    bits_true:
        Ground-truth bit sequence (binary array, length N).
    bits_pred:
        Decoded bit sequence (binary array, length N).

    Returns
    -------
    float in [0, 1].  0.0 = perfect decoding; 0.5 = random guessing.
    """
    bits_true = np.asarray(bits_true).ravel()
    bits_pred = np.asarray(bits_pred).ravel()

    if len(bits_true) != len(bits_pred):
        raise ValueError(
            f"Length mismatch: bits_true has {len(bits_true)} elements, "
            f"bits_pred has {len(bits_pred)} elements."
        )
    if len(bits_true) == 0:
        return 0.0

    return float(np.mean(bits_true != bits_pred))
