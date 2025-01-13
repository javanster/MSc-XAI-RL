from typing import cast

import numpy as np
from sklearn.metrics import accuracy_score


def binary_concept_probe_score(y_val: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the binary concept probe score based on validation and predicted labels.

    The score is calculated as `max(2 * (accuracy - 0.5), 0)`, where `accuracy` is the
    standard accuracy metric between `y_val` and `y_pred`. This ensures the score is
    non-negative and highlights performance above random chance, where accuracy = 0.5
    yields a score of 0.

    Parameters
    ----------
    y_val : np.ndarray
        The ground truth (validation) labels. Must be a binary array (0 or 1).
    y_pred : np.ndarray
        The predicted labels. Must be a binary array (0 or 1).

    Returns
    -------
    float
        The binary concept probe score, ranging from 0 to 1.
    """
    original_accuracy: float = cast(float, accuracy_score(y_val, y_pred))
    return max(2 * (original_accuracy - 0.5), 0)
