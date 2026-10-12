import numpy as np
from numpy.typing import NDArray


def classification_threshold_accuracy(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.float64],
    *,
    threshold: float = 0.5,
) -> float:
    predicted_labels = (y_pred > threshold).astype(int)
    return float(np.mean(predicted_labels == y_true))
