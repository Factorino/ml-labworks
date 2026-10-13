from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


MEAN_COLUMN = "среднее"


def classification_threshold_accuracy(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.float64],
    *,
    threshold: float = 0.5,
) -> float:
    predicted_labels = (y_pred > threshold).astype(int)
    return float(np.mean(predicted_labels == y_true))


def minmax_scores(scores: ArrayLike) -> NDArray[np.float64]:
    magnitudes = np.abs(np.asarray(scores, dtype=np.float64)).reshape(-1, 1)
    return np.asarray(MinMaxScaler().fit_transform(magnitudes), dtype=np.float64).ravel()


def ranking_table(scores: Mapping[str, ArrayLike], names: Sequence[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {method: minmax_scores(values) for method, values in scores.items()},
        index=pd.Index(names, name="признак"),
    )
    table[MEAN_COLUMN] = table.mean(axis=1)
    return table.sort_values(MEAN_COLUMN, ascending=False)


def top_features(table: pd.DataFrame, count: int = 4) -> list[str]:
    return [str(name) for name in table[MEAN_COLUMN].nlargest(count).index]
