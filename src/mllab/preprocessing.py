from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calendar_features(dates: pd.Series, parts: Sequence[str] = ("month",)) -> pd.DataFrame:
    return pd.DataFrame({part: getattr(dates.dt, part) for part in parts}, index=dates.index)


def log_target(regressor: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def build_preprocessor(categorical: Sequence[str], numeric: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("category", OneHotEncoder(drop="first"), list(categorical)),
            ("number", StandardScaler(), list(numeric)),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(
    regressor: Any,
    categorical: Sequence[str],
    numeric: Sequence[str],
) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [("preprocess", build_preprocessor(categorical, numeric)), ("regressor", regressor)]
    )
    return log_target(pipeline)
