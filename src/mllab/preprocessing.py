from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"month": dates.dt.month}, index=dates.index)


def log_target(regressor: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
