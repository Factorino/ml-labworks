from typing import Any

import pandas as pd


def make_table(rows: dict[str, Any], *, column: str = "значение") -> pd.DataFrame:
    return pd.DataFrame({column: list(rows.values())}, index=pd.Index(rows, name=""))
