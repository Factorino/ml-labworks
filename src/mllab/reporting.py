from typing import Any

import pandas as pd


def make_table(rows: dict[str, Any], *, column: str = "значение") -> pd.DataFrame:
    values = pd.Series(list(rows.values()), index=pd.Index(rows, name=""), dtype=object)
    return values.to_frame(column)
