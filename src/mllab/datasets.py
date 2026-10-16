from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from mllab import DATA_DIR


KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/{slug}"

TESLA_COLUMNS = {
    "Insider Trading": "insider",
    "Relationship": "relationship",
    "Date": "date",
    "Transaction": "transaction",
    "Cost": "cost",
    "Shares": "shares",
    "Value ($)": "value",
    "Shares Total": "shares_total",
    "SEC Form 4": "sec_form_4",
}


def fetch_kaggle(slug: str, member: str, path: Path) -> Path:
    if path.exists():
        return path

    with urlopen(KAGGLE_DOWNLOAD.format(slug=slug)) as response:  # noqa: S310
        archive = ZipFile(BytesIO(response.read()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive.read(member))
    return path


def load_tesla() -> pd.DataFrame:
    path = fetch_kaggle("ilyaryabov/tesla-insider-trading", "TSLA.csv", DATA_DIR / "tesla.csv")
    data = pd.read_csv(path, thousands=",", parse_dates=["Date"])
    return data.rename(columns=TESLA_COLUMNS)


def load_titanic() -> pd.DataFrame:
    path = fetch_kaggle("heptapod/titanic", "train_and_test2.csv", DATA_DIR / "titanic.csv")
    data = pd.read_csv(path, index_col="Passengerid")
    data = data.drop(columns=[column for column in data.columns if column.startswith("zero")])
    return data.rename(columns={"2urvived": "Survived"}).rename_axis("PassengerId")


def fetch_all() -> None:
    load_tesla()
    load_titanic()
