from pathlib import Path


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

__all__ = ["DATA_DIR", "PROJECT_ROOT", "RANDOM_STATE"]
