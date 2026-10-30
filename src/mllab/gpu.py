import ctypes
from importlib.util import find_spec
from pathlib import Path


def load_cusolver() -> None:
    spec = find_spec("nvidia.cusolver")
    if spec is None or not spec.submodule_search_locations:
        return

    for location in spec.submodule_search_locations:
        for library in sorted((Path(location) / "lib").glob("libcusolver.so.*")):
            ctypes.CDLL(str(library))
