import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_data import DEFAULT_N_SAMPLES, write_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Directory with ``data.parquet`` + ``metadata.json`` of 100 synthetic rows."""
    path = tmp_path_factory.mktemp("synthetic_dt4h")
    write_dataset(path, n_samples=DEFAULT_N_SAMPLES, seed=0)
    return path
