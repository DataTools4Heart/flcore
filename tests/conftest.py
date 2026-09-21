import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.synthetic_dt4h import make_dt4h_fixture, make_survival_fixture  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--save-golden",
        action="store_true",
        default=False,
        help="Write per-model aggregated-metric snapshots to tests/golden/ "
        "(the Phase-3 regression baseline) instead of only asserting the round completed.",
    )


@pytest.fixture
def sandbox_path(tmp_path):
    path = tmp_path / "sandbox"
    path.mkdir()
    return path


@pytest.fixture
def classification_data(tmp_path):
    return make_dt4h_fixture(tmp_path / "classification_data", task="classification")


@pytest.fixture
def regression_data(tmp_path):
    return make_dt4h_fixture(tmp_path / "regression_data", task="regression")


@pytest.fixture
def survival_data(tmp_path):
    return make_survival_fixture(tmp_path / "survival_data")
