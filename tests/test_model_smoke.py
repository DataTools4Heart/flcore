"""Phase-0 smoke tests: for each supported model key, build a real client/server
config against a small synthetic dataset, construct the real GetModelClient /
GetModelServerStrategy objects, and drive `num_rounds` of federated training +
evaluation through the real Strategy/Client code (see fed_driver.py) -- no
subprocesses, no gRPC, no certs.

This is the safety net later phases (argument standardization, cert/testing-mode
sanitation, aggregation-strategy homogenization) get regression-tested against.
Run with `--save-golden` to (re)write the tests/golden/ baseline snapshots.
"""
import json
from pathlib import Path

import pytest

from flcore.datasets import load_dataset
from flcore.utils import GetModelClient, GetModelServerStrategy

from config_builder import build_validated_config
from fed_driver import run_federated_rounds

GOLDEN_DIR = Path(__file__).parent / "golden"

# Known-broken model configs as of this session's run against the synthetic
# fixture (see CLAUDE.md Sec 5.3) -- xfail'd (strict=True) rather than left red so
# a genuine regression in a *passing* model still fails the suite, while these
# stay visible and self-documenting. strict=True means the day someone fixes the
# underlying bug, this test flips to XPASS and fails until the xfail is removed --
# that's the intended "please update this" signal, not a bug in the test.
_XFAIL_REASONS = {
    "logistic_regression": "flcore/models/linear_models/client.py fit() references undefined `ins` "
    "(leftover debug print from the old Client interface, never updated for NumPyClient.fit)",
    "linear_regression": "same linear_models/client.py `ins` NameError as logistic_regression",
    "weighted_random_forest": "flcore/models/weighted_random_forest/server.py reads nested "
    "config['weighted_random_forest'][...] keys from the old YAML config shape; "
    "server_cmd.py/client_cmd.py never produce that nested key -> KeyError",
    "xgb": "flcore/models/xgb/server.py::aggregate_bagging raises KeyError('iteration_indptr') "
    "merging a second client's trees; xgb's bagging aggregation is broken on the installed xgboost version",
    "nn": "flcore/models/nn/client.py fit() references an uninitialized `metrics` dict "
    "(training itself runs; only the running_time bookkeeping line crashes)",
    "cox": "flcore/models/cox/client.py fit() calls time.time() with no `import time` in the file",
    "gbs": "flcore/models/gbs/client.py fit() calls time.time() with no `import time` in the file",
}


def _case(model, task, data_fixture):
    if model in _XFAIL_REASONS:
        return pytest.param(
            model,
            task,
            data_fixture,
            marks=pytest.mark.xfail(reason=_XFAIL_REASONS[model], strict=True),
            id=model,
        )
    return pytest.param(model, task, data_fixture, id=model)


# (model key, task, dataset fixture name)
MODEL_CASES = [
    _case("logistic_regression", "classification", "classification_data"),
    _case("linear_regression", "regression", "regression_data"),
    _case("random_forest", "classification", "classification_data"),
    _case("weighted_random_forest", "classification", "classification_data"),
    _case("xgb", "classification", "classification_data"),
    _case("nn", "classification", "classification_data"),
    _case("cox", "survival", "survival_data"),
    _case("rsf", "survival", "survival_data"),
    _case("gbs", "survival", "survival_data"),
]


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


@pytest.mark.parametrize("model,task,data_fixture", MODEL_CASES)
def test_model_round_trip(model, task, data_fixture, request, sandbox_path):
    data_info = request.getfixturevalue(data_fixture)

    config = build_validated_config(
        model=model,
        task=task,
        data_dir=data_info["data_dir"],
        train_labels=data_info["train_labels"],
        target_labels=data_info["target_labels"],
        sandbox_path=sandbox_path,
        time_col=data_info.get("time_col"),
        event_col=data_info.get("event_col"),
        node_name="server",
    )

    _, strategy = GetModelServerStrategy(config)
    assert strategy is not None

    clients = []
    for i in range(config["num_clients"]):
        client_config = dict(config)
        client_config["node_name"] = f"client_{i}"
        data = load_dataset(client_config, i)
        clients.append(GetModelClient(client_config, data))

    result = run_federated_rounds(strategy, clients, num_rounds=config["num_rounds"])

    assert result["parameters"] is not None
    assert len(result["rounds"]) == config["num_rounds"]
    for round_info in result["rounds"]:
        assert not round_info["fit_failures"], round_info["fit_failures"]
        assert not round_info["eval_failures"], round_info["eval_failures"]

    if request.config.getoption("--save-golden"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "model": model,
            "task": task,
            "rounds": [
                {
                    "round": r["round"],
                    "fit_metrics": _to_jsonable(r["fit_metrics"]),
                    "loss": _to_jsonable(r["loss"]),
                    "eval_metrics": _to_jsonable(r["eval_metrics"]),
                }
                for r in result["rounds"]
            ],
        }
        (GOLDEN_DIR / f"{model}.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True))
