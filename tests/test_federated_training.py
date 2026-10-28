"""Every model, in every training scenario that applies to it.

Each case launches a real federation (1 server + 2 clients, 2 rounds) over
the synthetic dataset and checks that every process exits cleanly and that the
server completed both rounds with finite losses.
"""

import math

import pytest
from fl_runner import run_federation

NUM_CLIENTS = 2
NUM_ROUNDS = 2

BINARY = ["outcome_in_hospital_mortality"]  # BOOLEAN  -> binary classification
MULTICLASS = ["outcome_discharge_disposition"]  # NOMINAL (4) -> multiclass
REGRESSION = ["outcome_severity_score"]  # NUMERIC  -> regression
SURVIVAL_PATTERN = "cause_of_death_isAllCause_"  # 7d, 1mo, ... 5a horizon columns

TARGETS = {
    "binary": ("classification", BINARY),
    "multiclass": ("classification", MULTICLASS),
    "regression": ("regression", REGRESSION),
    "survival": ("survival", []),
}


def known_bug(reason):
    """Mark a scenario that currently fails because of a bug in flcore.

    Strict, so the suite goes red as soon as the bug is fixed and this mark
    needs to be removed. Run with ``--runxfail`` to see the actual failures.
    """
    return pytest.mark.xfail(reason=reason, strict=True)


def case(model, scenario, client_extra=(), id_suffix="", bug=None):
    marks = [known_bug(bug)] if bug else []
    return pytest.param(
        model,
        scenario,
        list(client_extra),
        id=f"{model}{id_suffix}-{scenario}",
        marks=marks,
    )


SGD_MULTICLASS_BUG = (
    "linear_models.utils.set_initial_params sets a scalar intercept_ for "
    "lsvc/svm, but SGDClassifier needs one intercept per class (multiclass)"
)
SVR_KERNEL_BUG = "SVR with a non-linear kernel has no fit_intercept / coef_"
XGB_BUG = "xgb server needs iteration_indptr (XGBoost >= 2.0, 1.7.5 is pinned)"
SURVIVAL_TIME_BUG = "survival client fit() uses `time` without importing it"

CASES = []

# Linear classifiers: binary + multiclass
for model in ["logistic_regression", "logistic_regression_elasticnet", "lsvc", "svm"]:
    sgd = model in ("lsvc", "svm")
    CASES.append(case(model, "binary"))
    CASES.append(case(model, "multiclass", bug=SGD_MULTICLASS_BUG if sgd else None))

# Linear regressors
for model in ["linear_regression", "lasso_regression", "ridge_regression"]:
    CASES.append(case(model, "regression"))
CASES.append(
    case(
        "linear_regression_elasticnet",
        "regression",
        bug="server GetModelServerStrategy rejects it as 'Unknown model'",
    )
)
for kernel in ["linear", "rbf", "poly", "sigmoid"]:
    CASES.append(
        case(
            "svr",
            "regression",
            client_extra=["--kernel", kernel],
            id_suffix=f"[{kernel}]",
            bug=None if kernel == "linear" else SVR_KERNEL_BUG,
        )
    )

# Tree ensembles and neural network
CASES += [
    case("random_forest", "binary"),
    case("random_forest", "multiclass"),
    case("random_forest", "regression"),
]
wrf_bug = "weighted_random_forest server reads config['weighted_random_forest'], never set by the CLI"
CASES += [
    case("weighted_random_forest", "binary", bug=wrf_bug),
    case("weighted_random_forest", "multiclass", bug=wrf_bug),
]
CASES += [case("xgb", s, bug=XGB_BUG) for s in ("binary", "multiclass", "regression")]
CASES += [case("nn", s) for s in ("binary", "multiclass", "regression")]

# Survival
CASES += [
    case("cox", "survival", bug=SURVIVAL_TIME_BUG),
    case("rsf", "survival"),
    case("gbs", "survival", bug=SURVIVAL_TIME_BUG),
]


@pytest.mark.parametrize("model,scenario,client_extra", CASES)
def test_federated_training(
    model, scenario, client_extra, synthetic_data_dir, tmp_path
):
    task, target_labels = TARGETS[scenario]
    if scenario == "survival":
        client_extra = [*client_extra, "--accumulative_pattern_col", SURVIVAL_PATTERN]
    result = run_federation(
        model=model,
        task=task,
        data_path=synthetic_data_dir,
        sandbox=tmp_path / "sandbox",
        target_labels=target_labels,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        client_extra_args=client_extra,
    )
    assert result.ok, result.describe()

    losses = result.history.get("losses_distributed", [])
    assert len(losses) == NUM_ROUNDS, (
        f"expected {NUM_ROUNDS} evaluated rounds, got {len(losses)}\n"
        + result.describe()
    )
    assert all(math.isfinite(x) for x in losses), f"non-finite losses: {losses}"
