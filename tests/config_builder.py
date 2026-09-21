"""Builds a single validated config dict for in-process federated smoke tests.

server_cmd.py and client_cmd.py each build their own config from their own argparse
defaults (they run as separate processes in production). A smoke test that drives
both a Client and a Strategy in one process needs ONE dict that satisfies both
flcore.utils.CheckClientConfig and flcore.utils.CheckServerConfig, so this module
merges the two CLI surfaces' defaults.

Where the two entry points currently disagree on a default (see CLAUDE.md "Known
issues" -- e.g. `balanced` defaults to `None` server-side vs `"True"` client-side,
or `dropout_method` defaults to Python `None` while the disabling check compares
against the *string* `"None"`, silently enabling dropout by default), this picks
one explicit, intended value rather than relying on either default. This is the
kind of drift Phase 1 (argument standardization) is meant to fix at the source.
"""
from pathlib import Path
from typing import Optional

from flcore.utils import CheckClientConfig, CheckServerConfig


def _defaults(sandbox_path: Path) -> dict:
    return {
        # shared server/client args
        "num_rounds": 2,
        "num_clients": 2,
        "min_fit_clients": 0,
        "min_evaluate_clients": 0,
        "min_available_clients": 0,
        "seed": 42,
        "sandbox_path": str(sandbox_path),
        "local_port": 8081,
        "production_mode": "False",
        "strategy": "FedAvg",
        "smooth_method": "EqualVoting",
        "smoothing_strenght": 0.5,
        # intended-disabled sentinel; see module docstring re: the None-vs-"None" bug
        "dropout_method": "None",
        "dropout_percentage": 0.0,
        "checkpoint_selection_metric": "precision",
        "metrics_aggregation": "weighted_average",
        "experiment_name": "smoke_test",
        # random forest / weighted random forest
        "balanced": "True",
        "n_estimators": 10,
        "max_depth": 2,
        "class_weight": "balanced",
        "levelOfDetail": "DecisionTree",
        "regression_criterion": "squared_error",
        # xgb
        "booster": "gbtree",
        "tree_method": "hist",
        "train_method": "bagging",
        "eta": 0.1,
        # cox
        "l1_penalty": 0.0,
        # dims -- recomputed by CheckClientConfig from metadata.json
        "n_feats": 0,
        "n_out": 0,
        # client-only args
        "node_name": "test_node_0",
        "certs_path": "/certs",
        "data_path": "",
        "dataset": "dt4h_format",
        "data_id": "",
        "normalization_method": "IQR",
        "train_labels": [],
        "target_labels": [],
        "train_size": 0.7,
        "validation_size": 0.2,
        "test_size": 0.1,
        "lr": 1e-3,
        "device": "cpu",
        "local_epochs": 2,
        "batch_size": 8,
        "penalty": "none",
        "save_every_n_rounds": 1,
        "solver": "saga",
        "l1_ratio": 0.5,
        "max_iter": 1000,
        "tol": 0.001,
        "kernel": "linear",
        "degree": 3,
        "gamma": "scale",
        "dropout_p": 0.0,
        "T": 5,
        "time_col": None,
        "event_col": None,
        "accumulative_pattern_col": None,
        "negative_duration_strategy": "clip",
    }


def build_validated_config(
    model: str,
    task: str,
    data_dir: str,
    train_labels: list,
    target_labels: list,
    sandbox_path: Path,
    time_col: Optional[str] = None,
    event_col: Optional[str] = None,
    **overrides,
) -> dict:
    """Build a config dict and run it through the real validators server_cmd.py /
    client_cmd.py use (CheckClientConfig then CheckServerConfig), so the smoke
    tests exercise the same alias-rewriting / dimension-inference / validation
    logic production actually runs, not a hand-rolled shortcut."""
    cfg = _defaults(sandbox_path)
    cfg.update(
        model=model,
        task=task,
        data_id=str(data_dir),
        data_path=str(data_dir),
        train_labels=list(train_labels),
        target_labels=list(target_labels),
        time_col=time_col,
        event_col=event_col,
    )
    cfg.update(overrides)
    cfg = CheckClientConfig(cfg)
    cfg = CheckServerConfig(cfg)
    return cfg
