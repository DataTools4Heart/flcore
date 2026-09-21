"""Synthetic DT4H-format dataset generators for in-process federated smoke tests.

Mirrors the shape flcore/datasets.py::load_dt4h and flcore/datasets.py::load_survival
(via flcore/utils.py::CheckClientConfig's metadata parsing) expect: a directory
containing one *.parquet file plus a metadata.json with an `entries[0]` block
describing each feature/outcome's `dataType` and precomputed stats.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["age", "systolic_bp", "weight"]
NOMINAL_FEATURE = "gender"
NOMINAL_CATEGORIES = ["male", "female"]


def _numeric_stats(series: pd.Series) -> dict:
    return {
        "numOfNotNull": int(series.notna().sum()),
        "min": float(series.min()),
        "max": float(series.max()),
        "q1": float(series.quantile(0.25)),
        "q2": float(series.quantile(0.5)),
        "q3": float(series.quantile(0.75)),
    }


def make_dt4h_fixture(out_dir: Path, task: str = "classification", n_rows: int = 80, seed: int = 0) -> dict:
    """Write a small dt4h_format parquet + metadata.json into out_dir.

    Returns {"data_dir", "train_labels", "target_labels"} ready to merge into a
    client/server config dict.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "age": rng.normal(55, 12, n_rows).round(1),
            "systolic_bp": rng.normal(130, 15, n_rows).round(1),
            "weight": rng.normal(75, 10, n_rows).round(1),
            NOMINAL_FEATURE: rng.choice(NOMINAL_CATEGORIES, n_rows),
        }
    )

    if task == "regression":
        target_name = "outcome_score"
        df[target_name] = (0.3 * df["age"] + 0.2 * df["systolic_bp"] + rng.normal(0, 5, n_rows)).round(2)
        target_dtype = "NUMERIC"
    else:
        target_name = "outcome_flag"
        prob = 1 / (1 + np.exp(-(df["age"] - 55) / 10))
        df[target_name] = (rng.random(n_rows) < prob).astype(bool)
        target_dtype = "BOOLEAN"

    train_labels = NUMERIC_FEATURES + [NOMINAL_FEATURE]
    target_labels = [target_name]

    features_meta = [{"name": name, "dataType": "NUMERIC"} for name in NUMERIC_FEATURES]
    feature_stats = {name: _numeric_stats(df[name]) for name in NUMERIC_FEATURES}

    features_meta.append({"name": NOMINAL_FEATURE, "dataType": "NOMINAL"})
    feature_stats[NOMINAL_FEATURE] = {
        "numOfNotNull": int(df[NOMINAL_FEATURE].notna().sum()),
        "valueSet": NOMINAL_CATEGORIES,
    }

    outcomes_meta = [{"name": target_name, "dataType": target_dtype}]
    if target_dtype == "NUMERIC":
        outcome_stats = {target_name: _numeric_stats(df[target_name])}
    else:
        outcome_stats = {target_name: {"numOfNotNull": int(df[target_name].notna().sum())}}

    metadata = {
        "entries": [
            {
                "features": features_meta,
                "outcomes": outcomes_meta,
                "datasetStats": {"featureStats": feature_stats, "outcomeStats": outcome_stats},
            }
        ]
    }

    df.to_parquet(out_dir / "data.parquet")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {"data_dir": str(out_dir), "train_labels": train_labels, "target_labels": target_labels}


def make_survival_fixture(out_dir: Path, n_rows: int = 80, seed: int = 0) -> dict:
    """Write a small survival-format parquet + a metadata.json into out_dir.

    load_survival() reads the parquet directly and ignores metadata.json content,
    but CheckClientConfig() unconditionally requires metadata.json to exist and be
    parseable regardless of dataset type, so one is still emitted here.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "age": rng.normal(60, 10, n_rows).round(1),
            "systolic_bp": rng.normal(130, 15, n_rows).round(1),
            "time": rng.exponential(200, n_rows).round(1),
            "event": (rng.random(n_rows) < 0.6).astype(bool),
        }
    )

    train_labels = ["age", "systolic_bp"]

    metadata = {
        "entries": [
            {
                "features": [
                    {"name": "age", "dataType": "NUMERIC"},
                    {"name": "systolic_bp", "dataType": "NUMERIC"},
                ],
                "outcomes": [],
                "datasetStats": {
                    "featureStats": {
                        "age": _numeric_stats(df["age"]),
                        "systolic_bp": _numeric_stats(df["systolic_bp"]),
                    },
                    "outcomeStats": {},
                },
            }
        ]
    }

    df.to_parquet(out_dir / "data.parquet")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "data_dir": str(out_dir),
        "train_labels": train_labels,
        "target_labels": [],
        "time_col": "time",
        "event_col": "event",
    }
