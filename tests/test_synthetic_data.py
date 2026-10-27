"""The synthetic generator must reproduce the layout of the DT4H sample files."""

import json

import numpy as np
import pandas as pd
import pytest
from synthetic_data import (
    ALL_COLUMNS,
    DEFAULT_N_SAMPLES as N_SAMPLES,
    DISPOSITIONS,
    SURVIVAL_COLUMNS,
    generate_dataset,
    write_dataset,
)

# Dtypes of synthetic_dt4h_dataset / synthetic_survival (strings are "object"
# or "str" depending on the pandas version, so they are compared by kind).
EXPECTED_KINDS = {
    "patient_demographics_gender": "O",
    "encounters_encounterClass": "O",
    "encounters_admissionYear": "i",
    "patient_demographics_age": "i",
    "encounters_lengthOfStay": "i",
    "vital_signs_weight_value_p6mo_min": "f",
    "med_admins_history_diuretics_loop_any": "b",
    "patient_flags_smoker": "b",
    "outcome_in_hospital_mortality": "b",
    "outcome_readmission_30d": "b",
    "outcome_discharge_disposition": "O",
    "outcome_severity_score": "f",
    **{c: "b" for c in SURVIVAL_COLUMNS},
}


@pytest.fixture(scope="module")
def df():
    return generate_dataset(N_SAMPLES, seed=0)


def test_has_100_rows_and_sample_file_columns(df):
    assert len(df) == 100
    assert list(df.columns) == ALL_COLUMNS
    assert len(ALL_COLUMNS) == 19


def test_dtypes_match_sample_files(df):
    for col, kind in EXPECTED_KINDS.items():
        assert df[col].dtype.kind == kind, f"{col}: {df[col].dtype}"


def test_no_missing_values(df):
    assert not df.isna().any().any()


def test_value_ranges_match_metadata_sample(df):
    assert df["patient_demographics_age"].between(18, 97).all()
    assert df["encounters_lengthOfStay"].between(1, 45).all()
    assert df["vital_signs_weight_value_p6mo_min"].between(42.5, 135.0).all()
    assert df["outcome_severity_score"].between(0.0, 10.0).all()
    assert set(df["outcome_discharge_disposition"]) == set(DISPOSITIONS)


def test_every_outcome_is_trainable_on_a_70_30_split(df):
    # Each class needs enough members to land in both train and test splits.
    assert df["outcome_in_hospital_mortality"].sum() >= 10
    assert (~df["outcome_in_hospital_mortality"]).sum() >= 10
    assert df["outcome_discharge_disposition"].value_counts().min() >= 10
    assert df["outcome_severity_score"].nunique() > 10
    assert df[SURVIVAL_COLUMNS[-1]].sum() >= 10  # events within 5 years


def test_survival_horizons_are_cumulative(df):
    flags = df[SURVIVAL_COLUMNS].to_numpy()
    assert (np.diff(flags.astype(int), axis=1) >= 0).all()


def test_death_disposition_matches_mortality(df):
    died = df["outcome_discharge_disposition"] == "death"
    assert (died == df["outcome_in_hospital_mortality"]).all()


def test_deterministic_for_a_seed():
    pd.testing.assert_frame_equal(generate_dataset(50, 7), generate_dataset(50, 7))
    assert not generate_dataset(50, 7).equals(generate_dataset(50, 8))


def test_written_dataset_roundtrip_and_metadata(tmp_path):
    written = write_dataset(tmp_path, n_samples=N_SAMPLES, seed=0)
    loaded = pd.read_parquet(tmp_path / "data.parquet")
    pd.testing.assert_frame_equal(written, loaded)

    entry = json.loads((tmp_path / "metadata.json").read_text())["entries"][0]
    stats = entry["datasetStats"]
    assert [f["name"] for f in entry["features"]] == ALL_COLUMNS[:8]
    assert [o["name"] for o in entry["outcomes"]] == ALL_COLUMNS[8:12]
    assert set(stats["featureStats"]) == set(ALL_COLUMNS[:8])
    assert set(stats["outcomeStats"]) == set(ALL_COLUMNS[8:12])

    age = stats["featureStats"]["patient_demographics_age"]
    assert age["numOfNotNull"] == 100
    assert age["min"] == loaded["patient_demographics_age"].min()
    assert age["q1"] <= age["q2"] <= age["q3"]
    disp = stats["outcomeStats"]["outcome_discharge_disposition"]
    assert sum(disp["cardinalityPerItem"].values()) == 100
    mortality = stats["outcomeStats"]["outcome_in_hospital_mortality"]
    assert mortality["numOfTrue"] == loaded["outcome_in_hospital_mortality"].sum()


def test_flcore_dt4h_loader_reads_the_dataset(synthetic_data_dir):
    """The generated files go through flcore's own loader without changes."""
    from flcore.datasets import load_dt4h

    config = {
        "metadata_file": str(synthetic_data_dir / "metadata.json"),
        "data_file": str(synthetic_data_dir / "data.parquet"),
        "normalization_method": "IQR",
        "train_size": 0.7,
        "task": "classification",
        "train_labels": [
            "patient_demographics_gender",
            "patient_demographics_age",
            "patient_flags_smoker",
        ],
        "target_labels": ["outcome_discharge_disposition"],
    }
    (X_train, y_train), (X_test, y_test) = load_dt4h(config)
    assert len(X_train) == 70 and len(X_test) == 30
    assert not X_train.isna().any().any()
    assert set(y_train) | set(y_test) == {0, 1, 2, 3}
