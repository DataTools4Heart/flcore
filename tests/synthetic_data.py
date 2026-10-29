"""Synthetic dataset generator that mirrors the DT4H dataset layout.

The real datasets cannot be shared, so the tests build their own from the
structure of the sample files (``synthetic_dt4h_dataset`` and
``synthetic_survival``):

* a table with the same 12 columns and dtypes (string / int / float / bool),
  plus the 7 ``cause_of_death_isAllCause_*`` horizon columns used by the
  survival models;
* a ``metadata.json`` with the same ``entries[0]`` layout (features, outcomes,
  datasetStats) whose statistics are computed from the generated table.

The generator is deterministic for a given seed, and guarantees that every
class / event appears often enough for each outcome to be trainable on a
70/30 split, whatever the sample size is (>= ~60 rows).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_N_SAMPLES = 100

GENDERS = ["female", "male"]
ENCOUNTER_CLASSES = ["inpatient", "outpatient", "emergency"]
ADMISSION_YEARS = ["2019", "2020", "2021", "2022", "2023"]
DISPOSITIONS = ["home", "rehab", "nursing_facility", "death"]

FEATURES = [
    ("patient_demographics_gender", "NOMINAL"),
    ("encounters_encounterClass", "NOMINAL"),
    ("encounters_admissionYear", "NOMINAL"),
    ("patient_demographics_age", "NUMERIC"),
    ("encounters_lengthOfStay", "NUMERIC"),
    ("vital_signs_weight_value_p6mo_min", "NUMERIC"),
    ("med_admins_history_diuretics_loop_any", "BOOLEAN"),
    ("patient_flags_smoker", "BOOLEAN"),
]
OUTCOMES = [
    ("outcome_in_hospital_mortality", "BOOLEAN"),
    ("outcome_readmission_30d", "BOOLEAN"),
    ("outcome_discharge_disposition", "NOMINAL"),
    ("outcome_severity_score", "NUMERIC"),
]

# Accumulative survival horizons, in the same order and naming as the sample
# survival file (suffix -> days).
SURVIVAL_PREFIX = "cause_of_death_isAllCause_"
SURVIVAL_HORIZONS = {
    "7d": 7,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1a": 365,
    "3a": 365 * 3,
    "5a": 365 * 5,
}

FEATURE_COLUMNS = [name for name, _ in FEATURES]
OUTCOME_COLUMNS = [name for name, _ in OUTCOMES]
SURVIVAL_COLUMNS = [SURVIVAL_PREFIX + suffix for suffix in SURVIVAL_HORIZONS]
ALL_COLUMNS = FEATURE_COLUMNS + OUTCOME_COLUMNS + SURVIVAL_COLUMNS


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() or 1.0)


def generate_dataset(n_samples=DEFAULT_N_SAMPLES, seed=0):
    """Return a DataFrame with ``n_samples`` rows and the DT4H column layout."""
    rng = np.random.default_rng(seed)
    n = n_samples

    gender = rng.choice(GENDERS, n, p=[0.36, 0.64])
    encounter = rng.choice(ENCOUNTER_CLASSES, n, p=[0.42, 0.35, 0.23])
    year = rng.choice([int(y) for y in ADMISSION_YEARS], n)
    age = np.clip(rng.normal(58, 16, n), 18, 97).round().astype(int)
    stay = np.clip(rng.lognormal(1.3, 0.7, n), 1, 45).round().astype(int)
    weight = np.clip(rng.normal(76.8, 14, n), 42.5, 135.0).round(1)
    diuretics = rng.random(n) < 0.12
    smoker = rng.random(n) < 0.23

    # One latent severity drives every outcome so the models have signal to learn.
    risk = (
        0.9 * _z(age)
        + 0.7 * _z(stay)
        + 0.6 * diuretics
        + 0.5 * (encounter == "emergency")
        + 0.3 * smoker
        + rng.normal(0, 0.6, n)
    )

    # Quantile thresholds (instead of random draws) pin the class balance.
    mortality = risk > np.quantile(risk, 0.80)
    readmission = (risk + rng.normal(0, 0.8, n)) > np.quantile(risk, 0.82)

    # Discharge: death == mortality, remaining patients split by risk rank.
    disposition = np.full(n, "home", dtype=object)
    survivors = np.flatnonzero(~mortality)
    ranked = survivors[np.argsort(risk[survivors])]
    n_home = int(round(len(ranked) * 0.65))
    n_rehab = int(round(len(ranked) * 0.20))
    disposition[ranked[n_home:n_home + n_rehab]] = "rehab"
    disposition[ranked[n_home + n_rehab:]] = "nursing_facility"
    disposition[mortality] = "death"

    severity = np.clip(4.8 + 1.8 * _z(risk) + rng.normal(0, 0.8, n), 0.0, 10.0).round(1)

    # Survival: time to death, shorter with higher risk; ~40% die within 5 years.
    time_to_death = rng.exponential(1.0, n) * np.exp(-0.8 * _z(risk)) * 365 * 9
    horizons = {
        SURVIVAL_PREFIX + suffix: time_to_death <= days
        for suffix, days in SURVIVAL_HORIZONS.items()
    }

    data = {
        "patient_demographics_gender": gender,
        "encounters_encounterClass": encounter,
        "encounters_admissionYear": year,
        "patient_demographics_age": age,
        "encounters_lengthOfStay": stay,
        "vital_signs_weight_value_p6mo_min": weight,
        "med_admins_history_diuretics_loop_any": diuretics,
        "patient_flags_smoker": smoker,
        "outcome_in_hospital_mortality": mortality,
        "outcome_readmission_30d": readmission,
        "outcome_discharge_disposition": disposition,
        "outcome_severity_score": severity,
        **horizons,
    }
    return pd.DataFrame(data)[ALL_COLUMNS]


def _numeric_stats(col):
    col = col.dropna().astype(float)
    q1, q2, q3 = np.quantile(col, [0.25, 0.5, 0.75])
    counts, edges = np.histogram(col, bins=5)
    return {
        "numOfNotNull": int(col.size),
        "min": float(col.min()),
        "max": float(col.max()),
        "avg": float(round(col.mean(), 3)),
        "q1": float(q1),
        "q2": float(q2),
        "q3": float(q3),
        "histogram": [
            {"bin": float(round(edges[i + 1], 3)), "count": int(counts[i])}
            for i in range(len(counts))
        ],
    }


def _nominal_stats(col, value_set):
    as_str = col.dropna().astype(str)
    return {
        "numOfNotNull": int(as_str.size),
        "valueSet": list(value_set),
        "cardinalityPerItem": {v: int((as_str == v).sum()) for v in value_set},
    }


def _boolean_stats(col):
    col = col.dropna()
    return {"numOfNotNull": int(col.size), "numOfTrue": int(col.astype(bool).sum())}


def build_metadata(df):
    """Build a ``metadata.json`` structure (``entries[0]`` layout) for ``df``."""
    value_sets = {
        "patient_demographics_gender": GENDERS,
        "encounters_encounterClass": ENCOUNTER_CLASSES,
        "encounters_admissionYear": ADMISSION_YEARS,
        "outcome_discharge_disposition": DISPOSITIONS,
    }

    def stats_for(name, dtype):
        if dtype == "NUMERIC":
            return _numeric_stats(df[name])
        if dtype == "NOMINAL":
            return _nominal_stats(df[name], value_sets[name])
        return _boolean_stats(df[name])

    return {
        "entries": [
            {
                "features": [{"name": n, "dataType": t} for n, t in FEATURES],
                "outcomes": [{"name": n, "dataType": t} for n, t in OUTCOMES],
                "datasetStats": {
                    "featureStats": {n: stats_for(n, t) for n, t in FEATURES},
                    "outcomeStats": {n: stats_for(n, t) for n, t in OUTCOMES},
                },
            }
        ]
    }


def write_dataset(directory, n_samples=DEFAULT_N_SAMPLES, seed=0):
    """Write ``data.parquet`` + ``metadata.json`` into ``directory``.

    The directory has the layout flcore expects for ``--data_path`` (a single
    parquet file next to ``metadata.json``). Returns the generated DataFrame.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_samples, seed)
    df.to_parquet(directory / "data.parquet", index=False)
    with open(directory / "metadata.json", "w") as f:
        json.dump(build_metadata(df), f, indent=2)
    return df
