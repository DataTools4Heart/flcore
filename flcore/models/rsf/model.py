# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple

import pickle
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
from scipy.interpolate import interp1d

from flcore.models.rsf.base_model import BaseSurvivalModel

class RSFModel(BaseSurvivalModel):
    def __init__(self, n_estimators=100, random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self.global_event_times_ = None  # unified time grid for federated evaluation

    def fit(self, data: dict):
        """Fit model locally (classic sklearn behavior)."""
        self.model.fit(data["X"], data["y"])
        return self

    def get_parameters(self):
        """Serialize trees and metadata for federated aggregation."""
        if not hasattr(self.model, "estimators_") or self.model.estimators_ is None:
            return []

        serialized_trees = [pickle.dumps(est) for est in self.model.estimators_]
        metadata = {
            "n_features_in_": self.model.n_features_in_,
            "n_outputs_": getattr(self.model, "n_outputs_", 1),
            "event_times_": getattr(self.model, "event_times_", None),
            "max_features_": getattr(self.model, "max_features_", None),
            "unique_times_": getattr(self.model, "unique_times_", None)
        }
        serialized_metadata = pickle.dumps(metadata)

        return serialized_trees + [serialized_metadata]

    def set_parameters(self, params_list):
        """Restore aggregated trees and metadata."""
        if not params_list:
            return

        try:
            # Restore trees
            self.model.estimators_ = [pickle.loads(est) for est in params_list[:-1]]
            self.model.n_estimators = len(self.model.estimators_)

            # Restore metadata
            metadata = pickle.loads(params_list[-1])
            self.model.n_features_in_ = metadata.get("n_features_in_", 0)
            self.model.n_outputs_ = metadata.get("n_outputs_", 1)
            self.model.event_times_ = metadata.get("event_times_", None)
            self.model.max_features_ = metadata.get("max_features_", None)
            self.model.unique_times_ = metadata.get("unique_times_", None)

            # Global event grid if present
            self.global_event_times_ = metadata.get("global_event_times_", None)

            print(f"[RSFModel] Restored {self.model.n_estimators} trees with {self.model.n_features_in_} features.")

        except Exception as e:
            print(f"[RSFModel] Error restoring RSF trees and metadata: {e}")

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return predicted risk scores (negative of survival)."""
        return self.model.predict(X)

    def predict_survival(self, X):
        """Federated-safe survival prediction with proper interpolation to global grid."""
        if not hasattr(self.model, "estimators_") or self.model.estimators_ is None:
            raise ValueError("Model has no trained trees.")

        # --- Determine common time grid ---
        if self.global_event_times_ is not None:
            time_grid = np.asarray(self.global_event_times_, dtype=float)
        else:
            # fallback: local event times from first tree
            time_grid = np.asarray([fn.x for fn in self.model.estimators_[0].predict_survival_function(X)]).flatten()

        # --- Interpolate all trees to the common grid ---
        all_survs = []
        for est in self.model.estimators_:
            tree_survs = est.predict_survival_function(X)
            for fn in tree_survs:
                f_interp = interp1d(fn.x, fn.y, bounds_error=False, fill_value=(1.0, 0.0))
                all_survs.append(f_interp(time_grid))

        # --- Average survival across trees ---
        n_samples = len(tree_survs)
        surv_matrix = np.mean(
            np.row_stack(all_survs).reshape(len(self.model.estimators_), n_samples, len(time_grid)),
            axis=0
        )

        # Return as list of Series
        return [pd.Series(surv_matrix[i], index=time_grid) for i in range(n_samples)]


    def evaluate(self, data: dict, client_id=None):
        """
        Federated-safe evaluation for RSF.
        Computes concordance index, Brier score, and Integrated Brier Score (IBS)
        using interpolated survival functions on a unified global time grid.
        """
        X_test = data["X_test"]
        y_test = data["y_test"]
        duration_col = data["duration_col"]
        event_col = data["event_col"]

        # --- Prepare structured y ---
        if isinstance(y_test, np.ndarray) and y_test.dtype.names is not None:
            y_test_df = pd.DataFrame({name: y_test[name] for name in y_test.dtype.names})
        else:
            y_test_df = y_test

        y_test_struct = Surv.from_dataframe(event_col, duration_col, y_test_df)

        # --- Primary metric: Concordance Index ---
        try:
            pred_risk = self.predict_risk(X_test)
            c_index = concordance_index_censored(
                y_test_struct[event_col],
                y_test_struct[duration_col],
                -pred_risk
            )[0]
        except Exception as e:
            print(f"[RSFModel] Could not compute concordance index: {e}")
            c_index = np.nan

        # --- Survival predictions ---
        try:
            surv_funcs = self.predict_survival(X_test)
        except Exception as e:
            print(f"[RSFModel] Could not compute survival functions: {e}")
            return {"c_index": float(c_index), "brier_score": np.nan, "ibs": np.nan}

        # --- Unified time grid clipped to test follow-up ---
        time_grid = np.asarray(surv_funcs[0].index, dtype=float)
        t_min = y_test_df[duration_col].min()
        t_max = y_test_df[duration_col].max()
        time_grid = time_grid[(time_grid >= t_min) & (time_grid <= t_max)]
        if len(time_grid) == 0:
            time_grid = np.linspace(t_min, t_max, 50)
        time_grid = np.unique(time_grid)

        # --- Convert survival functions to matrix ---
        try:
            surv_preds = np.row_stack([fn.values for fn in surv_funcs])
            if surv_preds.shape[1] != len(time_grid):
                # Interpolate if mismatch (safety)
                surv_preds_interp = []
                for fn in surv_funcs:
                    f = interp1d(fn.index, fn.values, bounds_error=False, fill_value=(1.0, 0.0))
                    surv_preds_interp.append(f(time_grid))
                surv_preds = np.row_stack(surv_preds_interp)
        except Exception as e:
            print(f"[RSFModel] Could not convert survival functions to matrix: {e}")
            return {"c_index": float(c_index), "brier_score": np.nan, "ibs": np.nan, 'accuracy': float(c_index)}

        # --- Integrated Brier Score ---
        try:
            ibs = integrated_brier_score(y_test_struct, y_test_struct, surv_preds, time_grid)
        except Exception as e:
            print(f"[RSFModel] Warning: could not compute IBS: {e}")
            ibs = np.nan

        # --- Brier Score at median time ---
        t_eval = np.median(time_grid)
        try:
            idx = np.argmin(np.abs(time_grid - t_eval))
            surv_at_t = surv_preds[:, idx].reshape(-1, 1)
            _, brier_arr = brier_score(y_test_struct, y_test_struct, surv_at_t, [time_grid[idx]])
            brier = float(np.mean(brier_arr))
        except Exception as e:
            print(f"[RSFModel] Warning: could not compute Brier at median time: {e}")
            brier = np.nan

        results = {"c_index": float(c_index), "brier_score": float(brier), "ibs": float(ibs), 'accuracy': float(c_index)}
        print(f"[RSFModel] Evaluation results: {results}")
        return results

    def save_model(self, path: str):
        """Save the model parameters to the specified path."""
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(self.get_parameters(), f)

    def load_model(self, path: str):
        """Load the model parameters from the specified path."""
        with open(path, 'rb') as f:
            import pickle
            self.set_parameters(pickle.load(f))

