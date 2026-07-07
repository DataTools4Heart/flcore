# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import pickle
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
from fpboost.models import FPBoost
from scipy.interpolate import interp1d
from flcore.models.gbs.base_model import BaseSurvivalModel

class GBSModel(BaseSurvivalModel):
    """
    Wrapper around FPBoost.FPBoost to be used in your federated client.
    """

    def __init__(self, n_estimators=100, learning_rate=0.01, random_state=42, **kwargs):
        print(f"[GBSModel] Initializing FPBoost with n_estimators={n_estimators}, lr={learning_rate}")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs

        # FPBoost signature in README: FPBoost(n_estimators=..., learning_rate=..., max_depth=..., random_state=...)
        self.model = FPBoost(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )

    def fit(self, data: dict):
        """
        Fit FPBoost on local client data (X, y).
        Expects y to be structured array / compatible with scikit-survival.
        """
        X = data["X"]
        y = data["y"]
        # FPBoost is scikit-survival compatible: directly fit
        self.model.fit(X, y)
        return self
    
    def evaluate(self, data: dict, client_id=None, round_id=None):
        """
        Safe evaluation for FPBoost (GBS) in a federated setting.
        Prevents IBS domain errors and ensures interpolation is valid.
        """

        X_test = data["X_test"]
        y_test = data["y_test"]
        duration_col = data["duration_col"]
        event_col = data["event_col"]

        # Convert structured array to DataFrame if needed
        if isinstance(y_test, np.ndarray) and y_test.dtype.names is not None:
            y_test_df = pd.DataFrame({name: y_test[name] for name in y_test.dtype.names})
        else:
            y_test_df = y_test

        # Structured survival array
        y_test_struct = Surv.from_dataframe(event_col, duration_col, y_test_df)

        # --- C-index ---
        pred_risk = self.model.predict(X_test)
        c_index = concordance_index_censored(
            y_test_struct[event_col],
            y_test_struct[duration_col],
            -pred_risk
        )[0]

        # Try survival prediction
        try:
            surv_funcs = self.model.predict_survival_function(X_test)
            has_surv = True
        except Exception as e:
            print(f"[GBSModel] Survival prediction unavailable: {e}")
            return {
                "c_index": float(c_index),
                "brier_score": np.nan,
                "ibs": np.nan,
                "n_estimators": getattr(self.model, "n_estimators", None),
            }

        # ---------------------------------------------------------------
        # ███ Safe GLOBAL IBS time grid computation
        # ---------------------------------------------------------------

        # Bounds of test follow-up (NOT the same as min/max durations!)
        follow_min = float(np.min(y_test_df[duration_col]))
        follow_max = float(np.max(y_test_df[duration_col]))

        # Domain of each predicted survival function
        domains_min = [float(fn.x[0]) for fn in surv_funcs]
        domains_max = [float(fn.x[-1]) for fn in surv_funcs]

        model_min = max(domains_min)  # Safe lower bound
        model_max = min(domains_max)  # Safe upper bound

        # IBS domain must satisfy: ibs_min < time < ibs_max
        ibs_min = max(follow_min, model_min)
        ibs_max = min(follow_max, model_max)

        # Ensure the upper bound is *strictly less* (open interval)
        ibs_max = ibs_max * 0.999999

        # If domain invalid → skip IBS
        if ibs_min >= ibs_max:
            print(f"[GBSModel] IBS skipped: invalid interval [{ibs_min}, {ibs_max}].")
            return {
                "c_index": float(c_index),
                "brier_score": np.nan,
                "ibs": np.nan,
                "n_estimators": getattr(self.model, "n_estimators", None),
            }

        # Create safe time grid fully inside the valid IBS domain
        time_grid = np.linspace(ibs_min, ibs_max, 200)

        # ---------------------------------------------------------------
        # ███ Interpolate survival curves onto safe time grid
        # ---------------------------------------------------------------

        surv_preds = []
        for fn in surv_funcs:
            f = interp1d(fn.x, fn.y, bounds_error=False, fill_value=(1.0, 0.0))
            surv_preds.append(f(time_grid))

        surv_preds = np.row_stack(surv_preds)

        # ---------------------------------------------------------------
        # ███ Compute IBS (always safe)
        # ---------------------------------------------------------------
        try:
            ibs = integrated_brier_score(
                y_test_struct,
                y_test_struct,
                surv_preds,
                time_grid
            )
        except Exception as e:
            print(f"[GBSModel] Warning: IBS failed even after strict clipping: {e}")
            ibs = np.nan

        # ---------------------------------------------------------------
        # ███ Brier Score at median of safe domain
        # ---------------------------------------------------------------
        t_eval = float(np.median(time_grid))
        try:
            idx = np.argmin(np.abs(time_grid - t_eval))
            surv_at_t = surv_preds[:, idx].reshape(-1, 1)
            _, brier_arr = brier_score(
                y_test_struct,
                y_test_struct,
                surv_at_t,
                [time_grid[idx]]
            )
            brier = float(np.mean(brier_arr))
        except Exception as e:
            print(f"[GBSModel] Warning: Brier computation failed at t={t_eval}: {e}")
            brier = np.nan

        # ---------------------------------------------------------------
        # ███ Final evaluation dictionary
        # ---------------------------------------------------------------
        results = {
            "c_index": float(c_index),
            "brier_score": float(brier),
            "ibs": float(ibs),
            "n_estimators": getattr(self.model, "n_estimators", None),
        }

        print(f"[GBSModel] Evaluation results: {results}")
        return results



    # -----------------------------
    # Federated parameter management
    # -----------------------------
    def get_parameters(self):
        """
        Serialize the FPBoost model object (pickle). Return a list to match your interface.
        """
        try:
            serialized_model = pickle.dumps(self.model)
            return [serialized_model]
        except Exception as e:
            print(f"[GBSModel] Serialization error: {e}")
            return []

    def set_parameters(self, params_list):
        """
        Deserialize the FPBoost model object sent from server.
        """
        if not params_list:
            print("[GBSModel] No parameters received to set.")
            return

        try:
            self.model = pickle.loads(params_list[0])
        except Exception as e:
            print(f"[GBSModel] Deserialization error: {e}")

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

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

    def explain(self, X, horizons=None):
        """
        Generate SHAP values for the GBS model (FPBoost).
        """
        import shap
        import pandas as pd
        
        if horizons is None:
            horizons = [1, 3, 5]
        
        try:
            # TreeExplainer is efficient for tree-based models
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            print(f"[GBSModel] TreeExplainer failed, falling back to Explainer: {e}")
            X_val = X.values if isinstance(X, pd.DataFrame) else X
            background = shap.sample(X_val, 100) if len(X_val) > 100 else X_val
            explainer = shap.Explainer(self.predict_risk, background)
            shap_values = explainer(X).values
            
        import numpy as np
        shap_values_np = np.array(shap_values)
        if shap_values_np.ndim > 2:
            shap_values_np = shap_values_np[:, :, 0]
            
        mean_abs_shap = np.abs(shap_values_np).mean(axis=0)
        total_shap = mean_abs_shap.sum()
        
        if total_shap > 0:
            relative_contribution = (mean_abs_shap / total_shap) * 100
        else:
            relative_contribution = np.zeros_like(mean_abs_shap)
            
        X_val = X.values if hasattr(X, "values") else X
        n_samples = X_val.shape[0]
        scores_at_horizons = {h: [] for h in horizons}
        
        try:
            from scipy.interpolate import interp1d
            surv_funcs = self.model.predict_survival_function(X) # pass original X in case of pd.DataFrame
            for fn in surv_funcs:
                for h in horizons:
                    f = interp1d(fn.x, fn.y, bounds_error=False, fill_value=(1.0, 0.0))
                    scores_at_horizons[h].append(float(f(h)))
        except Exception as e:
            print(f"[GBSModel] Fallback to risk scores for horizon evaluation: {e}")
            risk_scores = self.predict_risk(X)
            for h in horizons:
                scores_at_horizons[h] = risk_scores.tolist()

        results = []
        for i in range(n_samples):
            patient_result = []
            for h in horizons:
                patient_result.append({
                    "horizon": str(h),
                    "score": scores_at_horizons[h][i],
                    "shap_data": shap_values_np[i].tolist(),
                    "contribution_data": relative_contribution.tolist(),
                    "whatever_data": [],
                    "distribution_data": []
                })
            results.append(patient_result)
            
        return results if n_samples > 1 else results[0]
