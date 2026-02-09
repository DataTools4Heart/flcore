# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple
from flcore.models.cox.base_model import BaseSurvivalModel

class CoxPHModel(BaseSurvivalModel):
    """
    Implements the Cox Proportional Hazards model from scratch using
    Newton-Raphson optimization (via SciPy) of the partial log-likelihood.
    
    The max_iter is intentionally kept low (e.g., 5) to force partial updates.
    Supports L1 (Lasso) regularization.
    """
    
    def __init__(self, max_iter: int = 5, tol: float = 1e-1, verbose: bool = True, 
                 l1_penalty: float = 0.0):
        """
        Parameters:
        -----------
        max_iter : int
            Maximum number of optimization iterations per fit call
        tol : float
            Tolerance for optimization convergence
        verbose : bool
            Flag to control print statements
        l1_penalty : float
            L1 regularization strength (lambda). Default 0.0 means no regularization.
            Higher values increase regularization strength.
        """
        self.max_iter = max_iter  
        self.tol = tol           
        self.verbose = verbose
        self.l1_penalty = l1_penalty
        
        self.beta: Optional[np.ndarray] = None

    def _compute_nll_grad_hess(self, 
                               beta: np.ndarray, 
                               X: np.ndarray, 
                               time: np.ndarray, 
                               event: np.ndarray
                              ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes negative log-likelihood, gradient, and Hessian with L1 regularization.
        
        Note: L1 penalty is not differentiable at 0, so we use a smooth approximation
        for the gradient. The Hessian doesn't include L1 term as it would be 0 everywhere
        except at beta=0 where it's undefined.
        """
        n_samples, n_features = X.shape
        sort_idx = np.argsort(time)
        X_sorted, event_sorted = X[sort_idx], event[sort_idx]
        eta = X_sorted @ beta
        exp_eta = np.exp(eta)
        
        # Base negative log-likelihood
        nll = 0.0
        grad = np.zeros(n_features)
        hess = np.zeros((n_features, n_features))
        S0 = 0.0  
        S1 = np.zeros(n_features) 
        S2 = np.zeros((n_features, n_features))
        
        for i in range(n_samples - 1, -1, -1):
            exp_eta_i = exp_eta[i]
            X_i = X_sorted[i, :]
            
            S0 += exp_eta_i
            S1 += exp_eta_i * X_i
            S2 += exp_eta_i * np.outer(X_i, X_i)
            
            if event_sorted[i]:
                E1 = S1 / S0
                nll -= (eta[i] - np.log(S0))
                grad -= (X_i - E1)
                E2 = S2 / S0
                hess += (E2 - np.outer(E1, E1))
        
        # Add L1 regularization
        if self.l1_penalty > 0:
            # L1 penalty term: lambda * ||beta||_1
            nll += self.l1_penalty * np.sum(np.abs(beta))
            
            # Gradient of L1: lambda * sign(beta)
            # Using smooth approximation to avoid issues at beta=0
            epsilon = 1e-8
            grad += self.l1_penalty * (beta / (np.abs(beta) + epsilon))
            
            # Hessian doesn't change (L1 second derivative is 0 almost everywhere)
                
        return nll, grad, hess

    def _objective_func(self, beta, X, time, event):
        """Wrapper for SciPy optimizer to return NLL and Gradient."""
        nll, grad, _ = self._compute_nll_grad_hess(beta, X, time, event)
        return nll, grad

    def _hessian_func(self, beta, X, time, event):
        """Wrapper for SciPy optimizer to return Hessian."""
        _, _, hess = self._compute_nll_grad_hess(beta, X, time, event)
        return hess

    def get_parameters(self) -> List[np.ndarray]:
        """Returns the model parameters (coefficients) as a list of numpy arrays."""
        if self.beta is None:
            return []
        
        if self.verbose:
            print(f"[CoxPHModel] GET_PARAMS: Returning beta (shape {self.beta.shape}) to server.")
            print(f"    Snippet: {self.beta[:3]}") 
            
        return [self.beta]

    def set_parameters(self, params: List[np.ndarray]):
        """Sets the model parameters from a list of numpy arrays."""
        if not params:
            if self.verbose:
                print("[CoxPHModel] SET_PARAMS: Called with empty list. Model weights not set.")
            return
        
        self.beta = params[0]
        
        if self.verbose:
            print(f"[CoxPHModel] SET_PARAMS: Global beta received (shape {self.beta.shape}).")
            print(f"    Snippet: {self.beta[:3]}")

    def fit(self, data: dict):
        """Runs one round of optimization to fit the CoxPH model (partial update)."""
        
        # 1. Extract data
        X_df = data['X']
        y = data['y']
        event_col_name = data['event_col']
        time_col_name = data['duration_col']
        
        # 2. Convert to NumPy arrays
        X = X_df.values.astype(np.float64)
        event = y[event_col_name].astype(bool)
        time = y[time_col_name].astype(np.float64)
        
        # 3. Initialize parameters if this is the first run
        if self.beta is None:
            n_features = X.shape[1]
            self.beta = np.zeros(n_features)
            if self.verbose:
                print(f"[CoxPHModel] FIT: Initializing with {n_features} features (zeros).")
                if self.l1_penalty > 0:
                    print(f"    L1 penalty: {self.l1_penalty}")
        
        # Verbose print before optimization
        if self.verbose:
            print(f"[CoxPHModel] FIT: Starting local train (max_iter={self.max_iter}).")
            print(f"    Initial beta snippet: {self.beta[:3]}")
        
        # 4. Run the optimizer
        try:
            result = minimize(
                fun=self._objective_func,
                x0=self.beta,
                args=(X, time, event),
                method='Newton-CG',
                jac=True,
                hess=self._hessian_func,
                options={
                    'maxiter': self.max_iter,
                    'disp': self.verbose
                },
                tol=self.tol
            )
            
            if self.verbose:
                print("\n--- Optimizer Result ---")
                print(f"Success: {result.success}")
                print(f"Status: {result.status}")
                print(f"Message: {result.message}")
                print(f"Actual Iterations: {result.nit}")
                print(f"Final NLL: {result.fun:.6f}")
                if self.l1_penalty > 0:
                    print(f"L1 norm of beta: {np.sum(np.abs(result.x)):.6f}")
                    print(f"Non-zero coefficients: {np.sum(np.abs(result.x) > 1e-4)}/{len(result.x)}")
                print("------------------------\n")
            
            # 5. Update the model parameters
            self.beta = result.x
            
            if self.verbose:
                print(f"[CoxPHModel] FIT: Local train finished.")
                print(f"    Final beta snippet: {self.beta[:3]}")

        except np.linalg.LinAlgError as e:
            print(f"Error during optimization (often singular Hessian): {e}")
        except Exception as e:
            print(f"An unexpected error occurred during fit: {e}")

    def evaluate(self, data: dict) -> Dict[str, float]:
        """
        Evalúa el modelo CoxPH devolviendo un reporte completo con varias métricas.
        """
        X_test_df = data.get('X_test', data['X'])
        y_test = data.get('y_test', data['y'])
        event_col = data['event_col']
        duration_col = data['duration_col']

        if self.beta is None:
            if self.verbose:
                print("[CoxPHModel] EVALUATE: Modelo no entrenado. Devolviendo métricas por defecto.")
            return {
                "c_index": 0.5,
                "permissible_pairs": 0.0,
                "neg_log_likelihood": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "event_rate": np.nan,
                "mean_risk_score": np.nan,
            }

        X = X_test_df.values.astype(np.float64)
        event = y_test[event_col].astype(bool)
        time = y_test[duration_col].astype(np.float64)

        # C-index calculation
        risk_scores = X @ self.beta
        n_concordant = 0.0
        n_permissible = 0.0
        n_samples = len(time)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                eta_i = risk_scores[i]
                eta_j = risk_scores[j]

                is_perm = False
                is_conc = False
                is_tied = (eta_i == eta_j)

                if (time[i] < time[j]) and event[i]:
                    is_perm = True
                    if eta_i > eta_j:
                        is_conc = True
                elif (time[j] < time[i]) and event[j]:
                    is_perm = True
                    if eta_j > eta_i:
                        is_conc = True

                if is_perm:
                    n_permissible += 1
                    if is_tied:
                        n_concordant += 0.5
                    elif is_conc:
                        n_concordant += 1.0

        c_index = 0.5 if n_permissible == 0 else n_concordant / n_permissible

        # Additional metrics
        eta = risk_scores
        exp_eta = np.exp(eta)
        nll = 0.0
        for i in range(n_samples):
            if event[i]:
                risk_set = exp_eta[time >= time[i]]
                nll -= (eta[i] - np.log(np.sum(risk_set)))
        
        # Add L1 penalty to NLL for consistency
        if self.l1_penalty > 0:
            nll += self.l1_penalty * np.sum(np.abs(self.beta))
        
        neg_log_likelihood = nll

        # Information criteria
        k = len(self.beta)
        n = len(time)
        AIC = 2 * k + 2 * neg_log_likelihood
        BIC = np.log(n) * k + 2 * neg_log_likelihood

        event_rate = float(np.mean(event))
        mean_risk = float(np.mean(risk_scores))

        results = {
            "c_index": float(c_index),
            "permissible_pairs": float(n_permissible),
            "neg_log_likelihood": float(neg_log_likelihood),
            "AIC": float(AIC),
            "BIC": float(BIC),
            "event_rate": float(event_rate),
            "mean_risk_score": float(mean_risk),
        }

        if self.verbose:
            print(f"[CoxPHModel] Evaluation results: {results}")

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

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise ValueError("Model not trained or parameters not loaded.")
        return X @ self.beta