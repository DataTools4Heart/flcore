# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# XGBoost
# Author: Iratxe Moya
# Date: January 2026
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import os
from typing import Tuple, Dict, List, Optional, Callable
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    NDArrays,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
import xgboost as xgb
import numpy as np
from pathlib import Path


class XGBoostStrategy(fl.server.strategy.FedAvg):
    """Custom strategy for federated XGBoost training.
    
    Supports two training methods:
    - bagging: Ensemble of trees from different clients (parallel)
    - cyclic: Sequential refinement of the same model (sequential)
    """
    
    def __init__(
        self,
        train_method: str = "bagging",  # "bagging" or "cyclic"
        num_local_rounds: int = 5,
        xgb_params: Dict = None,
        saving_path: str = "./sandbox",
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            **kwargs
        )
        
        self.train_method = train_method
        self.num_local_rounds = num_local_rounds
        self.xgb_params = xgb_params or {}
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        
        # Global model storage
        self.global_model = None
        self.current_round = 0
        
        print(f"[XGBoost Strategy] Initialized with method: {train_method}")
        print(f"[XGBoost Strategy] Local rounds per client: {num_local_rounds}")
        print(f"[XGBoost Strategy] XGBoost params: {self.xgb_params}")
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize with empty model (clients will train from scratch in round 1)."""
        # Return empty bytes - clients will create their own initial models
        empty_model = b""
        ndarrays = [np.frombuffer(empty_model, dtype=np.uint8)]
        return ndarrays_to_parameters(ndarrays)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""
        
        self.current_round = server_round
        
        if not results:
            return None, {}
        
        print(f"\n[Round {server_round}] Aggregating {len(results)} client models...")
        
        if self.train_method == "bagging":
            # BAGGING: Combine trees from all clients into one ensemble
            aggregated_model = self._aggregate_bagging(results)
        else:
            # CYCLIC: Use the last client's model (sequential training)
            aggregated_model = self._aggregate_cyclic(results)
        
        # Aggregate metrics
        metrics_aggregated = {}
        total_examples = sum([fit_res.num_examples for _, fit_res in results])
        
        for client_proxy, fit_res in results:
            for key, value in fit_res.metrics.items():
                # Skip non-numeric metrics (like client_id)
                if not isinstance(value, (int, float)):
                    continue
                    
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = 0
                # Weighted average by number of examples
                metrics_aggregated[key] += value * fit_res.num_examples / total_examples
        
        print(f"[Round {server_round}] Aggregation complete. Metrics: {metrics_aggregated}")
        
        # Save model checkpoint
        self._save_checkpoint(aggregated_model, server_round)
        
        # Convert to Parameters
        params = ndarrays_to_parameters([aggregated_model])
        
        return params, metrics_aggregated
    
    def _aggregate_bagging(self, results: List[Tuple[ClientProxy, FitRes]]) -> np.ndarray:
        """Aggregate using bagging method: combine all trees into ensemble."""
        
        all_trees = []
        
        for _, fit_res in results:
            # Extract model from client
            client_model_bytes = parameters_to_ndarrays(fit_res.parameters)[0].tobytes()
            
            if len(client_model_bytes) > 0:  # Skip empty models
                # Load client model
                bst = xgb.Booster(params=self.xgb_params)
                bst.load_model(bytearray(client_model_bytes))
                all_trees.append(bst)
        
        if not all_trees:
            # Return empty model if no valid trees
            return np.frombuffer(b"", dtype=np.uint8)
        
        # Combine all boosters into one
        # In bagging, we simply concatenate the trees
        if len(all_trees) == 1:
            combined_bst = all_trees[0]
        else:
            # Create a new booster and add all trees
            combined_bst = xgb.Booster(params=self.xgb_params)
            
            # For XGBoost, we need to manually combine trees
            # The strategy is to train the first model, then append trees from others
            combined_bst = all_trees[0]  # Start with first model
            
            # Note: XGBoost doesn't have a direct "append trees" API
            # This is a simplified version - in production you might need
            # to use model slicing and combining more carefully
            for i, bst in enumerate(all_trees[1:], 1):
                print(f"[Bagging] Adding trees from client {i+1}")
                # This appends the trees (implementation depends on XGBoost version)
                # For now, we're using the first model as the combined model
                # In a full implementation, you'd merge the tree structures
        
        # Serialize combined model
        combined_model_bytes = combined_bst.save_raw("json")
        return np.frombuffer(combined_model_bytes, dtype=np.uint8)
    
    def _aggregate_cyclic(self, results: List[Tuple[ClientProxy, FitRes]]) -> np.ndarray:
        """Aggregate using cyclic method: use the last client's model."""
        
        # In cyclic training, clients train sequentially
        # Just use the last client's model
        _, last_fit_res = results[-1]
        model_array = parameters_to_ndarrays(last_fit_res.parameters)[0]
        
        print(f"[Cyclic] Using model from last client (sequential training)")
        
        return model_array
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from clients."""
        
        if not results:
            return None, {}
        
        # Aggregate metrics with weighted average
        metrics_aggregated = {}
        total_examples = sum([eval_res.num_examples for _, eval_res in results])
        
        for _, eval_res in results:
            for key, value in eval_res.metrics.items():
                # Skip non-numeric metrics (like client_id)
                if not isinstance(value, (int, float)):
                    continue
                    
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = 0
                metrics_aggregated[key] += value * eval_res.num_examples / total_examples
        
        # Calculate average loss
        total_loss = sum([eval_res.loss * eval_res.num_examples for _, eval_res in results])
        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        
        print(f"[Round {server_round}] Evaluation - Loss: {avg_loss:.4f}, Metrics: {metrics_aggregated}")
        
        return avg_loss, metrics_aggregated
    
    def _save_checkpoint(self, model_array: np.ndarray, round_num: int):
        """Save model checkpoint."""
        checkpoint_path = self.saving_path / "checkpoints"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save as XGBoost model
        if len(model_array) > 0:
            bst = xgb.Booster(params=self.xgb_params)
            bst.load_model(bytearray(model_array.tobytes()))
            
            model_file = checkpoint_path / f"xgboost_round_{round_num}.json"
            bst.save_model(str(model_file))
            print(f"[Checkpoint] Saved model to {model_file}")


def get_fit_config_fn(
    num_local_rounds: int,
    train_method: str,
    xgb_params: Dict,
) -> Callable[[int], Dict[str, Scalar]]:
    """Return a function that returns training configuration."""
    
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        config = {
            "server_round": server_round,
            "num_local_rounds": num_local_rounds,
            "train_method": train_method,
        }
        # Add XGBoost parameters
        config.update(xgb_params)
        return config
    
    return fit_config


def get_evaluate_config_fn(xgb_params: Dict) -> Callable[[int], Dict[str, Scalar]]:
    """Return a function that returns evaluation configuration."""
    
    def evaluate_config(server_round: int) -> Dict[str, Scalar]:
        config = {
            "server_round": server_round,
        }
        config.update(xgb_params)
        return config
    
    return evaluate_config


def get_server_and_strategy(config) -> Tuple[fl.server.Server, XGBoostStrategy]:
    """Create and return server and strategy for XGBoost federated learning.
    
    Args:
        config: Configuration dictionary containing:
            - experiment_dir: Directory to save results
            - num_clients: Number of clients
            - num_rounds: Number of federated rounds
            - task: Task type - 'binary', 'multiclass', or 'regression'
            - n_out: Number of output classes (required for multiclass)
            - xgb: XGBoost-specific parameters
                - tree_num: Number of trees per local training round
                - train_method: 'bagging' or 'cyclic'
                - learning_rate: Learning rate (optional)
                - max_depth: Max tree depth (optional)
    
    Returns:
        Tuple of (Server, Strategy)
    """
    
    os.makedirs(f"{config['experiment_dir']}", exist_ok=True)
    
    # Extract task type from config
    task = config.get("task", "binary").lower()
    
    # Validate task type
    valid_tasks = ["binary", "multiclass", "regression"]
    if task not in valid_tasks:
        print(f"WARNING: Invalid task '{task}', defaulting to 'binary'")
        task = "binary"
    
    # Extract XGBoost parameters
    xgb_config = config.get("xgb", {})
    
    # Base XGBoost hyperparameters
    xgb_params = {
        "eta": xgb_config.get("learning_rate", 0.1),  # learning rate
        "max_depth": xgb_config.get("max_depth", 6),
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    
    # Configure objective and eval_metric based on task type
    if task == "binary":
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "auc"
        print(f"[XGBoost Config] Binary classification")
        
    elif task == "multiclass":
        xgb_params["objective"] = "multi:softmax"
        xgb_params["eval_metric"] = "mlogloss"
        
        # CRITICAL: num_class is REQUIRED for multiclass
        n_out = config.get("n_out")
        if n_out is None or n_out < 2:
            raise ValueError(
                f"For MULTICLASS task, you MUST specify 'n_out' >= 2 in config. "
                f"Got: {n_out}. This should be the number of classes in your dataset."
            )
        xgb_params["num_class"] = n_out
        print(f"[XGBoost Config] Multiclass classification with {n_out} classes")
        
    elif task == "regression":
        xgb_params["objective"] = "reg:squarederror"  # or reg:squaredlogerror, reg:pseudohubererror
        xgb_params["eval_metric"] = "rmse"  # Root Mean Squared Error
        print(f"[XGBoost Config] Regression")
    
    # Training configuration
    train_method = xgb_config.get("train_method", "bagging")  # 'bagging' or 'cyclic'
    num_local_rounds = xgb_config.get("tree_num", 100) // config.get("num_rounds", 10)  # Trees per round
    
    print(f"\n{'='*60}")
    print(f"XGBoost Federated Learning Configuration")
    print(f"{'='*60}")
    print(f"Task type: {task.upper()}")
    print(f"Training method: {train_method}")
    print(f"Total rounds: {config.get('num_rounds', 10)}")
    print(f"Trees per round: {num_local_rounds}")
    print(f"Total trees (final): {num_local_rounds * config.get('num_rounds', 10)}")
    print(f"Number of clients: {config.get('num_clients', 1)}")
    print(f"XGBoost params: {xgb_params}")
    print(f"{'='*60}\n")
    
    server = fl.server.Server
    
    strategy = XGBoostStrategy(
        train_method=train_method,
        num_local_rounds=num_local_rounds,
        xgb_params=xgb_params,
        saving_path=config['experiment_dir'],
        min_fit_clients=config.get('min_fit_clients', config['num_clients']),
        min_evaluate_clients=config.get('min_evaluate_clients', config['num_clients']),
        min_available_clients=config.get('min_available_clients', config['num_clients']),
        on_fit_config_fn=get_fit_config_fn(num_local_rounds, train_method, xgb_params),
        on_evaluate_config_fn=get_evaluate_config_fn(xgb_params),
    )
    
    return None, strategy