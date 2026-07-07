"""
Fully Federated XGBoost - Flower Message-Based Server
"""

import json
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable, Iterable, Any

import numpy as np
import xgboost as xgb

from flwr.common import (
    ArrayRecord, 
    ConfigRecord, 
    Message, 
    MetricRecord, 
    RecordDict,
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server import Grid
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy


# ==========================================================
# BAGGING AGGREGATION (Tree-Level JSON Merge)
# ==========================================================

def _get_tree_nums(xgb_model_org: bytes):
    """Extract total tree numbers from XGBoost JSON model."""
    bst = json.loads(bytearray(xgb_model_org))
    model = bst["learner"]["gradient_booster"]["model"]
    tree_num = int(model["gbtree_model_param"]["num_trees"])
    paral_tree_num = int(model["gbtree_model_param"]["num_parallel_tree"])
    return tree_num, paral_tree_num

def aggregate_metricrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Perform weighted aggregation all MetricRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        # Because replies have been checked for consistency,
        # we can safely cast the weighting factor to float
        w = cast(float, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, weight_factors, strict=True):
        for record_item in record.metric_records.values():
            # aggregate in-place
            for key, value in record_item.items():
                if key == weighting_metric_name:
                    # We exclude the weighting key from the aggregated MetricRecord
                    continue
                if key not in aggregated_metrics:
                    if isinstance(value, list):
                        aggregated_metrics[key] = [v * weight for v in value]
                    else:
                        aggregated_metrics[key] = value * weight
                else:
                    if isinstance(value, list):
                        current_list = cast(list[float], aggregated_metrics[key])
                        aggregated_metrics[key] = [
                            curr + val * weight
                            for curr, val in zip(current_list, value, strict=True)
                        ]
                    else:
                        current_value = cast(float, aggregated_metrics[key])
                        aggregated_metrics[key] = current_value + value * weight

    return aggregated_metrics

def aggregate_bagging(
    bst_prev_org: bytes,
    bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if bst_prev_org == b"":
        return bst_curr_org

    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    previous_model = bst_prev["learner"]["gradient_booster"]["model"]
    previous_model["gbtree_model_param"]["num_trees"] = str(
        tree_num_prev + paral_tree_num_curr
    )

    iteration_indptr = previous_model["iteration_indptr"]
    previous_model["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]

    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        previous_model["trees"].append(trees_curr[tree_count])
        previous_model["tree_info"].append(0)

    return bytes(json.dumps(bst_prev), "utf-8")


# ==========================================================
# STRATEGY
# ==========================================================

class FedXgbFullyFederated(FedAvg):
    """Fully federated XGBoost strategy (bagging or cyclic)."""

    def __init__(
        self,
        num_local_rounds: int = 5,
        xgb_params: Dict = None,
        saving_path: str = "./sandbox",
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        train_method: str = "bagging",
        fraction_train=1.0,
        fraction_evaluate=1.0,

        # --> INHERITED
        **kwargs,
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
        self.xgb_params = xgb_params or {}
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)

        self.current_model: Optional[bytes] = b""

        print(f"[FedXgb] Training method: {train_method}")
        print(f"[FedXgb] XGBoost params: {self.xgb_params}")

    # ------------------------------------------------------
    # INITIALIZE
    # ------------------------------------------------------

    def initialize_parameters(self, client_manager):
        """Start with empty model."""
        empty = np.frombuffer(b"", dtype=np.uint8)
        return ndarrays_to_parameters([empty])

    # ------------------------------------------------------
    # AGGREGATE FIT (CRITICAL FIX)
    # ------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        print(f"\n[Round {server_round}] Aggregating {len(results)} clients")

        models: List[bytes] = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            model_bytes = ndarrays[0].tobytes()

            if model_bytes:
                models.append(model_bytes)

        if not models:
            return None, {}

        # -----------------------------------
        # BAGGING
        # -----------------------------------
        if self.train_method == "bagging":

            combined = self.current_model

            for m in models:
                combined = aggregate_bagging(combined, m)

        # -----------------------------------
        # CYCLIC
        # -----------------------------------
        else:
            combined = models[-1]

        self.current_model = combined

        # Save checkpoint
        self._save_checkpoint(combined, server_round)

        # Convert back to Parameters
        aggregated_params = ndarrays_to_parameters(
            [np.frombuffer(combined, dtype=np.uint8)]
        )

        # Weighted metric aggregation
        metrics_aggregated: Dict[str, Scalar] = {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        for _, fit_res in results:
            if "n_out" in fit_res.metrics:
                self.xgb_params["num_class"] = int(fit_res.metrics["n_out"])

            for key, value in fit_res.metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                metrics_aggregated.setdefault(key, 0.0)
                metrics_aggregated[key] += (
                    value * fit_res.num_examples / total_examples
                )

        print(f"[Round {server_round}] Aggregation done.")

        return aggregated_params, metrics_aggregated

    # ------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        total_examples = sum(eval_res.num_examples for _, eval_res in results)

        total_loss = sum(
            eval_res.loss * eval_res.num_examples
            for _, eval_res in results
        )

        if total_examples == 0:
            return None, {}

        avg_loss = total_loss / total_examples

        metrics_aggregated: Dict[str, Scalar] = {}

        for _, eval_res in results:
            for key, value in eval_res.metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                metrics_aggregated.setdefault(key, 0.0)
                metrics_aggregated[key] += (
                    value * eval_res.num_examples / total_examples
                )

        print(f"[Round {server_round}] Eval loss: {avg_loss:.4f}")

        return avg_loss, metrics_aggregated

    # ------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------

    def _save_checkpoint(self, model_bytes: bytes, round_num: int):

        if not model_bytes:
            return

        checkpoint_dir = self.saving_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        bst = xgb.Booster(params=self.xgb_params)
        bst.load_model(bytearray(model_bytes))

        model_path = checkpoint_dir / f"xgboost_round_{round_num}.json"
        bst.save_model(str(model_path))

        print(f"[Checkpoint] Saved {model_path}")


def get_fit_config_fn(
    num_local_rounds: int,
    train_method: str,
    xgb_params: Dict,
) -> Callable[[int], Dict[str, Any]]:
    """Return a function that returns training configuration."""
    
    def fit_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
            "num_local_rounds": num_local_rounds,
            "train_method": train_method,
        }
        # Add XGBoost parameters
        config.update(xgb_params)
        return config
    
    return fit_config

def get_evaluate_config_fn(xgb_params: Dict) -> Callable[[int], Dict[str, Any]]:
    """Return a function that returns evaluation configuration."""
    
    def evaluate_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
        }
        config.update(xgb_params)
        return config
    
    return evaluate_config

# ==========================================================
# SERVER FACTORY
# ==========================================================

def get_server_and_strategy(config: dict) -> FedXgbFullyFederated:
    """Create strategy from config dictionary."""

    os.makedirs(config["experiment_dir"], exist_ok=True)

    task = config.get("task", "binary").lower()
    xgb_config = config.get("xgb", {})

    xgb_params = {
        "eta": xgb_config.get("learning_rate", 0.1),
        "max_depth": xgb_config.get("max_depth", 6),
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    if task == "binary":
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "auc"

    elif task == "multiclass":
        n_out = config.get("n_out")
        xgb_params["objective"] = "multi:softmax"
        xgb_params["eval_metric"] = "mlogloss"
        if n_out is not None and n_out >= 2:
            xgb_params["num_class"] = n_out

    elif task == "regression":
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"

    train_method = xgb_config.get("train_method", "bagging")  # 'bagging' or 'cyclic'
    num_local_rounds = xgb_config.get("tree_num", 100) // config.get("num_rounds", 10)  # Trees per round
    

    print("\n" + "=" * 60)
    print("Federated XGBoost Configuration")
    print("=" * 60)
    print("Task:", task.upper())
    print("Train method:", xgb_config.get("train_method", "bagging"))
    print("Rounds:", config.get("num_rounds"))
    print("Clients:", config.get("num_clients"))
    print("XGBoost params:", xgb_params)
    print("=" * 60 + "\n")

    strategy = FedXgbFullyFederated(
        train_method=train_method,
        num_local_rounds=num_local_rounds,
        xgb_params=xgb_params,
        saving_path=config['experiment_dir'],
        min_fit_clients=config.get('min_fit_clients', config['num_clients']),
        min_evaluate_clients=config.get('min_evaluate_clients', config['num_clients']),
        min_available_clients=config.get('min_available_clients', config['num_clients']),
        on_fit_config_fn=get_fit_config_fn(num_local_rounds, train_method, xgb_params),
        on_evaluate_config_fn=get_evaluate_config_fn(xgb_params),
        fraction_train=1.0,
        fraction_evaluate=1.0
    )

    return None, strategy