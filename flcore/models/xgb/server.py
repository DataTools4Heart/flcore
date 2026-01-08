import flwr as fl
import numpy as np
import xgboost as xgb
from typing import Dict, Optional, List, Tuple

from datasets import load_dataset
from flwr.common import Parameters
from flwr.server.client_manager import ClientManager
from flcore.metrics import metrics_aggregation_fn

def fit_round( server_round: int ) -> Dict:
    """Send round number to client."""
    return { 'server_round': server_round }

def empty_parameters() -> Parameters:
    return fl.common.ndarrays_to_parameters(
        [np.frombuffer(b"", dtype=np.uint8)]
    )

def parameters_to_booster(parameters: Parameters, params: Dict) -> xgb.Booster:
    bst = xgb.Booster(params=params)
    raw = bytearray(parameters.tensors[0])
    if len(raw) > 0:
        bst.load_model(raw)
    return bst


def booster_to_parameters(bst: xgb.Booster) -> Parameters:
    raw = bst.save_raw("json")
    return fl.common.ndarrays_to_parameters(
        [np.frombuffer(raw, dtype=np.uint8)]
    )

class FedXgbStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        params: Dict,
        train_method: str,
        fraction_train: float,
        fraction_evaluate: float,
        test_dmatrix=None,
    ):
        self.params = params
        self.train_method = train_method
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.test_dmatrix = test_dmatrix

        self.global_bst: Optional[xgb.Booster] = None

    def initialize_parameters(self, client_manager: ClientManager):
        # Modelo vacío como en tu ejemplo
        return empty_parameters()

    def configure_fit(self, server_round, parameters, client_manager):
        num_clients = max(
            1, int(self.fraction_train * client_manager.num_available())
        )
        clients = client_manager.sample(num_clients)

        config = {"server-round": server_round}

        return [
            (client, fl.common.FitIns(parameters, config))
            for client in clients
        ]

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        if not results:
            return None, {}

        local_models = [
            parameters_to_booster(res.parameters, self.params)
            for _, res in results
        ]

        # --------- Bagging vs Cyclic ----------
        if self.global_bst is None:
            self.global_bst = local_models[0]

        else:
            if self.train_method == "bagging":
                # Concatenar árboles
                for bst in local_models:
                    self.global_bst = xgb.train(
                        params=self.params,
                        dtrain=None,
                        xgb_model=self.global_bst,
                        num_boost_round=bst.num_boosted_rounds(),
                    )
            else:
                # Cyclic: reemplazo completo
                self.global_bst = local_models[-1]

        return booster_to_parameters(self.global_bst), {}

    # -------------------------------------------------
    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.test_dmatrix is None:
            num_clients = max(
                1, int(self.fraction_evaluate * client_manager.num_available())
            )
            clients = client_manager.sample(num_clients)

            return [
                (client, fl.common.EvaluateIns(parameters, {}))
                for client in clients
            ]
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total = sum(r.num_examples for _, r in results)
        loss = sum(r.loss * r.num_examples for _, r in results) / total

        metrics = {}
        for _, r in results:
            for k, v in r.metrics.items():
                metrics[k] = metrics.get(k, 0.0) + v * r.num_examples

        for k in metrics:
            metrics[k] /= total

        return loss, metrics

    def evaluate(self, server_round, parameters):
        # ESTO NO TENDRIA QUE AGREGAR LAS METRICAS RECIBIDAS
        print("SERVER::EVALUATE::ENTRA")
        if self.test_dmatrix is None or server_round == 0:
            return None

        bst = parameters_to_booster(parameters, self.params)

        eval_results = bst.eval_set(
            evals=[(self.test_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        print("SERVER::EVALUATE::SALE")
        return 0.0, {"AUC": auc}

def get_server_and_strategy(config):
    if config["task"] == "classification":
        if config["n_out"] == 1: # Binario
            config["params"] = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": config["max_depth"],
                "eta": config["eta"],
                "tree_method": config["tree_method"],
                "subsample": config["test_size"],
                "colsample_bytree": 0.8,
                "tree_method": config["tree_method"],
                "seed": config["seed"],
            }
        elif config["n_out"] > 1: # Multivariable
            config["params"] = {
                "objective": "multi:softprob",
                "num_class": config["n_out"],
                "eval_metric": "mlogloss", # podria ser logloss
                "max_depth": config["max_depth"],
                "eta": config["eta"],
                "tree_method": config["tree_method"],
            }

    elif config["task"] == "regression":
            config["params"] = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": config["max_depth"],
                "eta": config["eta"],
                "tree_method": config["tree_method"],
            }

    strategy = FedXgbStrategy(
        params = config["params"],
        train_method = config["train_method"],
        fraction_train = config["train_size"],
        fraction_evaluate = config["validation_size"],
        test_dmatrix=None,
    )
    """
    min_available_clients = config['min_available_clients'],
    min_fit_clients = config['min_fit_clients'],
    min_evaluate_clients = config['min_evaluate_clients'],
    evaluate_metrics_aggregation_fn = metrics_aggregation_fn,
    on_fit_config_fn      = fit_round
    """


    """
    # El método dropout no está implementado. No creo que ni haga falta
    strategy.dropout_method = config['dropout_method']
    strategy.percentage_drop = config['dropout_percentage']
    strategy.smoothing_method = config['smooth_method']
    strategy.smoothing_strenght = config['smoothing_strenght']
    """
    return None, strategy