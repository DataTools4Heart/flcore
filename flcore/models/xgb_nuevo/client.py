# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# XGBoost Client for Flower
# Author: Jorge Fabila Fabian
# Fecha: January 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import warnings
from typing import List, Tuple, Dict

import flwr as fl
import numpy as np
import xgboost as xgb

from flwr.common import Parameters
from sklearn.metrics import log_loss
from flcore.metrics import calculate_metrics
from sklearn.metrics import mean_squared_error
from xgboost_comprehensive.task import load_data, replace_keys
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)

warnings.filterwarnings("ignore", category=UserWarning)

def _local_boost(bst_input, num_local_round, train_dmatrix, train_method):
    for _ in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    if train_method == "bagging":
        bst = bst_input[
            bst_input.num_boosted_rounds() - num_local_round :
            bst_input.num_boosted_rounds()
        ]
    else:  # cyclic
        bst = bst_input

    return bst

class XGBFlowerClient(fl.client.NumPyClient):
    def __init__(self, data, config):
        self.config = config

        self.train_method = config["train_method"]
        self.seed = config["seed"]
        self.test_fraction = config["test_fraction"]
        self.num_local_round = config["local_epochs"]

        self.bst = None

        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        if self.config["task"] == "classification":
            if self.config["n_out"] == 1: # Binario
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
            elif self.config["n_out"] > 1: # Multivariable
                config["params"] = {
                    "objective": "multi:softprob",
                    "num_class": config["n_out"],
                    "eval_metric": "mlogloss", # podria ser logloss
                    "max_depth": config["max_depth"],
                    "eta": config["eta"],
                    "tree_method": config["tree_method"],
                }

        elif self.config["task"] == "regression":
                config["params"] = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "max_depth": config["max_depth"],
                    "eta": config["eta"],
                    "tree_method": config["tree_method"],
                }

    def get_parameters(self, config):
        if self.bst is None:
            return []
        raw = self.bst.save_raw("json")
        return [np.frombuffer(raw, dtype=np.uint8)]

    def set_parameters(self, parameters: List[np.ndarray]):
        if not parameters:
            return
        self.bst = xgb.Booster(params=self.params)
        raw = bytearray(parameters[0].tobytes())
        self.bst.load_model(raw)


    def fit(self, parameters, config):
        server_round = config.get("server_round", 1)

        if server_round == 1 or not parameters:
            self.bst = xgb.train(
                self.params,
                self.dtrain,
                num_boost_round=self.num_local_round,
            )
        else:
            self.set_parameters(parameters)

            self.bst = _local_boost(
                self.bst,
                self.num_local_round,
                self.dtrain,
                self.train_method,
            )

        params = self.get_parameters({})
        metrics = {"num_examples": len(self.y_train)}

        return params, len(self.y_train), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.config["task"] == "classification":
            if self.config["n_out"] == 1: # Binario
                y_pred_prob = self.bst.predict(self.dtest)
                y_pred = (y_pred_prob > 0.5).astype(int)
                loss = log_loss(self.y_test, y_pred_prob)
            elif self.config["n_out"] > 1: # Multivariable
                y_pred_prob = self.bst.predict(self.dtest)
                y_pred = y_pred_prob.argmax(axis=1)
                loss = log_loss(self.y_test, y_pred_prob)
        elif self.config["task"] == "regression":
                y_pred = self.bst.predict(self.dtest)
                loss = mean_squared_error(self.y_test, y_pred)

        metrics = calculate_metrics(self.y_test, y_pred, self.config)
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.X_test),
            metrics=metrics,
        )