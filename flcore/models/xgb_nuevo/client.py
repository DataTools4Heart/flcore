# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# XGBoost Client for Flower
# Author: Jorge Fabila Fabian
# Fecha: January 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import flwr as fl
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from xgboost import XGBClassifier, XGBRegressor


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config, data):
        self.params = config

        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        # REVISA LAS VARIABLES NECESARIAS DESDE EL MAIN DEL CLIENTE
        # Se tendría que hacer lo mismo que en el random forest para clasificador regresor
        self.model = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config.get("subsample", 1.0),
            colsample_bytree=config.get("colsample_bytree", 1.0),
            objective="binary:logistic" if config["n_out"] == 1 else "multi:softmax",
            num_class=config["n_out"] if config["n_out"] > 1 else None,
            tree_method="hist"
        )


    def get_parameters(self, config):
        """Return model parameters as a list of numpy arrays."""
        booster = self.model.get_booster()
        return [np.frombuffer(booster.save_raw("json"), dtype=np.uint8)]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from list of numpy arrays."""
        raw = parameters[0].tobytes()
        self.model.load_model(raw)

    def fit(self, parameters, config):
        """Train XGBoost on local data."""
        self.set_parameters(parameters)

        self.model.fit(
            self.X_train,
            self.y_train,
            xgb_model=self.model,
            verbose=False
        )

        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        self.set_parameters(parameters)

        preds = self.model.predict(self.X_test)

        accuracy = (preds == self.y_test).mean()
        loss = float(np.mean((preds - self.y_test) ** 2))

        return loss, len(self.X_test), {"accuracy": float(accuracy)}

def get_client(config, data, client_id):
    return FlowerClient(config, data)
