# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import argparse
import os
import sys
import flwr as fl
from typing import Dict

from flcore.models.rsf.model import RSFModel
from flcore.models.rsf.data_formatter import get_numpy


class FLClient(fl.client.NumPyClient):
    def __init__(self, local_data: Dict, client_id: str = "client", saving_path: str = "/sandbox/"):
        self.model_wrapper = None  # will be set later
        self.local_data = local_data
        self.model_type = None  # will be set later
        self.id = client_id
        self.saving_path = saving_path
        os.makedirs(f"{self.saving_path}", exist_ok=True)
        os.makedirs(f"{self.saving_path}/models/", exist_ok=True)

    def get_parameters(self, config=None):
        if self.model_wrapper is None:
            return []
        return self.model_wrapper.get_parameters()

    def fit(self, parameters, config):
        # Get model type from server
    
        model_kwargs = {k: v for k, v in config.items() if k != "model_type"}
        if self.model_wrapper is None:
            self.model_wrapper = RSFModel(**model_kwargs)
            print(f"[Client] Initialized model type from server: rsf")

        if parameters:
            self.model_wrapper.set_parameters(parameters)

        data = self.local_data
        self.model_wrapper.fit(data)

        params = self.get_parameters()
        num_examples = data.get("num_examples", len(data.get("X", [])) if "X" in data else len(data.get("df")))
        return params, num_examples, {}

    def evaluate(self, parameters, config):
        model_kwargs = {k: v for k, v in config.items() if k != "model_type"}
        if self.model_wrapper is None:
            self.model_wrapper = RSFModel(**model_kwargs)
            print(f"[Client] Initialized model type from server (evaluate): rsf")

        if parameters:
            self.model_wrapper.set_parameters(parameters)

        data = self.local_data
        metrics = self.model_wrapper.evaluate(data)
        metrics['client_id'] = self.id

        num_examples = data.get("num_examples", len(data.get("X", [])) if "X" in data else len(data.get("df")))
        # Save model
        self.model_wrapper.save_model(f"{self.saving_path}/models/rsf.pkl")

        return 1 - metrics['c_index'], num_examples, metrics

def get_client(config, data, client_id="client") -> fl.client.Client:
    (X_train, y_train), (X_test, y_test), time, event = data
    local_data = get_numpy(X_train, y_train, X_test, y_test, time, event)
    return FLClient(local_data, client_id=client_id, saving_path=config["experiment_dir"])