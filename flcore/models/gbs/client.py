# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

# src/client/client.py
"""
Federated Survival Analysis Flower client.
Supports multiple model types (Cox PH, RSF, GBS) via external model factory.

Usage:
    python client.py
"""

import os
import sys
import json
import joblib
import argparse
import flwr as f
from typing import Dict
from pathlib import Path

from flcore.models.gbs.model import GBSModel
from flcore.models.gbs.data_formatter import get_numpy


# -------------------------------
# Flower client definition
# -------------------------------

class FLClient(fl.client.NumPyClient):
    def __init__(self, local_data: Dict, client_id: str = "client", saving_path: str = "/sandbox/"):
        self.model_wrapper = None  # will be set later
        self.local_data = local_data
        self.id = client_id
        self.saving_path = saving_path
        self.round = 0
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
            self.model_wrapper = GBSModel(**model_kwargs)
            print(f"[Client] Initialized model type from server: gbs")

        if parameters:
            self.model_wrapper.set_parameters(parameters)

        data = self.local_data
        self.model_wrapper.fit(data)

        params = self.get_parameters()
        num_examples = data.get("num_examples", len(data.get("X", [])) if "X" in data else len(data.get("df")))

        if self.round % self.config["save_every_n_rounds"] == 0:
            self.save_model()

        self.round += 1
        return params, num_examples, {}

    def evaluate(self, parameters, config):
        model_kwargs = {k: v for k, v in config.items() if k != "model_type"}
        if self.model_wrapper is None:
            self.model_wrapper = GBSModel(**model_kwargs)
            print(f"[Client] Initialized model type from server (evaluate): gbs")

        if parameters:
            self.model_wrapper.set_parameters(parameters)

        data = self.local_data
        metrics = self.model_wrapper.evaluate(data)
        metrics['client_id'] = self.id

        num_examples = data.get("num_examples", len(data.get("X", [])) if "X" in data else len(data.get("df")))
        return 1 - metrics['c_index'], num_examples, metrics

    def save_model(self):
        save_path = Path(self.config["sandbox_path"])/"model"
        save_path.mkdir(parents=True, exist_ok=True)
        model_name = self.config["model"]+"_"+self.config["task"]+"_round_"+str(self.round)
        model_path = save_path / f"{model_name}_model.pkl"        
        self.model_wrapper.save_model(model_path)

        data_metadata = json.load(open(self.config["metadata_file"], "r"))
        entity = data_metadata.get("entries", {})[0]
        features_list = entity.get("features", [])
        outcomes_list = entity.get("outcomes", [])
        dataset_stats = entity.get("datasetStats", {})
        feature_stats = dataset_stats.get("featureStats", {})
        outcome_stats = dataset_stats.get("outcomeStats", {})

        all_features_meta = {f['name']: f for f in features_list}
        all_outcomes_meta = {o['name']: o for o in outcomes_list}

        for f_name, f_meta in all_features_meta.items():
            stats = feature_stats.get(f_name, {})
            f_meta['stats'] = stats

        for o_name, o_meta in all_outcomes_meta.items():
            stats = outcome_stats.get(o_name, {})
            o_meta['stats'] = stats

        features_meta = {}
        for label in self.config["train_labels"]:
            if label in all_features_meta:
                features_meta[label] = all_features_meta[label]
            elif label in all_outcomes_meta:
                features_meta[label] = all_outcomes_meta[label]

        outcomes_meta = {}
        for label in self.config["target_labels"]:
            if label in all_outcomes_meta:
                outcomes_meta[label] = all_outcomes_meta[label]
            elif label in all_features_meta:
                outcomes_meta[label] = all_features_meta[label]

#>>> features_meta["patient_demographics_age"]["stats"]["min"]
        metadata = {
            "node_name": self.config["node_name"],
            "task": self.config["task"],
            "n_out": self.config["n_out"],
            "n_out": self.config["n_feats"],
            "model_type": self.config["model"],
            "feature_names": self.config["train_labels"],
            "target_names":self.config["target_labels"],
            "metrics": getattr(self, "last_metrics", None),
            "features_meta": features_meta,
            "outcomes_meta": outcomes_meta
        }

        metadata_path = save_path / f"{model_name}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Model and metadata saved for inference at {save_path}")

def get_client(config, data, client_id="client") -> fl.client.Client:
    (X_train, y_train), (X_test, y_test), time, event = data
    local_data = get_numpy(X_train, y_train, X_test, y_test, time, event)
    return FLClient(local_data, client_id=client_id, saving_path=config["experiment_dir"])