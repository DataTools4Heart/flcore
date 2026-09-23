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
import flwr as fl
from typing import Dict
from pathlib import Path

from flcore.models.gbs.model import GBSModel
from flcore.models.gbs.data_formatter import get_numpy

import shutil
import pickle
import sklearn
try:
    import imblearn
except ImportError:
    imblearn = None

# -------------------------------
# Flower client definition
# -------------------------------

class FLClient(fl.client.NumPyClient):
    def __init__(self, local_data, config):
        self.config = config
        self.model_wrapper = None  # will be set later
        self.local_data = local_data
        self.id = config["node_name"]
        self.saving_path = config["experiment_dir"]
        self.round = 0
        os.makedirs(f"{self.saving_path}/models/", exist_ok=True)

    def get_parameters(self, config=None):
        if self.model_wrapper is None:
            return []
        return self.model_wrapper.get_parameters()

    def fit(self, parameters, config):
        try:
            # Get model type from server
            start_time = time.time()
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

            elapsed_time = (time.time() - start_time)
            metrics["running_time"] = elapsed_time

            print(f"num_client {self.node_name} has an elapsed time {elapsed_time}")
            print(f"Training finished for round {self.round}")

            self.round += 1
            return params, num_examples, {}
        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error(
                "Model Fitting (Local Training)",
                e,
                config=getattr(self, "config", None),
                X=self.local_data.get("X") if isinstance(self.local_data, dict) else None,
                y=self.local_data.get("y") if isinstance(self.local_data, dict) else None
            )
            raise e

    def evaluate(self, parameters, config):
        try:
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
        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error(
                "Model Evaluation (Local Validation)",
                e,
                config=getattr(self, "config", None),
                X=self.local_data.get("X_test") if isinstance(self.local_data, dict) else None,
                y=self.local_data.get("y_test") if isinstance(self.local_data, dict) else None
            )
            raise e

    def save_model(self):
        save_path = Path(self.config["experiment_dir"]) / "models"
        save_path.mkdir(parents=True, exist_ok=True)
        is_final = self.round == self.config["num_rounds"] - 1

        model_name = self.config["model"] + "_" + self.config["task"] + "_round_" + str(self.round)

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
            f_meta['stats'] = feature_stats.get(f_name, {})

        for o_name, o_meta in all_outcomes_meta.items():
            o_meta['stats'] = outcome_stats.get(o_name, {})

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

        aggregate = self.model_wrapper.model
        aggregate.n_outputs_ = self.config["n_out"]
        aggregate.n_features_in_ = self.config["n_feats"]
        n_classes = self.config.get("n_classes")
        if n_classes is not None:
            aggregate.n_classes_ = n_classes

        metadata = {
            "node_name": self.config["node_name"],
            "task": self.config["task"],
            "n_out": self.config["n_out"],
            "n_feats": self.config["n_feats"],
            "model_type": self.config["model"],
            "feature_names": self.config["train_labels"],
            "target_names": self.config["target_labels"],
            "metrics": getattr(self, "last_metrics", None),
            "features_meta": features_meta,
            "outcomes_meta": outcomes_meta,
            "is_final": is_final,
            "round": self.round,
            "model_name": model_name,
            "sklearn_version": sklearn.__version__,
            "imblearn_version": getattr(imblearn, "__version__", None) if imblearn else None,
        }

        model_path = save_path / f"{model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model_wrapper, "metadata": metadata}, f)

        print(f"Model and metadata saved for inference at {model_path}")

        if is_final:
            final_name = f"{self.config['model']}_{self.config['task']}"
            final_path = save_path / f"{final_name}_model_final.pkl"
            shutil.copyfile(model_path, final_path)
            print(f"Final model marked at {final_path}")

def get_client(config, data) -> fl.client.Client:
    (X_train, y_train), (X_test, y_test), time, event = data
    local_data = get_numpy(X_train, y_train, X_test, y_test, time, event)
    return FLClient(local_data, config)
