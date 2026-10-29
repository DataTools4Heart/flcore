# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Uncertainty-Aware Neural Network
# Author: Jorge Fabila Fabian
# Fecha: September 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
import warnings
import flcore.models.linear_models.utils as utils
import flwr as fl
from sklearn.metrics import log_loss
from flcore.performance import measurements_metrics, get_metrics
from flcore.metrics import calculate_metrics
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ______________________________________________________________

import sys
import math
import json
import torch
import shutil
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple

from pathlib import Path

from collections import OrderedDict

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from flcore.models.nn.basic_nn import BasicNN
from flcore.models.nn.utils import uncertainty_metrics
from flcore.metrics import calculate_metrics

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config, data):
        self.config = config
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.epochs = config["local_epochs"]

        print("MODELS::NN:CLIENT::INIT")
        if torch.cuda.is_available() and self.config["device"] == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        self.X_train = torch.tensor(self.X_train.values, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test.values, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)

        # Criterio y conversión de tipos ANTES de crear datasets/loaders
        if self.config["task"] == "classification":
            if config["n_out"] == 1:  # Binario
                self.criterion = nn.BCEWithLogitsLoss()
            else:  # Multiclase
                self.criterion = nn.CrossEntropyLoss()
                self.y_train = self.y_train.long()
                self.y_test = self.y_test.long()
        elif self.config["task"] == "regression":
            pen = str(self.config.get("penalty", "l2")).lower()
            self.criterion = {
                "l1": nn.L1Loss(),
                "l2": nn.MSELoss(),
                "smooth_l1": nn.SmoothL1Loss(),
            }.get(pen, nn.MSELoss())
        else:
            raise ValueError(f"task desconocida: {self.config['task']}")

        train_ds = TensorDataset(self.X_train, self.y_train)
        test_ds = TensorDataset(self.X_test, self.y_test)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model = BasicNN(config["n_feats"], config["n_out"], config["dropout_p"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.round = 0
        self.last_metrics = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _forward_loss(self, X, y):
        """Forward + loss + predicciones, usando self.criterion."""
        out = self.model(X)
        if self.config["task"] == "classification":
            if self.config["n_out"] == 1:
                out = out.squeeze(1)
                loss = self.criterion(out, y)
                preds = (torch.sigmoid(out) > 0.5).long()
            else:
                loss = self.criterion(out, y)
                preds = out.argmax(dim=1)
        else:  # regression
            out = out.squeeze(-1) if out.dim() > y.dim() else out
            loss = self.criterion(out, y)
            preds = out
        return loss, preds

    def _predict(self, X):
        """Predicciones sobre X como numpy (formato tipo sklearn.predict)."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(X.to(self.device))
            if self.config["task"] == "classification":
                if self.config["n_out"] == 1:
                    preds = (torch.sigmoid(out.squeeze(1)) > 0.5).long()
                else:
                    preds = torch.argmax(out, dim=1)
            else:
                preds = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
        return preds.cpu().numpy()

    def _mean_entropy(self, X):
        """Entropía predictiva media, normalizada a [0, 1]."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(X.to(self.device))
            if self.config["n_out"] == 1:
                p = torch.sigmoid(out.squeeze(1)).clamp(1e-7, 1 - 1e-7)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                max_ent = math.log(2)
            else:
                logp = F.log_softmax(out, dim=1)
                ent = -(logp.exp() * logp).sum(dim=1)
                max_ent = math.log(out.shape[1])
        return float(ent.mean().item() / max_ent)

    @staticmethod
    def _clean_metrics(metrics):
        """Flower solo admite float/int/str/bytes/bool: castea numpy y descarta el resto."""
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (str, bytes)):
                clean[k] = v
            elif isinstance(v, (bool, int, float, np.number)):
                clean[k] = float(v)
            else:
                print(f"[metrics] '{k}' descartada (tipo {type(v).__name__} no soportado)")
        return clean

    # ------------------------------------------------------------------
    # Parámetros
    # ------------------------------------------------------------------
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, parameters, config):
        try:
            start_time = time.time()
            self.set_parameters(parameters)

            for epoch in range(self.epochs):
                self.model.train()
                total_loss, correct, total = 0.0, 0, 0

                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    loss, preds = self._forward_loss(X, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() * X.size(0)
                    total += y.size(0)
                    if self.config["task"] == "classification":
                        correct += (preds == y).sum().item()

                msg = f"Epoch {epoch + 1:02d} | Train Loss: {total_loss / total:.4f}"
                if self.config["task"] == "classification":
                    msg += f", Train Acc: {correct / total:.4f}"
                print(msg)

            # ---- Métricas sobre test ----
            y_pred = self._predict(self.X_test)
            y_true = self.y_test.cpu().numpy()
            metrics = calculate_metrics(y_true, y_pred, self.config)
            metrics = {f"personalized {key}": metrics[key] for key in metrics}

            # Clave sin prefijo: es la que lee UncertaintyWeightedFedAvg
            if self.config["task"] == "classification":
                metrics["entropy"] = self._mean_entropy(self.X_test)

            elapsed_time = time.time() - start_time
            metrics["running_time"] = elapsed_time
            metrics = self._clean_metrics(metrics)
            self.last_metrics = metrics  # antes de save_model

            is_last = self.round == self.config["num_rounds"] - 1
            if self.round % self.config["save_every_n_rounds"] == 0 or is_last:
                self.save_model()

            dataset_len = self.y_train.shape[0]

            print(f"num_client {self.config.get('node_name')} has an elapsed time {elapsed_time}")
            print(f"Training finished for round {self.round}")

            self.round += 1
            return self.get_parameters(config={}), dataset_len, metrics
        except Exception as e:
            from flcore.utils import log_detailed_error
            X_diag = self.X_train.cpu().numpy() if hasattr(self, "X_train") and hasattr(self.X_train, "cpu") else None
            y_diag = self.y_train.cpu().numpy() if hasattr(self, "y_train") and hasattr(self.y_train, "cpu") else None
            log_detailed_error("Model Fitting (Local Training)", e, config=getattr(self, "config", None), X=X_diag, y=y_diag)
            raise e

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            self.model.eval()
            y_true = self.y_test.cpu().numpy()

            y_pred = self._predict(self.X_test)
            metrics = calculate_metrics(y_true, y_pred, self.config)

            # MC Dropout only makes sense with class probabilities
            if self.config["dropout_p"] > 0.0 and self.config["task"] == "classification":
                metrics.update(uncertainty_metrics(
                    self.model, self.val_loader, device=self.device, T=int(self.config["T"])
                ))
                self.model.eval()  # uncertainty_metrics leaves dropout active

            total_loss, total = 0.0, 0
            with torch.no_grad():
                for X, y in self.test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    loss, _ = self._forward_loss(X, y)
                    total_loss += loss.item() * X.size(0)
                    total += y.size(0)

            test_loss = total_loss / total
            dataset_len = self.y_test.shape[0]
            return float(test_loss), dataset_len, self._clean_metrics(metrics)
        except Exception as e:
            from flcore.utils import log_detailed_error
            X_diag = self.X_test.cpu().numpy() if hasattr(self, "X_test") and hasattr(self.X_test, "cpu") else None
            y_diag = self.y_test.cpu().numpy() if hasattr(self, "y_test") and hasattr(self.y_test, "cpu") else None
            log_detailed_error("Model Evaluation (Local Validation)", e, config=getattr(self, "config", None), X=X_diag, y=y_diag)
            raise e

    # ------------------------------------------------------------------
    # Guardado
    # ------------------------------------------------------------------
    def save_model(self):
        save_path = Path(self.config["sandbox_path"]) / "model"
        save_path.mkdir(parents=True, exist_ok=True)

        current_round = getattr(self, "round", 0)
        is_final = current_round == self.config["num_rounds"] - 1

        model_name = f"{self.config['model']}_{self.config['task']}_round_{current_round}"

        with open(self.config["metadata_file"], "r") as f:
            data_metadata = json.load(f)

        entries = data_metadata.get("entries", [])
        if not entries:
            raise ValueError(f"'entries' vacío en {self.config['metadata_file']}")
        entity = entries[0]

        features_list = entity.get("features", [])
        outcomes_list = entity.get("outcomes", [])
        dataset_stats = entity.get("datasetStats", {})
        feature_stats = dataset_stats.get("featureStats", {})
        outcome_stats = dataset_stats.get("outcomeStats", {})

        all_features_meta = {f["name"]: f for f in features_list}
        all_outcomes_meta = {o["name"]: o for o in outcomes_list}

        for f_name, f_meta in all_features_meta.items():
            f_meta["stats"] = feature_stats.get(f_name, {})

        for o_name, o_meta in all_outcomes_meta.items():
            o_meta["stats"] = outcome_stats.get(o_name, {})

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

        metadata = {
            "node_name": self.config["node_name"],
            "task": self.config["task"],
            "n_out": self.config["n_out"],
            "n_feats": self.config["n_feats"],
            "model_type": self.config["model"],
            "feature_names": self.config["train_labels"],
            "target_names": self.config["target_labels"],
            "data_file": self.config["data_file"],
            "metrics": getattr(self, "last_metrics", None),
            "features_meta": features_meta,
            "outcomes_meta": outcomes_meta,
            "is_final": is_final,
            "round": current_round,
            "model_name": model_name,
            "torch_version": torch.__version__,
        }

        # Un solo archivo: state_dict + metadata
        model_path = save_path / f"{model_name}_model.pt"
        torch.save({"state_dict": self.model.state_dict(), "metadata": metadata}, model_path)
        print(f"Model and metadata saved for inference at {model_path}")

        if is_final:
            final_name = f"{self.config['model']}_{self.config['task']}"
            final_path = save_path / f"{final_name}_model_final.pt"
            shutil.copyfile(model_path, final_path)
            print(f"Final model marked at {final_path}")


def get_client(config, data) -> fl.client.Client:
    return FlowerClient(config, data)