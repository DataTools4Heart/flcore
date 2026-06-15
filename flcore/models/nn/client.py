# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Uncertainty-Aware Neural Network
# Author: Jorge Fabila Fabian
# Fecha: September 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import time
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
import json
import torch
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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config, data):
        self.config = config
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.epochs = config["local_epochs"]

        print("MODELS::NN:CLIENT::INIT")
        if torch.cuda.is_available() and self.config["device"] == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        self.X_train = torch.tensor(self.X_train.values, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test.values, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)

        train_ds = TensorDataset(self.X_train, self.y_train)
        test_ds = TensorDataset(self.X_test, self.y_test)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model = BasicNN( config["n_feats"], config["n_out"], config["dropout_p"] ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.round = 0

        if self.config["task"] == "classification":
            if config["n_out"] == 1:  # Binario
                self.criterion = nn.BCEWithLogitsLoss()
                #loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y)
                """
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs > 0.5).long()"""
            else:           # Multiclase
                self.criterion = nn.CrossEntropyLoss()
                self.y_train = self.y_train.long()
                self.y_test = self.y_test.long()
                #loss = F.cross_entropy(logits, y)
                #preds = torch.argmax(logits, dim=1)
            #return loss, preds
        elif self.config["task"] == "regression":
            if self.config["penalty"] == "l1":
                self.criterion = nn.L1Loss()
            elif self.config["penalty"] == "l2":
                self.criterion = nn.MSELoss()
            elif self.config["penalty"].lower() in ["smooth","smooth_l1","smoothl1"]:
                self.criterion = nn.SmoothL1Loss()

    def get_parameters(self, config): # config not needed at all
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters:List[np.ndarray]):
        self.model.train()
        # Si esto del self.model.train no funciona porque no reconoce la
        # función entonces deberías sustituírla por nuestra train:
        # train(self.model,params)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, params):
        try:
            self.set_parameters(parameters)
    # ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
            for epoch in range(self.epochs):
                self.model.train()
                total_loss, correct, total = 0, 0, 0

                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    if self.config["task"] == "classification":
                        logits = self.model(X)
                        if self.config["n_out"] == 1:  # Binario
                            loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y)
                            probs = torch.sigmoid(logits.squeeze(1))
                            preds = (probs > 0.5).long()
                        else:           # Multiclase
                            loss = F.cross_entropy(logits, y)
                            preds = torch.argmax(logits, dim=1)
                    elif self.config["task"] == "regression":
                        preds = self.model(X)
                        loss = F.mse_loss(preds, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() * X.size(0)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

                train_loss = total_loss / total
                train_acc = correct / total
                #test_loss, test_acc = self.evaluate()

                print(f"Epoch {epoch+1:02d} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ")
                #      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            dataset_len = self.y_train.shape[0]
    #       return get_weights(self.model), num_examples, metrics
            if self.round % self.config["save_every_n_rounds"] == 0:
                self.save_model()

            self.round += 1
            return self.get_parameters(config={}), dataset_len, {}
        except Exception as e:
            from flcore.utils import log_detailed_error
            X_diag = self.X_train.cpu().numpy() if hasattr(self, 'X_train') and hasattr(self.X_train, 'cpu') else None
            y_diag = self.y_train.cpu().numpy() if hasattr(self, 'y_train') and hasattr(self.y_train, 'cpu') else None
            log_detailed_error("Model Fitting (Local Training)", e, config=getattr(self, "config", None), X=X_diag, y=y_diag)
            raise e

#    @torch.no_grad()
    def evaluate(self, parameters, params):
        try:
            self.set_parameters(parameters)
    # ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
            self.model.eval()
            if self.config["dropout_p"] > 0.0:
                metrics = uncertainty_metrics(self.model, self.val_loader, device=self.device, T=int(self.config["T"]))
            else:
                pred = self.model(self.X_test)
                y_pred = pred[:,0]
                metrics = calculate_metrics(self.y_test, y_pred, self.config)

            total_loss, correct, total = 0, 0, 0
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                if self.config["task"] == "classification":
                    logits = self.model(X)
                    if self.config["n_out"] == 1:  # Binario
                        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y)
                        probs = torch.sigmoid(logits.squeeze(1))
                        preds = (probs > 0.5).long()
                    else:           # Multiclase
                            #y = y.squeeze()
                        y = y.long()
                        loss = F.cross_entropy(logits, y)
                        preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                elif self.config["task"] == "regression":
                    preds = self.model(X)
                    loss = F.mse_loss(preds, y)
                    #loss = F.l1_loss(preds, y)

                total_loss += loss.item() * X.size(0)
                total += y.size(0)

            test_loss = total_loss / total
            dataset_len = self.y_test.shape[0]

    #        return total_loss / total, correct / total
            return float(test_loss), dataset_len, metrics
        except Exception as e:
            from flcore.utils import log_detailed_error
            X_diag = self.X_test.cpu().numpy() if hasattr(self, 'X_test') and hasattr(self.X_test, 'cpu') else None
            y_diag = self.y_test.cpu().numpy() if hasattr(self, 'y_test') and hasattr(self.y_test, 'cpu') else None
            log_detailed_error("Model Evaluation (Local Validation)", e, config=getattr(self, "config", None), X=X_diag, y=y_diag)
            raise e

    def save_model(self):
        save_path = Path(self.config["sandbox_path"]) / "model"
        save_path.mkdir(parents=True, exist_ok=True)

        model_name = f"{self.config['model']}_{self.config['task']}_round_{getattr(self, 'round', 0)}"

        model_path = save_path / f"{model_name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        with open(self.config["metadata_file"], "r") as f:
            data_metadata = json.load(f)

        entity = data_metadata.get("entries", [])[0]

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

        metadata = {
            "node_name": self.config["node_name"],
            "task": self.config["task"],
            "n_out": self.config["n_out"],
            "n_feats": self.config["n_feats"],  # FIX
            "model_type": self.config["model"],
            "feature_names": self.config["train_labels"],
            "target_names": self.config["target_labels"],
            "metrics": getattr(self, "last_metrics", None),
            "features_meta": features_meta,
            "outcomes_meta": outcomes_meta,
        }

        metadata_path = save_path / f"{model_name}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[Client] NN model saved at {model_path}")
"""
model = BasicNN(...)
model.load_state_dict(torch.load("model.pt"))
model.eval()
"""

def get_client(config,data) -> fl.client.Client:
#    client = FlowerClient(params).to_client()
    return FlowerClient(config,data)
#_______________________________________________________________________________________
