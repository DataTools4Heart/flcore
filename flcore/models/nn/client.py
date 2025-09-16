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
        self.params = config
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.epochs = config["local_epochs"]

        print("MODELS::NN:CLIENT::INIT")
        if torch.cuda.is_available() and self.params["device"] == 'cuda':
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
#        self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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
        self.set_parameters(parameters)
        #train(self.model,self.params,self.dataset)
# ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
        for epoch in range(self.epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)

                if self.params["n_out"] == 1:  # Binario
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y)
                    probs = torch.sigmoid(logits.squeeze(1))
                    preds = (probs > 0.5).long()
                else:           # Multiclase
                    loss = F.cross_entropy(logits, y)
                    preds = torch.argmax(logits, dim=1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # métricas de incertidumbre en validación
                metrics = uncertainty_metrics(self.model, self.val_loader, device=self.device, T=int(self.params["T"]))
                # importante: el servidor usará 'entropy' y 'val_accuracy'
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
        return self.get_parameters(config={}), dataset_len, {}

#    @torch.no_grad()
    def evaluate(self, parameters, params):
        self.set_parameters(parameters)
# ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        for X, y in self.test_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            if self.params["n_out"] == 1:  # Binario
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y)
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs > 0.5).long()
            else:           # Multiclase
                loss = F.cross_entropy(logits, y)
                preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        test_loss = total_loss / total
        acc = correct / total
        dataset_len = self.y_test.shape[0]

#        return total_loss / total, correct / total
        return float(total_loss), dataset_len, {"accuracy": float(acc)}

def get_client(config,data,client_id) -> fl.client.Client:
#    client = FlowerClient(params).to_client()
    return FlowerClient(config,data)
#_______________________________________________________________________________________
