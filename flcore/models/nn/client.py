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

from train import train
from test import test
from utils import Parameters
from torch.utils.data import TensorDataset, DataLoader

from basic_nn import BasicNN

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config, data):
        self.params = config

        if torch.cuda.is_available() and self.params["device"] == 'cuda':
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)

        train_ds = TensorDataset(self.X_train, self.y_train)
        test_ds = TensorDataset(self.X_test, self.y_test)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.epochs = config["local_epochs"]

        self.model = BasicNN( config["n_feats"], config["n_out"], config["dropout_p"] ).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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
        print(" ***************************************** FIT self.params.client_id ", self.params)
        print(f"[Client {self.params.client_id}] fit")
        self.set_parameters(parameters)
        #train(self.model,self.params,self.dataset)
# ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
        for epoch in range(self.epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0
            
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # forward
                logits = self.model(X)
                loss = self.criterion(logits, y)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                """
                self.optimizer.step()
                # métricas de incertidumbre en validación
                metrics = uncertainty_metrics(self.model, self.val_loader, device=DEVICE, T=int(config.get("T", 20)))
                # importante: el servidor usará 'entropy' y 'val_accuracy'

                num_examples = len(self.train_loader.dataset)
                return get_weights(self.model), num_examples, metrics
                """                
                # métricas
                total_loss += loss.item() * X.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            train_loss = total_loss / total
            train_acc = correct / total
            test_loss, test_acc = self.evaluate()
            
            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        trainloader_dataset_len = self.dataset.train_size
        return self.get_parameters(config={}), trainloader_dataset_len, {}

#    @torch.no_grad()
    def evaluate(self, parameters, params):
# ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        for X, y in self.test_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        return total_loss / total, correct / total

# ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
set_weights(self.model, parameters)
self.model.eval()
criterion = nn.CrossEntropyLoss(reduction="sum")
loss_sum, total, correct = 0.0, 0, 0
with torch.no_grad():
for x, y in self.val_loader:
x, y = x.to(DEVICE), y.to(DEVICE)
logits = self.model(x)
loss = criterion(logits, y)
pred = logits.argmax(dim=-1)
correct += (pred == y).sum().item()
total += y.numel()
loss_sum += loss.item()
return float(loss_sum / max(1, total)), total, {"val_accuracy": correct / max(1, total)}

# ****** * * * * *  * *  *  *   *   *    *    *  * * * * * * * * ********
        # parameters es una lista y params un diccionario vacio
        # En principio aqui aceptamos params, pero no depende de nosotros pasar params,
        # flower pasa los parametros que le salen de los huevos
        print(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^[Client {self.params.client_id}] evaluate")
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.dataset)
        return float(loss), self.dataset.test_size, {"accuracy": float(accuracy)}

def get_client(config,data,client_id) -> fl.client.Client:
#    client = FlowerClient(params).to_client()
    return FlowerClient(config,data)

#_______________________________________________________________________________________
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl


from model import MCDropoutMLP
from data import load_mnist, make_client_datasets, get_loaders_from_indices
from utils import get_weights, set_weights, uncertainty_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FlowerClient(fl.client.NumPyClient):
def __init__(self, cid: str, idxs):
self.cid = cid
trainset, _ = load_mnist()
self.train_loader, self.val_loader = get_loaders_from_indices(trainset, idxs)
self.model = MCDropoutMLP(p=0.3).to(DEVICE)
self.criterion = nn.CrossEntropyLoss()
self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)


def get_parameters(self, config):
return get_weights(self.model)


def fit(self, parameters, config) -> Tuple[list, int, Dict]:
if parameters is not None:
set_weights(self.model, parameters)
self.model.train()
epochs = int(config.get("local_epochs", 1))
for _ in range(epochs):
for x, y in self.train_loader:
x, y = x.to(DEVICE), y.to(DEVICE)
self.optimizer.zero_grad()
logits = self.model(x)
loss = self.criterion(logits, y)
loss.backward()
self.optimizer.step()
# métricas de incertidumbre en validación
metrics = uncertainty_metrics(self.model, self.val_loader, device=DEVICE, T=int(config.get("T", 20)))
# importante: el servidor usará 'entropy' y 'val_accuracy'
num_examples = len(self.train_loader.dataset)
return get_weights(self.model), num_examples, metrics


def evaluate(self, parameters, config):
set_weights(self.model, parameters)
self.model.eval()
criterion = nn.CrossEntropyLoss(reduction="sum")
loss_sum, total, correct = 0.0, 0, 0
with torch.no_grad():
for x, y in self.val_loader:
x, y = x.to(DEVICE), y.to(DEVICE)
logits = self.model(x)
loss = criterion(logits, y)
pred = logits.argmax(dim=-1)
correct += (pred == y).sum().item()
total += y.numel()
loss_sum += loss.item()
return float(loss_sum / max(1, total)), total, {"val_accuracy": correct / max(1, total)}


def client_fn(cid: str):
# Mapear cid-> partición
num_clients = int(fl.common.parameters_dict_from_ndarrays([]).get("num_clients", 0) or 5)
parts = make_client_datasets(num_clients=num_clients, noniid=True, seed=0)
idxs = parts[int(cid)]
return FlowerClient(cid, idxs)


if __name__ == "__main__":
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FlowerClient("0", make_client_datasets(5)[0]))
