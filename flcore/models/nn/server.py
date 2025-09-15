# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Uncertainty-Aware Neural Network
# Author: Jorge Fabila Fabian
# Fecha: September 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from typing import Dict, Optional, Tuple, List, Any, Callable
import argparse
import numpy as np
import os
import flwr as fl
from flwr.common import Metrics, Scalar, Parameters
from sklearn.metrics import confusion_matrix
import functools

import flwr as fl
import flcore.models.linear_models.utils as utils
from flcore.metrics import metrics_aggregation_fn
from sklearn.metrics import log_loss
from typing import Dict
import joblib
from flcore.models.nn.FedCustomAggregator import UncertaintyWeightedFedAvg
from flcore.datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from flcore.models.linear_models.utils import get_model
from flcore.metrics import calculate_metrics
from flcore.models.nn.basic_nn import BasicNN
import torch

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def equal_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [ m["accuracy"] for num_examples, m in metrics]
    return {"accuracy": sum(accuracies) }


def get_server_and_strategy(config):
    if torch.cuda.is_available() and config["device"] == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    model_type = config['model']
    model = get_model(model_type)
    model = BasicNN( config["n_feats"], config["n_out"], config["dropout_p"] ).to(device)

    if config["metrics_aggregation"] == "weighted_average":
        metrics = weighted_average
    elif config["metrics_aggregation"] == "equal_average":
        metrics = equal_average

    if config["strategy"] == "FedAvg":
        print("================================")
        strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=metrics,
        min_fit_clients = config["min_fit_clients"],
        min_evaluate_clients = config["min_evaluate_clients"],
        min_available_clients = config["min_available_clients"])
    elif config["strategy"] == "FedOps":
        strategy = fl.server.strategy.FedOpt(evaluate_metrics_aggregation_fn=metrics,
        min_fit_clients = config["min_fit_clients"],
        min_evaluate_clients = config["min_evaluate_clients"],
        min_available_clients = config["min_available_clients"])
    elif config["strategy"] == "FedProx":
        strategy = fl.server.strategy.FedProx(evaluate_metrics_aggregation_fn=metrics,
        min_fit_clients = config["min_fit_clients"],
        min_evaluate_clients = config["min_evaluate_clients"],
        min_available_clients = config["min_available_clients"])
    elif config["strategy"] == "UncertaintyWeighted":
        strategy = UncertaintyWeightedFedAvg(
        min_fit_clients = config["min_fit_clients"],
        min_evaluate_clients = config["min_evaluate_clients"],
        min_available_clients = config["min_available_clients"])
    return None, strategy

