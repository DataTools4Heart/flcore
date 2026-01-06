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
import joblib
from flcore.models.nn.FedCustomAggregator import UncertaintyWeightedFedAvg
from flcore.metrics import calculate_metrics
from flcore.models.nn.basic_nn import BasicNN
import torch

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)

    metric_keys = metrics[0][1].keys()

    weighted_metrics = {}
    for key in metric_keys:
        weighted_sum = sum(
            num_examples * m[key] for num_examples, m in metrics
        )
        weighted_metrics[key] = weighted_sum / total_examples

    return weighted_metrics

def equal_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    # Número de clientes
    num_clients = len(metrics)

    # Asumimos que todas las métricas tienen las mismas keys
    metric_keys = metrics[0][1].keys()

    equal_metrics = {}
    for key in metric_keys:
        equal_sum = sum(
            m[key] for _, m in metrics
        )
        equal_metrics[key] = equal_sum / num_clients

    return equal_metrics


def get_server_and_strategy(config):
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

