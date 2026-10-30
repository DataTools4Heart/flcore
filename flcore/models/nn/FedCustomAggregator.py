from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np
import flwr.server.strategy.fedavg as fedav
import time
from flcore.dropout import select_clients
from flcore.smoothWeights import smooth_aggregate
import joblib

class UncertaintyWeightedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, epsilon: float = 0.1, gamma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon  # suaviza el peso de clientes con entropía ~0
        self.gamma = gamma      # 0 => FedAvg puro; mayor => más peso a baja entropía

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # As in FedAvg: do not aggregate if there were failures and they are not accepted
        if not self.accept_failures and failures:
            return None, {}

        ndarrays_list, raw_weights = [], []
        for _, fit_res in results:
            ndarrays_list.append(parameters_to_ndarrays(fit_res.parameters))

            entropy = fit_res.metrics.get("entropy")
            if entropy is None or not np.isfinite(entropy):
                entropy = 1.0  # peor caso (entropía normalizada) si falta o es inválida
                print(f"[round {server_round}] cliente sin 'entropy' válida, usando 1.0")

            raw_weights.append(fit_res.num_examples / (self.epsilon + entropy) ** self.gamma)

        alphas = np.asarray(raw_weights, dtype=np.float64)
        alphas /= alphas.sum() + 1e-12

        new_params = []
        for layer_idx in range(len(ndarrays_list[0])):
            layer = sum(a * nd[layer_idx].astype(np.float64) for a, nd in zip(alphas, ndarrays_list))
            new_params.append(layer.astype(ndarrays_list[0][layer_idx].dtype))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(new_params), metrics_aggregated