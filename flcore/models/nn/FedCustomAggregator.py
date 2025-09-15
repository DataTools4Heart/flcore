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
    def __init__(self, epsilon: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures):
        if not results:
            return None, {}
        # results es una lista con un único elemento que es una tupla que es fl.server.client_proxy
        # y fl.common.FitRes, failures es a parte
#        print(":::::::::::::::::::::::::::::::::::::",results[0][1])

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]


        weights_results = []
        agg_weights = []
        for _, fitres in results:
            ndarrays = fl.common.parameters_to_ndarrays(fitres.parameters)
            num_examples = fitres.num_examples
            entropy = fitres.metrics.get("entropy", 1.0)
            # peso = más datos y menor entropía => mayor confianza
            print(" *********************** ENTROPIA", entropy)
            w = num_examples / (self.epsilon + entropy)
            weights_results.append((ndarrays, w))
            agg_weights.append(w)

        wsum = np.sum(agg_weights) + 1e-12
        scaled = [(params, w / wsum) for params, w in weights_results]

        new_params = None
        for params, alpha in scaled:
            if new_params is None:
                new_params = [alpha * p for p in params]
            else:
                new_params = [np.add(acc, alpha * p) for acc, p in zip(new_params, params)]

        parameters_aggregated = ndarrays_to_parameters(new_params)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        """
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        """
        return parameters_aggregated, metrics_aggregated

