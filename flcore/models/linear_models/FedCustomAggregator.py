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


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class FedCustom(fl.server.strategy.FedAvg):
    """Configurable FedAvg strategy implementation."""
    
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
   
    def __init__(
        self,
        min_available_clients: int,
        min_fit_clients: int,
        min_evaluate_clients: int,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, fl.common.Scalar]]] = None,
        accept_failures: bool = False,
        dropout_method = None,
        percentage_drop = 0.0,
        smoothing_method = None,
        smoothing_strenght = 0.0,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        
        super().__init__(min_available_clients=min_available_clients, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, evaluate_fn=evaluate_fn, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, on_fit_config_fn=on_fit_config_fn, accept_failures=accept_failures)
        
        #DropOut center variable to get the initial execution time of the first round
        self.clients_first_round_time = {}
        self.time_server_round = time.time()
        self.clients_num_examples = {}
        self.accum_time = 0

        self.dropout_method = dropout_method
        self.percentage_drop = percentage_drop
        self.smoothing_method = smoothing_method
        self.smoothing_strenght = smoothing_strenght
        self.checkpoint_dir = checkpoint_dir


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        #Get the clients to train
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        #After the second round apply dropout if wanted
        if(self.dropout_method != 'None'):
            if(server_round>1):
                # Drop Out center
                clients = select_clients(self.dropout_method, self.percentage_drop,clients,self.clients_first_round_time,server_round,self.clients_num_examples)
                #(clients,self.fast_round) = Fast_at_odd_rounds(server_round,clients,self.clients_first_round_time, 25) #0)
                #clients = Fast_every_three(server_round,clients,self.clients_first_round_time, 25)
                #clients = random_dropout(server_round,clients,self.clients_first_round_time, 25)
                #clients = Less_participants_at_odd_rounds(server_round,clients, self.clients_num_examples,25)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        if self.smoothing_method == 'None': #(smoothing==0 | self.fast_round == True):
            weights_aggregated = aggregate(weights_results)
        else:
            weights_aggregated = smooth_aggregate(weights_results,self.smoothing_method,self.smoothing_strenght)

        # Save the aggregated weights
        joblib.dump(weights_aggregated, f"{self.checkpoint_dir}/round_{server_round}_weights.joblib")

        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)

        #DropOut Center: initially aggregate all execution times of all clients
        #ONLY THE FIRST ROUND is tracked the execution time to start further
        #rounds with dropout center if wanted
        if self.dropout_method != 'None':
            if server_round == 1:
                for client, res in results:
                    self.clients_first_round_time[client.cid] = res.metrics['running_time']
                    self.clients_num_examples[client.cid] = res.num_examples

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        elapsed_time =  (time.time() - self.time_server_round)
        self.accum_time = self.accum_time+ elapsed_time
        self.time_server_round = time.time()
        print(f"Elapsed time: {elapsed_time} for round {server_round}")

        filename = 'server_results.txt'
        with open(
        filename,
        "a",
        ) as f:
            f.write(f"Accumulated Time: {self.accum_time} for round {server_round}\n")

        return parameters_aggregated, metrics_aggregated
    

 