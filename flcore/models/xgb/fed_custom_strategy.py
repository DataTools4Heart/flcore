
from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

from flwr.common import (
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

from flwr.server.strategy import FedXgbNnAvg
from flwr.server.strategy.aggregate import aggregate

from flcore.dropout import select_clients
from flcore.smoothWeights import smooth_aggregate


class FedCustomStrategy(FedXgbNnAvg):
    """Configurable strategy for Center Dropout and weights smoothing."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        dropout_method: str = 'None',
        percentage_drop: float = 0,
        smoothing_method: str = 'None',
        smoothing_strenght: float = 0,

    ) -> None:
        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.dropout_method = dropout_method
        self.percentage_drop = percentage_drop
        self.smoothing_method = smoothing_method
        self.smoothing_strenght = smoothing_strenght
        self.clients_first_round_time = {}
        self.time_server_round = time.time()
        self.clients_num_examples = {}
        self.accum_time = 0

            
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        configure_clients = super().configure_fit(server_round, parameters, client_manager)
        clients = [client for client, fit_ins in configure_clients]
        fit_ins = [fit_ins for client, fit_ins in configure_clients]

        # #After the second round apply dropout if wanted
        if(self.dropout_method != 'None'):
            if(server_round>1):
                clients = select_clients(self.dropout_method, self.percentage_drop,clients, self.clients_first_round_time, server_round, self.clients_num_examples)
        
        print(f"Center Dropout, selected {len(clients)}  clients out of")
        # Return client/config pairs
        return list(zip(clients, fit_ins))
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Any], Dict[str, Scalar],]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters[0].parameters),  # type: ignore
                fit_res.num_examples,
            )
            for _, fit_res in results
        ]
        if(self.smoothing_method=='None' ): #(smoothing==0 | self.fast_round == True):
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        else:
            parameters_aggregated = ndarrays_to_parameters(smooth_aggregate(weights_results,self.smoothing_method,self.smoothing_strenght))

        #DropOut Center: initially aggregate all execution times of all clients
        #ONLY THE FIRST ROUND is tracked the execution time to start further
        #rounds with dropout center if wanted
        if(self.dropout_method != 'None'):
            if(server_round == 1):
                for client, res in results:
                    self.clients_first_round_time[client.cid] = res.metrics['running_time']
                    self.clients_num_examples[client.cid] = res.num_examples

        # Aggregate XGBoost trees from all clients
        trees_aggregated = [fit_res.parameters[1] for _, fit_res in results]  # type: ignore

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return [parameters_aggregated, trees_aggregated], metrics_aggregated