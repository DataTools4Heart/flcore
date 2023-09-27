#############################################################################
#RF Agregator Code implemented by Esmeralda Ruiz Pujadas                   ##
#The Federated RF aggregator is implemented with/without drop out center.  ##
#In this version, I implemented the weight of each tree in order to add    ##
#the smoothing weights                                                     ##
#The client will merge all the trees and weights from the server           ##
#and ensamble all the trees weighted if enable                             ##
#Fit does not ensamble. Only create a new tree with a new partition        ##
#Evaluation is where the ensamble is performed and                         ## 
#the result is sent to the server                                          ##
#Feel free to change the method or improve it                              ##
#############################################################################

from logging import WARNING
from typing import  Dict, List,Callable, Optional,Tuple,Union
#from dropout import Fast_at_odd_rounds

from flwr.common import  FitIns, FitRes,EvaluateRes, MetricsAggregationFn, NDArrays, Parameters,  Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from flcore.serialization_funs import serialize_RF, deserialize_RF

import numpy as np
from flcore.models.weighted_random_forest.utils import get_model
import random
import time
import flwr.server.strategy.fedavg as fedav
from flcore.dropout import select_clients
from flcore.smoothWeights import smooth_aggregate,computeSmoothedWeights

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""



class FedCustom(fl.server.strategy.FedAvg):
    """Configurable FedAvg strategy implementation."""
    #DropOut center variable to get the initial execution time of the first round
    clients_first_round_time = {}
    clients_num_examples = {}
    server_estimators = []
    time_server_round = time.time()
    bal_RF = None
    dropout_method = None
    percentage_drop = 0
    smoothing_method = None
    smoothing_strenght = 0
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    
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
                
            
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        #Deserialize to real parameter
        parameters_ndarrays = deserialize_RF(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

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
            (deserialize_RF(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        #If smoothing weights are enable compute the weights and add them
        #to the results for the client
        if(self.smoothing_method!= 'None'):
            weights_smooth = computeSmoothedWeights(weights_results,True,0.5)
            
            weights_results = [
                (deserialize_RF(fit_res[1].parameters), fit_res[1].num_examples,weight)
                for fit_res,weight in zip(results,weights_smooth)
            ]

 
        #Add the aggregated results
        aggregation_result = weights_results
        

        #ndarrays_to_parameters necessary to send the message
        parameters_aggregated = serialize_RF(aggregation_result)
        

        #DropOut Center: initially aggregate all execution times of all clients
        #ONLY THE FIRST ROUND is tracked the execution time to start further
        #rounds with dropout center if wanted
        if(server_round == 1):
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
        self.time_server_round = time.time()
        print(f"Elapsed time: {elapsed_time} for round {server_round}")

        return parameters_aggregated, metrics_aggregated
    

    def aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = fedav.weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
 

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


