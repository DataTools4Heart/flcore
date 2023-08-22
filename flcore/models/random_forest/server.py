from typing import Dict, Optional, Tuple, List, Any, Callable
import argparse
import numpy as np
import os
import flwr as fl
from flwr.common import Metrics
from sklearn.metrics import confusion_matrix

#from networks.arch_handler import Network

import warnings
#install pip install pyyaml
import yaml
from pathlib import Path

import flwr as fl
import flcore.models.random_forest.utils as utils
from sklearn.metrics import log_loss
from typing import Dict
import joblib
from flcore.models.random_forest.FedCustomAggregator import FedCustom
from sklearn.ensemble import RandomForestClassifier
from flcore.models.random_forest.utils import get_model



warnings.filterwarnings( 'ignore' )

def fit_round( server_round: int ) -> Dict:
    """Send round number to client."""
    return { 'server_round': server_round }


def get_server_and_strategy(config):
    bal_RF = False
    model = get_model(bal_RF) 
    utils.set_initial_params_server( model)

    # Pass parameters to the Strategy for server-side parameter initialization
    #strategy = fl.server.strategy.FedAvg(
    strategy = FedCustom(    
        #Have running the same number of clients otherwise it does not run the federated
        min_available_clients = config['num_clients'],
        min_fit_clients = config['num_clients'],
        min_evaluate_clients = config['num_clients'],
        #enable evaluate_fn  if we have data to evaluate in the server
        #evaluate_fn           = utils_RF.get_evaluate_fn( model ), #no data in server
        evaluate_metrics_aggregation_fn = utils.evaluate_metrics_aggregation_fn,
        on_fit_config_fn      = fit_round
    )

    return None, strategy


if __name__ == '__main__':
    bal_RF = False
    model = get_model(bal_RF) 
    utils.set_initial_params_server( model)

    # Pass parameters to the Strategy for server-side parameter initialization
    #strategy = fl.server.strategy.FedAvg(
    strategy = FedCustom(    
        #Have running the same number of clients otherwise it does not run the federated
        min_available_clients = utils.num_clients,
        #enable evaluate_fn  if we have data to evaluate in the server
        #evaluate_fn           = utils_RF.get_evaluate_fn( model ), #no data in server
        evaluate_metrics_aggregation_fn = utils.evaluate_metrics_aggregation_fn,
        on_fit_config_fn      = fit_round
    )



    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address = 'LOCALHOST:8080',
        config         = fl.server.ServerConfig( num_rounds = utils.num_rounds_server ), 
        strategy       = strategy,
        #client_manager= CenterDropoutClientManager()
        ##Socayna says to comment in local. It is for the docker
        #certificates   = (
        #    Path( '.cache/certificates/rootCA_cert.pem' ).read_bytes(),
        #    Path( '.cache/certificates/server_cert.pem' ).read_bytes(),
        #    Path( '.cache/certificates/server_key.pem'  ).read_bytes(),
        #),
    )
    