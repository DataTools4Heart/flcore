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
from flcore.metrics import metrics_aggregation_fn



warnings.filterwarnings( 'ignore' )

def fit_round( server_round: int ) -> Dict:
    """Send round number to client."""
    return { 'server_round': server_round }


def get_server_and_strategy(config):
    bal_RF = True if config['model'] == 'balanced_random_forest' else False
    model = get_model(bal_RF, config['random_forest']['tree_num']) 
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
        fit_metrics_aggregation_fn=metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn = metrics_aggregation_fn,
        on_fit_config_fn      = fit_round      
    )
    #Select normal RF or Balanced RF from config
    strategy.bal_RF= config['random_forest']['balanced_rf']
    strategy.dropout_method = config['dropout_method']
    strategy.percentage_drop = config['dropout']['percentage_drop']
    strategy.smoothing_method = config['smooth_method']
    strategy.smoothing_strenght = config['smoothWeights']['smoothing_strenght']

    filename = 'server_results.txt'
    with open(
    filename,
    "a",
    ) as f:
        f.write(f"Name Model Random Forest:  \n")
        f.write(f"Drop out Method: {strategy.dropout_method} \n")
        f.write(f"Drop out Method: {strategy.percentage_drop} \n")
        f.write(f"Smooth Method: {strategy.smoothing_method} \n")
        f.write(f"Smooth Strenght: {strategy.smoothing_strenght } \n")

    return None, strategy



    