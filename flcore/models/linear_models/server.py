#############################################################################
#Linear models implemented by Esmeralda Ruiz Pujadas                       ##
#The Linear Models are: LR, ElasticNet and LSVM                            ##
#You can select them from the params of config                             ##
#It is implemented with feature selection                                  ##
#In this implementation the first client selected by the server provides   ##
#the feature selection and is sent to the server and the server sends it   ##
#to all the clients as all the clients must use the same features          ##
#Feel free to implement more sophisticated feature selection               ##
#To disable the feature selection select the maximum features and all the  ##
#features will be used using n_features in config                          ##
#Params in config:                                                         ##
# Type: elastic_net,LSVC, LR                                               ##
# num_features                                                             ##
#Mising: Pipeline to deal with categorical                                 ##
#############################################################################

from typing import Dict, Optional, Tuple, List, Any, Callable
import argparse
import numpy as np
import os
import flwr as fl
from flwr.common import Metrics, Scalar, Parameters
from sklearn.metrics import confusion_matrix
import functools


#from networks.arch_handler import Network

import warnings
#install pip install pyyaml
import yaml
from pathlib import Path

import flwr as fl
import flcore.models.linear_models.utils as utils
from flcore.metrics import metrics_aggregation_fn
from sklearn.metrics import log_loss
from typing import Dict
import joblib
from flcore.models.linear_models.FedCustomAggregator import FedCustom
from flcore.datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from flcore.models.linear_models.utils import get_model
from flcore.metrics import calculate_metrics



warnings.filterwarnings( 'ignore' )

def fit_round( server_round: int ) -> Dict:
    """Send round number to client."""
    return { 'server_round': server_round }


def evaluate_held_out(
    server_round: int,
    parameters: fl.common.Parameters,
    kwargs: Dict[str, fl.common.Scalar],
    config: Dict[str, fl.common.Scalar],
) -> Tuple[float, Dict[str, float]]:
    
    """Evaluate the current model on the held-out validation set."""
    # Load held-out validation data
    client_id = 19
    model = get_model(config['model'])
    utils.set_model_params(model, parameters)
    (X_train, y_train), (X_test, y_test) = load_dataset(config, client_id)
    model.classes_ = np.unique(y_test)
    # Evaluate the model
    y_pred = model.predict(X_test)
    loss = log_loss(y_test, y_pred)
    metrics = calculate_metrics(y_test, y_pred)
    n_samples = len(y_test)
    metrics['n samples'] = n_samples
    metrics['client_id'] = client_id

    return loss, metrics


def get_server_and_strategy(config):
    model_type = config['model']
    model = get_model(model_type)
    n_features = config['linear_models']['n_features']
    utils.set_initial_params(model, n_features)

    # Pass parameters to the Strategy for server-side parameter initialization
    #strategy = fl.server.strategy.FedAvg(
    strategy = FedCustom(   
        #Have running the same number of clients otherwise it does not run the federated
        min_available_clients = config['num_clients'],
        min_fit_clients = config['num_clients'],
        min_evaluate_clients = config['num_clients'],
        #enable evaluate_fn  if we have data to evaluate in the server
        evaluate_fn=functools.partial(
            evaluate_held_out,
            config=config,
        ),
        fit_metrics_aggregation_fn = metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn = metrics_aggregation_fn,
        on_fit_config_fn = fit_round,
        checkpoint_dir = config["experiment_dir"] / "checkpoints",
        dropout_method = config['dropout_method'],
        percentage_drop = config['dropout']['percentage_drop'],
        smoothing_method = config['smooth_method'],
        smoothing_strenght = config['smoothWeights']['smoothing_strenght']
    )

    return None, strategy
