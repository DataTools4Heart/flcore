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
    client_id = config['held_out_center_id']
    # client_id = -1 # kaggle hf
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

    # Train personalized model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    personalized_metrics = calculate_metrics(y_test, y_pred)
    #Add 'personalized' to the metrics to identify them
    personalized_metrics = {f"personalized {key}": personalized_metrics[key] for key in personalized_metrics}
    metrics.update(personalized_metrics)


    if server_round == 0:
        local_model = utils.get_model(config['model'], local=True)
        utils.set_initial_params(local_model, config['linear_models']['n_features'])
        local_model.fit(X_train, y_train)
        y_pred = local_model.predict(X_test)
        local_metrics = calculate_metrics(y_test, y_pred)
        #Add 'local' to the metrics to identify them
        local_metrics = {f"local {key}": local_metrics[key] for key in local_metrics}
        metrics.update(local_metrics)

        # Train model in centralized way (if possible)
        (X_train, y_train), (X_test, y_test) = load_dataset(config, id=None)
        centralized_model = get_model(config['model'], local=True)
        utils.set_initial_params(centralized_model, config['linear_models']['n_features'])
        centralized_model.fit(X_train, y_train)
        y_pred = centralized_model.predict(X_test)
        centralized_metrics = calculate_metrics(y_test, y_pred)
        #Add 'centralized' to the metrics to identify them
        centralized_metrics = {f"centralized {key}": centralized_metrics[key] for key in centralized_metrics}
        metrics.update(centralized_metrics)

        per_center_metrics = []
        for i in range(0, config['num_clients']):
            # client_id = 10 + i
            client_id = i
            (X_train, y_train), (X_test, y_test) = load_dataset(config, client_id)
            y_pred = centralized_model.predict(X_test)
            center_metrics = calculate_metrics(y_test, y_pred)
            per_center_metrics.append(center_metrics)
        
        #Calculate the mean of the metrics
        non_weighted_centralized_metrics = {}
        for key in per_center_metrics[0]:
            non_weighted_centralized_metrics[key] = np.mean([center_metrics[key] for center_metrics in per_center_metrics])
        #Add 'centralized' to the metrics to identify them
        non_weighted_centralized_metrics = {f"non weighted centralized {key}": non_weighted_centralized_metrics[key] for key in non_weighted_centralized_metrics}
        metrics.update(non_weighted_centralized_metrics)


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
