# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from logging import WARNING
import argparse
import sys, os
import logging
import hashlib
import flwr as fl
from flwr.common.logger import log
from typing import List, Optional, Tuple, Union, Dict
# from flwr import weighted_loss_avg

import numpy as np
import pickle, json

from flcore.models.gbs.model import GBSModel
from flcore.models.gbs.aggregator import GBSAggregator


logger = logging.getLogger(__name__)


class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, rounds: int, saving_path :str = '/sandbox/', **kwargs):
        super().__init__(**kwargs)
        self.rounds = round
        self.results_history = {}
        self.saving_path = saving_path

    def _save_results_history(self):
        """Save the results history to a file."""
        with open(f"{self.saving_path}/history.json", "w") as f:
            json.dump(self.results_history, f)

    def aggregate_fit(self, rnd: int, results, failures):
        """
        results: list of (ClientProxy, FitRes)
        """
        if not results:
            return None, {}

        models = []
        weights = []

        for _, fit_res in results:
            # Convert Flower parameters to numpy arrays
            params_list = fl.common.parameters_to_ndarrays(fit_res.parameters)
            # Ensure each ndarray is converted back to bytes for legacy aggregators

            params_as_bytes = []
            for p in params_list:
                if isinstance(p, np.ndarray):
                    b = p.tobytes()
                    params_as_bytes.append(b)
                else:
                    params_as_bytes.append(p)
            models.append(params_as_bytes)
            
            weights.append(fit_res.num_examples)

        aggregator: BaseAggregator = GBSAggregator(models=models, weights=weights)
        aggregated_params = aggregator.aggregate()
        
        # Convert aggregated model back to Flower parameters
        parameters = fl.common.ndarrays_to_parameters(aggregated_params)
        
        # --- SAVE GLOBAL MODEL AFTER LAST ROUND ---
        if rnd == self.rounds:
            print(aggregated_params)
            model = GBSModel()
            model.set_parameters(aggregated_params)
            os.makedirs(f"{self.saving_path}/models/", exist_ok=True)
            with open(f"{self.saving_path}/models/gbs.pkl", "wb") as f:
                pickle.dump(model, f)
            
            model_bytes = pickle.dumps(model)
            model_md5 = hashlib.md5(model_bytes).hexdigest()
            self.results_history['MODEL_MD5'] = model_md5

        return parameters, {}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list,
        failures: list,
    ) -> tuple:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        round_results = {'CLIENTS': {}, 'ROUND_INFO': {}}
        for _, res in results:
            round_results['CLIENTS'][res.metrics['client_id']] = {key: value for key, value in res.metrics.items() if key != 'client_id'}
            round_results['CLIENTS'][res.metrics['client_id']]['num_examples'] = res.num_examples
            round_results['CLIENTS'][res.metrics['client_id']]['1-c_index(loss)'] = res.loss
        

        # Aggregate loss
        loss_aggregated = np.mean([evaluate_res.loss for _, evaluate_res in results])
        round_results['ROUND_INFO']['aggregated_loss'] = loss_aggregated

        # Aggregate custom metrics if aggregation fn was provided

        metrics_aggregated = {}
        for _, res in results:
            for key, value in res.metrics.items():
                if key == 'client_id':
                    continue
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = []
                metrics_aggregated[key].append(value)
        for key in metrics_aggregated:
            metrics_aggregated[key] = np.mean(metrics_aggregated[key])

        round_results['ROUND_INFO']['aggregated_metrics'] = metrics_aggregated
        
        self.results_history[f"ROUND {server_round}"] = round_results
        self.results_history['MODEL_TYPE'] = 'gbs'
        self._save_results_history()

        return loss_aggregated, metrics_aggregated

def get_fit_config_fn(estimators):
    def fit_config(rnd: int):
        conf = {"model_type": 'gbs', "n_estimators": estimators}
        return conf
    return fit_config


# -------------------------------
# Get server helper
# -------------------------------

def get_server_and_strategy(
    config
) -> Tuple[fl.server.Server, CustomStrategy]:

    os.makedirs(f"{config['experiment_dir']}", exist_ok=True)

    server = fl.server.Server
    strategy = CustomStrategy(
        on_fit_config_fn=get_fit_config_fn(config['n_estimators']),
        rounds = config['num_rounds'],
        min_available_clients=config['num_clients'],
        saving_path=config['experiment_dir'],
    )

    return None, strategy
