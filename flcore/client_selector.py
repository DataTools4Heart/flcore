import numpy as np

import flcore.models.linear_models as linear_models
import flcore.models.xgb as xgb
import flcore.models.random_forest as random_forest
import flcore.models.weighted_random_forest as weighted_random_forest
import flcore.models.nn as nn

def get_model_client(config, data, client_id):
    model = config["model"]

    if model in ("logistic_regression", "elastic_net", "lsvc"):
        client = linear_models.client.get_client(config,data,client_id)

    elif model == "random_forest":
        client = random_forest.client.get_client(config,data,client_id) 
    
    elif model == "weighted_random_forest":
        client = weighted_random_forest.client.get_client(config,data,client_id)

    elif model == "xgb":
        client = xgb.client.get_client(config, data, client_id)

    elif model == "nn":
        client = nn.client.get_client(config, data, client_id)

    else:
        raise ValueError(f"Unknown model: {model}")

    return client
