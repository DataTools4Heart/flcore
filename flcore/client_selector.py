import numpy as np

import flcore.models.logistic_regression as logistic_regression
import flcore.models.xgb as xgb
import flcore.models.random_forest as random_forest

def get_model_client(config, data, client_id):
    model = config["model"]

    if model == "logistic_regression":
        client = logistic_regression.client.get_client(data)

    elif model == "random_forest":
        client = random_forest.client.get_client(data,client_id) 

    elif model == "xgb":
        client = xgb.client.get_client(config, data, client_id)

    else:
        raise ValueError(f"Unknown model: {model}")

    return client
