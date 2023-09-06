import numpy as np

import flcore.models.logistic_regression as logistic_regression
import flcore.models.xgb as xgb
import flcore.models.bnn as bnn


def get_model_client(config, data, client_id):
    model = config["model"]

    if model == "logistic_regression":
        client = logistic_regression.client.get_client(data)

    elif model == "rf":
        pass

    elif model == "xgb":
        client = xgb.client.get_client(config, data, client_id)

    elif model == "":
        client = bnn.client.get_client(config, data, client_id)
    
    else:
        raise ValueError(f"Unknown model: {model}")

    return client
