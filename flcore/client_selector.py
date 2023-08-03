# import flcore.models.logistic_regression as logistic_regression
import flcore.models.logistic_regression as logistic_regression
import flcore.models.xgb as xgb

import numpy as np

def get_model_client(config, data, client_id):
    model = config[ 'model' ]
    X_train, y_train, X_test, y_test = data

    if model == 'logistic_regression':
        client = logistic_regression.client.MnistClient(data)

    elif model == 'rf':
        pass

    elif model == 'xgb':
        client = xgb.client.get_client(config, data, client_id)
        
    else:
        raise ValueError(f'Unknown model: {model}')
        
    return client
