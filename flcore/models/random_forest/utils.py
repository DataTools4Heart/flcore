from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
RFRegParams = RandomForestClassifier #Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

import flwr as fl
from sklearn.metrics import log_loss
from typing import Dict


import numpy.typing as npt
from typing import Any
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]
from typing import cast


def get_model(bal_RF):
    if(bal_RF == True):
        model = BalancedRandomForestClassifier(n_estimators=300,max_depth=10)
    else:
        model = RandomForestClassifier(n_estimators=300,max_depth=10,class_weight= "balanced_subsample")
    
    return model

def get_model_parameters(model: RandomForestClassifier) -> RFRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    params = [model]
    
    return params


def set_model_params(
    model: RandomForestClassifier, params: RFRegParams
) -> RandomForestClassifier:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.n_classes_ =2
    model.estimators_ = params[0]
    model.classes_ = np.array([i for i in range(model.n_classes_)])
    model.n_outputs_ = 1
    return model


def set_initial_params_server(model: RandomForestClassifier):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. 
    """
    model.estimators_ = 0


def set_initial_params_client(model: RandomForestClassifier,X_train, y_train):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch.
    """
    model.fit(X_train, y_train)  

#Evaluate in the aggregations evaluation with
#the client using client data and combine
#all the metrics of the clients
def evaluate_metrics_aggregation_fn(eval_metrics):
    print(eval_metrics[0][1].keys())
    keys_names = eval_metrics[0][1].keys()
    keys_names = list(keys_names)

    metrics ={}
    
    for kn in keys_names:
        results = [ evaluate_res[kn] for _, evaluate_res in eval_metrics]
        metrics[kn] = np.mean(results)

    return metrics


