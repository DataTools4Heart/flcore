from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression,SGDClassifier

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LinearMLParams = Union[XY, Tuple[np.ndarray]]
LinearClassifier = Union[LogisticRegression, SGDClassifier]
XYList = List[XY]


def get_model(model_name):
    match model_name:
        case "lsvc":
            #Linear classifiers (SVM, logistic regression, etc.) with SGD training.
            #If we use hinge, it implements SVM
            model = SGDClassifier( max_iter= 100000,n_iter_no_change=1000,average=True,random_state=42,class_weight= "balanced",warm_start=True,fit_intercept=True,loss="hinge", learning_rate='optimal')
        case "logistic_regression":
            model = LogisticRegression(
            penalty="l2",
            #max_iter=1,  # local epoch ==>> it doesn't work
            max_iter=100000,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
            class_weight= "balanced" #For unbalanced
        )
        case "elastic_net":
            model = LogisticRegression(
            l1_ratio=0.5,#necessary param for elasticnet otherwise error
            penalty="elasticnet",
            solver='saga', #necessary param for elasticnet otherwise error
            #max_iter=1,  # local epoch ==>> it doesn't work
            max_iter=100000,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
            class_weight= "balanced" #For unbalanced
        )

    
    return model

def get_model_parameters(model: LinearClassifier) -> LinearMLParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
            #For feature selection
            # model.features.astype(bool)
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LinearClassifier, params: LinearMLParams
) -> LinearClassifier:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    #For feature selection
    # model.features = params[2].astype(bool)  
    return model


def set_initial_params(model: LinearClassifier,n_features):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """    
    n_classes = 2  # MNIST has 10 classes
    #n_features = 9  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    if(isinstance(model,SGDClassifier)==True):
        model.coef_ = np.zeros((1, n_features))
        if model.fit_intercept:
            model.intercept_ = 0 
    else:
        model.coef_ = np.zeros((n_classes, n_features))
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,))


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
        #print(f"Metric {kn} in aggregation evaluate: {metrics[kn]}\n")

    # filename = 'server_results.txt'
    # with open(
    # filename,
    # "a",
    # ) as f:
    #     f.write(f"Accuracy: {metrics['accuracy']} \n")
    #     f.write(f"Sensitivity: {metrics['sensitivity']} \n")
    #     f.write(f"Specificity: {metrics['specificity']} \n")

    return metrics
        
