import numpy as np
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR, LinearSVR
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LinearMLParams = Union[XY, Tuple[np.ndarray]]
#LinearClassifier = Union[LogisticRegression, SGDClassifier]
XYList = List[XY]

def get_model(config):
    # Esto cubre clasificación con SVM y logistic regression con y sin elastic net
    if config["task"] == "classification":
        if config["model"] in ["lsvc","svm"]:
                #Linear classifiers (SVM, logistic regression, etc.) with SGD training.
                #If we use hinge, it implements SVM
                model = SGDClassifier(
                    max_iter=config["max_iter"],
                    n_iter_no_change=1000,
                    average=True,
                    random_state=config["seed"],
                    warm_start=True,
                    fit_intercept=True,
                    loss="hinge",
                    learning_rate='optimal')

        elif config["model"] == "logistic_regression":
                model = LogisticRegression(
                    penalty=config["penalty"],
                    solver=config["solver"], #necessary param for elasticnet otherwise error
                    l1_ratio=config["l1_ratio"],#necessary param for elasticnet otherwise error
                    #max_iter=1,  # local epoch ==>> it doesn't work
                    max_iter=config["max_iter"],
                    warm_start=True,  # prevent refreshing weights when fitting
                    random_state=config["seed"])
    #                class_weight= config["class_weight"],
    # Aqui cubrimos regresión con modelo lineal
    elif config["task"] == "regression":
        # nos solicitan tambien el pearson coefficiente:
        # from scipy.stats import pearsonr
        if config["model"] == "linear_regression":
            if config["penalty"] == "elasticnet":
                model = ElasticNet(
                    alpha=1.0, 
                    l1_ratio=config["l1_ratio"],
                    fit_intercept=True,
                    precompute=False,
                    max_iter=config["max_iter"],
                    copy_X=True,
                    tol=config["tol"],
                    warm_start=False,
                    positive=False,
                    random_state=config["seed"],
                    selection='cyclic')
            elif config["penalty"] == "l1":
                # ¿LASSOO?
                model = Lasso(
                    fit_intercept=True,
                    precompute=False,
                    copy_X=True,
                    max_iter=config["max_iter"],
                    tol=config["tol"],
                    warm_start=False,
                    positive=False,
                    random_state=config["seed"],
                    selection='cyclic')
            elif config["penalty"] == "l2":
                # ¿RIDGE?
                model = Ridge(
                    fit_intercept=True,
                    copy_X=True,
                    max_iter=config["max_iter"],
                    tol=config["tol"],
                    solver='auto',
                    positive=False,
                    random_state=config["seed"],
                    )
            elif config["penalty"] == "none" or config["penalty"] == None: 
                model = LinearRegression()
        elif config["model"] in ["svm", "svr"]:
            if config["kernel"] == "linear":
                model = LinearSVR(
                epsilon=0.0,
                tol=config["tol"],
                C=1.0,
                loss='epsilon_insensitive',
                fit_intercept=True,
                intercept_scaling=1.0,
                dual='auto',
                verbose=0,
                random_state=None,
                max_iter=config["max_iter"])
            else:
                model = SVR(
                    #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
                    kernel=config["kernel"],
                    degree=3,
                    gamma=config["gamma"],
                    coef0=0.0,
                    tol=config["tol"],
                    C=1.0,
                    epsilon=0.1,
                    shrinking=True,
                    cache_size=200,
                    verbose=False,
                    max_iter=config["max_iter"])
            
    else:
        # Invalid combinations: already managed by sanity check
        print("COMBINACIóN NO VÁLIDA: no debió llegar aquí")
        pass

    return model

def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model."""
    # AQUI DEBE DEVOLVER TAMBIEN PARA EL linear regression y los demas
    # AQUI FALLA POR ESO
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


def set_model_params(model, params):
    """Sets the parameters of a sklean LogisticRegression model."""
    # SUPONGO QUE AQUI TAMBIEN
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    #For feature selection
    # model.features = params[2].astype(bool)  
    return model


def set_initial_params(model,config):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """    
    #n_classes = 2  # MNIST has 10 classes
    n_classes = config["n_out"]  # MNIST has 10 classes
    n_features = config["n_feats"]
    #n_features = 9  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    if config["model"] == "logistic_regression": # buscar modelos compatibles
        model.coef_ = np.zeros((n_classes, n_features))
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,))
    elif config["model"] == "linear_regression": # idem
        model.coef_ = np.zeros((n_classes,n_features))
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,))
# .............................................................................................
    elif config["model"] in ["lsvc","svm","svr"]:
        if config["task"] == "classification":
            model.coef_ = np.zeros((n_classes, n_features))
            if model.fit_intercept:
                model.intercept_ = 0 
        elif config["task"] == "regression":
            if config["kernel"] == "linear":
                model.coef_ = np.zeros((n_classes, n_features))
                if model.fit_intercept:
                    model.intercept_ = 0 
            else:
                model.coef_ = np.zeros((1, n_features))
                if model.fit_intercept:
                    model.intercept_ = 0 

        #coef_ : of shape (1, n_features) if n_classes == 2 else (n_classes, n_features)
        model.coef_ = np.zeros((n_classes, n_features))
        if model.fit_intercept:
            model.intercept_ = 0 
    elif config["model"] in ["svm", ]:
        # parece que no encuentra los parametros:
        # 2025-12-20 15:21:35,575 - STDERR - ERROR - can't set attribute 'coef_'

        pass
    else:
        pass
# .............................................................................................

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
        
