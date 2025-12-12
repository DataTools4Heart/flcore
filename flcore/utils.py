import numpy as np

import flcore.models.linear_models as linear_models
import flcore.models.xgb as xgb
import flcore.models.random_forest as random_forest
import flcore.models.weighted_random_forest as weighted_random_forest
import flcore.models.nn as nn

linear_models_list = ["logistic_regression", "linear_regression", "lsvc",
                      "lasso_regression", "ridge_regression"]


def GetModelClient(config, data):
    model = config["model"]
    if model in linear_models_list:
        client = linear_models.client.get_client(config,data)

    elif model == "random_forest":
        client = random_forest.client.get_client(config,data) 
    
    elif model == "weighted_random_forest":
        client = weighted_random_forest.client.get_client(config,data)

    elif model == "xgb":
        client = xgb.client.get_client(config, data)

    elif model == "nn":
        client = nn.client.get_client(config, data)

    else:
        raise ValueError(f"Unknown model: {model}")

    return client

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        for line in message.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def SanityCheck(config):
    ###################### AQUI HAY QUE PONER LO DEL SANITY CHECK, concordancia entre task, modelo, etc
    """
    # Compaibilidad de logistic regression y elastic net con sus parámetros
    if config["model"] == "logistic_regression":
        if config["penalty"] == "elasticnet":
            if config["solver"] != "saga":
                config["solver"] = "saga"
            if config["l1_ratio"] == 0:
                print("Degenerate case equivalent to Penalty L1")
            elif config["l1_ratio"] == 1:
                print("Degenerate case equivalent to Penalty L2")
        if config["penalty"] == "L1":
            if config["l1_ratio"] != 0:
               config["l1_ratio"] = 0
            elif config["l1_ratio"] != 1:
               config["l1_ratio"] = 1                
            
        En el sanity check hay que poner que el uncertainty aware es solamente para NN
        Solvers como 'newton-cg', 'sag', 'lbfgs' — sólo soportan L2 o ninguna penalización. 

        Solvers 'liblinear' — soportan L1 y L2 (pero no elasticnet). 

        Solver 'saga' — soporta L1, L2 y elasticnet, por lo que es el más flexible entre ellos. 
    # Disponibilidad de clasificación / regresión según el modelo

    if config["model"] in ["lsvc", "logistic_regression"]:
        if config["task"] == "regression":
            print("The nature of the selected ML models does not allow to perform regression")
            print("if you want to perform regression with a linear model you can change to linear regression)
            # sys.exit()
    elif config["model"] == "linear_regression":
        if config["task"] == "classification":
            print("The nature of the selected ML model does not allow to perform classification")
            # sys.exit()
    """

    if config["model"] in ("logistic_regression", "elastic_net", "lsvc"):
        config["linear_models"] = {}
        n_feats = len(config["train_labels"])
        config['linear_models']['n_features'] = n_feats # config["n_features"]
        config["held_out_center_id"] = -1
    elif config["model"] == "nn": # in ("nn", "BNN"):
        config["n_feats"] = len(config["train_labels"])
        config["n_out"] = 1 # Quizás añadir como parámetro también
        config["dropout_p"] = config["neural_network"]["dropout_p"]
        config["device"] = config["neural_network"]["device"]
        config["batch_size"] = 32
        config["lr"] = 1e-3
        config["local_epochs"] = config["neural_network"]["local_epochs"]
# **************************************************************************************************************
#    parser.add_argument("--xgb", type=json.loads, default={"batch_size": 32,"num_iterations": 100,"task_type": "BINARY","tree_num": 500}, help="XGB parameters")
    elif config["model"] == "xgb":
        pass
# **************************************************************************************************************

    return config