import sys
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
    # Compaibilidad de logistic regression y elastic net con sus parámetros
    linear_regression_models_list = ["linear_regression","lasso_regression","ridge_regression","linear_regression_elasticnet"]
    if config["model"] == "logistic_regression":
        if config["task"] == "classification":
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
        elif config["task"] == "regression":
            print("The nature of the selected ML models does not allow to perform regression")
            print("if you want to perform regression with a linear model you can change to linear_regression")
            sys.exit()
    elif config["model"] == "lsvc":
        if config["task"] == "classification":
            pass
            # verificar variables
        elif config["task"] == "regression":
            print("The nature of the selected ML models does not allow to perform regression")
            sys.exit()
    elif config["model"] in linear_regression_models_list:
        if config["task"] == "classification":
            print("The nature of the selected ML model does not allow to perform classification")
            print("if you want to perform classification with a linear model you can change to logistic_regression")
            sys.exit()
        elif config["task"] == "regression":
            if config["model"] == "lasso_regression":
                config["model"] == "linear_regression"
                config["penalty"] = "l1"
            elif config["model"] == "ridge_regression":
                config["model"] == "linear_regression"
                config["penalty"] = "l2"
            elif config["model"] == "linear_regression_elasticnet":
                config["model"] == "linear_regression"
                config["penalty"] = "elasticnet"
    elif config["model"] == "logistic_regression_elasticnet":
        if config["task"] == "classification":
            config["model"] = "logistic_regression"
            config["penalty"] = "elasticnet"
            config["solver"] = "saga"
        elif config["task"] == "regression":
            print("The nature of the selected ML model does not allow to perform regression despite its name")
            sys.exit()
    elif config["model"] == "nn":
        config["n_feats"] = len(config["train_labels"])
        config["n_out"] = 1 # Quizás añadir como parámetro también
    elif config["model"] == "xgb":
        pass
    return config