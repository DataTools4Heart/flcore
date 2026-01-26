import os
import sys
import glob
import numpy as np
from pathlib import Path

import flcore.models.linear_models as linear_models
import flcore.models.xgb as xgb
import flcore.models.random_forest as random_forest
import flcore.models.weighted_random_forest as weighted_random_forest
import flcore.models.nn as nn

#import flcore.models.logistic_regression.server as logistic_regression_server
#import flcore.models.logistic_regression.server as logistic_regression_server
import flcore.models.xgb.server as xgb_server
import flcore.models.random_forest.server as random_forest_server
import flcore.models.linear_models.server as linear_models_server
import flcore.models.weighted_random_forest.server as weighted_random_forest_server
import flcore.models.nn.server as nn_server
import flcore.models.cox.server as cox_server
import flcore.models.rsf.server as rsf_server
import flcore.models.gbs.server as gbs_server

import flcore.models.cox as cox
import flcore.models.rsf as rsf
import flcore.models.gbs as gbs

linear_models_list = ["logistic_regression", "linear_regression", "lsvc", "svr", "svm",
                      "lasso_regression", "ridge_regression","logistic_regression_elasticnet"]
linear_regression_models_list = ["linear_regression","lasso_regression", "svr", "svm",
                        "ridge_regression","linear_regression_elasticnet"]


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
    elif model == "cox":
        client = cox.client.get_client(config, data, client_id)
    elif model == "rsf":
        client = rsf.client.get_client(config, data, client_id)
    elif model == "gbs":
        client = gbs.client.get_client(config, data, client_id)
    else:
        raise ValueError(f"Unknown model: {model}")
    return client

def GetModelServerStrategy(config):
    model = config["model"]
    if model in linear_models_list:
        server, strategy = linear_models_server.get_server_and_strategy(config)
    elif model == "random_forest":
        server, strategy = random_forest_server.get_server_and_strategy(config)
    elif model == "weighted_random_forest":
        server, strategy = weighted_random_forest_server.get_server_and_strategy(config)
    elif model == "xgb":
        server, strategy = xgb_server.get_server_and_strategy(config) #, data)
    elif model == "nn":
        server, strategy = nn_server.get_server_and_strategy(config)
    elif model == "cox":
        server, strategy = cox_server.get_server_and_strategy(config)
    elif model == "rsf":
        server, strategy = rsf_server.get_server_and_strategy(config)
    elif model == "gbs":
        server, strategy = gbs_server.get_server_and_strategy(config)
    else:
        raise ValueError(f"Unknown model: {model}")

    return server, strategy

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        for line in message.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def CheckClientConfig(config):
    # Compaibilidad de logistic regression y elastic net con sus parámetros
    assert config["task"] in ["classification","regression","none"], "Task not valid"

    if config["model"] == "logistic_regression":
        if (config["task"] == "classification" or config["task"].lower() == "none"):
            if config["task"].lower() == "none":
                print("Since this model only supports classification assigning task automatically to classification")
                config["task"] = "classification"
            if config["penalty"] == "none":
                print("LogisticRegression requieres a penalty and no input given, setting penalty to default L2")
                config["penalty"] = "l2"
                config["l1_ratio"] = 0
            elif config["penalty"] == "elasticnet":
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
        if (config["task"] == "classification"  or config["task"].lower() == "none"):
            if config["task"].lower() == "none":
                print("Since this model only supports classification assigning task automatically to classification")
            pass
            # verificar variables
        elif config["task"] == "regression":
            print("The nature of the selected ML models does not allow to perform regression")
            sys.exit()
    elif config["model"] in linear_regression_models_list:
        if config["task"] == "classification" and config["model"] != "svm":
            print("The nature of the selected ML model does not allow to perform classification")
            print("if you want to perform classification with a linear model you can change to logistic_regression")
            sys.exit()
        elif (config["task"] == "regression"  or config["task"].lower() == "none"):
            if config["task"].lower() == "none":
                print("Since this model only supports regression assigning task automatically to regression")

            if config["model"] == "lasso_regression":
                config["model"] == "linear_regression"
                config["penalty"] = "l1"
            elif config["model"] == "ridge_regression":
                config["model"] == "linear_regression"
                config["penalty"] = "l2"
            elif config["model"] == "linear_regression_elasticnet":
                config["model"] == "linear_regression"
                config["penalty"] = "elasticnet"
            elif config["model"] == "svm":
                if config["kernel"] != "linear":
                    print("The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to datasets")
                    print("with more than a couple of 10000 samples. Changing kernel for linear")
                    config["kernel"] = "linear"
    elif config["model"] == "logistic_regression_elasticnet":
        if (config["task"] == "classification"  or config["task"].lower() == "none"):
            if config["task"].lower() == "none":
                print("Since this model only supports classification assigning task automatically to classification")

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

    est = config["data_id"]
    id = est.split("/")[-1]
#    dir_name = os.path.dirname(config["data_id"])
    dir_name_parent = str(Path(config["data_id"]).parent)

#    config["metadata_file"] = os.path.join(dir_name_parent,"metadata.json")
    config["metadata_file"] = os.path.join(est,"metadata.json")

    pattern = "*.parquet"
    parquet_files = glob.glob(os.path.join(est, pattern))
    # Saniy check, empty list
    if len(parquet_files) == 0:
        print("No parquet files found in ",est)
        sys.exit()

    # ¿How to choose one of the list?
    config["data_file"] = parquet_files[-1]

    if len(config["train_labels"]) == 0:
        print("No training labels were provided")
        sys.exit()

    new = []
    for i in config["train_labels"]:
        parsed = i.replace("]", "").replace("[", "").replace(",", "")
        new.append(parsed)
    config["train_labels"] = new

    if len(config["target_labels"]) == 0:
        print("No target labels were provided")
        sys.exit()

    new = []        
    for i in config["target_labels"]:
        parsed = i.replace("]", "").replace("[", "").replace(",", "")
        new.append(parsed)
    config["target_labels"] = new

    # VERIFICAR EL TASK SI HACE FALTA CAMBIARLO SEGUN EL NUMERO DE LABELS, binario bmulticlase¿?¿?¿?¿?
    config["n_feats"] = len(config["train_labels"])
    config["n_out"] = len(config["target_labels"])

    if config["model"] in ["svm","svr","lsvr"]:
        if config["task"] == "regression":
            if config["kernel"] in ["poly", "rbf", "sigmoid", "precomputed"] and config["n_out"] > 1:
                print("Those kernels only support 1-variable as output")
                sys.exit()

    if config["task"].lower() == "none":
        print("Task not assigned. The  ML model  selection requieres a task to perform")
        sys.exit()  

    if config["penalty"] != "none":
        valid_values = ["l1", "l2"]
        if config["model"] in linear_models_list:
            valid_values.append("elasticnet")
        elif config["model"] == "nn":
            valid_values.append("SmoothL1Loss")
        elif config["model"] == "random_forest":
            print("Random forest does not admit L1, L2 or ElasticNet regularization ... ignoring this variable")
            sys.exit()
        assert config["penalty"] in valid_values, "Penalty is not valid or available for the selected model"
    return config


def CheckServerConfig(config):
    assert isinstance(config['num_clients'], int), 'num_clients should be an int'
    assert isinstance(config['num_rounds'], int), 'num_rounds should be an int'
    if(config['smooth_method'] != 'None'):
        assert config['smoothing_strenght'] >= 0 and config['smoothing_strenght'] <= 1, 'smoothing_strenght should be betwen 0 and 1'
    #if(config['dropout_method'] != 'None' or config["dropout_method"] is not None):
    #    assert config['percentage_drop'] >= 0 and config['percentage_drop'] < 100, 'percentage_drop should be betwen 0 and 100'

    assert (config['smooth_method']== 'EqualVoting' or \
        config['smooth_method']== 'SlowerQuartile' or \
        config['smooth_method']== 'SsupperQuartile' or \
        config['smooth_method']== 'None'), 'the smooth methods are not correct: EqualVoting, SlowerQuartile and SsupperQuartile'

    """if(config['model'] == 'weighted_random_forest'):
         assert (config['weighted_random_forest']['levelOfDetail']== 'DecisionTree' or \
            config['weighted_random_forest']['levelOfDetail']== 'RandomForest'), 'the levels of detail for weighted RF are not correct: DecisionTree and RandomForest '
    """
# _________________________________________________________________________________________________--
    if config["min_fit_clients"] == 0:
        config["min_fit_clients"] = config["num_clients"]
    if config["min_evaluate_clients"] == 0:
        config["min_evaluate_clients"] = config["num_clients"]
    if config["min_available_clients"] == 0:
        config["min_available_clients"] = config["num_clients"]

    # Specific for models:
    if config["model"] == "random_forest":
        assert isinstance(config['balanced'], str), 'Balanced is a parameter required when random forest model is used '
        assert config["balanced"].lower() == "true" or config["balanced"].lower() == "false", "Balanced is required to be True or False "
        assert isinstance(config["task"], str), "Task is a parameter required when random forest model is used"
    """
    Se tendrían que añadir también
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--max_depth", type=int, default=2, help="Max depth")
    parser.add_argument("--class_weight", type=str, default="balanced", help="Class weight")
    parser.add_argument("--levelOfDetail", type=str, default="DecisionTree", help="Level of detail")
    parser.add_argument("--regression_criterion", type=str, default="squared_error", help="Criterion for training")
    """
    if config["strategy"] == "UncertaintyWeighted":
        if config["model"] == "nn":
            pass
        else:
           print("UncertaintyWeighted is only available for NN")
           print("Changing strategy to FedAvg")
           config["strategy"] = "FedAvg"

    # si XGB train_method debe ser bagging o cyclic
# Tendriamos que añadir que se verifique que las tasks sean consistentes con los label y el tipo de dato
    return config