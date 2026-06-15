import os
import sys
import glob
import json
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
survival_models_list = ["cox","rsf","gbs"]

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
        client = cox.client.get_client(config, data)
    elif model == "rsf":
        client = rsf.client.get_client(config, data)
    elif model == "gbs":
        client = gbs.client.get_client(config, data)
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
        for line in message.splitlines():
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        pass

def CheckClientConfig(config):
    # Compaibilidad de logistic regression y elastic net con sus parámetros
    assert config["task"] in ["classification","regression","none","survival"], "Task not valid"

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
                config["model"] = "linear_regression"
                config["penalty"] = "l1"
            elif config["model"] == "ridge_regression":
                config["model"] = "linear_regression"
                config["penalty"] = "l2"
            elif config["model"] == "linear_regression_elasticnet":
                config["model"] = "linear_regression"
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
    elif config["model"] in survival_models_list:
        config["dataset"] = "survival"

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
#    config["data_file"] = "/home/jorge/workdir/flcore-suite/dataset/bucarest_sintetico/synthetic_dt4h_dataset.csv"

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

    if len(config["target_labels"]) == 0 and config["task"] != "survival":
        print("No target labels were provided")
        sys.exit()

    new = []        
    for i in config["target_labels"]:
        parsed = i.replace("]", "").replace("[", "").replace(",", "")
        new.append(parsed)
    config["target_labels"] = new

# ____________________________________________________________________
    with open(config["metadata_file"]) as f:
        meta = json.load(f)

    entries = meta.get("entries", [])
    if entries:
        entry = entries[0]
        feature_stats = entry["datasetStats"]["featureStats"]
        outcome_stats = entry["datasetStats"]["outcomeStats"]
        features_meta = {o["name"]: o for o in entry["features"]}
        outcomes_meta = {o["name"]: o for o in entry["outcomes"]}
    else:
        dataset_stats = meta.get("datasetStats", {})
        feature_stats = dataset_stats.get("featureStats", {})
        outcome_stats = dataset_stats.get("outcomeStats", {})
        features_meta = {o["name"]: o for o in meta.get("features", [])}
        outcomes_meta = {o["name"]: o for o in meta.get("outcomes", [])}
# ____________________________________________________________________

    n_out = 0
    for target in config["target_labels"]:
        if target in outcomes_meta.keys():
            dtype = outcomes_meta[target]["dataType"]
            stats = outcome_stats.get(target, {})

        elif target in features_meta.keys():
            dtype = features_meta[target]["dataType"]
            stats = feature_stats.get(target, {})
        else:
            raise ValueError(f"Target {target} no encontrado en metadata['outcomes']")

        if dtype == "BOOLEAN":
            n_out += 1
        elif dtype == "NOMINAL":
            n_out += len(stats.get("valueSet", []))
        elif dtype == "NUMERIC":
            n_out += 1

        if config["task"] == "regression":
            if dtype != "NUMERIC":
                raise ValueError(
                    f"Inconsistent configuration: task='regression' but target '{target}' "
                    f"has dtype '{dtype}'. Regression requires NUMERIC targets."
                )

        elif config["task"] == "classification":
            if dtype == "NUMERIC":
                raise ValueError(
                    f"Inconsistent configuration: task='classification' but target '{target}' "
                    f"has dtype NUMERIC. Classification requires BOOLEAN or NOMINAL targets."
                )

    config["n_out"] = n_out
    config["n_feats"] = len(config["train_labels"])

    if config["model"] in ["svm","svr","lsvr"]:
        if config["task"] == "regression":
            if config["kernel"] in ["poly", "rbf", "sigmoid", "precomputed"] and config["n_out"] > 1:
                print("Those kernels only support 1-variable as output")
                sys.exit()

    if config["model"] in survival_models_list:
        if config["time_col"] == "None" or config["event_col"] == "None":
            print("Time col and Event col needed when survival model is choosen")
            sys.exit()
        else:
            config["survival"] = {}
            config["survival"]["time_col"] = config["time_col"]
            config["survival"]["event_col"] = config["event_col"]
            config['survival']['negative_duration_strategy'] = config["negative_duration_strategy"]

    # Create experiment directory
    experiment_dir = Path(os.path.join(config["sandbox_path"],config["experiment_name"]))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config["experiment_dir"] = experiment_dir

# CUANDO SURVIVAL MODEL TASK NO ES NECESAIRO

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

    # si XGB train_method debe ser bagging o cyclicç    
    if config["model"] == "xgb":
        if config["strategy"] != "bagging":
            config["strategy"] = "bagging"
# Tendriamos que añadir que se verifique que las tasks sean consistentes con los label y el tipo de dato
    return config


def log_detailed_error(stage_name, exception, config=None, X=None, y=None, data_path=None):
    """
    Logs a highly visible and detailed error message, including system information,
    configuration variables, dataset characteristics, and a clean traceback.
    """
    import os
    import sys
    import logging
    import traceback
    import numpy as np
    import pandas as pd

    logger = logging.getLogger("ERROR_DIAGNOSTICS")
    
    border = "=" * 80
    logger.error(border)
    logger.error(f"  CRITICAL ERROR IN STAGE: {stage_name.upper()}  ".center(80, "="))
    logger.error(border)
    
    # Error Message
    logger.error(f"Error Type: {type(exception).__name__}")
    logger.error(f"Error Message: {str(exception)}")
    logger.error(border)
    
    # Configuration Diagnostics
    if config:
        logger.error("  CONFIGURATION PARAMETERS  ".center(80, "-"))
        for key in ["model", "task", "dataset", "data_id", "data_path", "train_labels", "target_labels", "train_size"]:
            if key in config:
                logger.error(f"  * {key}: {config[key]}")
        logger.error(border)

    # Data Diagnostics
    if X is not None or y is not None:
        logger.error("  DATASET DIAGNOSTICS  ".center(80, "-"))
        
        # Diagnostics for X
        if X is not None:
            if isinstance(X, (pd.DataFrame, pd.Series)):
                logger.error(f"  * X type: {type(X)}")
                logger.error(f"  * X shape: {X.shape}")
                logger.error(f"  * X columns: {list(X.columns) if hasattr(X, 'columns') else 'No columns'}")
                # Check NaNs
                nan_cols = X.isna().sum()
                nan_total = nan_cols.sum()
                logger.error(f"  * X total missing (NaN) values: {nan_total}")
                if nan_total > 0:
                    logger.error(f"    - Columns with NaNs: {nan_cols[nan_cols > 0].to_dict()}")
                # Check infinite values (for numeric columns only)
                num_cols = X.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    try:
                        inf_total = np.isinf(X[num_cols]).sum().sum()
                        logger.error(f"  * X total infinite values: {inf_total}")
                    except Exception:
                        pass
            elif isinstance(X, np.ndarray):
                logger.error(f"  * X type: numpy.ndarray")
                logger.error(f"  * X shape: {X.shape}")
                try:
                    nan_total = np.isnan(X).sum()
                    logger.error(f"  * X total missing (NaN) values: {nan_total}")
                    if np.issubdtype(X.dtype, np.number):
                        logger.error(f"  * X total infinite values: {np.isinf(X).sum()}")
                except Exception:
                    pass
            else:
                logger.error(f"  * X type (raw): {type(X)}")
                try:
                    logger.error(f"  * X length: {len(X)}")
                except Exception:
                    pass

        # Diagnostics for y
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                logger.error(f"  * y type: {type(y)}")
                logger.error(f"  * y shape: {y.shape}")
                try:
                    nan_total = y.isna().sum().sum() if isinstance(y, pd.DataFrame) else y.isna().sum()
                    logger.error(f"  * y total missing (NaN) values: {nan_total}")
                except Exception:
                    pass
                # Class / label distribution
                try:
                    unique_vals = y.value_counts().to_dict()
                    logger.error(f"  * y class distribution / values: {unique_vals}")
                except Exception:
                    pass
            elif isinstance(y, np.ndarray):
                logger.error(f"  * y type: numpy.ndarray")
                logger.error(f"  * y shape: {y.shape}")
                try:
                    logger.error(f"  * y total missing (NaN) values: {np.isnan(y).sum()}")
                except Exception:
                    pass
                try:
                    vals, counts = np.unique(y, return_counts=True)
                    logger.error(f"  * y class distribution: {dict(zip(vals.tolist(), counts.tolist()))}")
                except Exception:
                    pass
            else:
                logger.error(f"  * y type (raw): {type(y)}")
                try:
                    logger.error(f"  * y length: {len(y)}")
                except Exception:
                    pass
        logger.error(border)

    # System/File diagnostics
    if data_path:
        logger.error("  FILE SYSTEM DIAGNOSTICS  ".center(80, "-"))
        logger.error(f"  * target file/dir path: {data_path}")
        try:
            exists = os.path.exists(data_path)
            logger.error(f"  * path exists: {exists}")
            if exists:
                logger.error(f"  * is file: {os.path.isfile(data_path)}")
                logger.error(f"  * is directory: {os.path.isdir(data_path)}")
        except Exception as file_err:
            logger.error(f"  * failed to run path checks: {file_err}")
        logger.error(border)

    # Detailed traceback
    logger.error("  DETAILED TRACEBACK  ".center(80, "-"))
    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    for line in "".join(tb_lines).splitlines():
        logger.error(f"    {line}")
    logger.error(border)